"""Build a TRAIN/VAL cache = official rows + raw-CE spatial features at the same timestamps.

The official cache stays the single source of devices, timestamps, labels and follow-up;
raw events only add columns. Devices without a raw CSV keep NaN raw features (abstention,
not zero). Multiprocess over cache shards; deterministic output; manifest hashes files.
"""
import argparse
import io
import json
import os
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import raw_events

def representation(name):
    require(name in ("v1","spatial_v3"), "Unknown representation")
    if name=="spatial_v3":
        import raw_spatial_v3
        return raw_spatial_v3
    return raw_events

from common import read, save, sha, digest, require
from history_math import verify

META = ['device', 'time', 'failure', 'observed_start', 'observed_end', 'history_rows']


def _raw_reader(source):
    source = Path(source)
    if source.suffix == '.zip':
        z = zipfile.ZipFile(source)
        names = {Path(n).stem: n for n in z.namelist() if n.startswith('type_A/') and n.endswith('.csv')}
        return names, lambda sn: z.read(names[sn])
    names = {p.stem: p for p in source.glob('*.csv')}
    return names, lambda sn: names[sn].read_bytes()


def _shard(args):
    part, src, dst, raw_source, sn_names, cutoff, rep = args
    raw_events = representation(rep)
    names, load = _raw_reader(raw_source)
    table = pq.read_table(src).to_pandas()
    require((np.diff(table.device.to_numpy()) >= 0).all(), 'Cache rows must be sorted by device')
    extra = np.full((len(table), len(raw_events.FEATURES)), np.nan, np.float32)
    stats = dict(devices=0, with_raw=0, raw_events=0, raw_events_after_cutoff=0)
    codes = table.device.to_numpy()
    cuts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1, len(table)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        stats['devices'] += 1
        sn = sn_names[int(codes[a])]
        if sn not in names:
            continue
        frame = pd.read_csv(io.BytesIO(load(sn)), usecols=raw_events.RAW_COLUMNS,
                            dtype={'burst_info': str, 'error_type_full_name': str}, low_memory=False)
        # Never read beyond the cache cutoff (TEST period stays unopened).
        late = pd.to_numeric(frame.LogTime) >= cutoff
        stats['raw_events_after_cutoff'] += int(late.sum())
        frame = frame.loc[~late]
        ev = raw_events.events(frame)
        stats['with_raw'] += 1
        stats['raw_events'] += len(ev)
        times = table.time.to_numpy()[a:b]
        require((np.diff(times) >= 0).all(), 'Device rows must be time-sorted')
        extra[a:b] = raw_events.features_at(ev, times)
    for name in raw_events.FEATURES:
        require(name not in table, 'Raw feature collides with official column: ' + name)
    table = pd.concat([table, pd.DataFrame(extra, columns=raw_events.FEATURES, index=table.index)], axis=1)
    keep = [c for c in table.columns if c not in META] + META
    pq.write_table(pa.Table.from_pandas(table[keep], preserve_index=False), dst, compression='zstd',
                   row_group_size=65536)
    stats['part'] = part
    return stats


def smoke(source_cache, raw_source, out, parts, rep="v1"):
    """Process the first `parts` shards into a scratch folder (no manifest) and report stats."""
    source_cache, out = Path(source_cache), Path(out)
    m = verify(source_cache)
    out.mkdir(parents=True, exist_ok=True)
    registry = pd.read_parquet(source_cache / 'device_metadata.parquet')
    sn_names = registry.set_index('device').sn_name.to_dict()
    files = sorted((source_cache / 'rows').glob('part-*.parquet'))[:parts]
    tick = time.monotonic()
    for p in files:
        s = _shard((p.name, str(p), str(out / p.name), str(raw_source), sn_names, m['windows']['val'][1], rep))
        print('SMOKE', json.dumps(s), f'{time.monotonic()-tick:.0f}s', flush=True)


def build(source_cache, raw_source, out, workers, final_test=False, rep="v1"):
    """final_test: source_cache is the official TEST cache (test_opened); raw events up to the
    TEST end are used, the parser and features are identical (frozen raw_events module)."""
    source_cache, out = Path(source_cache), Path(out)
    m = verify(source_cache)
    raw_events = representation(rep)
    spec_hashes = {Path(raw_events.__file__).name:sha(Path(raw_events.__file__)), 'raw_events.py':sha(Path(representation('v1').__file__))}
    require(out.resolve()!=source_cache.resolve() and not out.resolve().is_relative_to(source_cache.resolve()), 'Output must be outside source cache')
    require(bool(m['test_opened']) == bool(final_test), 'Cache partition does not match --final-test')
    if (out / 'manifest.json').exists():
        done = verify(out)
        require(done['raw_feature_spec']['source_manifest'] == sha(source_cache / 'manifest.json'), 'Raw cache source changed')
        require(done['raw_feature_spec'].get('representation', 'v1')==rep, 'Representation changed: use a new output directory')
        require(done['raw_feature_spec'].get('parser_hashes')==spec_hashes, 'Parser identity differs: use a new output directory')
        print('RAW_CACHE_REUSED', out, flush=True)
        return done
    tmp = out.with_name(out.name + '.partial')
    require(not out.exists() and not tmp.exists(), 'Output or partial cache exists; preserve it and choose a new output path')
    (tmp / 'rows').mkdir(parents=True)
    registry = pd.read_parquet(source_cache / 'device_metadata.parquet')
    sn_names = registry.set_index('device').sn_name.to_dict()
    cutoff = m['windows']['test' if final_test else 'val'][1]
    parts = sorted((source_cache / 'rows').glob('part-*.parquet'))
    jobs = [(p.name, str(p), str(tmp / 'rows' / p.name), str(raw_source), sn_names, cutoff, rep) for p in parts]
    tick = time.monotonic()
    totals = dict(devices=0, with_raw=0, raw_events=0, raw_events_after_cutoff=0)
    with ProcessPoolExecutor(workers) as pool:
        for i, s in enumerate(pool.map(_shard, jobs), 1):
            for k in totals:
                totals[k] += s[k]
            if i % 10 == 0 or i == len(jobs):
                print(f'RAW_CACHE_PROGRESS shards={i}/{len(jobs)} devices={totals["devices"]} '
                      f'with_raw={totals["with_raw"]} seconds={int(time.monotonic()-tick)}', flush=True)
    for name in ('device_metadata.parquet', 'devices.json', 'excluded.json'):
        shutil.copy2(source_cache / name, tmp / name)
    manifest = dict(m)
    manifest['features'] = m['features'] + raw_events.FEATURES
    manifest['raw_feature_spec'] = dict(version=raw_events.VERSION, representation=rep, parser_hashes=spec_hashes, source_manifest=sha(source_cache / 'manifest.json'),
                                        raw_source=str(raw_source), cutoff=cutoff, **totals,
                                        final_test=bool(final_test), raw_events_module_sha256=sha(Path(raw_events.__file__)),
                                        policy='Source CSVs are read, then rows at/after the partition end are discarded before feature computation. Features use events at or before each anchor.')
    manifest['raw_feature_descriptions'] = raw_events.mapping_features()
    manifest['files'] = {p.relative_to(tmp).as_posix(): sha(p) for p in tmp.rglob('*.parquet')}
    manifest['files'].update({n: sha(tmp / n) for n in ('devices.json', 'excluded.json')})
    save(tmp / 'manifest.json', manifest)
    tmp.rename(out)
    print('RAW_CACHE_READY', out, json.dumps(totals), flush=True)
    return verify(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-cache', required=True)
    p.add_argument('--raw', required=True, help='stage1_csv.zip or a directory of type_A CSVs')
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument('--smoke-parts', type=int, default=0)
    p.add_argument('--final-test', action='store_true')
    p.add_argument('--representation', choices=['v1','spatial_v3'], default='v1')
    a = p.parse_args()
    if a.smoke_parts:
        smoke(a.source_cache, a.raw, a.out, a.smoke_parts,a.representation)
    else:
        build(a.source_cache, a.raw, a.out, a.workers, a.final_test,a.representation)
