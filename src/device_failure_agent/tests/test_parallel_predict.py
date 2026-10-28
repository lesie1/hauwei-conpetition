"""Parallel scoring (score_workers>1) must reproduce the serial predict() exactly."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from common import save, sha, DAY
from settings import DEFAULTS, policy
from history_math import Writer
from experiment_contract import initial_plans
from prediction import predict

T = 1704067200


def build(root, domain, n=90):
    cache = root / 'cache'
    writer = Writer(cache / 'rows')
    registry = []
    rng = np.random.default_rng(3)
    for i in range(n):
        fd = 10 + (i * 5) % 45 if i % 4 == 0 else None
        failure = T + fd * DAY + 600 * (i % 9) if fd else -1
        end = failure if fd else T + 60 * DAY
        registry.append(dict(device=i, sn_name=str(i), failure=failure, observed_start=T, observed_end=end))
        if i % 17 == 5:   # registry devices with no rows at all
            continue
        ts = np.unique(np.sort(T + rng.integers(0, (fd or 60) * DAY, 20 + 3 * i))).astype(np.int64)
        if fd: ts = ts[ts < failure]
        x = (np.maximum(0, 5 - (failure - ts) / DAY) + rng.random(len(ts))) if fd else rng.random(len(ts)) * 2
        writer.add(pd.DataFrame(dict(x=x.astype(np.float32), z=rng.random(len(ts)).astype(np.float32), device=i, time=ts,
                                     failure=failure, observed_start=T, observed_end=end, history_rows=np.arange(1, len(ts) + 1))))
        if i % 7 == 6: writer.flush()   # many shards, never splitting a device (as the normalizer writes)
    writer.flush()
    pd.DataFrame(registry).to_parquet(cache / 'device_metadata.parquet', index=False)
    save(cache / 'devices.json', [str(i) for i in range(n)]); save(cache / 'excluded.json', [])
    m = dict(features=['x', 'z'], included_devices=n, test_opened=False, coverage={'basis': 'Synthetic'},
             windows=dict(train=[T, T + 36 * DAY], val=[T + 36 * DAY, T + 48 * DAY], test=[T + 48 * DAY, T + 60 * DAY]))
    m['files'] = {p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob('*') if p.is_file()}
    save(cache / 'manifest.json', m)
    c = dict(DEFAULTS, **policy(domain), domain=domain, data_dir=str(root / 'raw'), workspace=str(root / 'w'),
             output_root=str(root / 's'), active_cache=str(cache), device='cpu', num_threads=1)
    return c, m


def model():
    rng = np.random.default_rng(0)
    x = rng.random((400, 2)).astype(np.float32); y = (x[:, 0] > .6).astype(int)
    return xgb.train(dict(objective='binary:logistic', max_depth=3, nthread=1), xgb.DMatrix(x, label=y), 5)


class ParallelPredict(unittest.TestCase):
    def test_identical_tables_and_hashes(self):
        b = model()
        for domain in ('memory', 'hdd'):
            with tempfile.TemporaryDirectory() as tmp:
                c, m = build(Path(tmp), domain)
                self.assertGreater(len(list((Path(c['active_cache']) / 'rows').glob('*.parquet'))), 5)
                recipes = []
                for kind in ('window_max', 'window_mean', 'window_second', 'latest', 'all_history_max'):
                    r = copy.deepcopy(initial_plans()[0]); r['score_aggregation'] = kind; recipes.append(r)
                for fs in ('history', 'recent'):
                    r = copy.deepcopy(initial_plans()[0])
                    r.update(sample_unit='device_day', feature_set=fs, score_aggregation='latest', prediction_interval_hours=24)
                    recipes.append(r)
                for r in recipes:
                    if r['feature_set'] != 'base':
                        from history_store import feature_names
                        n = len(feature_names(m['features'], r['feature_set']))
                        rng = np.random.default_rng(1); x = rng.random((300, n)).astype(np.float32)
                        bm = xgb.train(dict(objective='binary:logistic', max_depth=3, nthread=1), xgb.DMatrix(x, label=(x[:, 0] > .5).astype(int)), 4)
                    else:
                        bm = b
                    with self.subTest(domain=domain, kind=r['score_aggregation'], fs=r['feature_set']):
                        serial, s1 = predict(bm, dict(c, score_workers=1), r, m, 'val', threshold=.5)
                        par, s2 = predict(bm, dict(c, score_workers=4), r, m, 'val', threshold=.5)
                        self.assertEqual(s1, s2)
                        pd.testing.assert_frame_equal(serial, par)


if __name__ == '__main__':
    unittest.main()
