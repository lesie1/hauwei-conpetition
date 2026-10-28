"""Cross-validation scorer equals the frozen predict(); folds never leak devices; paired CV works."""
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
from sample_builder import build_training
import crossval

T = 1704067200


def fixture(root, n=40):
    cache = root / 'cache'
    writer = Writer(cache / 'rows')
    registry = []
    rng = np.random.default_rng(0)
    for i in range(n):
        fd = 8 + 3 * (i % 11) if i % 3 == 0 else None     # faults spread across TRAIN and VAL
        failure = T + fd * DAY + 3600 * (i % 5) if fd else -1
        horizon = (fd or 60)
        ts = np.sort(T + rng.integers(0, horizon * DAY, 30 + i)).astype(np.int64)
        ts = np.unique(ts)
        if fd:
            ts = ts[ts < failure]
        vals = (np.maximum(0, 6 - (failure - ts) / DAY) if fd else rng.random(len(ts)) * 2).astype(np.float32)
        end = failure if fd else T + 60 * DAY
        writer.add(pd.DataFrame(dict(x=vals, z=rng.random(len(ts)).astype(np.float32), device=i, time=ts,
                                     failure=failure, observed_start=T, observed_end=end,
                                     history_rows=np.arange(1, len(ts) + 1))))
        registry.append(dict(device=i, sn_name=str(i), failure=failure, observed_start=T, observed_end=end))
    writer.flush()
    pd.DataFrame(registry).to_parquet(cache / 'device_metadata.parquet', index=False)
    save(cache / 'devices.json', [str(i) for i in range(n)])
    save(cache / 'excluded.json', [])
    m = dict(features=['x', 'z'], included_devices=n, test_opened=False, coverage={'basis': 'Synthetic'},
             windows=dict(train=[T, T + 40 * DAY], val=[T + 40 * DAY, T + 50 * DAY], test=[T + 50 * DAY, T + 60 * DAY]))
    m['files'] = {p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob('*') if p.is_file()}
    save(cache / 'manifest.json', m)
    c = dict(DEFAULTS, **policy('memory'), domain='memory', data_dir=str(root / 'raw'), workspace=str(root / 'w'),
             output_root=str(root / 'session'), active_cache=str(cache), device='cpu', num_threads=1,
             memory_limit_gb=2, rounds=6, eval_every=2, patience=2, min_free_disk_gb=.01,
             selection_window=[T + 40 * DAY, T + 45 * DAY], confirmation_window=[T + 45 * DAY, T + 50 * DAY])
    (root / 'session').mkdir()
    return c, m


def booster(c, recipe, m, out, rounds=6):
    x, y, w, names, ratio, audit = build_training(c, recipe, m, out, 900, 2)
    dt = xgb.DMatrix(np.array(x), label=y, weight=w)
    del x
    p = dict(recipe['params']); p['eta'] = p.pop('learning_rate')
    p.update(objective='binary:logistic', tree_method='hist', nthread=1, scale_pos_weight=ratio, base_score=.5, seed=1)
    return xgb.train(p, dt, rounds)


class CrossValidation(unittest.TestCase):
    def test_scorer_matches_predict_for_every_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c, m = fixture(root)
            recipe = initial_plans()[0]
            out = root / 'fit'; out.mkdir()
            b = booster(c, recipe, m, out)
            window = [T + 30 * DAY, T + 35 * DAY]
            plist = [[T + 20 * DAY, T + 25 * DAY], window, [T + 35 * DAY, T + 40 * DAY]]
            steps = [2, 4, 6]
            for kind in ('window_max', 'window_mean', 'window_second', 'latest', 'all_history_max'):
                r = copy.deepcopy(recipe); r['score_aggregation'] = kind
                got = crossval.score_devices(b, steps, c, r, m, plist, list(range(40)))
                for s in steps:
                  for pi, w in enumerate(plist):
                    with self.subTest(kind=kind, step=s, period=pi):
                        mm = dict(m); mm['windows'] = dict(m['windows'], val=w)
                        table, _ = predict(b[:s], c, r, mm, 'val')
                        e = table.loc[table.status.eq('eligible')].set_index('device')
                        g = got.loc[(got.step == s) & (got.period == pi)].set_index('device')
                        self.assertEqual(sorted(g.index), sorted(e.index), kind)
                        np.testing.assert_allclose(g.max_score.reindex(e.index), e.max_score, rtol=1e-6, equal_nan=True, err_msg=kind)
                        np.testing.assert_allclose(g.max_hit_score.reindex(e.index), e.max_hit_score, rtol=1e-6, equal_nan=True, err_msg=kind)
                        self.assertTrue((g.positive.reindex(e.index) == e.positive).all())

    def test_folds_disjoint_stratified_and_mask_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c, m = fixture(root)
            reg = pd.read_parquet(root / 'cache/device_metadata.parquet')
            f = crossval.folds(reg, m['windows']['train'], 5)
            self.assertEqual(sorted(set(f)), [0, 1, 2, 3, 4])
            lo, hi = m['windows']['train']
            fails = ((reg.failure >= lo) & (reg.failure < hi)).to_numpy()
            counts = np.bincount(f[fails], minlength=5)
            self.assertLessEqual(counts.max() - counts.min(), 1)
            np.testing.assert_array_equal(f, crossval.folds(reg, m['windows']['train'], 5))   # deterministic
            recipe = initial_plans()[0]
            out = root / 'mask'; out.mkdir()
            mask = f != 0
            x, y, w, names, ratio, audit = build_training(c, recipe, m, out, 900, 2, device_mask=mask)
            held = set(reg.device[f == 0])
            full_out = root / 'full'; full_out.mkdir()
            xf, yf, wf, *_ = build_training(c, recipe, m, full_out, 900, 2)
            self.assertLess(len(y), len(yf))
            self.assertEqual(audit['sampling']['cross_validation_devices'], int(mask.sum()))
            # The masked matrix equals the full matrix restricted to non-held devices.
            from sample_builder import targets
            from history_store import training_batches
            dev = []
            for bt in training_batches(c, recipe, m, meta_only=True):
                mm = bt.to_pandas(); use, _ = targets(mm, 7, hi, 900)
                use &= (mm.time >= lo).to_numpy() & (mm.history_rows > 0).to_numpy()
                dev.append(mm.device.to_numpy()[use])
            dev = np.concatenate(dev)
            keep = ~np.isin(dev, list(held))
            np.testing.assert_array_equal(np.asarray(x), np.asarray(xf)[keep])
            np.testing.assert_array_equal(y, yf[keep])
            del x, xf   # release memmaps before the temporary directory is removed (Windows)

    def test_end_to_end_oof_and_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c, m = fixture(root)
            c['cv_rounds'] = 4
            r1 = initial_plans()[0]
            r2 = copy.deepcopy(r1); r2['params']['max_depth'] = 2
            a = crossval.oof_tables(c, r1, root / 'cv_a', k=3)
            b = crossval.oof_tables(c, r2, root / 'cv_b', k=3)
            self.assertEqual(a['folds'], 3)
            self.assertGreater(a['fault_units'], 0)
            oof = pd.read_parquet(root / 'cv_a/oof.parquet')
            # every (period, device) unit appears once per step and comes from exactly one fold
            per = oof.groupby(['step', 'period', 'device']).size()
            self.assertTrue((per == 1).all())
            self.assertTrue((oof.groupby('device').fold.nunique() == 1).all())
            reg = pd.read_parquet(root / 'cache/device_metadata.parquet')
            f = crossval.folds(reg, m['windows']['train'], 3)
            np.testing.assert_array_equal(oof.fold.to_numpy(), f[oof.device.to_numpy()])
            again = crossval.oof_tables(c, r1, root / 'cv_a', k=3)
            self.assertEqual(again['oof_sha256'], a['oof_sha256'])
            p = crossval.paired_cv(root / 'cv_a', root / 'cv_b', bootstraps=50)
            self.assertLessEqual(p['interval_95'][0], p['interval_95'][1])
            self.assertEqual(p['fault_units'], a['fault_units'])
            same = crossval.paired_cv(root / 'cv_a', root / 'cv_a', bootstraps=20)
            self.assertEqual(same['interval_95'], [0.0, 0.0])
            for period in a['periods']:
                self.assertGreaterEqual(period[0], m['windows']['train'][0])
                self.assertLessEqual(period[1], m['windows']['train'][1])


if __name__ == '__main__':
    unittest.main()
