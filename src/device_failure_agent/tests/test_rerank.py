"""Two-stage gated reranking: score composition, serialization, leakage control, end to end."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from common import save, read, sha, DAY
from settings import DEFAULTS, policy
from history_math import Writer
from experiment_contract import initial_plans, validate_recipe, identity, differences, with_defaults
import rerank
import crossval

T = 1704067200


def booster(seed, n=500):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 2)).astype(np.float32); y = (x[:, 0] + .2 * rng.random(n) > .7).astype(int)
    return xgb.train(dict(objective='binary:logistic', max_depth=3, nthread=1), xgb.DMatrix(x, label=y), 6)


def fixture(root, n=90):
    cache = root / 'cache'; writer = Writer(cache / 'rows'); registry = []
    rng = np.random.default_rng(1)
    for i in range(n):
        fd = 6 + (i * 7) % 52 if i % 3 == 0 else None   # faults in TRAIN, selection and confirmation periods
        failure = T + fd * DAY + 1800 * (i % 7) if fd else -1
        end = failure if fd else T + 60 * DAY
        ts = np.unique(np.sort(T + rng.integers(0, (fd or 60) * DAY, 60 + i))).astype(np.int64)
        if fd: ts = ts[ts < failure]
        x = (np.maximum(0, 5 - (failure - ts) / DAY) + rng.random(len(ts))) if fd else rng.random(len(ts)) * 4.5
        z = (rng.random(len(ts)) + (1.5 if fd else 0)).astype(np.float32)   # separates faults inside the high-x region
        writer.add(pd.DataFrame(dict(x=x.astype(np.float32), z=z, device=i, time=ts, failure=failure,
                                     observed_start=T, observed_end=end, history_rows=np.arange(1, len(ts) + 1))))
        registry.append(dict(device=i, sn_name=str(i), failure=failure, observed_start=T, observed_end=end))
    writer.flush(); pd.DataFrame(registry).to_parquet(cache / 'device_metadata.parquet', index=False)
    save(cache / 'devices.json', [str(i) for i in range(n)]); save(cache / 'excluded.json', [])
    m = dict(features=['x', 'z'], included_devices=n, test_opened=False, coverage={'basis': 'Synthetic'},
             windows=dict(train=[T, T + 40 * DAY], val=[T + 40 * DAY, T + 50 * DAY], test=[T + 50 * DAY, T + 60 * DAY]))
    m['files'] = {p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob('*') if p.is_file()}
    save(cache / 'manifest.json', m)
    (root / 'session').mkdir()
    c = dict(DEFAULTS, **policy('memory'), domain='memory', data_dir=str(root / 'raw'), workspace=str(root / 'w'),
             output_root=str(root / 'session'), active_cache=str(cache), device='cpu', num_threads=1, memory_limit_gb=2,
             rounds=4, eval_every=2, patience=2, min_free_disk_gb=.01, cv_rounds=4, rerank_min_rows=5,
             rerank_min_positive_devices=2, selection_window=[T + 40 * DAY, T + 45 * DAY],
             confirmation_window=[T + 45 * DAY, T + 50 * DAY])
    return c, m


class Composition(unittest.TestCase):
    def test_score_rule_ranking_and_roundtrip(self):
        s1, s2 = booster(0), booster(1)
        comp = rerank.Composite(s1, s2, .6)
        x = np.random.default_rng(2).random((300, 2)).astype(np.float32)
        a, b, out = s1.inplace_predict(x), s2.inplace_predict(x), comp.inplace_predict(x)
        g = a >= .6
        np.testing.assert_allclose(out[~g], a[~g], rtol=1e-7)
        np.testing.assert_allclose(out[g], .6 + .4 * b[g], rtol=1e-6)
        if g.any() and (~g).any():
            self.assertGreaterEqual(out[g].min(), out[~g].max())   # gated rows outrank ungated rows
        self.assertTrue(((out >= 0) & (out <= 1)).all())
        again = rerank.from_raw(comp.save_raw())
        np.testing.assert_array_equal(again.inplace_predict(x), out)
        plain = rerank.from_raw(s1.save_raw(raw_format='ubj'))
        np.testing.assert_array_equal(plain.inplace_predict(x), a)
        with self.assertRaises(ValueError):
            rerank.Composite(s1, s2, 1.0)

    def test_recipe_contract(self):
        base = initial_plans()[0]
        old = {k: v for k, v in base.items() if k not in ('rerank', 'rerank_gate')}   # pre-rerank receipt
        self.assertEqual(identity(old), identity(base))
        self.assertEqual(differences(old, base), [])
        r = dict(base, rerank='gated', rerank_gate=.8); validate_recipe(r)
        self.assertEqual(set(differences(base, r)), {'rerank', 'rerank_gate'})
        self.assertEqual(rerank.stage1_identity(r), rerank.stage1_identity(base))
        for bad in (dict(base, rerank_gate=.8), dict(r, rerank_gate=.65),
                    dict(r, sample_unit='device_day', feature_set='history', score_aggregation='latest')):
            with self.assertRaises(ValueError):
                validate_recipe(with_defaults(bad))


class NestedCV(unittest.TestCase):
    def test_fold_stage2_never_sees_its_fold(self):
        rng = np.random.default_rng(5)
        n = 1200; x = rng.random((n, 2)).astype(np.float32)
        y = (x[:, 0] + .3 * rng.random(n) > .8).astype(np.uint8); w = np.ones(n, np.float32)
        dev = np.repeat(np.arange(120), 10); rowfold = dev % 4
        c = dict(DEFAULTS, device='cpu', num_threads=1, rerank_min_rows=3, rerank_min_positive_devices=1)
        recipe = dict(initial_plans()[0], rerank='gated', rerank_gate=.5)
        b1, a1 = rerank.fold_stage2(x, y, w, dev, rowfold, 2, 4, recipe, c, 3, 4)
        # Poison every fold-2 row (features and labels): stage 2 for fold 2 must be byte-identical.
        x2, y2 = x.copy(), y.copy(); hold = rowfold == 2
        x2[hold] = rng.random((hold.sum(), 2)) * 9; y2[hold] = 1 - y2[hold]
        b2, a2 = rerank.fold_stage2(x2, y2, w, dev, rowfold, 2, 4, recipe, c, 3, 4)
        self.assertEqual(bytes(b1.save_raw(raw_format='ubj')), bytes(b2.save_raw(raw_format='ubj')))
        self.assertEqual(a1, a2)
        # Poisoning another fold does change it (the test is sensitive).
        x3 = x.copy(); x3[rowfold == 1] = 0
        b3, _ = rerank.fold_stage2(x3, y, w, dev, rowfold, 2, 4, recipe, c, 3, 4)
        self.assertNotEqual(bytes(b1.save_raw(raw_format='ubj')), bytes(b3.save_raw(raw_format='ubj')))

    def test_gate_with_too_few_rows_is_refused(self):
        y = np.r_[np.ones(3), np.zeros(50)].astype(np.uint8); s1 = np.r_[np.full(3, .95), np.full(50, .1)]
        c = dict(rerank_min_rows=10, rerank_min_positive_devices=1)
        with self.assertRaisesRegex(ValueError, 'Lower the gate'):
            rerank.gated_rows(y, s1, .9, np.arange(53), c)


class EndToEnd(unittest.TestCase):
    def test_full_trial_and_nested_cv(self):
        from xgb_runner import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c, m = fixture(root)
            base = initial_plans()[0]
            parent = root / 'parent'
            run(c, base, parent)
            pcv = crossval.oof_tables(c, base, root / 'parent_cv', k=3)
            r = dict(base, rerank='gated', rerank_gate=.5)
            out = root / 'gated'
            res = run(c, r, out, stage1_model=str(parent), stage1_cv=str(root / 'parent_cv'))
            self.assertTrue((out / 'model.ubj').read_bytes().startswith(rerank.MAGIC))
            ra = res['sample_audit']['rerank']
            self.assertEqual(ra['stage1_trial_model_sha256'], read(parent / 'result.json')['model_sha256'])
            self.assertGreater(ra['gated_positive_rows'], 0)
            # Reloaded composite reproduces the saved validation metrics through predict().
            from prediction import predict
            from evaluation import metrics
            from history_store import checked_manifest
            m = checked_manifest(c)   # trials score the selection window, as the runner does
            model = rerank.load_model(out / 'model.ubj')
            table, _ = predict(model, c, r, m, 'val', res['metrics']['threshold'])
            self.assertEqual(metrics(table, res['metrics']['threshold'])['f1'], res['metrics']['f1'])
            # Parallel scoring handles composites identically.
            t1, s1 = predict(model, dict(c, score_workers=1), r, m, 'val', .5)
            t2, s2 = predict(model, dict(c, score_workers=3), r, m, 'val', .5)
            self.assertEqual(s1, s2); pd.testing.assert_frame_equal(t1, t2)
            cv = rerank.oof_tables(c, r, root / 'gated_cv', root / 'parent_cv')
            self.assertEqual(cv['folds'], pcv['folds']); self.assertEqual(cv['periods'], pcv['periods'])
            self.assertEqual(cv['fault_units'], pcv['fault_units'])
            p = crossval.paired_cv(root / 'gated_cv', root / 'parent_cv', 50)
            self.assertLessEqual(p['interval_95'][0], p['interval_95'][1])
            with self.assertRaises(ValueError):   # stage-1 parent mismatch is refused
                run(c, dict(r, params=dict(r['params'], max_depth=3)), root / 'bad', stage1_model=str(parent),
                    stage1_cv=str(root / 'parent_cv'))


def fixture_loop(root, loop_fixture):
    """The v2-loop session, with its cache replaced by the rerank fixture (more devices, gated
    rows of both classes in every fold)."""
    session, c = loop_fixture(root)
    rc, _ = fixture(root / 'rr')
    c = read(session / 'config.json')
    c.update(active_cache=rc['active_cache'], selection_window=rc['selection_window'],
             confirmation_window=rc['confirmation_window'])
    mapping = read(session / 'mapping.json')
    mapping['observations'][0]['features'] = [dict(name='x', description='rising degradation'),
                                              dict(name='z', description='fault-specific signal')]
    from common import digest
    save(session / 'config.json', c); save(session / 'mapping.json', mapping)
    s = read(session / 'state.json'); s.update(config_hash=digest(c), mapping_hash=digest(mapping)); save(session / 'state.json', s)
    return session, c


class LoopIntegration(unittest.TestCase):
    """Controller path: analysis proposes rerank, stage 1 is wired from the parent, nested CV decides."""

    def test_agent_rerank_experiment(self):
        self._run_loop(min_rows=3)

    def test_agent_rerank_infeasible_gate_is_recorded(self):
        self._run_loop(min_rows=10 ** 6)

    def _run_loop(self, min_rows):
        from tests.test_v2_loop import fixture as loop_fixture, Mock
        from agent import Loop
        from common import digest
        with tempfile.TemporaryDirectory(prefix='rr_') as tmp:
            session, c = fixture_loop(Path(tmp), loop_fixture)
            c = read(session / 'config.json')
            c.update(rerank_allowed=True, rerank_min_rows=min_rows, rerank_min_positive_devices=1); save(session / 'config.json', c)
            s = read(session / 'state.json'); s['config_hash'] = digest(c); save(session / 'state.json', s)

            class RerankMock(Mock):
                def call(self, role, payload, folder, **kw):
                    if role == 'analysis':
                        self.payloads.setdefault(('analysis', 'analysis'), []).append(payload)
                        save(Path(folder) / 'usage.json', {'total_tokens': 5})
                        n = sum(t['status'] == 'complete' for t in payload['history'])
                        if n == 1:
                            self.scope = payload['scope']
                            return dict(action='experiment', parent='trial_001', changes={'rerank': 'gated', 'rerank_gate': 0.5},
                                        reason='Separate TP from FP in the high-score region', hypothesis='h', expected='e', falsification='f')
                        if self.features_asked < 1:
                            return dict(action='features', parent='trial_001', hypothesis_id='H1', question='q', reason='r',
                                        expected='e', falsification='f')
                        return dict(action='stop', reason='done')
                    return super().call(role, payload, folder, **kw)

            fake = RerankMock(); fake.features_asked = 1   # skip the null feature in the base mock
            loop = Loop(session, fake)
            try:
                loop.run()
            except ValueError:
                for p in sorted(session.rglob('cv.log')):
                    print('=== ' + str(p) + '\n' + p.read_text(errors='replace')[-3000:])
                raise
            self.assertTrue(fake.scope['rerank']['allowed'])
            gated = loop.s['trials'][1]
            self.assertEqual(gated['recipe']['rerank'], 'gated'); self.assertEqual(gated['parent'], 'trial_001')
            if min_rows >= 100:
                # Infeasible gate on full TRAIN: the trial fails cleanly, analysis sees the reason.
                self.assertEqual(gated['status'], 'failed'); self.assertIn('Lower the gate', gated['error'])
                self.assertNotEqual(loop.s['best'], gated['id'])
                return
            self.assertTrue((loop.folder(gated) / 'model.ubj').read_bytes().startswith(rerank.MAGIC))
            expected = ('cv_paired', 'screened_out') if min_rows < 100 else ('rerank_cv_infeasible',)
            self.assertIn(gated['acceptance']['rule'], expected)
            if gated['acceptance']['rule'] == 'rerank_cv_infeasible':
                self.assertFalse(gated['acceptance']['accepted']); self.assertIn('Lower the gate', gated['acceptance']['error'])
            if gated['acceptance']['rule'] == 'cv_paired':
                cvr = read(loop.folder(gated) / 'cv/oof_complete.json')
                self.assertEqual(cvr['rerank']['method'], 'nested out-of-fold gating')
            # A feature request on a gated parent is refused by the controller.
            loop.s['best'] = gated['id'] if gated['status'] == 'complete' else loop.s['best']
            with self.assertRaises(ValueError):
                loop.check_action(dict(action='features', parent=gated['id'], hypothesis_id='H1', question='q', reason='r',
                                       expected='e', falsification='f'))


if __name__ == '__main__':
    unittest.main()
