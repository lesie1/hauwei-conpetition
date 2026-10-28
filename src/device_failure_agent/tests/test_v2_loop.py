"""v2 loop end to end with mocked DeepSeek roles and real CPU XGBoost/CV."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from common import save, read, sha, digest, code_hashes, DAY
from settings import DEFAULTS, policy
from history_math import Writer
from experiment_contract import source_for
from agent import Loop

T = 1704067200


class Mock:
    def __init__(self, bad_feature_first=True):
        self.calls = []; self.payloads = {}; self.bad = bad_feature_first; self.features_asked = 0

    def call(self, role, payload, folder, **kw):
        task = payload.get('task')
        self.calls.append((role, task)); self.payloads.setdefault((role, task), []).append(payload)
        save(Path(folder) / 'usage.json', {'total_tokens': 5})
        if role == 'code':
            return dict(source=source_for(payload['accepted_plan']))
        if role == 'feature' and task == 'hypotheses':
            names = [f['name'] for f in payload['fields']]
            return dict(hypotheses=[dict(id=f'H{i}', mechanism='m', basis='b', fields=[n], expected_direction='higher_in_failing',
                                         falsified_if='AUC interval includes 0.5') for i, n in enumerate(names[:3], 1)])
        if role == 'feature':
            self.features_asked += 1
            if self.features_asked == 1:
                # pure noise column: prescreen must reject it without training
                return dict(source="def transform(frame):\n    n = past.observation_count(frame) * 0 + 1\n    return pd.DataFrame({'fe_const': n}, index=frame.index)",
                            features=['fe_const'], reason='H1 null control')
            return dict(source="def transform(frame):\n    r = frame['x'].rolling('3D', min_periods=1).max()\n    return pd.DataFrame({'fe_xmax3d': r}, index=frame.index)",
                        features=['fe_xmax3d'], reason='H1 rising signal')
        n = sum(t['status'] == 'complete' for t in payload['history'])
        best = payload['best']['id']
        if n == 1 and not payload['evidence']:
            return dict(action='diagnose', parent=best, kind='high_score_contrast', question='What separates TP from FP?', reason='evidence')
        if n == 1:
            return dict(action='experiment', parent=best, changes={'params.max_depth': 3}, reason='control',
                        hypothesis='depth', expected='e', falsification='f')
        if self.features_asked < 2:
            return dict(action='features', parent=best, hypothesis_id='H1', question='Does a trailing max separate TP from FP?',
                        reason='r', expected='e', falsification='f')
        return dict(action='stop', reason='done')


def fixture(root, n=60):
    cache = root / 'cache'; writer = Writer(cache / 'rows'); registry = []
    rng = np.random.default_rng(1)
    for i in range(n):
        fd = 6 + (i * 7) % 50 if i % 3 == 0 else None
        failure = T + fd * DAY + 1800 * (i % 7) if fd else -1
        end = failure if fd else T + 60 * DAY
        ts = np.unique(np.sort(T + rng.integers(0, (fd or 60) * DAY, 40 + i))).astype(np.int64)
        if fd: ts = ts[ts < failure]
        x = (np.maximum(0, 5 - (failure - ts) / DAY) + rng.random(len(ts))) if fd else rng.random(len(ts)) * 1.5
        writer.add(pd.DataFrame(dict(x=x.astype(np.float32), z=rng.random(len(ts)).astype(np.float32),
                                     q=rng.random(len(ts)).astype(np.float32), device=i, time=ts,
                                     failure=failure, observed_start=T, observed_end=end, history_rows=np.arange(1, len(ts) + 1))))
        registry.append(dict(device=i, sn_name=str(i), failure=failure, observed_start=T, observed_end=end))
    writer.flush(); pd.DataFrame(registry).to_parquet(cache / 'device_metadata.parquet', index=False)
    save(cache / 'devices.json', [str(i) for i in range(n)]); save(cache / 'excluded.json', [])
    m = dict(features=['q', 'x', 'z'], included_devices=n, test_opened=False, coverage={'basis': 'Synthetic'},
             windows=dict(train=[T, T + 40 * DAY], val=[T + 40 * DAY, T + 50 * DAY], test=[T + 50 * DAY, T + 60 * DAY]))
    m['files'] = {p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob('*') if p.is_file()}
    save(cache / 'manifest.json', m)
    session = root / 'session'; session.mkdir()
    c = dict(DEFAULTS, **policy('memory'), domain='memory', data_dir=str(root / 'raw'), workspace=str(root / 'w'),
             output_root=str(session), active_cache=str(cache), device='cpu', num_threads=1, memory_limit_gb=2,
             rounds=4, eval_every=2, patience=2, max_trials=5, min_free_disk_gb=.01, comparison_bootstraps=5,
             cv_folds=3, cv_rounds=4, cv_bootstraps=50, accept_lower_bound=-1.0, screen_margin=1.0,
             selection_window=[T + 40 * DAY, T + 45 * DAY], confirmation_window=[T + 45 * DAY, T + 50 * DAY],
             instructions='Synthetic v2 test')
    mapping = dict(observations=[dict(features=[dict(name='x', description='rising degradation'), dict(name='z', description='noise'),
                                                dict(name='q', description='noise')])],
                   description='Synthetic')
    save(session / 'config.json', c); save(session / 'mapping.json', mapping); save(session / 'code_seal.json', code_hashes())
    save(session / 'state.json', dict(version=1, status='ready', config_hash=digest(c), mapping_hash=digest(mapping), dataset='syn',
                                      trials=[], calls=0, tokens=0, best=None, pending=None, analyze_next=None, diagnostics={},
                                      feature_jobs=0, experience=None, experience_tried=True, last_failure=None))
    return session, c


class V2Loop(unittest.TestCase):
    def test_full_loop(self):
        with tempfile.TemporaryDirectory(prefix='v2_') as tmp:
            session, c = fixture(Path(tmp))
            fake = Mock(); loop = Loop(session, fake); loop.run()
            self.assertEqual(loop.s['status'], 'stopped')
            # hypotheses first, labels never shown
            self.assertEqual(fake.calls[0], ('code', 'training'))
            self.assertIn(('feature', 'hypotheses'), fake.calls)
            h = read(session / 'hypotheses.json'); self.assertFalse(h['labels_visible'])
            hp = fake.payloads[('feature', 'hypotheses')][0]
            self.assertNotIn('history', hp); self.assertNotIn('metrics', str(hp))
            trials = loop.s['trials']
            # baseline accepted with CV; every analysis payload carries the acceptance rule and hypotheses
            self.assertEqual(trials[0]['acceptance']['rule'], 'baseline')
            self.assertTrue((loop.folder(trials[0]) / 'cv/oof_complete.json').exists())
            for p in fake.payloads[('analysis', 'analysis')]:
                self.assertEqual(p['acceptance_rule']['rule'], 'cv_paired'); self.assertIn('hypotheses', p)
            # contrast evidence produced from CV units
            ev = read(session / 'diagnostics' / trials[0]['id'] / 'high_score_contrast' / 'evidence.json')
            self.assertEqual(ev['contrast']['kind'], 'high_score_contrast')
            # the constant feature was rejected by prescreen without training
            fails = [x for x in loop.s['trials'] if 'fe_const' in str(x)]
            self.assertFalse(fails)
            self.assertTrue((session / 'feature_jobs/job_001/prescreen.json').exists())
            self.assertFalse(read(session / 'feature_jobs/job_001/prescreen.json')['prescreen_passed'])
            # challengers decided by paired CV
            for t in trials[1:]:
                self.assertIn(t['acceptance']['rule'], ('cv_paired', 'screened_out'))
                if t['acceptance']['rule'] == 'cv_paired':
                    self.assertTrue((loop.folder(t) / 'cv_comparison_best.json').exists())
                    lo, hi = t['acceptance']['cv_interval_95']; self.assertLessEqual(lo, hi)
            # best is either baseline or an accepted challenger, never an unaccepted one
            best = loop.trial(loop.s['best'])
            self.assertTrue(best['acceptance']['accepted'])
            # feature request without hypothesis_id is refused
            bad = dict(action='features', parent=best['id'], question='q', reason='r', expected='e', falsification='f')
            loop.s['feature_jobs'] = 0
            with self.assertRaises(ValueError):
                loop.check_action(bad)

    def test_strict_bound_rejects_ties(self):
        with tempfile.TemporaryDirectory(prefix='v2b_') as tmp:
            session, c = fixture(Path(tmp))
            c = read(session / 'config.json'); c.update(accept_lower_bound=0.5); save(session / 'config.json', c)
            s = read(session / 'state.json'); s['config_hash'] = digest(c); save(session / 'state.json', s)
            fake = Mock(); loop = Loop(session, fake); loop.run()
            # With an impossible bound no challenger may replace the baseline.
            self.assertEqual(loop.s['best'], loop.s['trials'][0]['id'])
            self.assertTrue(all(not t['acceptance']['accepted'] for t in loop.s['trials'][1:]))


if __name__ == '__main__':
    unittest.main()
