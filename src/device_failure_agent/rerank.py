"""Two-stage gated reranking: stage 1 recalls, stage 2 separates true faults from false alarms.

Stage 2 is trained only on TRAIN rows whose OUT-OF-FOLD stage-1 score reaches the gate g, so
its objective targets exactly the high-score region where false alarms and true faults mix.
Row score = s1 if s1 < g else g + (1 - g) * s2. Every gated row outranks every ungated row;
ordering inside the gated region is stage 2's. Pooling, schedule, cohort and the scorer are
unchanged because the composite only replaces the row-level score.

Leakage control
- Full-data trial: gating uses OOF stage-1 scores from the parent's device-grouped CV folds;
  inference uses the parent's own saved stage-1 model.
- Cross-validation: outer fold k's stage 2 is gated with NESTED OOF scores, each produced by a
  model trained without fold k, so no fold-k label reaches its own stage 2 (tested by
  poisoning fold k and checking stage 2 is byte-identical).
"""
import base64
import copy
import gc
import json
import time
from pathlib import Path
import numpy as np
import xgboost as xgb
from common import require, read, save, sha, digest

MAGIC = b'COMPOSITE1\n'


# ---------------------------------------------------------------- model objects

class Composite:
    """Row scorer with the Booster calls the pipeline uses (inplace_predict, save_raw, ...)."""

    def __init__(self, stage1, stage2, gate, stage1_steps=None):
        require(0 < float(gate) < 1, 'Rerank gate must lie in (0, 1)')
        self.stage1, self.stage2 = stage1, stage2
        self.gate = float(gate)
        self.stage1_steps = int(stage1_steps) if stage1_steps else None

    def inplace_predict(self, x, iteration_range=(0, 0)):
        """iteration_range applies to stage 2 (checkpoints); stage 1 is fixed."""
        x = np.asarray(x)
        s1 = np.asarray(self.stage1.inplace_predict(x, iteration_range=(0, self.stage1_steps or 0)), np.float64)
        out = s1.copy()
        gated = s1 >= self.gate
        if gated.any():
            s2 = np.asarray(self.stage2.inplace_predict(x[gated], iteration_range=iteration_range), np.float64)
            out[gated] = self.gate + (1. - self.gate) * s2
        return np.clip(out, 0., 1.)

    def get_score(self, importance_type='gain'):
        return self.stage2.get_score(importance_type=importance_type)

    def set_param(self, params):
        self.stage1.set_param(params); self.stage2.set_param(params)

    def save_raw(self, raw_format='ubj'):
        doc = dict(gate=self.gate, stage1_steps=self.stage1_steps,
                   stage1=base64.b64encode(bytes(self.stage1.save_raw(raw_format='ubj'))).decode(),
                   stage2=base64.b64encode(bytes(self.stage2.save_raw(raw_format='ubj'))).decode())
        return bytearray(MAGIC + json.dumps(doc, sort_keys=True).encode())

    def save_model(self, path):
        Path(path).write_bytes(bytes(self.save_raw()))


def _booster(raw, threads):
    b = xgb.Booster(); b.load_model(bytearray(raw)); b.set_param(dict(device='cpu', nthread=threads)); return b


def from_raw(raw, threads=1):
    """Plain UBJ booster bytes or a Composite blob -> CPU scorer."""
    raw = bytes(raw)
    if raw.startswith(b'PARALLEL1\n'):
        from parallel_branch import from_raw as parallel_from_raw
        return parallel_from_raw(raw, threads)
    if not raw.startswith(MAGIC):
        return _booster(raw, threads)
    doc = json.loads(raw[len(MAGIC):])
    return Composite(_booster(base64.b64decode(doc['stage1']), threads), _booster(base64.b64decode(doc['stage2']), threads),
                     doc['gate'], doc['stage1_steps'])


def load_model(path, threads=1):
    return from_raw(Path(path).read_bytes(), threads)


def cpu_copy(booster, threads):
    return _booster(booster.save_raw(raw_format='ubj'), threads)


# ---------------------------------------------------------------- training helpers

def stage1_identity(recipe):
    """Training identity of the stage-1 model a rerank recipe builds on."""
    from experiment_contract import training_identity, FACTORS
    r = copy.deepcopy(recipe); r['rerank'] = 'none'; r['rerank_gate'] = FACTORS['rerank_gate'][0]
    return training_identity(r)


def positive_weight(y, w, policy):
    pos = w[y == 1].sum(dtype=np.float64); neg = w[y == 0].sum(dtype=np.float64)
    require(pos > 0 and neg > 0, 'Training subset needs both classes')
    ratio = float(neg / pos)
    return {'ratio': ratio, 'sqrt_ratio': ratio ** .5, 'unit': 1.}[policy]


def params(recipe, c, spw):
    p = dict(recipe['params']); p['eta'] = p.pop('learning_rate')
    p.update(objective='binary:logistic', tree_method='hist', device=c['device'], nthread=c['num_threads'],
             subsample=1., scale_pos_weight=spw, base_score=.5, seed=recipe['seed'], disable_default_eval_metric=1)
    return p


def fit(x, y, w, recipe, c, rounds):
    """Same objective/params as the main runner; returns a CPU copy."""
    d = xgb.QuantileDMatrix(x, label=y, weight=w, max_bin=recipe['params']['max_bin'], nthread=c['num_threads'])
    b = xgb.Booster(params(recipe, c, positive_weight(y, w, recipe['weight_policy'])), cache=[d])
    for i in range(rounds):
        b.update(d, i)
    out = cpu_copy(b, c['num_threads'])
    del d, b; gc.collect()
    return out


def gated_rows(y, s1, gate, dev, c):
    g = s1 >= gate
    pos = int((y[g] == 1).sum()); neg = int((y[g] == 0).sum())
    posdev = int(len(np.unique(dev[g & (y == 1)])))
    need_rows = c.get('rerank_min_rows', 100); need_dev = c.get('rerank_min_positive_devices', 20)
    require(pos >= need_rows and neg >= need_rows and posdev >= need_dev,
            f'Rerank gate {gate}: too few gated TRAIN samples (positive rows {pos}, positive devices {posdev}, '
            f'negative rows {neg}; need >= {need_rows} rows each and >= {need_dev} positive devices). Lower the gate.')
    return g, dict(gate=float(gate), gated_rows=int(g.sum()), gated_share=float(g.mean()), gated_positive_rows=pos,
                   gated_negative_rows=neg, gated_positive_devices=posdev)


def _matrix(c, recipe, m, folder):
    """Whole TRAIN matrix in memory plus the device of every row (scratch memmap removed)."""
    from sample_builder import build_training
    from scratch import cleanup_matrix
    x, y, w, names, _, audit, dev = build_training(c, recipe, m, folder, c['minimum_lead_seconds'],
                                                   c['memory_limit_gb'], return_devices=True)
    x = np.array(x, dtype=np.float32)
    cleanup_matrix(folder)
    return x, np.asarray(y), np.asarray(w), dev, audit


def _folds(c, m, k):
    import pandas as pd
    from crossval import folds
    registry = pd.read_parquet(Path(c['active_cache']) / 'device_metadata.parquet')
    return registry, folds(registry, m['windows']['train'], k)


# ---------------------------------------------------------------- full-data trial

def train_full(c, recipe, m, out, stage1_folder, stage1_cv):
    """Stage 2 on gated TRAIN rows, checkpoints scored on the selection window like any trial."""
    from prediction import predict
    from evaluation import select
    out, stage1_folder, stage1_cv = Path(out), Path(stage1_folder), Path(stage1_cv)
    prev = read(stage1_folder / 'result.json')
    require(prev['recipe'].get('rerank', 'none') == 'none', 'Stage-1 parent must not itself rerank')
    require(stage1_identity(prev['recipe']) == stage1_identity(recipe), 'Stage-1 parent recipe must equal this recipe without reranking')
    require(sha(stage1_folder / 'model.ubj') == prev['model_sha256'], 'Stage-1 model changed')
    cvr = read(stage1_cv / 'oof_complete.json')
    require(stage1_identity(cvr['recipe']) == stage1_identity(recipe), 'Stage-1 cross-validation recipe differs')
    tick = time.monotonic()
    x, y, w, dev, audit = _matrix(c, recipe, m, out)
    registry, fold_of = _folds(c, m, cvr['folds'])
    rowfold = fold_of[dev]; steps1 = cvr['best_step']
    s1 = np.full(len(y), np.nan)
    for k in range(cvr['folds']):
        sel = rowfold == k
        if sel.any():
            s1[sel] = load_model(stage1_cv / f'fold_{k}' / 'model.ubj', c['num_threads']).inplace_predict(x[sel], iteration_range=(0, steps1))
    require(np.isfinite(s1).all(), 'Out-of-fold stage-1 scores missing for some TRAIN rows')
    gate = recipe['rerank_gate']
    g, gate_audit = gated_rows(y, s1, gate, dev, c)
    xg, yg, wg = x[g], y[g], w[g]
    del x; gc.collect()
    stage1 = load_model(stage1_folder / 'model.ubj', c['num_threads'])
    spw = positive_weight(yg, wg, recipe['weight_policy'])
    p = params(recipe, c, spw)
    d = xgb.QuantileDMatrix(xg, label=yg, weight=wg, max_bin=recipe['params']['max_bin'], nthread=c['num_threads'])
    b2 = xgb.Booster(p, cache=[d]); best = None; stale = 0; history = []
    feature_seconds = time.monotonic() - tick; boost = 0.; infer = 0.
    for step in range(1, c['rounds'] + 1):
        t0 = time.monotonic(); b2.update(d, step - 1); boost += time.monotonic() - t0
        if step != 1 and step % c['eval_every'] and step != c['rounds']:
            continue
        t0 = time.monotonic()
        comp = Composite(stage1, cpu_copy(b2, c['num_threads']), gate)
        print(f'Step {step}: validating reranked {recipe["prediction_interval_hours"]}h schedule...', flush=True)
        table, stats = predict(comp, c, recipe, m, 'val'); metric, grid = select(table)
        infer += time.monotonic() - t0
        history.append(dict(step=step, val_period_unweighted_bce=stats['val_period_unweighted_bce'], **metric))
        save(out / 'loss_history.json', history)
        if best is None or metric['f1'] > best['metrics']['f1'] + 1e-12:
            best = dict(step=step, metrics=metric, score_sha256=stats['score_sha256']); stale = 0
            comp.save_model(out / 'model.ubj'); grid.to_csv(out / 'threshold_search_val.csv', index=False, float_format='%.17g')
        else:
            stale += 1
        print(f'Step {step}: device F1={metric["f1"]:.6f}; TP={metric["TP"]} FP={metric["FP"]} FN={metric["FN"]}; best={best["step"]}', flush=True)
        if stale >= c['patience']:
            break
    actual = json.loads(b2.save_config())['learner']['generic_param']['device']
    del d; gc.collect()
    audit = dict(audit, rerank=dict(gate_audit, stage1_trial_model_sha256=prev['model_sha256'], stage1_oof_steps=steps1,
                                    stage1_cv_oof_sha256=cvr['oof_sha256'], stage2_spw=spw,
                                    note='Stage 2 trained on TRAIN rows whose out-of-fold stage-1 score >= gate.'))
    return best, audit, p, actual, dict(feature_and_matrix_seconds=feature_seconds, boost_updates_seconds=boost,
                                        validation_prediction_seconds=infer)


# ---------------------------------------------------------------- nested cross-validation

def fold_stage2(x, y, w, dev, rowfold, fold, k, recipe, c, steps1, rounds):
    """Stage 2 for outer fold `fold`, using only rows of the other folds.

    Gating scores for rows of inner fold j come from a stage-1 model trained on folds
    other than j and `fold`, so neither the gate nor the fit sees fold `fold`.
    """
    pool = rowfold != fold
    inner = np.full(len(y), np.nan)
    for j in range(k):
        if j == fold:
            continue
        train = pool & (rowfold != j)
        score = rowfold == j
        if score.any():
            inner[score] = fit(x[train], y[train], w[train], recipe, c, steps1).inplace_predict(x[score])
    idx = np.flatnonzero(pool)
    g, audit = gated_rows(y[idx], inner[idx], recipe['rerank_gate'], dev[idx], c)
    idx = idx[g]
    return fit(x[idx], y[idx], w[idx], recipe, c, rounds), audit


def oof_tables(c, recipe, out, parent_cv, rounds=None):
    """Nested-CV out-of-fold units for a rerank recipe, in the crossval.oof_tables format."""
    import pandas as pd
    from history_store import checked_manifest
    from feature_programs import materialize
    from crossval import periods, score_devices, summarize, SEED
    out, parent_cv = Path(out), Path(parent_cv)
    out.mkdir(parents=True, exist_ok=True)
    c, m = materialize(c, recipe['feature_program'], checked_manifest(c))
    rounds = rounds or c.get('cv_rounds', 400)
    pr = read(parent_cv / 'oof_complete.json')
    require(stage1_identity(pr['recipe']) == stage1_identity(recipe), 'Parent cross-validation recipe differs from the stage-1 recipe')
    k = pr['folds']; plist = periods(m, c)
    require(plist == pr['periods'], 'Cross-validation periods differ from the parent run')
    key = digest(dict(recipe=recipe, rounds=rounds, eval_every=c['eval_every'], periods=plist, k=k, seed=SEED,
                      manifest=sha(Path(c['active_cache']) / 'manifest.json'), parent=pr['oof_sha256']))
    done = out / 'oof_complete.json'
    if done.exists() and read(done)['key'] == key:
        return read(done)
    tick = time.monotonic()
    x, y, w, dev, audit = _matrix(c, recipe, m, out)
    registry, fold_of = _folds(c, m, k)
    rowfold = fold_of[dev]; steps1 = pr['best_step']
    steps = sorted({s for s in range(c['eval_every'], rounds + 1, c['eval_every'])} | {rounds})
    rows = []; fold_audits = []
    for fold in range(k):
        print(f'Rerank CV fold {fold+1}/{k}: nested stage-1 gating and stage-2 fit', flush=True)
        b2, gate_audit = fold_stage2(x, y, w, dev, rowfold, fold, k, recipe, c, steps1, rounds)
        stage1 = load_model(parent_cv / f'fold_{fold}' / 'model.ubj', c['num_threads'])
        comp = Composite(stage1, b2, recipe['rerank_gate'], stage1_steps=steps1)
        held = registry.device.to_numpy()[fold_of == fold]
        t = score_devices(comp, steps, c, recipe, m, plist, held)
        t['fold'] = fold; rows.append(t); fold_audits.append(dict(fold=fold, **gate_audit))
        print(f'Rerank CV fold {fold+1}/{k}: scored {len(steps)} checkpoints ({int(time.monotonic()-tick)}s)', flush=True)
    del x; gc.collect()
    oof = pd.concat(rows, ignore_index=True)
    oof.to_parquet(out / 'oof.parquet', index=False)
    result = dict(key=key, recipe=recipe, periods=plist, folds=k, rounds=rounds, seconds=time.monotonic() - tick,
                  oof_sha256=sha(out / 'oof.parquet'), rerank=dict(stage1_steps=steps1, parent_oof_sha256=pr['oof_sha256'],
                  folds=fold_audits, method='nested out-of-fold gating'), **summarize(oof))
    save(done, result)
    return result
