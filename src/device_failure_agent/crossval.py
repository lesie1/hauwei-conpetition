"""Device-grouped, time-blocked cross-validation inside TRAIN with the frozen device-F1 scorer.

Why: the 15-day selection window holds about 51 eligible faults, too few to tell trials apart.
Each fold k holds out a disjoint device group G_k (stratified by whether the device ever fails
in TRAIN). Its model trains on TRAIN rows of devices outside G_k. Every held-out device is then
scored on consecutive 15-day TRAIN periods with exactly the validation cohort rules, schedule,
aggregation and lead bounds. Pooling out-of-fold (period, device) units gives several hundred
fault units for paired comparison.

Leakage control: devices never cross folds. Labels at a training row use only that device's
own future within TRAIN; held-out devices contribute nothing to their own fold's model.
"""
import argparse
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from common import read, save, sha, digest, require
from history_store import checked_manifest
from evaluation import period_cohort, select, classify
from sample_builder import build_training

FOLDS = 5
SEED = 20261027


def periods(manifest, c):
    """Consecutive TRAIN periods of the selection-window length, ending at the TRAIN boundary.

    The first periods are dropped while devices still lack history (cohorts need coverage
    from the period start), keeping at most five.
    """
    lo, hi = manifest['windows']['train']
    width = c['selection_window'][1] - c['selection_window'][0]
    out = []
    end = hi
    while end - width >= lo + width and len(out) < 5:
        out.append([end - width, end])
        end -= width
    require(out, 'TRAIN too short for cross-validation periods')
    return out[::-1]


def folds(registry, train_window, k=FOLDS, seed=SEED):
    """Device -> fold, stratified by TRAIN failure so every fold holds faults."""
    lo, hi = train_window
    fails = ((registry.failure >= lo) & (registry.failure < hi)).to_numpy()
    rng = np.random.default_rng(seed)
    out = np.full(len(registry), -1, np.int64)
    for flag in (True, False):
        ix = registry.device.to_numpy()[fails == flag]
        ix = ix[rng.permutation(len(ix))]
        out[ix] = np.arange(len(ix)) % k
    require((out >= 0).all(), 'Unassigned devices in CV folds')
    return out


UNIT_COLUMNS = ['device', 'positive', 'period', 'step', 'max_score', 'max_hit_score',
                'timely_prediction_opportunities', 'decision_time']


def score_devices(booster, steps, c, recipe, m, plist, devices):
    """Per (step, period, device): max_score and max_hit_score exactly as prediction.predict.

    Only the given devices' rows are read; checkpoints are iteration_range prefixes of one
    booster. Schedule (anchors), pooling (pool_scores), cohort (period_cohort) and lead bounds
    are the frozen validation rules; tests assert equality with predict() on the same window.
    """
    from history_store import cache_path, anchors
    from prediction import pool_scores
    import pyarrow.parquet as pq
    require(recipe['feature_set'] == 'base', 'Cross-validation scorer supports event (base) models')
    cache = cache_path(c)
    base = m['features']
    kind = recipe['score_aggregation']
    step = recipe['prediction_interval_hours'] * 3600
    registry = pd.read_parquet(cache / 'device_metadata.parquet')
    devices = np.asarray(sorted(set(int(d) for d in devices)), np.int64)
    # Window pools need (point-step, point]; latest/all-history need the whole past.
    first = min(p[0] for p in plist) - step if kind.startswith('window') else None
    last = max(p[1] for p in plist)
    filters = [('device', 'in', devices.tolist()), ('time', '<', int(last))]
    if first is not None:
        filters.append(('time', '>=', int(first)))
    files = sorted((cache / 'rows').glob('*.parquet'))
    data = pq.read_table([str(p) for p in files], columns=base + ['device', 'time'], filters=filters).to_pandas()
    data = data.sort_values(['device', 'time'], kind='stable').reset_index(drop=True)
    x = data[base].to_numpy(np.float32)
    raw = {s: booster.inplace_predict(x, iteration_range=(0, s)).astype(np.float64) if len(x) else np.zeros(0)
           for s in steps}
    times = data.time.to_numpy(np.int64)
    codes = data.device.to_numpy()
    cuts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1, len(data)] if len(data) else np.zeros(1, np.int64)
    spans = {int(codes[a]): (a, b) for a, b in zip(cuts[:-1], cuts[1:])}
    lead, maximum = c['minimum_lead_seconds'], c.get('maximum_lead_seconds')
    # Eligible (period, device) units with their schedule; a device's periods are disjoint and
    # increasing, so its anchors concatenate and each pooled score is computed once per step.
    units = {}
    for pi, window in enumerate(plist):
        cohort = period_cohort(registry, window, c).set_index('device').loc[devices]
        cohort = cohort.loc[cohort.status.eq('eligible')]
        for d, positive, failure, start, end in zip(cohort.index.to_numpy(), cohort.positive.to_numpy(),
                                                     cohort.failure.to_numpy(), cohort.observed_start.to_numpy(),
                                                     cohort.observed_end.to_numpy()):
            points = anchors(window, m['windows']['train'][0], recipe['prediction_interval_hours'], int(failure))
            points = points[(points >= start) & (points <= end)]
            timely = np.zeros(len(points), bool)
            if positive:
                timely = points <= failure - lead
                if maximum is not None:
                    timely &= failure - points <= maximum
            units.setdefault(int(d), []).append((pi, bool(positive), points, timely))
    out = {k: [] for k in UNIT_COLUMNS}
    for d, parts in units.items():
        points = np.concatenate([p[2] for p in parts])
        require((np.diff(points) > 0).all(), 'CV periods must be disjoint and increasing')
        cuts = np.cumsum([0] + [len(p[2]) for p in parts])
        a, b = spans.get(d, (0, 0))
        for s in steps:
            pooled = pool_scores(times[a:b], raw[s][a:b], points, step, kind)[0] if b > a else np.full(len(points), np.nan)
            for (pi, positive, pts, timely), lo, hi in zip(parts, cuts[:-1], cuts[1:]):
                p = pooled[lo:hi]
                fin = np.isfinite(p)
                hit = fin & timely
                out['device'].append(d); out['positive'].append(positive); out['period'].append(pi)
                out['step'].append(int(s))
                out['max_score'].append(float(p[fin].max()) if fin.any() else np.nan)
                out['max_hit_score'].append(float(p[hit].max()) if hit.any() else np.nan)
                out['timely_prediction_opportunities'].append(int(timely.sum()))
                # Decision time of the unit's most confident relevant score (for error-case evidence).
                rel = hit if positive else fin
                out['decision_time'].append(int(pts[rel][np.argmax(p[rel])]) if rel.any() else -1)
    return pd.DataFrame(out, columns=UNIT_COLUMNS).sort_values(['period', 'device', 'step'], kind='stable').reset_index(drop=True)


def _train(c, recipe, m, out, device_mask, rounds):
    x, y, w, names, ratio, audit = build_training(c, recipe, m, out, c['minimum_lead_seconds'],
                                                  c['memory_limit_gb'], device_mask=device_mask)
    dt = xgb.QuantileDMatrix(x, label=y, weight=w, max_bin=recipe['params']['max_bin'], nthread=c['num_threads'])
    del x
    gc.collect()
    from scratch import cleanup_matrix
    cleanup_matrix(out)
    params = dict(recipe['params'])
    params['eta'] = params.pop('learning_rate')
    params.update(objective='binary:logistic', tree_method='hist', device=c['device'], nthread=c['num_threads'],
                  subsample=1., scale_pos_weight=ratio, base_score=.5, seed=recipe['seed'], disable_default_eval_metric=1)
    model = xgb.Booster(params, cache=[dt])
    for step in range(1, rounds + 1):
        model.update(dt, step - 1)
    del dt
    gc.collect()
    b = xgb.Booster()
    b.load_model(model.save_raw(raw_format='ubj'))
    b.set_param(dict(device='cpu', nthread=c['num_threads']))
    b.save_model(Path(out) / 'model.ubj')
    steps = sorted({s for s in range(c['eval_every'], rounds + 1, c['eval_every'])} | {rounds})
    return b, steps, audit


def oof_tables(c, recipe, out, rounds=None, k=None):
    """Out-of-fold device tables per (step, period); cached on disk by recipe/config identity."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m0 = checked_manifest(c)
    from feature_programs import materialize
    c, m = materialize(c, recipe['feature_program'], m0)
    rounds = rounds or c.get('cv_rounds', 400)
    k = k or c.get('cv_folds', FOLDS)
    plist = periods(m, c)
    registry = pd.read_parquet(Path(c['active_cache']) / 'device_metadata.parquet')
    fold_of = folds(registry, m['windows']['train'], k)
    key = digest(dict(recipe=recipe, rounds=rounds, eval_every=c['eval_every'], periods=plist, k=k,
                      manifest=sha(Path(c['active_cache']) / 'manifest.json'), seed=SEED))
    done = out / 'oof_complete.json'
    if done.exists() and read(done)['key'] == key:
        return read(done)
    tick = time.monotonic()
    rows = []
    for fold in range(k):
        held = registry.device.to_numpy()[fold_of == fold]
        train_mask = fold_of != fold
        folder = out / f'fold_{fold}'
        folder.mkdir(exist_ok=True)
        print(f'CV fold {fold+1}/{k}: training on {int(train_mask.sum())} devices', flush=True)
        booster, steps, audit = _train(c, recipe, m, folder, train_mask, rounds)
        t = score_devices(booster, steps, c, recipe, m, plist, held)
        t['fold'] = fold
        rows.append(t)
        save(folder / 'audit.json', dict(audit=audit, held_out_devices=int(len(held)),
                                         scored_units=int((t.step == steps[-1]).sum())))
        print(f'CV fold {fold+1}/{k}: scored {len(steps)} checkpoints x {len(plist)} periods '
              f'({int(time.monotonic()-tick)}s)', flush=True)
        del booster
        gc.collect()
    oof = pd.concat(rows, ignore_index=True)
    oof.to_parquet(out / 'oof.parquet', index=False)
    summary = summarize(oof)
    result = dict(key=key, recipe=recipe, periods=plist, folds=k, rounds=rounds, seconds=time.monotonic() - tick,
                  oof_sha256=sha(out / 'oof.parquet'), **summary)
    save(done, result)
    return result


def pooled_select(units):
    """Threshold maximizing pooled device-F1 over (period, device) units; ties -> fewer FP.

    Each (period, device) unit counts like one device in one evaluation period, so pooled
    F1 is the device F1 of the concatenated periods.
    """
    t = units.copy()
    t['status'] = 'eligible'
    t['protocol'] = 'cv_pooled'
    return select(t)


def summarize(oof):
    curves = []
    for step, g in oof.groupby('step'):
        m, _ = pooled_select(g)
        curves.append(dict(step=int(step), f1=m['f1'], TP=m['TP'], FP=m['FP'], FN=m['FN'],
                           precision=m['precision'], recall=m['recall'], threshold=m['threshold']))
    curves.sort(key=lambda r: (-r['f1'], r['FP'], r['step']))
    best = curves[0]
    units = oof.loc[oof.step == best['step']]
    from contrast import error_counts
    return dict(best_step=best['step'], cv_metrics=best, error_counts=error_counts(oof,best['step'],best['threshold']), curve=sorted(curves, key=lambda r: r['step']),
                fault_units=int(units.positive.sum()), eligible_units=len(units))


def unit_outcomes(oof, step, threshold):
    t = oof.loc[oof.step == step].copy()
    t['status'] = 'eligible'
    t = classify(t, threshold)
    return t.set_index(['period', 'device'])[['positive', 'hit', 'fp', 'fn']].sort_index()


def paired_cv(current, reference, bootstraps=1000, seed=SEED):
    """Paired bootstrap of pooled CV F1 at each model's own CV-selected step and threshold.

    Resamples devices with all their periods, stratified by ever-fault status;
    both models see the same resampled clusters. Step/threshold selection is exploratory.
    """
    a = pd.read_parquet(Path(current) / 'oof.parquet')
    b = pd.read_parquet(Path(reference) / 'oof.parquet')
    ra, rb = read(Path(current) / 'oof_complete.json'), read(Path(reference) / 'oof_complete.json')
    ua = unit_outcomes(a, ra['best_step'], ra['cv_metrics']['threshold'])
    ub = unit_outcomes(b, rb['best_step'], rb['cv_metrics']['threshold'])
    require(ua.index.equals(ub.index) and ua.positive.equals(ub.positive), 'CV units differ between models')
    va = ua[['hit', 'fp', 'fn']].to_numpy(np.int64)
    vb = ub[['hit', 'fp', 'fn']].to_numpy(np.int64)
    # A device can span several periods: resample its joint outcome totals as a
    # cluster. Stratify by ever-fault-in-these-periods, keeping all its periods together.
    joined=pd.DataFrame(np.c_[va,vb],index=ua.index)
    device_totals=joined.groupby(level='device').sum()
    device_fault=ua.positive.groupby(level='device').any().reindex(device_totals.index)
    rng = np.random.default_rng(seed)
    totals = np.zeros((bootstraps, 6), np.int64)
    for flag in (False, True):
        ix = np.flatnonzero(device_fault.to_numpy() == flag)
        if not len(ix):
            continue
        pats, counts = np.unique(device_totals.to_numpy(np.int64)[ix], axis=0, return_counts=True)
        totals += rng.multinomial(len(ix), counts / counts.sum(), size=bootstraps) @ pats

    def f1(v):
        den = 2 * v[:, 0] + v[:, 1] + v[:, 2]
        return np.divide(2 * v[:, 0], den, out=np.zeros(len(v)), where=den > 0)

    delta = f1(totals[:, :3]) - f1(totals[:, 3:])
    lo, hi = np.quantile(delta, [.025, .975])
    return dict(delta_f1=ra['cv_metrics']['f1'] - rb['cv_metrics']['f1'], interval_95=[float(lo), float(hi)],
                probability_better=float((delta > 0).mean()), fault_units=int(ua.positive.sum()),
                new_hits=int((ua.hit & ~ub.hit).sum()), lost_hits=int((~ua.hit & ub.hit).sum()),
                new_false_alarms=int((ua.fp & ~ub.fp).sum()), removed_false_alarms=int((~ua.fp & ub.fp).sum()),
                current=ra['cv_metrics'], reference=rb['cv_metrics'],
                note='Paired device-cluster bootstrap keeps all periods of a device together; stratified by ever-fault status. Reused CV selection remains exploratory.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--parent-cv', help='stage-1 parent CV folder (required for rerank=gated)')
    a = p.parse_args()
    from experiment_contract import validate_recipe, with_defaults
    recipe = validate_recipe(with_defaults(read(a.plan)))
    if recipe['rerank'] in ('gated','parallel'):
        require(a.parent_cv, 'Gated rerank cross-validation needs --parent-cv')
        import rerank, parallel_branch
        module = parallel_branch if recipe['rerank']=='parallel' else rerank
        r = module.oof_tables(read(a.config), recipe, a.out, a.parent_cv)
    else:
        r = oof_tables(read(a.config), recipe, a.out)
    print('CV_DONE best_step=%s f1=%.4f fault_units=%s' % (r['best_step'], r['cv_metrics']['f1'], r['fault_units']), flush=True)


if __name__ == '__main__':
    main()
