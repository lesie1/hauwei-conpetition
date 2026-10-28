"""Independent all-row branch. Device-balanced shallow boosting, NOT M2-MFP.

The second branch sees every eligible TRAIN row, irrespective of stage-1 score.
Its representation is the current numeric event-row input (possibly already aggregated).
Device weights total one per device; fixed depth 2 / sqrt class weight differ deliberately
from the parent. This is a named multi-factor architecture control, not a weight ablation.
Max fusion can promote low stage-1 scores. Threshold/step selection remains unchanged.
"""
import base64
import copy
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from common import require, read, save, sha, digest
from rerank import _booster, _matrix, _folds, stage1_identity, positive_weight, params, fit, load_model, cpu_copy

MAGIC = b'PARALLEL1\n'


class Parallel:
    def __init__(self, primary, auxiliary, primary_steps=None):
        self.primary, self.auxiliary = primary, auxiliary
        self.primary_steps = primary_steps

    def inplace_predict(self, x, iteration_range=(0, 0)):
        a = self.primary.inplace_predict(x, iteration_range=(0, self.primary_steps or 0))
        b = self.auxiliary.inplace_predict(x, iteration_range=iteration_range)
        return np.maximum(a, b)

    def get_score(self, importance_type='gain'):
        # Importance is for the auxiliary branch; primary remains in its parent receipt.
        return self.auxiliary.get_score(importance_type=importance_type)

    def set_param(self, p):
        self.primary.set_param(p); self.auxiliary.set_param(p)

    def save_raw(self, raw_format='ubj'):
        doc = dict(primary_steps=self.primary_steps, fusion='max',
                   primary=base64.b64encode(bytes(self.primary.save_raw(raw_format='ubj'))).decode(),
                   auxiliary=base64.b64encode(bytes(self.auxiliary.save_raw(raw_format='ubj'))).decode())
        return bytearray(MAGIC + json.dumps(doc, sort_keys=True).encode())

    def save_model(self, path):
        Path(path).write_bytes(bytes(self.save_raw()))


def from_raw(raw, threads=1):
    doc = json.loads(bytes(raw)[len(MAGIC):])
    require(doc['fusion'] == 'max', 'Unknown fusion')
    return Parallel(_booster(base64.b64decode(doc['primary']), threads),
                    _booster(base64.b64decode(doc['auxiliary']), threads), doc['primary_steps'])


def branch_recipe(recipe):
    r = copy.deepcopy(recipe)
    r.update(rerank='none', rerank_gate=.5, base_weight='device', weight_policy='sqrt_ratio')
    r['params'].update(max_depth=2, min_child_weight=1.)
    return r


def balance(w, dev):
    """Normalize only the training subset; held-out devices never enter these totals."""
    _, inverse = np.unique(dev, return_inverse=True)
    total = np.bincount(inverse, weights=w)
    require((total > 0).all(), 'Device weight totals must be positive')
    return (w / total[inverse]).astype(np.float32)


def fold_auxiliary(x, y, w, dev, rowfold, fold, recipe, c, rounds):
    keep = rowfold != fold
    return fit(x[keep], y[keep], balance(w[keep], dev[keep]), branch_recipe(recipe), c, rounds)


def train_full(c, recipe, m, out, stage1_folder, stage1_cv):
    from prediction import predict
    from evaluation import select
    out, stage1_folder = Path(out), Path(stage1_folder)
    prev = read(stage1_folder / 'result.json')
    require(prev['recipe'].get('rerank','none') == 'none', 'Parallel parent must be single-stage')
    require(stage1_identity(prev['recipe']) == stage1_identity(recipe), 'Parallel parent recipe differs')
    require(sha(stage1_folder / 'model.ubj') == prev['model_sha256'], 'Parent model changed')
    tick = time.monotonic()
    x, y, w, dev, audit = _matrix(c, recipe, m, out)
    w = balance(w, dev); br = branch_recipe(recipe)
    p = params(br, c, positive_weight(y,w,br['weight_policy']))
    dt = xgb.QuantileDMatrix(x,label=y,weight=w,max_bin=br['params']['max_bin'],nthread=c['num_threads'])
    del x; gc.collect()
    b = xgb.Booster(p, cache=[dt]); primary = load_model(stage1_folder/'model.ubj', c['num_threads'])
    best = None; stale = 0; history = []; boost = 0.; infer = 0.
    feature_seconds = time.monotonic()-tick
    for step in range(1,c['rounds']+1):
        t = time.monotonic(); b.update(dt,step-1); boost += time.monotonic()-t
        if step != 1 and step % c['eval_every'] and step != c['rounds']:continue
        t = time.monotonic(); model = Parallel(primary,cpu_copy(b,c['num_threads']))
        table, stats = predict(model,c,recipe,m,'val'); metric, grid = select(table)
        infer += time.monotonic()-t
        history.append(dict(step=step,val_period_unweighted_bce=stats['val_period_unweighted_bce'],**metric))
        save(out/'loss_history.json',history)
        if best is None or metric['f1']>best['metrics']['f1']+1e-12:
            best=dict(step=step,metrics=metric,score_sha256=stats['score_sha256']); stale=0
            model.save_model(out/'model.ubj'); grid.to_csv(out/'threshold_search_val.csv',index=False,float_format='%.17g')
        else:stale+=1
        print(f'Parallel step {step}: F1={metric["f1"]:.6f} TP={metric["TP"]} FP={metric["FP"]} FN={metric["FN"]}',flush=True)
        if stale>=c['patience']:break
    actual=json.loads(b.save_config())['learner']['generic_param']['device']
    require(not c['device'].startswith('cuda') or actual.startswith('cuda'),'Unexpected CPU fallback')
    del dt; gc.collect()
    audit=dict(audit,parallel=dict(primary_model_sha256=prev['model_sha256'],fusion='max',
        branch_recipe=br,all_training_rows=True,device_weight_total=1.,
        feature_importance='auxiliary only',note='Generic shallow branch, not raw-event rule-tree reproduction'))
    return best,audit,p,actual,dict(feature_and_matrix_seconds=feature_seconds,boost_updates_seconds=boost,validation_prediction_seconds=infer)


def oof_tables(c, recipe, out, parent_cv, rounds=None):
    from history_store import checked_manifest
    from feature_programs import materialize
    from crossval import periods, score_devices, summarize, SEED
    out,parent_cv=Path(out),Path(parent_cv);out.mkdir(parents=True,exist_ok=True)
    c,m=materialize(c,recipe['feature_program'],checked_manifest(c))
    pr=read(parent_cv/'oof_complete.json')
    require(stage1_identity(pr['recipe'])==stage1_identity(recipe),'Parallel CV parent differs')
    require(pr['recipe'].get('rerank','none')=='none','Parallel CV parent must be single-stage')
    require(sha(parent_cv/'oof.parquet')==pr['oof_sha256'],'Parent OOF changed')
    rounds=rounds or c.get('cv_rounds',400);k=pr['folds'];plist=periods(m,c)
    require(plist==pr['periods'],'CV periods differ')
    key=digest(dict(recipe=recipe,rounds=rounds,eval_every=c['eval_every'],periods=plist,k=k,seed=SEED,
                    manifest=sha(Path(c['active_cache'])/'manifest.json'),parent=pr['oof_sha256'],version=1))
    done=out/'oof_complete.json'
    if done.exists() and read(done)['key']==key:return read(done)
    tick=time.monotonic(); x,y,w,dev,audit=_matrix(c,recipe,m,out)
    registry,fold_of=_folds(c,m,k);rowfold=fold_of[dev]
    steps=sorted({s for s in range(c['eval_every'],rounds+1,c['eval_every'])}|{rounds});rows=[]
    for fold in range(k):
        auxiliary=fold_auxiliary(x,y,w,dev,rowfold,fold,recipe,c,rounds)
        primary=load_model(parent_cv/f'fold_{fold}'/'model.ubj',c['num_threads'])
        model=Parallel(primary,auxiliary,pr['best_step'])
        t=score_devices(model,steps,c,recipe,m,plist,registry.device.to_numpy()[fold_of==fold])
        t['fold']=fold;rows.append(t)
        folder=out/f'fold_{fold}';folder.mkdir(exist_ok=True);model.save_model(folder/'model.ubj')
    del x;gc.collect()
    oof=pd.concat(rows,ignore_index=True);oof.to_parquet(out/'oof.parquet',index=False)
    result=dict(key=key,recipe=recipe,periods=plist,folds=k,rounds=rounds,seconds=time.monotonic()-tick,
        oof_sha256=sha(out/'oof.parquet'),parallel=dict(fusion='max',all_training_rows=True,
        primary_step=pr['best_step'],note='Both fits exclude the held-out devices. Reused CV selects steps/thresholds; exploratory, not a fresh final test.'),**summarize(oof))
    save(done,result);return result
