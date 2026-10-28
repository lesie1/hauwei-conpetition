"""Error-case evidence for the agents: what separates high-score true faults from false alarms.

Works on cross-validation out-of-fold units inside TRAIN, so no validation period is spent.
High-score units are those crossing the CV-selected threshold at the CV-selected step: faults
with a timely score above it (TP-like) and healthy devices above it (FP-like). For each unit,
take the latest feature row at its most confident decision time and compare the groups per
feature by AUC with a bootstrap interval. Also serves as a cheap pre-screen for a proposed
feature program before full training.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import read, save, require
from history_store import cache_path

SEED = 20261027


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_interval(pos, neg, n=300, seed=SEED):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    v = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]


def high_units(oof, step, threshold):
    """Units at the chosen step whose relevant max score crosses the CV threshold."""
    u = oof.loc[oof.step == step].copy()
    tp = u.positive & (u.max_hit_score >= threshold)
    fp = ~u.positive & (u.max_score >= threshold)
    u = u.loc[tp | fp].copy()
    u['group'] = np.where(u.positive, 'TP', 'FP')
    return u


def feature_snapshot(c, units, names):
    """Latest feature row available at each unit's decision_time (its most confident score)."""
    cache = cache_path(c)
    files = sorted((cache / 'rows').glob('*.parquet'))
    devices = sorted(set(units.device.tolist()))
    data = pq.read_table([str(p) for p in files], columns=['device', 'time'] + names,
                         filters=[('device', 'in', devices)]).to_pandas().sort_values(['device', 'time'])
    by = {d: (g.time.to_numpy(), g[names].to_numpy(np.float64)) for d, g in data.groupby('device', sort=False)}
    out = np.full((len(units), len(names)), np.nan)
    for i, (d, at) in enumerate(zip(units.device.to_numpy(), units.decision_time.to_numpy())):
        if int(d) not in by or at < 0:
            continue
        t, v = by[int(d)]
        j = np.searchsorted(t, at, side='right') - 1
        if j >= 0:
            out[i] = v[j]
    return pd.DataFrame(out, columns=names, index=units.index)


def low_units(oof, step, threshold):
    u = oof.loc[oof.step == step].copy()
    fn = u.positive & ~(u.max_hit_score >= threshold)
    tn = ~u.positive & ~(u.max_score >= threshold)
    u = u.loc[fn | tn].copy()
    u['group'] = np.where(u.positive, 'FN', 'TN')
    return u


def error_counts(oof, step, threshold):
    u=oof.loc[oof.step==step]
    positive=u.positive.to_numpy(bool)
    hit=u.max_hit_score.to_numpy()>=threshold
    alarm=u.max_score.to_numpy()>=threshold
    fn=positive & ~hit
    no_clock=u.timely_prediction_opportunities.to_numpy()==0
    no_score=~np.isfinite(u.max_hit_score.to_numpy())
    return dict(TP=int((positive & hit).sum()),
        FP=int((alarm & (~positive | ~hit)).sum()), FN=int(fn.sum()),
        TN=int((~positive & ~alarm).sum()),
        fn_no_timely_clock=int((fn & no_clock).sum()),
        fn_no_timely_score=int((fn & ~no_clock & no_score).sum()),
        fn_scored_below_threshold=int((fn & ~no_score).sum()),
        fn_with_outside_window_alarm=int((fn & alarm).sum()),
        note='Outside-window alarms overlap FN categories and also count as FP. Missing timely scores do not prove no raw data.')


def cluster_interval(pos, neg, pdev, ndev, n=200):
    # A device can appear in several periods, even in both groups. Resample it once
    # jointly rather than pretending period-device units are independent devices.
    devices=np.unique(np.r_[pdev,ndev]);rng=np.random.default_rng(SEED)
    vals=[]
    for _ in range(n):
        sampled=rng.choice(devices,len(devices),replace=True)
        ids,counts=np.unique(sampled,return_counts=True)
        mult=dict(zip(ids,counts))
        a=np.repeat(pos,[mult.get(d,0) for d in pdev])
        b=np.repeat(neg,[mult.get(d,0) for d in ndev])
        value=auc(a,b)
        if np.isfinite(value):vals.append(value)
    return [float(v) for v in np.quantile(vals,[.025,.975])] if len(vals)>=n//2 else [np.nan,np.nan]


def contrast(c, names, cv_dir, top=25, kind='high_score_contrast'):
    cv_dir=Path(cv_dir);r=read(cv_dir/'oof_complete.json')
    from common import sha
    if r.get('oof_sha256'):
        require(sha(cv_dir/'oof.parquet')==r['oof_sha256'],'OOF evidence changed')
    oof=pd.read_parquet(cv_dir/'oof.parquet')
    threshold=r['cv_metrics']['threshold'];step=r['best_step']
    high=kind=='high_score_contrast'; positive_name,negative_name=('TP','FP') if high else ('FN','TN')
    full=(high_units if high else low_units)(oof,step,threshold)
    counts={g:int((full.group==g).sum()) for g in (positive_name,negative_name)}
    # Bounded, deterministic diagnostic sampling only; scorer and cohorts are unchanged.
    cap=c.get('contrast_max_units_per_group',1000)
    parts=[g.sample(min(len(g),cap),random_state=SEED) for _,g in full.groupby('group',sort=True)]
    units=pd.concat(parts).sort_index() if parts else full
    names=list(names)
    x=feature_snapshot(c,units,names) if len(units) else pd.DataFrame(columns=names)
    pos=units.group.eq(positive_name).to_numpy();neg=units.group.eq(negative_name).to_numpy()
    devices=units.device.to_numpy()
    rows=[]
    for name in names:
        v=x[name].to_numpy(float);a=auc(v[pos],v[neg])
        rows.append(dict(feature=name,auc_tp_vs_fp=a if high else None,auc_positive_vs_negative=a,
            separation=abs(a-.5) if np.isfinite(a) else 0.,
            positive_median=float(np.nanmedian(v[pos])) if np.isfinite(v[pos]).any() else None,
            negative_median=float(np.nanmedian(v[neg])) if np.isfinite(v[neg]).any() else None,
            positive_missing=float(np.isnan(v[pos]).mean()) if pos.any() else None,
            negative_missing=float(np.isnan(v[neg]).mean()) if neg.any() else None))
    rows.sort(key=lambda d:-d['separation'])
    for d in rows[:top]:
        v=x[d['feature']].to_numpy(float)
        pp=pos & np.isfinite(v); nn=neg & np.isfinite(v)
        enough=len(np.unique(devices[pp]))>=3 and len(np.unique(devices[nn]))>=3
        ci=cluster_interval(v[pp],v[nn],devices[pp],devices[nn]) if enough else [np.nan,np.nan]
        d['auc_interval_95']=ci
        d['credible']=bool(enough and np.isfinite(ci).all() and (ci[0]>.5 or ci[1]<.5))
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        return None if isinstance(v,float) and not np.isfinite(v) else v
    return clean(dict(kind=kind,step=step,threshold=threshold,cv_metrics=r['cv_metrics'],
        tp_units=counts.get('TP',0),fp_units=counts.get('FP',0),group_counts=counts,
        sampled_group_counts={g:int((units.group==g).sum()) for g in counts},
        features=rows[:top],credible_features=[d['feature'] for d in rows[:top] if d.get('credible')],
        eligible_units=int((oof.step==step).sum()),error_counts=error_counts(oof,step,threshold),
        note='Exploratory TRAIN OOF evidence; device-cluster bootstrap, no multiple-testing correction. '
             'Features are the latest causal row at the best relevant decision time, not attribution of a pooled score. '
             'Fault decision times use outcomes for retrospective diagnosis only. Unscored units stay in counts; '
             'small groups or missing snapshots are insufficient evidence, not proof of absent signal.'))


def atlas(c,names,cv_dir,top=25):
    high=contrast(c,names,cv_dir,top,'high_score_contrast')
    low=contrast(c,names,cv_dir,top,'missed_fault_contrast')
    return dict(kind='error_atlas',error_counts=high['error_counts'],precision=high,recall=low)


def prescreen(c,cv_dir,new_names):
    high=contrast(c,new_names,cv_dir,len(new_names),'high_score_contrast')
    low=contrast(c,new_names,cv_dir,len(new_names),'missed_fault_contrast')
    passed=sorted(set(high['credible_features']+low['credible_features']))
    return dict(kind='feature_prescreen',prescreen_passed=bool(passed),passed_features=passed,
        high_score=high,missed_faults=low,tp_units=high['tp_units'],fp_units=high['fp_units'],
        features=high['features'],note='A pass permits training; it does not establish model improvement.')


def screen_decision(screen,c,request,used):
    mode=c.get('prescreen_mode','balanced')
    high=screen.get('high_score',screen)
    passed=bool(high.get('credible_features')) if mode=='strict_high' else screen['prescreen_passed']
    if passed:return dict(train=True,route='evidence',reason='Univariate diagnostic signal; verify in a model control')
    reason=request.get('exploration_reason')
    already=request.get('job') in used
    if mode=='balanced' and isinstance(reason,str) and reason.strip() and (already or len(used)<c.get('feature_exploration_budget',2)):
        return dict(train=True,route='exploration',reason=reason)
    return dict(train=False,route='deferred',reason='No univariate evidence in the permitted contrasts. '
        'Not proof of uselessness: a hypothesis-linked recall/interaction experiment may use a remaining exploration slot with exploration_reason.')


def main():
    import argparse
    from history_store import checked_manifest
    from feature_programs import materialize,program
    p=argparse.ArgumentParser();p.add_argument('task',choices=['contrast','missed','atlas','prescreen'])
    for field in ('config','program','cv','out'):p.add_argument('--'+field,required=True)
    a=p.parse_args();c=read(a.config);cc,m=materialize(c,a.program,checked_manifest(c))
    if a.task=='prescreen':result=prescreen(cc,a.cv,program(c,a.program)['features'])
    elif a.task=='atlas':result=atlas(cc,m['features'],a.cv)
    else:result=contrast(cc,m['features'],a.cv,kind='high_score_contrast' if a.task=='contrast' else 'missed_fault_contrast')
    save(a.out,result)


if __name__=='__main__':main()
