"""An immutable evaluation identity and transparent, paired model comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import read, save, digest, sha, ROOT, require
from evaluation import period_cohort, metrics, select
from history_store import cache_path
from experiment_contract import differences


def evaluation_contract(c,m,split):
    registry=pd.read_parquet(cache_path(c,split=='test')/'device_metadata.parquet')
    cohort=period_cohort(registry,m['windows'][split],c)
    fields=['device','sn_name','failure','observed_start','observed_end','status','positive']
    return dict(version='caafe_device_comparison_v1',scorer=c['evaluation_protocol'],
        scorer_sha256=sha(ROOT/'evaluation.py'),split=split,period=list(m['windows'][split]),
        minimum_lead_seconds=c['minimum_lead_seconds'],maximum_lead_seconds=c['maximum_lead_seconds'],
        cohort_sha256=digest(cohort[fields].to_dict('records')),
        base_features=m['features'],source_files_digest=digest(m['files']),
        threshold_selection='maximum validation device F1, ties fewer FP then higher threshold',
        missing_policy='eligible faults remain FN; unavailable predictions abstain',
        schedule_policy='actual decision time; no retrospective alarm credit',
        followup_basis=m['coverage'])


def seal_evaluation(c,m,split):
    spec=evaluation_contract(c,m,split);path=Path(c['output_root'])/('evaluation_'+split+'.json')
    if path.exists():require(read(path)==spec,'Evaluation contract or cohort changed')
    else:save(path,spec)
    return digest(spec)


def extended_metrics(table,threshold):
    d=metrics(table,threshold)
    from evaluation import classify
    t=classify(table,threshold);t=t.loc[t.status.eq('eligible')]
    normal=~t.positive.astype(bool)
    d.update(false_alarm_fraction=d['FP']/(d['TP']+d['FP']) if d['TP']+d['FP'] else 0.,
        miss_rate=d['FN']/(d['TP']+d['FN']) if d['TP']+d['FN'] else 0.,
        nonfault_devices=int(normal.sum()),normal_false_alarm_devices=int((normal&t.alarm).sum()),
        normal_device_false_positive_rate=float(t.loc[normal,'alarm'].mean()) if normal.any() else None,
        late_only_fault_alarms=int((t.positive&t.alarm&~t.hit).sum()),
        no_prediction_devices=int(t.max_score.isna().sum()),
        no_prediction_fault_devices=int((t.positive&t.max_score.isna()).sum()))
    return d


def operating_points(table):
    _,grid=select(table);out=[]
    for budget in (50,100,200):
        candidates=grid.loc[grid.FP<=budget].sort_values(['recall','FP','threshold'],ascending=[False,True,False])
        row=candidates.iloc[0]
        out.append(dict(kind='max_fp',target=budget,feasible=True,threshold=float(row.threshold),
            metrics=extended_metrics(table,float(row.threshold))))
    for target in (.1,.2,.3,.5):
        candidates=grid.loc[grid.recall>=target].sort_values(['FP','recall','threshold'],ascending=[True,False,False])
        if not len(candidates):out.append(dict(kind='min_recall',target=target,feasible=False,threshold=None));continue
        row=candidates.iloc[0]
        out.append(dict(kind='min_recall',target=target,feasible=True,threshold=float(row.threshold),
            metrics=extended_metrics(table,float(row.threshold))))
    return out


def frozen_operating_points(table,points):
    return [dict(kind=p['kind'],target=p['target'],feasible=p['feasible'],threshold=p['threshold'],
                 metrics=extended_metrics(table,p['threshold']) if p['feasible'] else None,
                 note='Threshold frozen on validation; test may exceed budget or miss recall target') for p in points]


def paired(current,reference,bootstraps=200):
    a=read(Path(current)/'result.json');b=read(Path(reference)/'result.json')
    require(a['evaluation_id']==b['evaluation_id'],'Incomparable evaluation IDs')
    t=pd.read_csv(Path(current)/'devices_val.csv').set_index('device').sort_index()
    p=pd.read_csv(Path(reference)/'devices_val.csv').set_index('device').sort_index()
    require(t.index.equals(p.index),'Different device sets')
    require(t[['status','positive']].equals(p[['status','positive']]),'Different outcomes')
    t=t.loc[t.status.eq('eligible')];p=p.loc[t.index]
    out=dict(reference=str(reference),changed_factors=differences(b['recipe'],a['recipe']),
        delta={k:a['metrics'][k]-b['metrics'][k] for k in ('f1','precision','recall','TP','FP','FN')},
        new_hits=int((t.hit&~p.hit).sum()),lost_hits=int((~t.hit&p.hit).sum()),
        new_false_alarms=int((t.fp&~p.fp).sum()),removed_false_alarms=int((~t.fp&p.fp).sum()),
        model_identical=a['model_sha256']==b['model_sha256'],
        same_training_samples=(a['sample_audit']['sample_membership_sha256']==b['sample_audit']['sample_membership_sha256']
            if a['sample_audit'].get('sample_membership_sha256') and b['sample_audit'].get('sample_membership_sha256') else None))
    if bootstraps:
        # Pair by device, stratify by true fault status; not independent-test evidence.
        a1=t[['hit','fp','fn']].to_numpy(np.int64);b1=p[['hit','fp','fn']].to_numpy(np.int64)
        groups=[np.flatnonzero(t.positive.to_numpy()==x) for x in (False,True)]
        # Compress identical outcomes to avoid huge device bootstrap matrices.
        rng=np.random.default_rng(20261021);delta=np.zeros(bootstraps)
        totals=np.zeros((bootstraps,6),np.int64)
        for ix in groups:
            if not len(ix):continue
            patterns,counts=np.unique(np.c_[a1[ix],b1[ix]],axis=0,return_counts=True)
            totals+=rng.multinomial(len(ix),counts/counts.sum(),size=bootstraps)@patterns
        def f1(v):
            den=2*v[:,0]+v[:,1]+v[:,2]
            return np.divide(2*v[:,0],den,out=np.zeros(len(v)),where=den>0)
        delta=f1(totals[:,:3])-f1(totals[:,3:])
        out['conditional_paired_f1_interval_95']=[float(x) for x in np.quantile(delta,[.025,.975])]
        out['uncertainty_note']='Descriptive paired bootstrap at selected validation thresholds. Does not correct adaptive search or threshold selection; not a generalization guarantee.'
    return out
