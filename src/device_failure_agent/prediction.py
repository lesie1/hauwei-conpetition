"""Bounded-memory scheduled inference. All scores are timestamped when available."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from common import DAY, require
from history_store import (cache_path, iter_devices, snapshot_features, anchors, Writer,
                           feature_names, batches)
from evaluation import period_cohort, annotate, classify


def pool_scores(times,values,points,step,kind):
    """Windows are [previous clock tick, current tick); alarm at current tick.

    latest/full-history permit an observation exactly at the decision time.
    Window rules match the old absolute hour-end convention, never backdate a peak.
    """
    p=np.full(len(points),np.nan);support=np.zeros(len(points),np.int64)
    end=np.searchsorted(times,points,side='right' if kind in ('latest','all_history_max') else 'left')
    if kind in ('latest','all_history_max'):
        use=end>0
        v=values if kind=='latest' else np.maximum.accumulate(values)
        p[use]=v[end[use]-1];support=end
        return p,support
    start=np.searchsorted(times,points-step,side='left');support=end-start
    # Vectorized over windows: list every window's elements once (windows may overlap),
    # then reduce per window segment. Equal to the per-window loop (tests compare both).
    idx=np.flatnonzero(end>start)
    if not len(idx):return p,support
    lengths=(end-start)[idx];offsets=np.r_[0,np.cumsum(lengths)[:-1]]
    seg=np.repeat(np.arange(len(idx)),lengths)
    vals=np.asarray(values,np.float64)[np.repeat(start[idx],lengths)+np.arange(lengths.sum())-np.repeat(offsets,lengths)]
    if kind=='window_max':p[idx]=np.maximum.reduceat(vals,offsets)
    elif kind=='window_mean':p[idx]=np.add.reduceat(vals,offsets)/lengths
    elif kind=='window_second':
        ok=lengths>=2
        ordered=vals[np.lexsort((vals,seg))]
        p[idx[ok]]=ordered[(offsets+lengths)[ok]-2]
    return p,support


def scored_devices(model,cache,base):
    """One batched inference per Parquet batch rather than per device."""
    pieces=[];current=None
    for b in batches(cache,'rows',base+['device','time']):
        f=b.to_pandas();v=model.inplace_predict(f[base].to_numpy(np.float32))
        a=pd.DataFrame(dict(device=f.device,time=f.time,score=v))
        codes=a.device.to_numpy();cuts=np.r_[0,np.flatnonzero(np.diff(codes))+1,len(a)]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            code=int(codes[lo])
            if current is not None and code!=current:
                joined=pd.concat(pieces,ignore_index=True)
                yield current,joined.time.to_numpy(np.int64),joined.score.to_numpy(np.float64)
                pieces=[]
            current=code;pieces.append(a.iloc[lo:hi])
    if pieces:
        joined=pd.concat(pieces,ignore_index=True)
        yield current,joined.time.to_numpy(np.int64),joined.score.to_numpy(np.float64)


def _score_shard(args):
    """Worker: raw per-device pooled scores for one cache shard (no threshold state).

    Returns per-device arrays only; the parent merges them in device order, so the
    result is identical to the serial path (tests compare full tables and hashes).
    """
    model_raw,files,c,recipe,manifest,split=args
    import xgboost as xgb
    from rerank import from_raw
    model=from_raw(model_raw,1)   # plain booster or two-stage composite
    cache=cache_path(c,split=='test');base=manifest['features']
    registry=pd.read_parquet(cache/'device_metadata.parquet')
    cohort=period_cohort(registry,manifest['windows'][split],c).set_index('device',drop=False)
    step=recipe['prediction_interval_hours']*3600;out=[]
    import pyarrow.parquet as pq
    for f in files:
        frame=pq.read_table(f,columns=base+['device','time']).to_pandas()
        if not len(frame):continue
        codes=frame.device.to_numpy();cuts=np.r_[0,np.flatnonzero(np.diff(codes))+1,len(frame)]
        if recipe['feature_set']=='base':
            v=model.inplace_predict(frame[base].to_numpy(np.float32)).astype(np.float64)
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            code=int(codes[lo]);r=cohort.loc[code]
            points=anchors(manifest['windows'][split],manifest['windows']['train'][0],recipe['prediction_interval_hours'],int(r.failure))
            points=points[(points>=r.observed_start)&(points<=r.observed_end)]
            t=frame.time.to_numpy(np.int64)[lo:hi]
            if recipe['feature_set']=='base':
                p,support=pool_scores(t,v[lo:hi],points,step,recipe['score_aggregation'])
            else:
                a,support=snapshot_features(t,frame[base].to_numpy(np.float32)[lo:hi],points,base,recipe['feature_set'])
                p=np.full(len(points),np.nan);active=support>0
                if active.any():p[active]=model.inplace_predict(a[active]).astype(np.float64)
            out.append((code,t,points,support,p))
    return out


def _parallel_scores(model,c,recipe,manifest,split,workers):
    """Yields (code, times, points, support, scores) in ascending device order across shards."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    cache=cache_path(c,split=='test')
    files=sorted(str(p) for p in (cache/'rows').glob('*.parquet'))
    raw=bytes(model.save_raw(raw_format='ubj'))
    groups=[files[i:i+1] for i in range(len(files))]
    # spawn, never fork: the parent already holds XGBoost/OpenMP threads, and a forked child
    # can inherit a held lock and deadlock (observed on Linux). Spawned workers start clean.
    # Bounded window: at most 2*workers shard results are in flight or buffered, consumed in
    # shard order. Executor.map would submit every shard at once and buffer all finished
    # results while the single-threaded merge lags, which exhausted a 90 GiB container.
    from collections import deque
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn')) as pool:
        last=-1;pending=deque();jobs=iter(groups)
        def submit():
            g=next(jobs,None)
            if g is not None:pending.append(pool.submit(_score_shard,(raw,g,c,recipe,manifest,split)))
        for _ in range(2*workers):submit()
        while pending:
            part=pending.popleft().result();submit()
            for item in part:
                # A device never spans shards (Writer flushes whole devices); order must ascend.
                require(item[0]>last,'Parallel scoring requires devices sorted and unique across shards')
                last=item[0];yield item
            del part


def predict(model,c,recipe,manifest,split,threshold=None,score_folder=None):
    cache=cache_path(c,split=='test');base=manifest['features']
    registry=pd.read_parquet(cache/'device_metadata.parquet')
    cohort=period_cohort(registry,manifest['windows'][split],c).set_index('device',drop=False)
    require(cohort.index.tolist()==list(range(len(cohort))),'Device IDs must be contiguous')
    ndev=len(cohort);step=recipe['prediction_interval_hours']*3600
    counts={k:np.zeros(ndev,np.int64) for k in ('daily_predictions','timely_prediction_opportunities',
        'scheduled_predictions','history_opportunities','timely_history_opportunities',
        'empty_window_opportunities','insufficient_window_opportunities','scored_timely_opportunities')}
    maxima=np.full(ndev,-np.inf);hits=np.full(ndev,-np.inf)
    first_alarm=np.full(ndev,np.inf);first_hit=np.full(ndev,np.inf)
    score_hash=hashlib.sha256();loss_sum=0.;loss_n=0
    writer=Writer(score_folder) if score_folder else None
    def schedule(code):
        r=cohort.loc[code]
        points=anchors(manifest['windows'][split],manifest['windows']['train'][0],
                       recipe['prediction_interval_hours'],int(r.failure))
        physical=(points>=r.observed_start)&(points<=r.observed_end)
        return points[physical]
    # Every registry member is retained even when there are no source rows.
    schedules={}
    for code,r in cohort.iterrows():
        points=schedule(code)
        counts['scheduled_predictions'][code]=len(points)
        if r.status=='eligible' and r.positive:
            ok=points<=r.failure-c['minimum_lead_seconds']
            if c.get('maximum_lead_seconds') is not None:ok &= (r.failure-points<=c['maximum_lead_seconds'])
            counts['timely_prediction_opportunities'][code]=int(ok.sum())

    def collect(meta,p):
        nonlocal loss_sum,loss_n
        if not len(meta):return
        m=annotate(meta,cohort.reset_index(drop=True),c['minimum_lead_seconds'])
        p=np.asarray(p,np.float64)
        require((np.isnan(p)|((p>=0)&(p<=1))).all(),'Invalid model scores')
        codes=m.device.to_numpy(np.int64);finite=np.isfinite(p);timely=m.timely_opportunity.to_numpy()
        # Missing window scores are deliberate abstention; inference errors raise above.
        for key,mask in [('daily_predictions',finite),('history_opportunities',m.history_rows.to_numpy()>0),
            ('timely_history_opportunities',timely&(m.history_rows.to_numpy()>0)),
            ('scored_timely_opportunities',timely&finite),
            ('empty_window_opportunities',(m.support_rows.to_numpy()==0)&(m.history_rows.to_numpy()>0)),
            ('insufficient_window_opportunities',(m.support_rows.to_numpy()==1)&~finite)]:
            np.add.at(counts[key],codes[mask],1)
        np.maximum.at(maxima,codes[finite],p[finite])
        np.maximum.at(hits,codes[finite&timely],p[finite&timely])
        if threshold is not None:
            alarm=finite&(p>=threshold)
            np.minimum.at(first_alarm,codes[alarm],m.time.to_numpy()[alarm])
            np.minimum.at(first_hit,codes[alarm&timely],m.time.to_numpy()[alarm&timely])
        use=m.primary_eligible.to_numpy()&finite
        q=np.clip(p[use],1e-12,1-1e-12);y=m.label.to_numpy()[use]
        loss_sum+=float(-(y*np.log(q)+(1-y)*np.log1p(-q)).sum());loss_n+=int(use.sum())
        # Row records are interleaved before hashing, independent of batch boundaries.
        record=np.empty(len(m),dtype=[('device','<i8'),('time','<i8'),('score','<f8')])
        record['device']=codes;record['time']=m.time.to_numpy();record['score']=p
        score_hash.update(record.tobytes())
        if writer:
            m['score']=p;writer.add(m)

    def make_meta(code,times,points,support):
        return pd.DataFrame(dict(device=code,time=points,
            history_rows=np.searchsorted(times,points,side='right'),support_rows=support))

    workers=int(c.get('score_workers',1) or 1)
    if workers>1:
        # Same per-device computation as below, sharded over processes; merged in device order.
        frames=[];values=[];length=0
        for code,t,points,support,p in _parallel_scores(model,c,recipe,manifest,split,workers):
            frames.append(make_meta(code,t,points,support));values.append(p);length+=len(points)
            if length>=65536:
                collect(pd.concat(frames,ignore_index=True),np.concatenate(values));frames=[];values=[];length=0
        if frames:collect(pd.concat(frames,ignore_index=True),np.concatenate(values))
    elif recipe['feature_set']=='base':
        frames=[];values=[];length=0
        for code,t,v in scored_devices(model,cache,base):
            points=schedule(code)
            p,support=pool_scores(t,v,points,step,recipe['score_aggregation'])
            frames.append(make_meta(code,t,points,support));values.append(p);length+=len(points)
            if length>=65536:
                collect(pd.concat(frames,ignore_index=True),np.concatenate(values));frames=[];values=[];length=0
        if frames:collect(pd.concat(frames,ignore_index=True),np.concatenate(values))
    else:
        frames=[];features=[];length=0
        def flush():
            m=pd.concat(frames,ignore_index=True);x=np.concatenate(features)
            p=np.full(len(m),np.nan);active=m.history_rows.to_numpy()>0
            if active.any():p[active]=model.inplace_predict(x[active])
            collect(m,p)
        for code,t,x in iter_devices(cache,base):
            points=schedule(code)
            a,n=snapshot_features(t,x,points,base,recipe['feature_set'])
            frames.append(make_meta(code,t,points,n));features.append(a);length+=len(points)
            if length>=65536:flush();frames=[];features=[];length=0
        if frames:flush()
    if writer:writer.flush()
    table=cohort.reset_index(drop=True)
    table['max_score']=np.where(np.isfinite(maxima),maxima,np.nan)
    table['max_hit_score']=np.where(np.isfinite(hits),hits,np.nan)
    for k,a in counts.items():table[k]=a
    if threshold is not None:
        table=classify(table,threshold)
        table['first_alarm_time']=np.where(np.isfinite(first_alarm),first_alarm,np.nan)
        table['first_timely_alarm_time']=np.where(np.isfinite(first_hit),first_hit,np.nan)
        table['lead_days']=np.where(table.hit,(table.failure-table.first_timely_alarm_time)/DAY,np.nan)
    return table,dict(score_sha256=score_hash.hexdigest(),
        val_period_unweighted_bce=loss_sum/loss_n if loss_n else None,scored_rows=loss_n)
