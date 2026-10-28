"""Opt-in memory adapter: preserve v1 columns, add short-window shapes and chip-local topology.

No fitted state or labels. These are selected structural descriptors, not full Multi-BSFE.
Other device domains keep their own adapters; no DQ assumptions enter the generic controller.
"""
import numpy as np
import pandas as pd
import raw_events as base

VERSION='raw_ce_spatial_v3'
RAW_COLUMNS=base.RAW_COLUMNS
WINDOWS={'15m':900,'1h':3600,'6h':21600}
SHAPES=['dq_groups_max','dq_run_max','beat_groups_max','beat_run_max',
        'row_bits_mean','row_bits_max','col_bits_mean','col_bits_max']
LOCAL=['chip_cells','chip_rows','chip_cols','chip_banks']
EXTRA=([f'raw_{w}_{k}' for w in ('15m','1h') for k in base.WINDOW_KEYS]+
       [f'spatial_{w}_{k}_{op}' for w in WINDOWS for k in SHAPES for op in ('mean','max')]+
       [f'spatial_{w}_distinct_{k}' for w in WINDOWS for k in LOCAL]+
       [f'spatial_{w}_bit_observed_events' for w in WINDOWS])
FEATURES=base.FEATURES+EXTRA


def _runs(v):
    padded=np.r_[False,v,False].astype(np.int8)
    starts=np.flatnonzero(np.diff(padded)==1);ends=np.flatnonzero(np.diff(padded)==-1)
    return len(starts),int((ends-starts).max()) if len(starts) else 0


def shape(text):
    grid=base.decode_bits(text)
    if grid is None or not grid.any():return [np.nan]*len(SHAPES)
    rows=[_runs(v) for v in grid];cols=[_runs(v) for v in grid.T]
    return [max(v[0] for v in rows),max(v[1] for v in rows),
            max(v[0] for v in cols),max(v[1] for v in cols),
            grid.sum(1).mean(),grid.sum(1).max(),grid.sum(0).mean(),grid.sum(0).max()]


def events(frame):
    ordered=frame.sort_values('LogTime',key=lambda s:pd.to_numeric(s),kind='stable').reset_index(drop=True)
    ev=base.events(ordered)
    codes,texts=pd.factorize(ordered.burst_info.fillna('').astype(str))
    values=np.array([shape(t) for t in texts],float).reshape(len(texts),len(SHAPES))
    for i,name in enumerate(SHAPES):ev[name]=values[codes,i]
    return ev


def features_at(ev,anchors):
    anchors=np.asarray(anchors,np.int64)
    old=base.features_at(ev,anchors)
    out=np.full((len(anchors),len(EXTRA)),np.nan,np.float32)
    if ev is None or not len(ev):return np.c_[old,out]
    t=ev.time.to_numpy(np.int64);hi=np.searchsorted(t,anchors,side='right');idx={n:i for i,n in enumerate(EXTRA)}
    rank,bg,bank,row,col,chip=(ev[n].to_numpy() for n in ('RankId','BankgroupId','BankId','RowId','ColumnId','deviceID'))
    # Physical locations use chip identity; unknown addresses are omitted, not merged as one location.
    parts={'chip_cells':(rank,chip,bg,bank,row,col),'chip_rows':(rank,chip,bg,bank,row),
           'chip_cols':(rank,chip,bg,bank,col),'chip_banks':(rank,chip,bg,bank)}
    def sums(v,lo):
        cs=np.r_[0.,np.cumsum(v)];return cs[hi]-cs[lo]
    for w,width in WINDOWS.items():
        lo=np.searchsorted(t,anchors-width,side='right');present=hi>lo
        for name in SHAPES:
            v=ev[name].to_numpy(float);valid=np.isfinite(v);count=sums(valid,lo)
            mean=np.divide(sums(np.where(valid,v,0),lo),count,out=np.full(len(hi),np.nan),where=count>0)
            out[:,idx[f'spatial_{w}_{name}_mean']]=mean
            out[:,idx[f'spatial_{w}_{name}_max']]=base.range_reduce(v,lo,hi,np.maximum)
        out[:,idx[f'spatial_{w}_bit_observed_events']]=sums(np.isfinite(ev.err_bits.to_numpy()),lo)
        for name,columns in parts.items():
            valid=np.logical_and.reduce([v>=0 for v in columns]);codes=base._codes(*columns)
            # Invalid positions have no location identity. Count on the filtered timeline.
            vlo=np.searchsorted(t[valid],anchors-width,side='right');vhi=np.searchsorted(t[valid],anchors,side='right')
            count=base.window_distinct(codes[valid],vlo,vhi)
            out[:,idx[f'spatial_{w}_distinct_{name}']]=np.where(vhi>vlo,count,np.nan)
        if w=='6h':continue
        prefix=f'raw_{w}_'
        out[:,idx[prefix+'events']]=hi-lo
        for key,colname,op in [('max_err_bits','err_bits',np.maximum),('max_dq_count','dq_count',np.maximum),
                ('max_beat_count','beat_count',np.maximum),('min_dq_interval','min_dq_interval',np.minimum),
                ('max_beat_interval','max_beat_interval',np.maximum)]:
            out[:,idx[prefix+key]]=base.range_reduce(ev[colname].to_numpy(float),lo,hi,op)
        for key,v in dict(adjacent_dq_events=ev.adjacent_dq_pairs>0,multi_dq_events=ev.dq_count>1,
                         multi_beat_events=ev.beat_count>1,risky_events=ev.risky.fillna(0)).items():
            out[:,idx[prefix+key]]=sums(np.asarray(v,float),lo)
        for key,columns in dict(cells=(rank,bg,bank,row,col),rows=(rank,bg,bank,row),cols=(rank,bg,bank,col),
                                banks=(rank,bg,bank),chips=(rank,chip)).items():
            out[:,idx[prefix+'distinct_'+key]]=base.window_distinct(base._codes(*columns),lo,hi)
    return np.c_[old,out].astype(np.float32)


def mapping_features():
    return base.mapping_features()+[dict(name=n,column=n,kind='numeric_unknown',unit=None,
        description='Memory-only v3 causal descriptor: '+n+'; chip-local counts include deviceID; unavailable locations remain missing') for n in EXTRA]
