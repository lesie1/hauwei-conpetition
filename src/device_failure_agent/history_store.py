"""One normalized registry; train views are lazy and feature-version aware."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import DAY,read,save,sha,digest,require
from history_math import Writer,cumulative_features,batches,verify
import history_math as math

META=['device','time','failure','observed_start','observed_end','history_rows']


def cache_path(c,final_test=False):
    return Path(c.get('active_cache') or Path(c['output_root'])/('cache_test' if final_test else 'cache'))


def checked_manifest(c,final_test=False):
    require(not final_test, 'TEST is disabled in these exploration releases')
    m=verify(cache_path(c,False))
    require(not m['test_opened'], 'Wrong cache partition')
    if c.get('selection_window'):
        import copy
        m=copy.deepcopy(m)
        lo,hi=c['selection_window'];a,b=m['windows']['val']
        require(a<=lo<hi<=b, 'Invalid selection validation window')
        m['windows']['val']=[lo,hi]
    return m


def windows(c):return read(cache_path(c)/'manifest.json')['windows']


def iter_devices(cache,features):
    pieces=[];current=None
    for b in batches(cache,'rows',features+META):
        f=b.to_pandas();codes=f.device.to_numpy()
        cuts=np.r_[0,np.flatnonzero(np.diff(codes))+1,len(f)]
        for a,z in zip(cuts[:-1],cuts[1:]):
            code=int(codes[a])
            if current is not None and current!=code:
                require(code>current,'Rows must be sorted by device')
                joined=pd.concat(pieces,ignore_index=True)
                yield current,joined.time.to_numpy(np.int64),joined[features].to_numpy(np.float32)
                pieces=[]
            current=code;pieces.append(f.iloc[a:z])
    if pieces:
        f=pd.concat(pieces,ignore_index=True)
        yield current,f.time.to_numpy(np.int64),f[features].to_numpy(np.float32)


def anchors(window,origin,interval_hours,failure=-1):
    lo,hi=window;step=interval_hours*3600
    first=origin+((lo-origin)//step+1)*step
    return np.arange(first,min(hi,failure) if failure>=0 else hi,step,dtype=np.int64)


def feature_names(base,kind):
    if kind=='base':return list(base)
    names=math.feature_names(base)
    if kind=='recent':names += [n+'__recent_mean_1d' for n in base]+[n+'__prior_mean_7d' for n in base]+['rows_recent_1d','rows_prior_7d']
    return names


def snapshot_features(t,x,points,base,kind):
    n=np.searchsorted(t,points,side='right')
    if kind=='base':
        a=np.full((len(points),len(base)),np.nan,np.float32);use=n>0;a[use]=x[n[use]-1];return a,n
    a,n=cumulative_features(t,x,points)
    if kind=='recent':
        ok=np.isfinite(x);s=np.vstack([np.zeros(x.shape[1]),np.cumsum(np.where(ok,x,0),axis=0,dtype=np.float64)])
        counts=np.vstack([np.zeros(x.shape[1]),np.cumsum(ok,axis=0)])
        middle=np.searchsorted(t,points-DAY,side='right');start=np.searchsorted(t,points-8*DAY,side='right')
        def avg(i,j):
            den=counts[j]-counts[i]
            return np.divide(s[j]-s[i],den,out=np.full_like(den,np.nan),where=den>0)
        a=np.c_[a,avg(middle,n),avg(start,middle),n-middle,middle-start].astype(np.float32)
    return a,n


def training_batches(c,r,m,meta_only=False):
    cache=cache_path(c);base=m['features']
    if r['sample_unit']=='event':
        yield from batches(cache,'rows',META if meta_only else base+META);return
    key=digest(dict(source=sha(cache/'manifest.json'),kind=r['feature_set'],window=m['windows']['train']))
    folder=Path(c['output_root'])/'training_views'/key;manifest=folder/'manifest.json'
    if manifest.exists():
        expected=read(manifest)
        for name,h in expected['files'].items():require(sha(folder/name)==h,'Training feature view changed')
    else:
        folder.mkdir(parents=True,exist_ok=True)
        for p in folder.glob('part-*.parquet'):p.unlink()
        writer=Writer(folder);registry=pd.read_parquet(cache/'device_metadata.parquet').set_index('device')
        names=feature_names(base,r['feature_set'])
        for code,t,x in iter_devices(cache,base):
            row=registry.loc[code];points=anchors(m['windows']['train'],m['windows']['train'][0],24,int(row.failure))
            values,n=snapshot_features(t,x,points,base,r['feature_set'])
            f=pd.DataFrame(values,columns=names)
            f['device']=code;f['time']=points;f['history_rows']=n
            for k in ('failure','observed_start','observed_end'):f[k]=int(row[k])
            writer.add(f)
        writer.flush();save(manifest,dict(files={p.name:sha(p) for p in folder.glob('part-*.parquet')}))
    yield from batches(folder.parent,folder.name,META if meta_only else feature_names(base,r['feature_set'])+META)
