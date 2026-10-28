"""CAAFE-style generated numerical programs, immutable versions and prefix checks.

The AST restriction is an interface guard, not a hostile-code security sandbox.
Programs run in the credential-free, timed worker. Only past-looking primitives
are exposed; all input is one device, ordered by availability time, without labels.
"""
from pathlib import Path
import ast
import copy
import shutil
import time
import numpy as np
import pandas as pd
from common import save,read,digest,sha,require
from history_store import cache_path,iter_devices,Writer
from history_math import verify
from feature_contract import METHODS,NP,REDUCE,HELPERS,Past,VERSION


def validate_source(source):
    require(isinstance(source,str) and len(source)<24000,'Feature source too large')
    tree=ast.parse(source);require(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef),'Use one def transform(frame)')
    fn=tree.body[0]
    require(fn.name=='transform' and [a.arg for a in fn.args.args]==['frame'] and not fn.decorator_list and not fn.args.defaults
        and not fn.args.kwonlyargs and not fn.args.kwarg and not fn.args.vararg and not fn.args.posonlyargs
        and fn.returns is None and fn.args.args[0].annotation is None,'Use exactly def transform(frame)')
    require(isinstance(fn.body[-1],ast.Return),'Return a DataFrame of new features')
    for statement in fn.body:
        require(isinstance(statement,(ast.Assign,ast.Return)),'Only numerical assignments and return are supported')
        if isinstance(statement,ast.Assign):
            require(all(isinstance(t,ast.Name) and not t.id.startswith('_') and t.id not in ('frame','pd','np','past') for t in statement.targets),'Assign only new local variables; frame/pd/np/past are reserved')
    forbidden=(ast.Import,ast.ImportFrom,ast.For,ast.While,ast.Lambda,ast.ListComp,ast.DictComp,ast.GeneratorExp,
               ast.With,ast.Try,ast.ClassDef,ast.Global,ast.Nonlocal,ast.Await,ast.Yield,ast.NamedExpr)
    parents={id(child):parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        require(not isinstance(node,forbidden),'Unsupported feature-code construct '+type(node).__name__+' at line '+str(getattr(node,'lineno',0))+'; use explicit local assignments and literal column names')
        if isinstance(node,ast.Name):require(not node.id.startswith('_'),'Private names are unavailable')
        if isinstance(node,ast.Attribute):
            require(node.attr in METHODS|NP|HELPERS|{'DataFrame','index','nan'},'Unsupported attribute '+node.attr+' at line '+str(node.lineno)+'; see feature_api; use arithmetic operators and past.gap_seconds(frame)')
            if node.attr=='index':
                p=parents.get(id(node))
                require(isinstance(p,ast.keyword) and p.arg=='index','frame.index is only for output alignment; use past.gap_seconds(frame) for observation gaps')
        if isinstance(node,ast.Subscript):
            require(isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str),'Select columns by literal name; positional access is unavailable')
        if isinstance(node,ast.Call):
            require(isinstance(node.func,ast.Attribute),'Only pd/np/past-looking Series calls are available')
            name=node.func.attr;obj=node.func.value
            require(not any(k.arg is None for k in node.keywords),'Expanded keyword arguments unavailable')
            if isinstance(obj,ast.Name) and obj.id=='np':require(name in NP,'Unknown NumPy operation: '+name)
            elif isinstance(obj,ast.Name) and obj.id=='pd':require(name=='DataFrame','Unsupported pd.'+name+'; only pd.DataFrame is exposed; inputs are already numeric')
            elif isinstance(obj,ast.Name) and obj.id=='past':
                require(name in HELPERS and len(node.args)==1 and isinstance(node.args[0],ast.Name) and node.args[0].id=='frame' and not node.keywords,'Use past.gap_seconds(frame) or past.observation_count(frame) exactly')
            else:require(name in METHODS,'Unsupported Series operation: '+name+' at line '+str(node.lineno)+'; see feature_api; use operators for arithmetic, past.gap_seconds(frame) for time gaps')
            if name in ('shift','diff'):
                val=node.args[0] if node.args else next((k.value for k in node.keywords if k.arg=='periods'),ast.Constant(1))
                require(isinstance(val,ast.Constant) and type(val.value) is int and val.value>=0,'Only nonnegative literal lag is causal')
                require(not any(k.arg=='freq' for k in node.keywords),'Timestamp-shifting freq is unavailable')
            if name=='rolling':
                require(not any(k.arg=='center' and not (isinstance(k.value,ast.Constant) and k.value.value is False) for k in node.keywords),'Centered windows use future data')
                require(len(node.args)<=1,'Use keyword arguments for rolling options')
            if name in REDUCE:
                require(isinstance(obj,ast.Call) and isinstance(obj.func,ast.Attribute) and obj.func.attr in ('rolling','expanding','ewm'),
                        'Aggregation must follow rolling/expanding/ewm; whole-history reductions leak the future')
            if name in ('fillna','replace'):require(not any(k.arg=='method' for k in node.keywords),'Use ffill() or an explicit value')
            if name=='replace':
                for argument in list(node.args)+[k.value for k in node.keywords]:
                    try:ast.literal_eval(argument)
                    except (ValueError,TypeError):raise ValueError('replace mappings must be literals, not whole data series')
    return tree


def compile_transform(source):
    tree=validate_source(source);env={'__builtins__':{},'np':np,'pd':pd,'past':Past}
    exec(compile(tree,'<generated-feature>','exec'),env)
    return env['transform']


def apply(source,times,values,names,expected,transform=None):
    transform=transform or compile_transform(source)
    frame=pd.DataFrame(values.copy(),columns=names,index=pd.to_datetime(times,unit='s',utc=True))
    frame.index.name='available_at'
    answer=transform(frame)
    require(isinstance(answer,pd.DataFrame) and answer.index.equals(frame.index),'Output must preserve device rows and timestamps')
    require(list(answer.columns)==expected,'Output feature columns differ from declaration')
    require(not set(names)&set(expected),'New features must have unique names')
    a=answer.to_numpy(dtype=np.float32)
    require(not np.isinf(a).any(),'Generated feature overflow; repair the formula')
    return a


def register(c,answer,parent,samples):
    require(set(answer)=={'source','features','reason'},'Feature response needs source, features, reason')
    columns=answer['features'];require(isinstance(columns,list) and 1<=len(columns)<=c['max_features_per_program'],'Add a small feature batch')
    import re
    require(len(set(columns))==len(columns) and all(isinstance(x,str) and re.fullmatch(r'fe_[A-Za-z0-9_]+',x) for x in columns),'Use unique fe_ feature names')
    validate_source(answer['source']);checks=[];tick=time.monotonic()
    require(samples,'No training histories available to validate generated features')
    for times,values,names in samples:
        require(len(names)+len(columns)<=c['max_total_features'],'Total feature budget exceeded')
        full=apply(answer['source'],times,values,names,columns)
        for cut in sorted({1,max(1,len(times)//2),max(1,len(times)-1)}):
            before=apply(answer['source'],times[:cut],values[:cut],names,columns)
            require(np.allclose(full[:cut],before,equal_nan=True,rtol=1e-5,atol=1e-6),'Future rows changed past features')
        checks.append(dict(rows=len(times),finite_values=int(np.isfinite(full).sum())))
    require(any(x['finite_values'] for x in checks),'No finite feature outputs in diverse TRAIN prefixes; this is bounded sample coverage, not proof of globally missing data. Check required-column coverage and warm-up length before changing the formula.')
    spec=dict(parent=parent,source=answer['source'],features=columns)
    key=digest(spec);folder=Path(c['output_root'])/'feature_programs'/key
    if folder.exists():require(read(folder/'program.json')['spec']==spec,'Feature hash collision or changed program')
    else:
        folder.mkdir(parents=True);save(folder/'program.json',dict(id=key,spec=spec,reason=answer['reason'],
            checks=checks,sample_validation_seconds=time.monotonic()-tick,labels_visible=False,
            limitation='Causal API restrictions plus sampled prefix checks; runtime resource budgets also apply'))
        (folder/'transform.py').write_text(answer['source'],encoding='utf-8')
    return key


def program(c,key):
    require(key=='base' or len(key)==64 and all(x in '0123456789abcdef' for x in key),'Invalid feature ID')
    p=Path(c['output_root'])/'feature_programs'/key/'program.json';value=read(p)
    require(digest(value['spec'])==key and (p.parent/'transform.py').read_text(encoding='utf-8')==value['spec']['source'],'Feature artifact changed')
    return value['spec']


def materialize(c,key,base_manifest,final_test=False,depth=0):
    if key=='base':return c,base_manifest
    require(depth<12,'Feature program chain too deep');spec=program(c,key)
    parent_c,parent=materialize(c,spec['parent'],base_manifest,final_test,depth+1)
    out=Path(c['output_root'])/'feature_views'/('test' if final_test else 'trainval')/key
    receipt=out/'manifest.json';source=cache_path(parent_c,final_test)
    stamp=dict(parent=sha(source/'manifest.json'),program=key,test=final_test)
    if receipt.exists():
        m=verify(out);require(m['feature_view_spec']==stamp,'Feature cache identity changed')
        return dict(c,active_cache=str(out)),m
    out.mkdir(parents=True,exist_ok=True)
    for p in (out/'rows').glob('part-*.parquet'):p.unlink()
    writer=Writer(out/'rows');names=parent['features']+spec['features']
    require(len(names)<=c['max_total_features'],'Feature budget exceeded')
    registry=pd.read_parquet(source/'device_metadata.parquet').set_index('device')
    from data_ingest import check_disk
    tick=time.monotonic();notice=tick;transform=compile_transform(spec['source']);rows=0
    print('FEATURE_VIEW_START program='+key+' base cache preserved',flush=True)
    for count,(code,t,x) in enumerate(iter_devices(source,parent['features']),1):
        require(time.monotonic()-tick<c['feature_timeout_seconds'],'Feature materialization timeout; simplify the feature program')
        require(len(t)<=c['max_device_rows'],'Device exceeds feature row budget')
        check_disk(out,c,int(x.nbytes*2))
        extra=apply(spec['source'],t,x,parent['features'],spec['features'],transform)
        f=pd.DataFrame(np.c_[x,extra],columns=names);f['device']=code;f['time']=t
        f['history_rows']=np.arange(1,len(t)+1)
        for name in ('failure','observed_start','observed_end'):f[name]=int(registry.loc[code,name])
        writer.add(f)
        rows+=len(t)
        if time.monotonic()-notice>=30:
            print('FEATURE_VIEW_PROGRESS devices='+str(count)+' rows='+str(rows)+' elapsed_seconds='+str(int(time.monotonic()-tick)),flush=True);notice=time.monotonic()
    writer.flush()
    for name in ('device_metadata.parquet','devices.json','excluded.json'):shutil.copy2(source/name,out/name)
    m=copy.deepcopy(parent);m.update(features=names,feature_view_spec=stamp)
    m['files']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*.parquet')}
    m['files'].update({n:sha(out/n) for n in ('devices.json','excluded.json')})
    save(receipt,m);return dict(c,active_cache=str(out)),m


def training_samples(c,key,m,limit=12,required=None):
    """Coverage-directed deterministic TRAIN sampling; no outcomes or validation rows.

    Probe eight spread shards, retain varied missingness/column coverage, then
    fetch COMPLETE prefixes of selected devices (not fragments at shard edges).
    The audit is descriptive and must never be interpreted as population coverage.
    """
    import pyarrow.parquet as pq
    cc,mm=materialize(c,key,m);cache=cache_path(cc);names=mm['features']
    required=sorted(set(required or []) & set(names))
    stamp=dict(version=VERSION,manifest=sha(cache/'manifest.json'),limit=limit,required=required)
    folder=Path(c['output_root'])/'feature_samples'/digest(stamp)
    if (folder/'samples.npz').exists() and (folder/'audit.json').exists():
        audit=read(folder/'audit.json');require(sha(folder/'samples.npz')==audit['sha256'],'Feature sample cache changed')
        with np.load(folder/'samples.npz',allow_pickle=False) as data:
            return [(data['t'+str(i)],data['x'+str(i)],names) for i in range(audit['devices'])]
    files=sorted(p for p in mm['files'] if p.startswith('rows/') and p.endswith('.parquet'))
    lo,hi=m['windows']['train'];positions=sorted(set(np.linspace(0,len(files)-1,min(8,len(files)),dtype=int))) if files else []
    candidates={};probed=[];spread=[];target=[names.index(n) for n in required] or list(range(len(names)))
    for pos in positions:
        path=cache/files[pos];probed.append(files[pos])
        frame=pq.read_table(path,columns=['device','time']+names,filters=[('time','>=',lo),('time','<',hi)]).to_pandas()
        if frame.empty:continue
        finite=frame[names].notna();finite['device']=frame.device
        coverage=finite.groupby('device',sort=True).sum();counts=frame.groupby('device',sort=True).size()
        # Bounded summaries; include dense devices and spread device IDs per shard.
        rank=coverage.iloc[:,target].gt(0).sum(axis=1).sort_values(ascending=False,kind='stable')
        ids=list(rank.index[:128])+list(coverage.index[np.linspace(0,len(coverage)-1,min(128,len(coverage)),dtype=int)])
        ids+=list(coverage.idxmax(axis=0))
        eligible=[int(code) for code in rank.index if counts.loc[code]>=2]
        if eligible:spread.append(eligible[0])
        for code in ids:
            if counts.loc[code]<2:continue
            v=coverage.loc[code].to_numpy(dtype=np.int64)
            candidates[int(code)]=np.maximum(candidates.get(int(code),np.zeros(len(names),dtype=np.int64)),v)
    chosen=[];uncovered=set(target)
    for code in dict.fromkeys(spread):
        if code not in candidates or len(chosen)>=max(1,limit//2):continue
        v=candidates.pop(code);chosen.append(code);uncovered-={j for j in uncovered if v[j]>0}
    while candidates and len(chosen)<max(1,limit-2):
        code=max(candidates,key=lambda k:(sum(candidates[k][j]>0 for j in uncovered),sum(candidates[k][j]>0 for j in target),int(candidates[k][target].sum()),-k))
        v=candidates.pop(code);chosen.append(code);uncovered-={j for j in uncovered if v[j]>0}
    # Keep sparse cases too: checks must include absent sensors and empty prefixes.
    for code in sorted(candidates,key=lambda k:(np.count_nonzero(candidates[k]),k))[:max(0,limit-len(chosen))]:chosen.append(code)
    if not chosen:return []
    data=pq.read_table([str(cache/p) for p in files],columns=['device','time']+names,
                       filters=[('device','in',chosen),('time','>=',lo),('time','<',hi)]).to_pandas()
    out=[];device_ids=[]
    for code in chosen:
        frame=data.loc[data.device==code].sort_values('time',kind='stable').iloc[:8192]
        if len(frame)<2:continue
        out.append((frame.time.to_numpy(np.int64),frame[names].to_numpy(np.float32),names));device_ids.append(code)
    folder.mkdir(parents=True,exist_ok=True)
    arrays={}
    for i,(t,x,_) in enumerate(out):arrays['t'+str(i)]=t;arrays['x'+str(i)]=x
    np.savez_compressed(folder/'samples.npz',**arrays)
    save(folder/'audit.json',dict(stamp=stamp,devices=len(out),device_ids=device_ids,probed_shards=probed,
        rows=[len(t) for t,x,n in out],finite_by_column={n:sum(int(np.isfinite(x[:,i]).sum()) for t,x,_ in out) for i,n in enumerate(names)},
        sha256=sha(folder/'samples.npz'),labels_visible=False,limitation='Bounded diverse training prefixes; not population prevalence. Maximum prefix length 8192.'))
    print('TRAIN_SAMPLE_VALIDATED devices='+str(len(out))+' shards='+str(len(probed))+' labels_visible=False',flush=True)
    return out


def required_columns(source):
    tree=validate_source(source)
    return sorted({n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='frame'})
