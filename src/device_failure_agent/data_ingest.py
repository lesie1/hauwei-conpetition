"""Streaming, declarative adapters. The feature agent supplies the mapping; data stay local."""
from pathlib import Path
import fnmatch
import json
import re
import sqlite3
import shutil
import hashlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from common import require,save,read,digest,sha
from history_math import Writer,verify

FORMATS={'.csv','.parquet','.feather','.xlsx','.json','.jsonl','.log'}
META={'device','time','failure','observed_start','observed_end','history_rows','label','target',
      'failure_time','failed_at','alarm_time','__time'}


def paths(root,pattern):
    root=Path(root).resolve()
    require(isinstance(pattern,str) and not Path(pattern).is_absolute() and '..' not in Path(pattern).parts,'Use relative source globs')
    result=sorted(p for p in root.glob(pattern) if p.is_file())
    require(result,'No source files match '+pattern)
    require(all(p.resolve().is_relative_to(root) for p in result),'Source symlink escapes data root')
    return result


def frames(path,limit_mb=128,log_regex=None):
    """One bounded chunk at a time. Excel uses read-only streaming."""
    path=Path(path);ext=path.suffix.lower()
    if ext=='.csv':
        yield from pd.read_csv(path,chunksize=32768,dtype=str,keep_default_na=True)
    elif ext=='.parquet':
        for b in pq.ParquetFile(path).iter_batches(batch_size=32768):yield b.to_pandas()
    elif ext=='.feather':
        with pa.memory_map(str(path),'r') as f:
            r=ipc.open_file(f)
            for i in range(r.num_record_batches):
                b=r.get_batch(i)
                for lo in range(0,len(b),32768):yield b.slice(lo,32768).to_pandas()
    elif ext=='.xlsx':
        import openpyxl
        book=openpyxl.load_workbook(path,read_only=True,data_only=True)
        try:
            require(len(book.sheetnames)==1,'Multiple Excel sheets: export or select a single table first')
            rows=iter(book.active.values);header=list(next(rows));buf=[]
            require(len(header)==len(set(header)) and all(isinstance(x,str) for x in header),'Invalid Excel header')
            for row in rows:
                buf.append(row)
                if len(buf)>=32768:yield pd.DataFrame(buf,columns=header);buf=[]
            if buf:yield pd.DataFrame(buf,columns=header)
        finally:book.close()
    elif ext=='.json':
        require(path.stat().st_size<=limit_mb*1024**2,'Large JSON array: convert to JSONL for streaming')
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        require(isinstance(value,list),'JSON must contain an array of records')
        for i in range(0,len(value),32768):yield pd.json_normalize(value[i:i+32768])
    elif ext in ('.jsonl','.log'):
        pattern=re.compile(log_regex) if log_regex else None;buf=[]
        with path.open(encoding='utf-8-sig') as f:
            for n,line in enumerate(f,1):
                if not line.strip():continue
                if pattern:
                    match=pattern.fullmatch(line.strip())
                    require(match is not None,f'Log line {n} did not match declared named-group regex')
                    row=match.groupdict()
                else:row=json.loads(line)
                require(isinstance(row,dict),'JSONL record must be an object');buf.append(row)
                if len(buf)>=32768:yield pd.json_normalize(buf);buf=[]
            if buf:yield pd.json_normalize(buf)
    else:raise ValueError('Unsupported format: '+ext)


def inventory(data,limit_mb=128):
    root=Path(data).resolve();files=[];groups={}
    require(root.is_dir(),'Dataset directory not found: '+str(root))
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in FORMATS:continue
        require(p.resolve().is_relative_to(root),'External symlink in dataset')
        try:
            chunk=next(frames(p,limit_mb));columns=[str(x) for x in chunk.columns]
            require(len(columns)==len(set(columns)),'Duplicate columns')
        except StopIteration:columns=[]
        except Exception as e:
            if p.suffix.lower()!='.log':raise ValueError(f'Cannot inspect {p.name}: {e}') from e
            columns=['__unparsed_log__']
        key=digest(columns);relative=p.relative_to(root).as_posix()
        files.append(dict(path=relative,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,schema=key))
        group=groups.setdefault(key,dict(columns=columns,file_count=0,examples=[]))
        group['file_count']+=1
        # Names and headers only: profiling never sends held-out feature values or labels.
        if len(group['examples'])<8:group['examples'].append(relative)
        if len(files)%500==0:print(f'Inventory: {len(files)} files; {len(groups)} schemas',flush=True)
    require(files,'No supported files found; unpack archives first')
    return dict(files=files,schemas=groups,fingerprint=digest(files),
                note='All file schemas inspected; no raw values or test labels sent to the agent')


def seconds(series,unit,timezone='UTC'):
    require(not series.isna().any(),'Missing time values')
    if unit in ('s','ms','us','ns'):
        a=pd.to_numeric(series,errors='raise').to_numpy(np.float64)/{'s':1,'ms':1000,'us':1e6,'ns':1e9}[unit]
    else:
        require(unit in ('datetime','date'),'Time unit must be s/ms/us/ns/datetime/date')
        dt=pd.to_datetime(series,errors='raise',format='mixed')
        if dt.dt.tz is None:
            require(bool(timezone),'Naive dates require a declared timezone')
            dt=dt.dt.tz_localize(timezone,ambiguous='raise',nonexistent='raise')
        a=dt.dt.tz_convert('UTC').astype('int64').to_numpy(np.float64)/1e9
    require(np.isfinite(a).all() and np.equal(a,np.floor(a)).all(),'Time precision must resolve to whole seconds')
    require(((a>=946684800)&(a<4102444800)).all(),'Timestamp outside 2000-2099; check units')
    return a.astype(np.int64)


def apply_filters(f,spec):
    for col,val in spec.get('filters',{}).items():
        require(col in f,'Filter column absent: '+col);f=f.loc[f[col].astype(str)==str(val)]
    return f


def ids(f,spec,path):
    col=spec.get('id_column')
    if isinstance(col,list):
        require(col and all(k in f for k in col),'Composite device ID columns absent')
        require(not f[col].isna().any().any(),'Missing composite device identifier')
        sep=spec.get('id_separator','_')
        a=f[col].astype(str).agg(sep.join,axis=1)
    else:a=f[col] if col else pd.Series(str(Path(path).stem),index=f.index)
    require(not a.isna().any(),'Missing device identifiers');a=a.astype(str)
    require(a.str.len().gt(0).all(),'Empty device identifiers')
    return a.to_numpy()


def validate_mapping(m,root):
    require(isinstance(m,dict) and set(m)=={'observations','failure','followup','description'},'Mapping needs observations, failure, followup, description')
    require(isinstance(m['observations'],list) and m['observations'],'Need observation mappings')
    seen={};names={};source_labels=set()
    fault=m['failure'];require(fault['mode'] in ('tickets','indicator'),'Failure mode tickets/indicator')
    require(fault.get('definition'),'Explain the actual failure definition')
    if fault['mode']=='indicator':source_labels.add(fault['column'])
    follow=m['followup'];require(follow['mode'] in ('last_observation','file','complete_coverage'),'Invalid follow-up policy')
    require(follow.get('basis'),'Follow-up needs a documented basis')
    for spec in m['observations']:
        require(spec.get('time_column') and spec.get('time_unit'),'Observation time required')
        files=paths(root,spec['glob'])
        for p in files:
            require(p not in seen,'Overlapping observation globs: '+str(p));seen[p]=True
        require(spec['features'],'Need numeric input features')
        require(len({x['name'] for x in spec['features']})==len(spec['features']),'Duplicate feature name within source mapping')
        for col in spec['features']:
            require(set(col)=={'name','column','kind','unit','description'},'Feature mapping fields differ')
            require(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',col['name']) and col['name'] not in META,'Invalid feature name')
            idcols=spec.get('id_column');idcols=idcols if isinstance(idcols,list) else [idcols]
            forbidden=META|source_labels|{spec['time_column']}|set(idcols)
            require(col['column'] not in forbidden,'Outcome/identifier/time cannot be an input feature')
            require(col['kind'] in ('gauge','counter','event_count','numeric_unknown'),'Unsupported feature kind')
            if col['name'] in names:require(names[col['name']]=={k:col[k] for k in ('kind','unit')},'Feature meaning differs between batches')
            names[col['name']]={k:col[k] for k in ('kind','unit')}
    require(names,'No features');return sorted(names)


def check_disk(folder,c,extra=0):
    free=shutil.disk_usage(folder).free
    require(free>c['min_free_disk_gb']*1024**3+extra,'Insufficient free scratch disk; preserve results, free space or change workspace')


def normalize(c,m,inventory_hash,final_test=False):
    """SQLite external sort handles daily HDD files without loading all devices into RAM."""
    root=Path(c['output_root']);cache=root/('cache_test' if final_test else 'cache')
    cache.mkdir(parents=True,exist_ok=True)
    spec=dict(mapping=digest(m),inventory=inventory_hash,coverage=[c.get('coverage_start'),c.get('coverage_end')],test=final_test)
    if (cache/'manifest.json').exists():
        previous=verify(cache);require(previous['spec']==spec,'Data changed: initialize a new session');return previous
    features=validate_mapping(m,c['data_dir']);require(len(features)<=c['max_total_features'],'Too many base features')
    db=cache/'staging.sqlite';db.unlink(missing_ok=True)
    for p in (cache/'rows').glob('part-*.parquet'):p.unlink()
    con=sqlite3.connect(db);con.execute('PRAGMA journal_mode=OFF');con.execute('PRAGMA temp_store=FILE')
    con.execute('CREATE TABLE meta (id TEXT PRIMARY KEY, first INTEGER, last INTEGER)')
    con.execute('CREATE TABLE obs (id TEXT,t INTEGER,x BLOB)')
    receipts=[];min_time=None;max_time=None
    def visit_metadata():
        nonlocal min_time,max_time
        for group in m['observations']:
            for p in paths(c['data_dir'],group['glob']):
                stat=p.stat();h=sha(p);receipts.append(dict(path=str(p.relative_to(c['data_dir'])),sha256=h))
                for frame in frames(p,c['max_json_file_mb'],group.get('log_regex')):
                    f=apply_filters(frame,group)
                    if not len(f):continue
                    times=seconds(f[group['time_column']],group['time_unit'],group.get('timezone','UTC'))
                    sn=ids(f,group,p);a=pd.DataFrame(dict(id=sn,t=times)).groupby('id').t.agg(['min','max'])
                    con.executemany('INSERT INTO meta VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET first=min(first,excluded.first),last=max(last,excluded.last)',
                        [(str(i),int(r['min']),int(r['max'])) for i,r in a.iterrows()])
                    min_time=int(times.min()) if min_time is None else min(min_time,int(times.min()))
                    max_time=int(times.max()) if max_time is None else max(max_time,int(times.max()))
                require((p.stat().st_size,p.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns),'Input changed while scanning')
                con.commit();check_disk(cache,c)
    try:
        visit_metadata();require(min_time is not None,'No observations after filters')
        lo=int(pd.Timestamp(c['coverage_start']).timestamp()) if c.get('coverage_start') else min_time
        hi=int(pd.Timestamp(c['coverage_end']).timestamp()) if c.get('coverage_end') else max_time+1
        require(hi-lo>10*86400,'Coverage too short for chronological failure experiments')
        edges=[lo,lo+(hi-lo)*6//10,lo+(hi-lo)*8//10,hi]
        win={s:[edges[i],edges[i+1]] for i,s in enumerate(('train','val','test'))}
        cutoff=hi if final_test else win['val'][1]
        failure={};rawrows=0
        for group in m['observations']:
            for p in paths(c['data_dir'],group['glob']):
                for frame in frames(p,c['max_json_file_mb'],group.get('log_regex')):
                    f=apply_filters(frame,group)
                    if not len(f):continue
                    t=seconds(f[group['time_column']],group['time_unit'],group.get('timezone','UTC'))
                    use=t<cutoff;f=f.loc[use];t=t[use]
                    if not len(f):continue
                    sn=ids(f,group,p)
                    if m['failure']['mode']=='indicator':
                        column=m['failure']['column'];require(column in f,'Failure indicator absent')
                        mask=f[column].astype(str).isin([str(v) for v in m['failure']['positive_values']]).to_numpy()
                        for device,at in zip(sn[mask],t[mask]):failure[device]=min(failure.get(device,int(at)),int(at))
                    values=np.full((len(f),len(features)),np.nan,np.float32)
                    for field in group['features']:
                        if field['column'] not in f:continue # declared missing column is represented by NaN
                        v=pd.to_numeric(f[field['column']],errors='raise').to_numpy(np.float32)
                        v[~np.isfinite(v)]=np.nan;values[:,features.index(field['name'])]=v
                    check_disk(cache,c,values.nbytes*2)
                    con.executemany('INSERT INTO obs VALUES (?,?,?)',[(str(d),int(at),v.tobytes()) for d,at,v in zip(sn,t,values)])
                    rawrows+=len(f)
                con.commit()
        if m['failure']['mode']=='tickets':
            fault=m['failure']
            for p in paths(c['data_dir'],fault['glob']):
                receipts.append(dict(path=str(p.relative_to(c['data_dir'])),sha256=sha(p)))
                for frame in frames(p,c['max_json_file_mb']):
                    f=apply_filters(frame,fault)
                    if not len(f):continue
                    t=seconds(f[fault['time_column']],fault['time_unit'],fault.get('timezone','UTC'));sn=ids(f,fault,p)
                    for device,at in zip(sn[t<cutoff],t[t<cutoff]):failure[device]=min(failure.get(device,int(at)),int(at))
        followed={};follow=m['followup']
        if follow['mode']=='file':
            for p in paths(c['data_dir'],follow['glob']):
                receipts.append(dict(path=str(p.relative_to(c['data_dir'])),sha256=sha(p)))
                for f in frames(p,c['max_json_file_mb']):
                    a=seconds(f[follow['start_column']],follow['time_unit'],follow.get('timezone','UTC'))
                    b=seconds(f[follow['end_column']],follow['time_unit'],follow.get('timezone','UTC'))
                    for device,start,end in zip(ids(f,follow,p),a,b):
                        require(device not in followed and start<=end,'Duplicate/invalid follow-up');followed[device]=(int(start),int(end))
        metadata={str(i):(int(a),int(b)) for i,a,b in con.execute('SELECT id,first,last FROM meta')}
        allids=sorted(set(metadata)|set(failure)|set(followed));registry=[]
        if final_test:
            original=read(root/'cache/devices.json')
            allids=original+sorted(set(allids)-set(original))
        for code,sn in enumerate(allids):
            if follow['mode']=='file':start,end=followed.get(sn,(cutoff,cutoff-1))
            elif follow['mode']=='complete_coverage':start,end=lo,cutoff-1
            else:start,end=metadata.get(sn,(cutoff,cutoff-1))
            end=min(end,cutoff-1);ue=failure.get(sn,-1)
            require(ue<0 or start<=ue<=end,'Failure lies outside observed follow-up: '+sn)
            registry.append(dict(device=code,sn_name=sn,failure=ue,observed_start=start,observed_end=end))
        require(registry,'Empty registry');check_disk(cache,c)
        con.execute('CREATE INDEX obs_sort ON obs(id,t)');con.commit()
        writer=Writer(cache/'rows');duplicate_rows=0
        for meta in registry:
            sn=meta['sn_name'];records=[]
            cursor=con.execute('SELECT t,x FROM obs WHERE id=? ORDER BY t',(sn,))
            for at,blob in cursor:
                if at>meta['observed_end'] or meta['failure']>=0 and at>=meta['failure']:continue
                records.append((at,blob));require(len(records)<=c['max_device_rows'],'Per-device row budget exceeded: '+sn)
            if records:
                times=np.array([a for a,b in records],np.int64)
                values=np.stack([np.frombuffer(b,dtype=np.float32) for a,b in records])
                unique=np.r_[True,np.diff(times)!=0]
                if not unique.all():
                    for j in np.flatnonzero(~unique):require(np.array_equal(values[j],values[j-1],equal_nan=True),'Conflicting duplicate device timestamp')
                    duplicate_rows+=int((~unique).sum());times=times[unique];values=values[unique]
                f=pd.DataFrame(values,columns=features)
                for k in ('device','failure','observed_start','observed_end'):f[k]=meta[k]
                f['time']=times;f['history_rows']=np.arange(1,len(f)+1);writer.add(f)
            if (meta['device']+1)%500==0:print(f'Normalize: {meta["device"]+1}/{len(registry)} devices',flush=True)
        writer.flush();pd.DataFrame(registry).to_parquet(cache/'device_metadata.parquet',index=False)
        save(cache/'devices.json',allids);save(cache/'excluded.json',[])
        files={p.relative_to(cache).as_posix():sha(p) for p in cache.rglob('*.parquet')}
        files.update({n:sha(cache/n) for n in ('devices.json','excluded.json')})
        manifest=dict(spec=spec,features=features,windows=win,files=files,included_devices=len(registry),
            coverage=follow,source_receipts=receipts,test_opened=final_test,duplicate_rows_collapsed=duplicate_rows,
            rows_before_failure_filter=rawrows,source_inventory=inventory_hash,
            metadata_policy='Full source timestamps define observation bounds; future feature values and failure outcomes are excluded before normalization. Feature-agent examples use TRAIN only.')
        save(cache/'manifest.json',manifest);return manifest
    finally:
        con.close();db.unlink(missing_ok=True)
