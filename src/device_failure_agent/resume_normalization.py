"""Audited recovery of a completed SQLite ingest; never rescans feature values.

The live migration supports CSV indicator/last-observation mappings only. It
rebuilds small failure metadata, retains completed whole-device Parquet shards,
and uses the same normalization rules as data_ingest.normalize.
"""
from pathlib import Path
import argparse
import sqlite3
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import read, save, sha, digest, require, RunLock, code_hashes
from data_ingest import paths, validate_mapping, apply_filters, seconds, ids, check_disk
from history_math import Writer, verify
from settings import validate_config


def context(session):
    root=Path(session).resolve(); c=validate_config(read(root/'config.json'))
    require(Path(c['output_root']).resolve()==root,'Session path changed')
    state=read(root/'state.json');m=read(root/'mapping.json');inv=read(root/'inventory.json')
    require(state['config_hash']==digest(c),'Configuration seal differs')
    require(state['mapping_hash']==digest(m),'Mapping seal differs')
    require(not state.get('trials') and not state.get('pending'),'Only pre-training sessions may be recovered')
    require(state['status']=='mapped','Recovery requires mapped status')
    return c,m,inv


def data_spec(c,m,inv):
    return dict(mapping=digest(m),inventory=inv['fingerprint'],
                coverage=[c.get('coverage_start'),c.get('coverage_end')],test=False)


def source_stats(c,inv):
    for item in inv['files']:
        p=Path(c['data_dir'])/item['path'];st=p.stat()
        require((st.st_size,st.st_mtime_ns)==(item['size'],item['mtime_ns']),
                'Source file changed: '+item['path'])


def prepare_checkpoint(c,m,inv,output):
    """Read only the closed ingest tables; the old device writer may keep running."""
    require(m['failure']['mode']=='indicator' and m['followup']['mode']=='last_observation',
            'Live recovery supports indicator + last_observation only')
    features=validate_mapping(m,c['data_dir']);source_stats(c,inv)
    db=Path(c['output_root'])/'cache/staging.sqlite';before=db.stat()
    con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True,timeout=2)
    try:
        indexes=[r[1] for r in con.execute('PRAGMA index_list(obs)')]
        require('obs_sort' in indexes,'Ingestion has not completed; do not recover this stage')
        metadata={str(i):(int(a),int(b)) for i,a,b in con.execute('SELECT id,first,last FROM meta')}
        rawrows=int(con.execute('SELECT COALESCE(MAX(rowid),0) FROM obs').fetchone()[0])
    finally:con.close()
    require(metadata,'Empty metadata');lo=min(x[0] for x in metadata.values());hi=max(x[1] for x in metadata.values())+1
    if c.get('coverage_start'):lo=int(pd.Timestamp(c['coverage_start']).timestamp())
    if c.get('coverage_end'):hi=int(pd.Timestamp(c['coverage_end']).timestamp())
    require(hi-lo>10*86400,'Coverage too short')
    edges=[lo,lo+(hi-lo)*6//10,lo+(hi-lo)*8//10,hi]
    win={s:[edges[i],edges[i+1]] for i,s in enumerate(('train','val','test'))};cutoff=win['val'][1]
    failure={};receipts=[];fault=m['failure'];scanned_rows=0
    expected_files=sum(len(paths(c['data_dir'],g['glob'])) for g in m['observations']);done=0
    for group in m['observations']:
        idcols=group.get('id_column');idcols=idcols if isinstance(idcols,list) else ([idcols] if idcols else [])
        columns=set(idcols+[group['time_column'],fault['column']]+list(group.get('filters',{})))
        for p in paths(c['data_dir'],group['glob']):
            require(p.suffix.lower()=='.csv','Live recovery supports CSV observations only')
            st=p.stat();h=sha(p);receipts.append(dict(path=str(p.relative_to(c['data_dir'])),sha256=h))
            for frame in pd.read_csv(p,usecols=lambda x:x in columns,chunksize=32768,dtype=str,keep_default_na=True):
                f=apply_filters(frame,group)
                if not len(f):continue
                t=seconds(f[group['time_column']],group['time_unit'],group.get('timezone','UTC'))
                use=t<cutoff;f=f.loc[use];t=t[use];scanned_rows+=len(f)
                if not len(f):continue
                sn=ids(f,group,p);mask=f[fault['column']].astype(str).isin([str(v) for v in fault['positive_values']]).to_numpy()
                for device,at in zip(sn[mask],t[mask]):failure[device]=min(failure.get(device,int(at)),int(at))
            require((p.stat().st_size,p.stat().st_mtime_ns)==(st.st_size,st.st_mtime_ns),'Source changed while recovering')
            done+=1
            if done%10==0 or done==expected_files:print(f'Recovery metadata: {done}/{expected_files} files',flush=True)
    require(scanned_rows==rawrows,'Staging row count differs from recovered source selection')
    allids=sorted(set(metadata)|set(failure));registry=[]
    for code,sn in enumerate(allids):
        start,end=metadata.get(sn,(cutoff,cutoff-1));end=min(end,cutoff-1);ue=failure.get(sn,-1)
        require(ue<0 or start<=ue<=end,'Failure outside observed follow-up')
        registry.append(dict(device=code,sn_name=sn,failure=ue,observed_start=start,observed_end=end))
    print('Recovery: hashing immutable staging database',flush=True)
    db_hash=sha(db);after=db.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Ingest database is still changing')
    source_stats(c,inv)
    checkpoint=dict(version=1,spec=data_spec(c,m,inv),config_hash=digest(c),features=features,
                    windows=win,source_receipts=receipts,registry=registry,rows_before_failure_filter=rawrows,
                    database=dict(size=after.st_size,mtime_ns=after.st_mtime_ns,sha256=db_hash))
    save(output,checkpoint);save(str(output)+'.sha256.json',dict(sha256=sha(output)))
    print('RECOVERY_CHECKPOINT_READY:',len(registry),'devices;',rawrows,'observations',flush=True)
    return checkpoint


def inspect_prefix(cache,registry,features):
    """Writer.add flushes only between devices, so each completed shard is resumable."""
    parts=sorted((cache/'rows').glob('part-*.parquet'));last=-1;sealed={}
    columns=features+['device','failure','observed_start','observed_end','time','history_rows']
    for i,p in enumerate(parts):
        require(p.name==f'part-{i:05d}.parquet','Non-contiguous output parts')
        parquet=pq.ParquetFile(p)
        require(parquet.schema_arrow.names==columns,'Output schema differs')
        f=parquet.read(columns=['device','failure','observed_start','observed_end','time','history_rows']).to_pandas()
        require(len(f)>0 and f.device.is_monotonic_increasing,'Invalid device order')
        require(int(f.device.iloc[0])>last and int(f.device.iloc[-1])<len(registry),'Output boundary overlaps')
        for code,g in f.groupby('device',sort=False):
            meta=registry[int(code)]
            require(np.array_equal(g.history_rows.to_numpy(),np.arange(1,len(g)+1)),'Partial device history')
            require(g.time.is_monotonic_increasing and not g.time.duplicated().any(),'Invalid normalized timestamps')
            for k in ('failure','observed_start','observed_end'):require(g[k].eq(meta[k]).all(),'Recovered registry differs: '+k)
            require(g.time.le(meta['observed_end']).all(),'Rows beyond follow-up')
            require(meta['failure']<0 or g.time.lt(meta['failure']).all(),'Rows at/after failure')
        last=int(f.device.iloc[-1]);sealed[p.name]=sha(p)
    return last,len(parts),sealed


def finish_checkpoint(c,m,inv,checkpoint_path):
    cp=read(checkpoint_path);require(sha(checkpoint_path)==read(str(checkpoint_path)+'.sha256.json')['sha256'],'Checkpoint changed')
    require(cp['version']==1 and cp['spec']==data_spec(c,m,inv) and cp['config_hash']==digest(c),'Checkpoint protocol differs')
    root=Path(c['output_root']);cache=root/'cache'
    if (cache/'manifest.json').exists():return verify(cache)
    source_stats(c,inv);db=cache/'staging.sqlite';st=db.stat();expected=cp['database']
    require(st.st_size==expected['size'] and st.st_mtime_ns==expected['mtime_ns'],'Staging database changed')
    print('Recovery: verifying staging database',flush=True)
    require(sha(db)==expected['sha256'],'Staging database hash differs')
    features=cp['features'];registry=cp['registry'];last,part,sealed=inspect_prefix(cache,registry,features)
    print(f'Recovery: keeping {part} completed shards; resume at device {last+1}/{len(registry)}',flush=True)
    save(root/'recovery_prefix.json',dict(last_device=last,parts=sealed,checkpoint_sha256=sha(checkpoint_path)))
    con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True,timeout=2);duplicate_rows=0
    writer=Writer(cache/'rows');writer.part=part
    try:
        if last>=0:
            # Count only identical duplicate timestamps already accepted in retained shards.
            by_id={x['sn_name']:x for x in registry[:last+1]}
            for sn,at,n in con.execute('SELECT id,t,COUNT(*) FROM obs WHERE id<=? GROUP BY id,t HAVING COUNT(*)>1',(registry[last]['sn_name'],)):
                meta=by_id[sn]
                if at<=meta['observed_end'] and (meta['failure']<0 or at<meta['failure']):duplicate_rows+=n-1
        start=time.monotonic()
        for meta in registry[last+1:]:
            records=[]
            for at,blob in con.execute('SELECT t,x FROM obs WHERE id=? ORDER BY t',(meta['sn_name'],)):
                if at>meta['observed_end'] or meta['failure']>=0 and at>=meta['failure']:continue
                records.append((at,blob));require(len(records)<=c['max_device_rows'],'Per-device row budget exceeded')
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
            if (meta['device']+1)%500==0:
                elapsed=time.monotonic()-start;rate=(meta['device']-last)/max(elapsed,.001)
                print(f'Normalize: {meta["device"]+1}/{len(registry)} devices; {rate:.1f} devices/s since resume',flush=True)
                check_disk(cache,c)
        writer.flush();pd.DataFrame(registry).to_parquet(cache/'device_metadata.parquet',index=False)
        save(cache/'devices.json',[x['sn_name'] for x in registry]);save(cache/'excluded.json',[])
        files={p.relative_to(cache).as_posix():sha(p) for p in cache.rglob('*.parquet')}
        files.update({n:sha(cache/n) for n in ('devices.json','excluded.json')})
        for name,h in sealed.items():require(sha(cache/'rows'/name)==h,'Retained output modified')
        manifest=dict(spec=cp['spec'],features=features,windows=cp['windows'],files=files,included_devices=len(registry),
            coverage=m['followup'],source_receipts=cp['source_receipts'],test_opened=False,duplicate_rows_collapsed=duplicate_rows,
            rows_before_failure_filter=cp['rows_before_failure_filter'],source_inventory=inv['fingerprint'],
            metadata_policy='Full source timestamps define observation bounds; future feature values and failure outcomes are excluded before normalization. Feature-agent examples use TRAIN only.')
        source_stats(c,inv);save(cache/'manifest.json',manifest)
        save(root/'recovery_complete.json',dict(checkpoint_sha256=sha(checkpoint_path),retained_parts=len(sealed),
             retained_last_device=last,manifest_sha256=sha(cache/'manifest.json')))
        print('NORMALIZATION_RECOVERY_COMPLETE',flush=True)
        return manifest
    finally:con.close()


def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','finish']);a.add_argument('--session',required=True);a.add_argument('--checkpoint',required=True)
    args=a.parse_args();c,m,inv=context(args.session)
    if args.mode=='prepare':prepare_checkpoint(c,m,inv,args.checkpoint)
    else:
        require(read(Path(args.session)/'code_seal.json')==code_hashes(),'Runtime seal differs')
        with RunLock(args.session):finish_checkpoint(c,m,inv,args.checkpoint)


if __name__=='__main__':main()
