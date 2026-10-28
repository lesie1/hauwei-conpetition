"""Credential-free worker for feature profile, validation and cache creation."""
import argparse
from pathlib import Path
import numpy as np
from common import read,save,require
from history_store import checked_manifest
from feature_programs import training_samples,register,materialize,required_columns
from feature_contract import VERSION


def profile(c,key):
    m=checked_manifest(c);samples=training_samples(c,key,m);names=samples[0][2] if samples else [];stats={}
    for i,name in enumerate(names):
        v=np.concatenate([x[:,i] for t,x,n in samples]);finite=v[np.isfinite(v)]
        stats[name]=dict(sample_rows=len(v),finite_rows=len(finite),missing_fraction=round(float(np.isnan(v).mean()),5),
            minimum=float(finite.min()) if len(finite) else None,maximum=float(finite.max()) if len(finite) else None,
            mean=round(float(finite.mean()),6) if len(finite) else None,
            devices_with_values=sum(bool(np.isfinite(x[:,i]).any()) for t,x,n in samples))
    return dict(feature_program=key,feature_names=names,interface_version=VERSION,column_summary=stats,
        training_examples=[dict(rows=len(t),median_gap_seconds=float(np.median(np.diff(t)))) for t,x,n in samples],labels_visible=False,
        source='Diverse training-only device prefixes across spread shards. Missing in sample does NOT mean missing in population.')


def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['profile','validate'])
    p.add_argument('--config',required=True);p.add_argument('--parent',required=True);p.add_argument('--out',required=True)
    p.add_argument('--answer');a=p.parse_args();c=read(a.config)
    if a.task=='profile':result=profile(c,a.parent)
    else:
        m=checked_manifest(c);answer=read(a.answer)
        key=register(c,answer,a.parent,training_samples(c,a.parent,m,required=required_columns(answer['source'])))
        print('FEATURE_SAMPLE_AND_CAUSALITY_PASSED program='+key,flush=True)
        materialize(c,key,m)
        result=dict(feature_program=key,validated=True,source_id=a.parent)
    save(a.out,result)


if __name__=='__main__':main()
