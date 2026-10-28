"""Post-search later-VAL confirmation. No Agent calls and no TEST access."""
import argparse
import copy
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from common import read,save,sha,require
from history_store import checked_manifest
from feature_programs import materialize
from prediction import predict
from evaluation import metrics
from agent import receipt


def confirm(session,out):
    import xgboost as xgb
    session,out=Path(session),Path(out)
    state=read(session/'state.json');c=read(session/'config.json')
    require(state['status'] in ('stopped','trial_budget_complete'),'Search must finish before later-VAL confirmation')
    require(state['best'] and state['trials'][0]['status']=='complete','Need complete baseline and winner')
    names=list(dict.fromkeys([state['trials'][0]['id'],state['best']]))
    entries=[]
    for name in names:
        folder=session/'trials'/name/'attempt_01';r=receipt(folder)
        entries.append(dict(trial=name,model_sha256=r['model_sha256'],threshold=r['metrics']['threshold'],recipe=r['recipe']))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'frozen.json').exists():require(read(out/'frozen.json')==entries,'Frozen entries changed')
    else:save(out/'frozen.json',entries)
    reports=[]
    for e in entries:
        target=out/e['trial'];target.mkdir(exist_ok=True)
        if (target/'report.json').exists():reports.append(read(target/'report.json'));continue
        base=checked_manifest(c);cc,m=materialize(c,e['recipe']['feature_program'],base)
        m=copy.deepcopy(m);m['windows']['val']=c['confirmation_window']
        from rerank import load_model
        model=load_model(session/'trials'/e['trial']/'attempt_01/model.ubj',2)
        table,stats=predict(model,cc,e['recipe'],m,'val',e['threshold'])
        result=dict(trial=e['trial'],model_sha256=e['model_sha256'],metrics=metrics(table,e['threshold']),
            confirmation_window=c['confirmation_window'],threshold_reselected=False,test_opened=False,
            note='Later VAL was withheld from this search; earlier historical VAL use means it is not a fresh independent test.')
        table.to_csv(target/'devices.csv',index=False);save(target/'report.json',result);reports.append(result)
    save(out/'summary.json',dict(reports=reports,test_opened=False,agent_feedback=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();confirm(a.session,a.out)
