"""One persistent Agent action; the shared scheduler alternates independent sessions."""
import argparse
import copy
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from common import ROOT,read,save,sha,digest,code_hashes,require,RunLock,event
from agent import Loop
from settings import policy,validate_config
from experiment_contract import initial_plans

DONE={'stopped','trial_budget_complete'}


def initialize(source,session,active_cache=None,arm_path=None,data_dir=None):
    source,session=Path(source).resolve(),Path(session).resolve()
    arm=read(Path(arm_path) if arm_path else ROOT/'arm_policy.json')
    session.mkdir(parents=True,exist_ok=True)
    if (session/'state.json').exists():
        Loop(session)
        return
    original=read(source/'config.json')
    domain=original['domain']
    source_cache=Path(original.get('active_cache') or source/'cache').resolve()
    cache=Path(active_cache).resolve() if active_cache else source_cache
    m=read(cache/'manifest.json')
    require(not m['test_opened'],'Need unopened TRAIN/VAL cache')
    require(cache!=session and not cache.is_relative_to(session),'Cache must remain outside the new session')
    require(m['windows']==read(source_cache/'manifest.json')['windows'],'Active cache must keep the source split')
    lo,hi=m['windows']['val'];mid=(lo+hi)//2
    c=copy.deepcopy(original)
    c.update(policy(domain))
    if data_dir:c['data_dir']=str(Path(data_dir).resolve())
    from settings import DEFAULTS,V2_KEYS
    c.update({k:DEFAULTS[k] for k in V2_KEYS})
    c.update(output_root=str(session),workspace=str(session.parent/'workspace'),active_cache=str(cache),
        selection_window=[lo,mid],confirmation_window=[mid,hi],original_validation_window=[lo,hi],
        search_variant=arm['name'],sampling_reference=arm.get('sampling_reference',original.get('sampling_reference')),
        rounds=600,eval_every=50,patience=4,num_threads=8,memory_limit_gb=120,
        max_trials=14,max_agent_calls=120,max_reported_tokens=3000000,
        max_feature_programs=6,max_features_per_program=4,max_total_features=len(m['features'])+24,
        max_diagnostics_per_trial=3,
        trial_timeout_seconds=21600,feature_timeout_seconds=5400,min_free_disk_gb=5,
        agent_timeout_seconds=1200,agent_max_tokens=65536,agent_max_tokens_cap=131072,
        instructions=arm['description'])
    # Per-arm resource overrides (e.g. HDD needs more RAM, threads, scoring workers).
    allowed={'memory_limit_gb','num_threads','score_workers','cv_rounds','cv_folds','max_trials','trial_timeout_seconds','feature_timeout_seconds',
             'rerank_allowed','max_feature_programs','parallel_allowed','device_weight_allowed',
             'prescreen_mode','feature_exploration_budget','contrast_max_units_per_group'}
    require(set(arm.get('config',{}))<=allowed,'Arm config may only override resource settings')
    c.update(arm.get('config',{}))
    validate_config(c)
    mapping=read(source/'mapping.json')
    save(session/'config.json',c);save(session/'mapping.json',mapping)
    save(session/'code_seal.json',code_hashes())
    save(session/'inventory.json',dict(fingerprint=read(source/'state.json')['dataset'],reused_source=str(source)))
    save(session/'state.json',dict(version=1,status='ready',config_hash=digest(c),
        dataset=read(source/'state.json')['dataset'],mapping_hash=digest(mapping),calls=0,tokens=0,
        trials=[],best=None,pending=None,analyze_next=None,diagnostics={},feature_jobs=0,
        experience=None,experience_tried=True,last_failure=None))
    inputs={str(source/n):sha(source/n) for n in ['config.json','mapping.json']}
    inputs[str(source_cache/'manifest.json')]=sha(source_cache/'manifest.json')
    save(session/'bootstrap.json',dict(source=str(source),source_hashes=inputs,
        raw_preprocessing_reused=True,model_weights_imported=False,test_opened=False,
        selection_period=[lo,mid],confirmation_period=[mid,hi],
        note='Prior validation has exploratory history. Later VAL is kept outside this new Agent search, not claimed as a fresh independent test.'))
    Loop(session).prepare()


def step(session):
    loop=Loop(session)
    with RunLock(loop.root):
        if loop.s['status'] in DONE:
            return
        try:
            loop.s['status']='running';loop.commit()
            pending=loop.s.get('pending')
            if pending:
                action=pending['action']
                if action=='train':loop.train()
                elif action=='experiment':loop.queue(pending['recipe'],pending['parent'],pending['reason'],pending)
                elif action=='features':loop.features(pending)
                elif action=='diagnose':loop.diagnostic(pending)
                else:
                    loop.s.update(status='stopped',stop_reason=pending['reason'],pending=None);loop.commit()
            elif len(loop.s['trials'])>=loop.c['max_trials']:
                loop.s['status']='trial_budget_complete';loop.commit()
            elif not loop.s['trials']:
                loop.queue(initial_plans()[0])
            else:
                require(loop.s['best'],'Baseline did not complete; inspect failure')
                if loop.c.get('acceptance')=='cv_paired':loop.hypotheses()
                loop.analyze()
            event(loop.root,'scheduler_step_complete',status=loop.s['status'],best=loop.s['best'],test_opened=False)
        except Exception as e:
            loop.s.update(status='needs_attention',last_error=str(e));loop.commit();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--initialize-from')
    p.add_argument('--active-cache');p.add_argument('--arm');p.add_argument('--data-dir');a=p.parse_args()
    if a.initialize_from:initialize(a.initialize_from,a.session,a.active_cache,a.arm,a.data_dir)
    else:step(a.session)
