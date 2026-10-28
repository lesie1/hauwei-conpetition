"""Persistent three-role experiment controller. Dataset sessions never share models."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import zipfile
import pandas as pd
from common import ROOT, read, save, sha, digest, require, event, process, RunLock, code_hashes
from settings import DEFAULTS, policy, validate_config
from providers import DeepSeek, ReplyTruncated
from experiment_contract import (initial_plans, validate_recipe, identity, training_identity,
                                 differences, validate_source, source_for, BOUNDS, FACTORS)
from data_ingest import inventory, validate_mapping, normalize
from history_store import checked_manifest
from comparison import paired, seal_evaluation
from feature_contract import contract as feature_contract


def compact_error(folder, error):
    log=Path(folder)/'execution.log'
    tail=log.read_text(encoding='utf-8',errors='replace')[-9000:] if log.exists() else ''
    return str(error)+'\n'+tail


def receipt(folder):
    folder=Path(folder);r=read(folder/'complete.json')
    require(r.get('files'),'Missing completion receipt')
    for name,h in r['files'].items():
        p=(folder/name).resolve()
        require(p.is_relative_to(folder.resolve()) and p.is_file() and sha(p)==h,'Completion receipt differs: '+name)
    return read(folder/'result.json')


class Loop:
    def __init__(self,session,provider=None):
        self.root=Path(session).resolve();self.c=validate_config(read(self.root/'config.json'))
        require(Path(self.c['output_root'])==self.root,'Session moved: use its original mounted workspace path')
        self.s=read(self.root/'state.json');self.provider=provider or DeepSeek(self.c)
        require(self.s['config_hash']==digest(self.c),'Configuration changed; create a new session for changed protocols/budgets')
        require(read(self.root/'code_seal.json')==code_hashes(),'Program changed; keep the release used to create this session')

    def commit(self):save(self.root/'state.json',self.s)

    def call(self,role,payload,*,max_tokens=None):
        if self.s['calls']>=self.c['max_agent_calls']:raise RuntimeError('Agent call budget reached; increase limits or export preserved results')
        if self.s['tokens']>=self.c['max_reported_tokens']:raise RuntimeError('Token budget reached; increase limits or export preserved results')
        self.s['calls']+=1;self.commit()
        folder=self.root/'calls'/f'{self.s["calls"]:05d}_{role}'
        folder.mkdir(parents=True,exist_ok=True);save(folder/'payload.json',payload)
        event(self.root,'agent_call',role=role,number=self.s['calls'])
        try:
            if max_tokens is None:answer=self.provider.call(role,payload,folder)
            else:answer=self.provider.call(role,payload,folder,max_tokens=max_tokens)
            save(folder/'response.json',answer);return answer
        finally:
            if (folder/'usage.json').exists():self.s['tokens']+=int(read(folder/'usage.json').get('total_tokens',0))
            self.commit()

    def validated_call(self,role,payload,check):
        error=None;last=None;repairs=0;truncations=0;max_tokens=None
        while True:
            current=dict(payload)
            if error:current.update(repair=dict(error=error,previous_response=last,
                instruction='Repair this response only; do not change parent, question, or accepted plan.'))
            try:
                last=self.call(role,current,max_tokens=max_tokens);return check(last)
            except ReplyTruncated as e:
                cap=self.c.get('agent_max_tokens_cap',65536)
                next_limit=min(e.max_tokens*2,cap)
                if truncations>=self.c.get('max_truncation_retries',1) or next_limit<=e.max_tokens:
                    raise RuntimeError(str(e)+f'; generation cap={cap} or retry limit reached; '
                        'session preserved; inspect calls before changing the generation budget') from None
                truncations+=1;max_tokens=next_limit
                event(self.root,'reply_budget_increased',role=role,attempt=truncations,
                    old_max_tokens=e.max_tokens,new_max_tokens=next_limit)
            except (ValueError,KeyError,TypeError,SyntaxError) as e:
                error=str(e);repairs+=1
                event(self.root,'response_repair',role=role,attempt=repairs,error=error)
                if repairs>self.c['max_repairs']:
                    raise ValueError(role+' response validation failed after bounded repair: '+str(error)) from e

    def adapt(self,mapping_path=None):
        inv=read(self.root/'inventory.json')
        def check(a):
            if a.get('needs_information'):raise RuntimeError('Data semantics need clarification: '+str(a['needs_information']))
            require(set(a)=={'mapping'},'Return only {mapping: ...}')
            validate_mapping(a['mapping'],self.c['data_dir'])
            require(a['mapping']['followup']['mode']!='complete_coverage' or self.c['complete_failure_coverage'],
                    'Complete failure coverage is not asserted; use documented follow-up or last_observation')
            return a['mapping']
        if mapping_path:
            m=check({'mapping':read(mapping_path)})
        else:
            example=read(ROOT/'examples/mapping_example.json')
            m=self.validated_call('feature',dict(task='adaptation',instructions=self.c['instructions'],
                domain=self.c['domain'],schemas=inv['schemas'],mapping_example=example,
                complete_failure_coverage=self.c['complete_failure_coverage']),check)
        save(self.root/'mapping.json',m);self.s['mapping_hash']=digest(m);self.s['status']='mapped';self.commit()

    def prepare(self):
        require(digest(read(self.root/'mapping.json'))==self.s['mapping_hash'],'Mapping changed')
        m=checked_manifest(self.c)
        seal_evaluation(self.c,m,'val')
        self.s['status']='ready';self.commit();return m

    def trial(self,name):
        t=next((x for x in self.s['trials'] if x['id']==name),None)
        require(t is not None and t['status']=='complete','Parent must be a completed trial in this session')
        return t

    def folder(self,t):return self.root/'trials'/t['id']/'attempt_01'

    def scope(self):
        rerank=self.c.get('rerank_allowed',False)
        parallel=self.c.get('parallel_allowed',True)
        return dict(level=2,allowed_changes=['params.'+k for k in BOUNDS]+['seed']+(['rerank','rerank_gate'] if rerank or parallel else [])+(['base_weight'] if self.c.get('device_weight_allowed',True) else []),
            features_allowed=True, controls_without_best_gain=0,
            parallel=dict(allowed=parallel, mode='parallel', gate=0.5,
                note='Independent shallow device-balanced branch trains on ALL TRAIN rows, including low-score rows. '
                     'Fuses row scores by max; same scorer, no unconditional acceptance. Not the M2-MFP rule tree.'),
            feature_screen=dict(mode=self.c.get('prescreen_mode','balanced'),
                exploration_budget=self.c.get('feature_exploration_budget',2),
                exploration_used=len(self.s.get('feature_explorations',[]))),
            rerank=dict(allowed=rerank,gates=list(FACTORS['rerank_gate']),
                note='rerank=gated adds a stage-2 model trained only on TRAIN rows whose out-of-fold stage-1 score >= rerank_gate; '
                     'it reorders the high-score region to separate true faults from false alarms. Set both rerank and rerank_gate '
                     'in one experiment whose parent is the single-stage model to build on; vary rerank_gate on a gated parent. '
                     'Stage 2 uses the parent recipe params.') if rerank else dict(allowed=False),
            explanation='Use evidence to choose parameter, causal-feature, permitted device-weight, or composite controls. Labels, sampling, feature_set, schedule and scorer remain fixed.')

    def check_action(self,a):
        require(isinstance(a,dict),'Return an action object')
        action=a.get('action');require(action in ('experiment','features','diagnose','stop'),'Choose experiment/features/diagnose/stop')
        require(isinstance(a.get('reason'),str) and a['reason'].strip(),'Need a short reason')
        completed=sum(t['status']=='complete' for t in self.s['trials'])
        if action=='stop':
            require(completed>=3 and self.s['feature_jobs']>=1,'Complete a parameter and feature attempt before stopping; resource failures are reported separately')
            return dict(action=action,reason=a['reason'])
        if completed==1 and self.s['feature_jobs']==0:
            require(action in ('experiment','diagnose'),'First propose one controlled parameter change')
        if completed>=2 and self.s['feature_jobs']==0:
            require(action in ('features','diagnose'),'A feature-agent experiment is required after the first parameter control')
        parent=self.trial(a.get('parent'));scope=self.scope()
        if action=='experiment':
            require(isinstance(a.get('changes'),dict) and a['changes'],'Provide partial changes')
            require(set(a['changes'])<=set(scope['allowed_changes']),'Changes outside current search scope: '+str(scope))
            from experiment_contract import with_defaults
            r=with_defaults(copy.deepcopy(parent['recipe']))
            for key,val in a['changes'].items():
                if key.startswith('params.'):r['params'][key[7:]]=val
                else:r[key]=val
            validate_recipe(r);diff=differences(parent['recipe'],r)
            require(r['rerank']!='parallel' or self.c.get('parallel_allowed',True), 'Parallel branch outside scope')
            require(r['rerank']!='gated' or self.c.get('rerank_allowed',False), 'Gated rerank outside scope')
            if with_defaults(parent['recipe'])['rerank']=='parallel':
                raise ValueError('Parallel controls start from the single-stage parent; do not stack branches')
            require(diff and len(diff)<=3,'Use one factor, or at most three required coupled representation settings')
            coupled=({'sample_unit','feature_set','score_aggregation'},{'rerank','rerank_gate'})
            if len(diff)>1:require(any(set(diff)<=g for g in coupled),'Several unrelated factors changed')
            if r['rerank']=='gated' and with_defaults(parent['recipe'])['rerank']=='gated':
                require(diff==['rerank_gate'],'On a gated parent change only rerank_gate; other changes belong on the single-stage parent')
                # The stage-1 model is the gated parent's own stage-1 parent.
                a=dict(a,parent=parent['parent'])
            require(identity(r) not in {identity(t['recipe']) for t in self.s['trials']},'Recipe already attempted; inspect history')
            for k in ('hypothesis','expected','falsification'):require(isinstance(a.get(k),str) and a[k].strip(),'Need '+k)
            if r['rerank'] in ('gated','parallel'):
                stage1=self.trial(a['parent'])
                require(with_defaults(stage1['recipe'])['rerank']=='none','Gated rerank needs a single-stage parent')
            return dict(a,recipe=r)
        require(isinstance(a.get('question'),str) and a['question'].strip(),'Need a concrete question')
        if action=='features':
            require(scope['features_allowed'],'First run a controlled parameter experiment or request evidence')
            require(parent['recipe'].get('rerank','none')=='none','Feature experiments need a single-stage parent; use the gated trial\'s stage-1 parent')
            require(self.s['feature_jobs']<self.c['max_feature_programs'],'Feature-job budget exhausted')
            require(a.get('target','balanced') in ('precision','recall','interaction','balanced'), 'Unknown feature target')
            if a.get('exploration_reason') is not None:
                require(isinstance(a['exploration_reason'],str) and a['exploration_reason'].strip(), 'Need nonempty exploration_reason')
            for k in ('expected','falsification'):require(isinstance(a.get(k),str) and a[k].strip(),'Need '+k)
            if (self.root/'hypotheses.json').exists():
                ids={h['id'] for h in read(self.root/'hypotheses.json')['hypotheses']}
                require(a.get('hypothesis_id') in ids,'Name the hypothesis_id this feature request tests: '+','.join(sorted(ids)))
        else:
            require(a.get('kind') in ('errors','coverage','score_distribution','feature_profiles','loss_curve','paired_devices','high_score_contrast','missed_fault_contrast','error_atlas'),'Unknown diagnostic kind')
            used=self.s['diagnostics'].get(parent['id'],[])
            require(len(used)<self.c['max_diagnostics_per_trial'] and a['kind'] not in used,'Diagnostic already done or per-trial budget exhausted; use saved evidence')
        return a

    def queue(self,recipe,parent=None,reason='baseline',decision=None):
        validate_recipe(recipe)
        t=dict(id=f'trial_{len(self.s["trials"])+1:03d}',status='pending',recipe=recipe,parent=parent,reason=reason)
        self.s['trials'].append(t);self.s['pending']=dict(action='train',trial=t['id'],decision=decision)
        self.commit();return t

    def train(self):
        task=self.s['pending'];t=next(x for x in self.s['trials'] if x['id']==task['trial']);out=self.folder(t)
        out.mkdir(parents=True,exist_ok=True);plan=t['recipe']
        if (out/'complete.json').exists():r=receipt(out)
        else:
            save(out/'plan.json',plan)
            if not (out/'candidate.py').exists():
                def check(a):
                    require(set(a)=={'source'},'Return only {source: ...}')
                    validate_source(a['source'],plan);return a['source']
                source=self.validated_call('code',dict(task='training',accepted_plan=plan,
                    source_example=source_for(plan),reason=t['reason'],decision=task.get('decision')),check)
                (out/'candidate.py').write_text(source,encoding='utf-8')
            validate_source((out/'candidate.py').read_text(encoding='utf-8'),plan)
            args=[str(ROOT/'xgb_runner.py'),'--config',str(self.root/'config.json'),'--source',str(out/'candidate.py'),
                  '--plan',str(out/'plan.json'),'--out',str(out)]
            if plan.get('rerank','none') in ('gated','parallel'):
                # Stage 1 = the parent; its device-grouped CV supplies out-of-fold gating scores.
                stage1=self.trial(t['parent']);receipt(self.folder(stage1))
                from rerank import stage1_identity
                require(stage1['recipe'].get('rerank','none')=='none' and stage1_identity(stage1['recipe'])==stage1_identity(plan),
                        'Rerank parent must be the single-stage recipe it builds on')
                args+=['--stage1-model',str(self.folder(stage1)),'--stage1-cv',str(self.cross_validate(stage1))]
            elif t['parent']:
                parent=self.trial(t['parent']);receipt(self.folder(parent))
                if training_identity(parent['recipe'])==training_identity(plan):args+=['--parent-model',str(self.folder(parent)),'--require-reuse']
            t['status']='running';self.commit();event(self.root,'training_started',trial=t['id'],log=str(out/'execution.log'))
            try:process(args,out/'execution.log',self.c['trial_timeout_seconds'])
            except (ValueError,TimeoutError) as e:
                error=compact_error(out,e);t['error']=error
                infrastructure=any(k in error.lower() for k in ('disk','scratch','memory','cuda','exceeded','no space'))
                t['status']='pending' if infrastructure else 'failed'
                if not infrastructure:self.s['pending']=None;self.s['last_failure']=dict(trial=t['id'],error=error)
                self.commit();event(self.root,'training_failed',trial=t['id'],infrastructure=infrastructure,error=str(e))
                if infrastructure:raise RuntimeError('Resource/timeout problem; results preserved. Fix resources and resume; no repeated automatic training retry.') from e
                return
            r=receipt(out)
        t.update(status='complete',metrics=r['metrics'],model_reused_from=r['model_reused_from'])
        if t['parent']:
            comparison=paired(out,self.folder(self.trial(t['parent'])),self.c['comparison_bootstraps'])
            save(out/'comparison_parent.json',comparison);t['comparison']=comparison
        self.commit()
        decision=self.accept(t)
        t['acceptance']=decision
        if decision['accepted']:self.s['best']=t['id']
        self.s['pending']=None;self.s['analyze_next']=t['id'];self.s['last_failure']=None;self.commit()
        event(self.root,'experiment_complete',trial=t['id'],accepted=decision['accepted'],
              rule=decision['rule'],**r['metrics']);self.publish_experience()

    def cross_validate(self,t):
        """Grouped TRAIN cross-validation for a completed trial (cached by recipe identity)."""
        dest=self.folder(t)/'cv'
        if not (dest/'oof_complete.json').exists():
            save(self.folder(t)/'cv_plan.json',t['recipe'])
            event(self.root,'cross_validation_started',trial=t['id'],folds=self.c['cv_folds'])
            args=[str(ROOT/'crossval.py'),'--config',str(self.root/'config.json'),'--plan',str(self.folder(t)/'cv_plan.json'),'--out',str(dest)]
            if t['recipe'].get('rerank','none') in ('gated','parallel'):
                # Nested CV reuses the stage-1 parent's fold models; parent CV runs first.
                args+=['--parent-cv',str(self.cross_validate(self.trial(t['parent'])))]
            process(args,self.folder(t)/'cv.log',self.c['trial_timeout_seconds'])
        r=read(dest/'oof_complete.json')
        t['cv']=dict(best_step=r['best_step'],metrics=r['cv_metrics'],fault_units=r['fault_units'],error_counts=r.get('error_counts'))
        self.commit();return dest

    def accept(self,t):
        """Adopt t as best only with paired CV evidence (v2), or legacy selection-F1 gain.

        cv_paired: the baseline is always accepted. A challenger is first screened on the
        selection window; only if its selection F1 is not clearly worse than best's
        (within screen_margin) does it get the costly CV run. It is accepted when the lower
        end of the paired-bootstrap 95% interval of pooled CV F1 exceeds accept_lower_bound
        and it lost no fault hits net (new_hits >= lost_hits).
        """
        best=self.s['best']
        if self.c.get('acceptance','selection_f1')=='selection_f1':
            ok=not best or t['metrics']['f1']>self.trial(best)['metrics']['f1']
            return dict(accepted=ok,rule='selection_f1')
        if not best:
            self.cross_validate(t);return dict(accepted=True,rule='baseline',cv=t['cv'])
        b=self.trial(best);gap=t['metrics']['f1']-b['metrics']['f1']
        if gap< -self.c['screen_margin']:
            return dict(accepted=False,rule='screened_out',selection_f1_gap=gap,
                        note='Selection-window F1 clearly below best; CV not run')
        theirs=self.cross_validate(b)
        try:mine=self.cross_validate(t)
        except ValueError as e:
            if t['recipe'].get('rerank','none')!='gated':raise
            # A gate that leaves a fold without enough of both classes is evidence, not a crash.
            log=self.folder(t)/'cv.log';tail=log.read_text(encoding='utf-8',errors='replace')[-1500:] if log.exists() else str(e)
            return dict(accepted=False,rule='rerank_cv_infeasible',reference=best,selection_f1_gap=gap,
                        error=tail,note='Nested cross-validation could not gate enough TRAIN rows; try a lower rerank_gate')
        from crossval import paired_cv
        p=paired_cv(mine,theirs,self.c['cv_bootstraps'])
        save(self.folder(t)/'cv_comparison_best.json',dict(p,reference_trial=best))
        ok=p['interval_95'][0]>self.c['accept_lower_bound'] and p['new_hits']>=p['lost_hits']
        return dict(accepted=bool(ok),rule='cv_paired',reference=best,selection_f1_gap=gap,
                    cv_delta_f1=p['delta_f1'],cv_interval_95=p['interval_95'],
                    probability_better=p['probability_better'],new_hits=p['new_hits'],lost_hits=p['lost_hits'],
                    new_false_alarms=p['new_false_alarms'],removed_false_alarms=p['removed_false_alarms'],
                    fault_units=p['fault_units'])

    def publish_experience(self):
        best=self.trial(self.s['best']);base=self.s['trials'][0]
        value=dict(version=1,session=self.root.name,domain=self.c['domain'],dataset=self.s['dataset'],
            params=best['recipe']['params'],metrics=best['metrics'],
            gain_from_own_baseline=best['metrics']['f1']-base['metrics']['f1'],
            semantics=read(self.root/'mapping.json')['description'],
            feature_notes=[read(p).get('reason','') for p in (self.root/'feature_jobs').glob('*/answer.json')],
            transfer_policy='Candidate hyperparameters and semantic ideas only; never copy trees, thresholds or labels.')
        save(Path(self.c['workspace'])/'experience'/f'{self.root.name}.json',value)

    def diagnostic(self,a):
        t=self.trial(a['parent']);out=self.folder(t);kind=a['kind'];dest=self.root/'diagnostics'/t['id']/kind
        dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'evidence.json').exists():
            d=pd.read_csv(out/'devices_val.csv');eligible=d.loc[d.status.eq('eligible')]
            result=dict(kind=kind,question=a['question'],trial=t['id'])
            if kind in ('errors','coverage'):
                result.update(counts=t['metrics'],status_counts={str(k):int(v) for k,v in d.status.value_counts().items()},
                    faulty_without_score=int((eligible.positive&eligible.max_score.isna()).sum()),
                    faulty_scored_below_hit_threshold=int((eligible.fn&eligible.max_hit_score.notna()).sum()))
            elif kind=='score_distribution':
                result['quantiles']={key:{str(q):float(v) if pd.notna(v) else None for q,v in eligible.loc[eligible[key],col].dropna().quantile([0,.25,.5,.75,1]).items()}
                    for key,col in [('hit','max_hit_score'),('fp','max_score'),('fn','max_hit_score')]}
            elif kind in ('high_score_contrast','missed_fault_contrast','error_atlas'):
                cv=self.cross_validate(t)
                process([str(ROOT/'contrast.py'),{'high_score_contrast':'contrast','missed_fault_contrast':'missed','error_atlas':'atlas'}[kind],'--config',str(self.root/'config.json'),
                    '--program',t['recipe']['feature_program'],'--cv',str(cv),'--out',str(dest/'contrast.json')],
                    dest/'execution.log',self.c['feature_timeout_seconds'])
                result['contrast']=read(dest/'contrast.json')
            elif kind=='loss_curve':result['history']=read(out/'loss_history.json')
            elif kind=='paired_devices':
                result['comparison']=t.get('comparison',{})
                result['examples']=eligible[['device','positive','hit','fp','fn']].head(30).to_dict('records')
            else:
                args=[str(ROOT/'feature_worker.py'),'profile','--config',str(self.root/'config.json'),
                    '--parent',t['recipe']['feature_program'],'--out',str(dest/'profile.json')]
                process(args,dest/'execution.log',self.c['feature_timeout_seconds']);result['profile']=read(dest/'profile.json')
            save(dest/'evidence.json',result)
        used=self.s['diagnostics'].setdefault(t['id'],[])
        if kind not in used:used.append(kind)
        self.s['pending']=None;self.s['analyze_next']=t['id'];self.commit()
        event(self.root,'diagnostic_complete',trial=t['id'],kind=kind,path=str(dest/'evidence.json'))

    def features(self,a):
        if not a.get('job'):
            self.s['feature_jobs']+=1;a['job']=f'job_{self.s["feature_jobs"]:03d}';self.s['pending']=a;self.commit()
        job=self.root/'feature_jobs'/a['job'];job.mkdir(parents=True,exist_ok=True);save(job/'request.json',a)
        parent=self.trial(a['parent']);key=parent['recipe']['feature_program']
        if not (job/'profile.json').exists():
            process([str(ROOT/'feature_worker.py'),'profile','--config',str(self.root/'config.json'),
                '--parent',key,'--out',str(job/'profile.json')],job/'execution.log',self.c['feature_timeout_seconds'])
        stage=read(job/'progress.json') if (job/'progress.json').exists() else dict(attempts=0,error=None)
        while not (job/'validated.json').exists() and stage['attempts']<=self.c['max_repairs']:
            payload=dict(task='features',question=a['question'],reason=a['reason'],expected=a['expected'],
                parent=key,max_features=self.c['max_features_per_program'],profile=read(job/'profile.json'),
                feature_api=feature_contract(),
                semantic_mapping=read(self.root/'mapping.json'),prior_experience=self.s.get('experience'),
                hypotheses=read(self.root/'hypotheses.json')['hypotheses'] if (self.root/'hypotheses.json').exists() else None,
                hypothesis_id=a.get('hypothesis_id'),
                error_evidence=[read(p) for p in sorted((self.root/'diagnostics'/parent['id']).glob('*/evidence.json'))],
                target=a.get('target','balanced'),exploration_reason=a.get('exploration_reason'),
                error=stage['error'],previous_response=read(job/'answer.json') if (job/'answer.json').exists() else None)
            answer=self.call('feature',payload);save(job/'answer.json',answer)
            stage['attempts']+=1;save(job/'progress.json',stage)
            try:
                process([str(ROOT/'feature_worker.py'),'validate','--config',str(self.root/'config.json'),
                    '--parent',key,'--answer',str(job/'answer.json'),'--out',str(job/'validated.json')],
                    job/'execution.log',self.c['feature_timeout_seconds'])
            except (ValueError,TimeoutError) as e:
                stage['error']=compact_error(job,e);save(job/'progress.json',stage)
                event(self.root,'feature_repair',job=a['job'],attempt=stage['attempts'],error=str(e))
                if any(w in stage['error'].lower() for w in ('disk','scratch','no space','exceeded')):
                    raise RuntimeError('Feature resource budget exceeded; restore resources and resume') from e
        if not (job/'validated.json').exists():
            self.s['last_failure']=dict(kind='features',question=a['question'],error=stage['error'])
            self.s['pending']=None;self.s['analyze_next']=parent['id'];self.commit();return
        r=copy.deepcopy(parent['recipe']);r['feature_program']=read(job/'validated.json')['feature_program']
        if identity(r) in {identity(t['recipe']) for t in self.s['trials']}:
            self.s['last_failure']=dict(kind='features',error='Duplicate feature experiment rejected',job=a['job'])
            self.s['pending']=None;self.commit();return
        if self.c.get('prescreen_features') and self.c.get('acceptance')=='cv_paired':
            # Cheap check before a full fit: do the new columns separate the parent's
            # high-score true faults from its false alarms on TRAIN out-of-fold units?
            if not (job/'prescreen.json').exists():
                cv=self.cross_validate(parent)
                process([str(ROOT/'contrast.py'),'prescreen','--config',str(self.root/'config.json'),
                    '--program',r['feature_program'],'--cv',str(cv),'--out',str(job/'prescreen.json')],
                    job/'prescreen.log',self.c['feature_timeout_seconds'])
            screen=read(job/'prescreen.json')
            event(self.root,'feature_prescreen',job=a['job'],passed=screen['prescreen_passed'],features=screen['passed_features'])
            from contrast import screen_decision
            decision=screen_decision(screen,self.c,a,self.s.get('feature_explorations',[]))
            save(job/'screen_decision.json',decision)
            if decision['route']=='exploration':
                used=self.s.setdefault('feature_explorations',[])
                if a['job'] not in used:used.append(a['job'])
                self.commit()
            if not decision['train']:
                self.s['last_failure']=dict(kind='feature_prescreen',job=a['job'],question=a['question'],
                    error=decision['reason'],evidence=screen)
                self.s['pending']=None;self.s['analyze_next']=parent['id'];self.commit();return
        self.queue(r,parent['id'],'Feature-only contrast: '+a['question'],a)

    def fallback(self,error):
        """Finite, logged controls when JSON repair fails; never pretend the model chose them."""
        parent=self.trial(self.s['best']);scope=self.scope();choices=[]
        for k,values in {'params.learning_rate':[.05,.15],'params.max_depth':[3,6],
            'params.reg_lambda':[5.,10.],'params.colsample_bytree':[.5,1.],
            'label_horizon_days':[15,30],'weight_policy':['sqrt_ratio','unit'],
            'score_aggregation':['window_mean','window_second']}.items():
            if k in scope['allowed_changes']:
                for value in values:choices.append({k:value})
        for changes in choices:
            a=dict(action='experiment',parent=parent['id'],changes=changes,
                reason='Controller fallback after invalid analysis response',hypothesis='A bounded single-factor control may improve validation device F1',
                expected='Compare with the saved parent under identical scoring',falsification='No validation improvement')
            try:a=self.check_action(a)
            except ValueError:continue
            event(self.root,'controller_fallback',error=str(error),changes=changes);return a
        return dict(action='stop',reason='Analysis repair exhausted and no untried safe fallback remains: '+str(error))

    def hypotheses(self):
        """Knowledge prior: the feature role states failure mechanisms before seeing outcomes.

        Input is field semantics only (mapping descriptions, device domain, the raw-parser
        description); no labels, metrics or values. Output is a falsifiable list that the
        analysis and feature roles must reference. Saved once per session.
        """
        path=self.root/'hypotheses.json'
        if path.exists():return read(path)
        m=read(self.root/'mapping.json');fields={}
        for group in m.get('observations',[]):
            for f in group['features']:fields[f['name']]=f['description']
        manifest=checked_manifest(self.c)
        # Controller-added columns (e.g. raw-event parser) carry their own descriptions.
        for f in manifest.get('raw_feature_descriptions',[]):fields[f['name']]=f['description']
        for name in manifest['features']:fields.setdefault(name,'model input column '+name)
        fields=[dict(name=k,description=fields[k]) for k in manifest['features']]
        least=min(3,len(fields))
        def check(a):
            require(set(a)=={'hypotheses'} and isinstance(a['hypotheses'],list) and least<=len(a['hypotheses'])<=12,f'Return {{hypotheses:[{least}..12 items]}}')
            names={f['name'] for f in fields}
            for i,h in enumerate(a['hypotheses']):
                require(isinstance(h,dict) and set(h)=={'id','mechanism','basis','fields','expected_direction','falsified_if'},'Each hypothesis needs id, mechanism, basis, fields, expected_direction, falsified_if')
                require(isinstance(h['fields'],list) and h['fields'] and set(h['fields'])<=names,'Hypothesis fields must be existing feature names: '+str(h.get('id')))
                require(h['expected_direction'] in ('higher_in_failing','lower_in_failing','pattern'),'expected_direction: higher_in_failing/lower_in_failing/pattern')
                for k in ('id','mechanism','basis','falsified_if'):require(isinstance(h[k],str) and h[k].strip(),'Empty '+k)
            require(len({h['id'] for h in a['hypotheses']})==len(a['hypotheses']),'Duplicate hypothesis id')
            return a
        answer=self.validated_call('feature',dict(task='hypotheses',domain=self.c['domain'],
            instructions=self.c['instructions'],fields=fields,mapping_description=m.get('description','')),check)
        value=dict(answer,labels_visible=False,source='field semantics only')
        save(path,value);event(self.root,'hypotheses_recorded',count=len(value['hypotheses']));return value

    def analyze(self):
        current=self.s.get('analyze_next') or self.s['best'];t=self.trial(current)
        evidence=[read(p) for p in sorted((self.root/'diagnostics'/current).glob('*/evidence.json'))]
        payload=dict(task='analysis',current=t,best=self.trial(self.s['best']),history=self.s['trials'],
            scope=self.scope(),evidence=evidence,last_failure=self.s.get('last_failure'),
            policy=policy(self.c['domain']),feature_requests_used=self.s['feature_jobs'],
            diagnostic_kinds_used=self.s['diagnostics'].get(current,[]),
            diagnostic_budget=self.c['max_diagnostics_per_trial'],
            allowed_factor_values=FACTORS,hyperparameter_bounds=BOUNDS,
            remaining_trials=self.c['max_trials']-len(self.s['trials']))
        payload['diagnostic_kinds']=['errors','coverage','score_distribution','feature_profiles','loss_curve','paired_devices','high_score_contrast','missed_fault_contrast','error_atlas']
        payload['feature_api']=feature_contract()
        if (self.root/'hypotheses.json').exists():payload['hypotheses']=read(self.root/'hypotheses.json')['hypotheses']
        payload['acceptance_rule']=dict(rule=self.c.get('acceptance','selection_f1'),
            lower_bound=self.c.get('accept_lower_bound'),screen_margin=self.c.get('screen_margin'),
            explanation='best changes only when paired CV bootstrap 95% lower bound > lower_bound and net hits are not lost' if self.c.get('acceptance')=='cv_paired' else 'selection-window F1 gain')
        payload['task_context']=dict(variant=self.c.get('search_variant'),instructions=self.c['instructions'],
            label_days=initial_plans()[0]['label_horizon_days'],maximum_lead_seconds=self.c['maximum_lead_seconds'],
            sampling_fixed=bool(self.c.get('sampling_reference')),selection_period=self.c.get('selection_window'),
            later_validation_hidden=True,test_opened=False)
        try:a=self.validated_call('analysis',payload,self.check_action)
        except ValueError as e:a=self.fallback(e)
        self.s['pending']=a;self.s['analyze_next']=None;self.commit()
        event(self.root,'decision_accepted',action=a['action'],parent=a.get('parent'),reason=a['reason'])

    def run(self):
        self.prepare();self.s['status']='running';self.commit()
        while True:
            a=self.s.get('pending')
            if a:
                if a['action']=='train':self.train()
                elif a['action']=='experiment':self.queue(a['recipe'],a['parent'],a['reason'],a)
                elif a['action']=='features':self.features(a)
                elif a['action']=='diagnose':self.diagnostic(a)
                else:
                    self.s['status']='stopped';self.s['stop_reason']=a['reason'];self.s['pending']=None;self.commit();break
                continue
            if len(self.s['trials'])>=self.c['max_trials']:
                self.s['status']='trial_budget_complete';self.commit();break
            if not self.s['trials']:self.queue(initial_plans()[0]);continue
            # Past datasets contribute a recipe candidate, NEVER fitted trees or thresholds.
            exp=self.s.get('experience')
            if exp and not self.s.get('experience_tried') and self.s['best']:
                self.s['experience_tried']=True;self.commit();r=copy.deepcopy(initial_plans()[0]);r['params']=exp['params']
                if identity(r) not in {identity(x['recipe']) for x in self.s['trials']}:
                    self.queue(r,self.s['trials'][0]['id'],'Historical hyperparameter candidate; fresh fit, source '+exp['session']);continue
            require(self.s['best'],'Baseline training failed; inspect execution.log and data semantics before continuing')
            if self.c.get('acceptance')=='cv_paired':self.hypotheses()
            self.analyze()
        event(self.root,'loop_complete',status=self.s['status'],best=self.s['best'],test_opened=False)


def create_session(data,workspace,domain,instructions,config=None,mapping=None,complete_coverage=False,provider=None):
    data=Path(data).resolve();workspace=Path(workspace).resolve();workspace.mkdir(parents=True,exist_ok=True)
    require(not workspace.is_relative_to(data),'Workspace must be outside input data')
    inv=inventory(data);name=time.strftime('%Y%m%d_%H%M%S')+'_'+domain+'_'+uuid.uuid4().hex[:8]
    root=workspace/'sessions'/name;root.mkdir(parents=True)
    override=read(config) if config else {};require(set(override)<=set(DEFAULTS),'Config may only override documented runtime settings')
    c=copy.deepcopy(DEFAULTS);c.update(override);c.update(policy(domain))
    c.update(data_dir=str(data),workspace=str(workspace),output_root=str(root),domain=domain,
             instructions=instructions,complete_failure_coverage=complete_coverage)
    validate_config(c);save(root/'config.json',c);save(root/'code_seal.json',code_hashes());save(root/'inventory.json',inv)
    experience=None
    for p in sorted((workspace/'experience').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        e=read(p)
        if e.get('domain')==domain and e.get('dataset')!=inv['fingerprint']:
            r=copy.deepcopy(initial_plans()[0]);r['params']=e.get('params')
            try:validate_recipe(r)
            except (ValueError,KeyError,TypeError):continue
            experience=e;break
    save(root/'state.json',dict(version=1,status='created',config_hash=digest(c),dataset=inv['fingerprint'],
        calls=0,tokens=0,trials=[],best=None,pending=None,analyze_next=None,diagnostics={},feature_jobs=0,
        experience=experience,experience_tried=False,last_failure=None))
    print('SESSION: '+str(root),flush=True)
    Loop(root,provider).adapt(mapping);return root


def freeze(loop):
    require(loop.s['best'] and not loop.s['pending'],'Finish or explicitly stop pending work before freezing')
    selected=[loop.s['trials'][0]['id'],loop.s['best']];selected=list(dict.fromkeys(selected))
    entries=[]
    for name in selected:
        t=loop.trial(name);r=receipt(loop.folder(t));entries.append(dict(trial=name,model_sha256=r['model_sha256'],
            recipe=r['recipe'],threshold=r['metrics']['threshold'],receipt_sha256=sha(loop.folder(t)/'complete.json')))
    save(loop.root/'frozen.json',dict(models=entries,policy=policy(loop.c['domain']),
        mapping_hash=loop.s['mapping_hash'],config_hash=digest(loop.c)))
    loop.s['status']='frozen';loop.commit();return entries


def update_limits(loop,changes):
    allowed={'memory_limit_gb','trial_timeout_seconds','feature_timeout_seconds','max_trials','max_agent_calls','max_reported_tokens'}
    require(set(changes)<=allowed and changes,'Only documented resource/call limits may be changed')
    require(not (loop.root/'frozen.json').exists(),'Frozen session cannot increase optimization budgets')
    before={k:loop.c[k] for k in changes};candidate=dict(loop.c,**changes);validate_config(candidate)
    save(loop.root/'config.json',candidate);loop.c=candidate;loop.s['config_hash']=digest(candidate);loop.commit()
    event(loop.root,'limits_changed',before=before,after=changes)


def final_test(loop):
    raise RuntimeError("TEST disabled: user authorized validation-only exploration")
    frozen=read(loop.root/'frozen.json');require(frozen['config_hash']==digest(loop.c),'Frozen settings changed')
    old=checked_manifest(loop.c)
    # Exact source bytes and mapping are frozen. New quarterly data require a new session.
    for x in old['source_receipts']:require(sha(Path(loop.c['data_dir'])/x['path'])==x['sha256'],'Source data changed before final test')
    require(digest(read(loop.root/'mapping.json'))==frozen['mapping_hash'],'Frozen mapping changed')
    normalize(loop.c,read(loop.root/'mapping.json'),read(loop.root/'inventory.json')['fingerprint'],True)
    reports=[]
    for entry in frozen['models']:
        t=loop.trial(entry['trial']);folder=loop.folder(t);r=receipt(folder)
        require(r['recipe']==entry['recipe'] and r['model_sha256']==entry['model_sha256'] and
            r['metrics']['threshold']==entry['threshold'] and sha(folder/'complete.json')==entry['receipt_sha256'],'Frozen model or threshold changed')
        out=loop.root/'test'/t['id'];out.mkdir(parents=True,exist_ok=True)
        if not (out/'complete.json').exists():
            process([str(ROOT/'xgb_runner.py'),'--config',str(loop.root/'config.json'),'--source',str(folder/'candidate.py'),
                '--plan',str(folder/'plan.json'),'--out',str(out),'--test-model',str(folder)],
                out/'execution.log',loop.c['trial_timeout_seconds'])
        reports.append(dict(trial=t['id'],report=read(out/'complete.json')['report']))
    save(loop.root/'test_summary.json',reports);loop.s['status']='tested';loop.commit();return reports


def export_report(loop):
    dest=loop.root.parent/(loop.root.name+'_report.zip')
    omit={'cache','cache_test','feature_views','training_views','scores_val','scores_test','__pycache__'}
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in loop.root.rglob('*'):
            if not p.is_file() or any(x in omit for x in p.relative_to(loop.root).parts):continue
            if p.name=='run.lock' or p.suffix in ('.ubj','.npy','.parquet'):continue
            z.write(p,p.relative_to(loop.root))
    return dest


def main():
    p=argparse.ArgumentParser(description='Three DeepSeek roles; one independent session per new dataset')
    sub=p.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init');init.add_argument('--data',required=True);init.add_argument('--workspace',required=True)
    init.add_argument('--domain',choices=['memory','hdd','generic'],required=True)
    init.add_argument('--instructions',default='');init.add_argument('--description-file');init.add_argument('--config');init.add_argument('--mapping')
    init.add_argument('--complete-failure-coverage',action='store_true')
    for cmd in ('adapt','prepare','run','doctor','status','export','freeze','test','limits'):
        q=sub.add_parser(cmd);q.add_argument('--session',required=True)
        if cmd=='prepare':q.add_argument('--mapping')
        if cmd=='doctor':q.add_argument('--api-check',action='store_true')
        if cmd=='limits':
            for key in ('memory_limit_gb','trial_timeout_seconds','feature_timeout_seconds','max_trials','max_agent_calls','max_reported_tokens'):
                q.add_argument('--'+key.replace('_','-'),type=float if key=='memory_limit_gb' else int)
    a=p.parse_args()
    if a.command=='init':
        text=a.instructions
        if a.description_file:text+='\n'+Path(a.description_file).read_text(encoding='utf-8-sig')
        create_session(a.data,a.workspace,a.domain,text,a.config,a.mapping,a.complete_failure_coverage);return
    loop=Loop(a.session)
    if a.command=='status':print(json.dumps(loop.s,ensure_ascii=False,indent=2));return
    with RunLock(loop.root):
        try:
            if a.command=='adapt':
                if not (loop.root/'mapping.json').exists():loop.adapt()
                else:require(digest(read(loop.root/'mapping.json'))==loop.s['mapping_hash'],'Mapping changed')
                print('MAPPING READY:',loop.root/'mapping.json',flush=True)
            elif a.command=='prepare':
                if a.mapping or not (loop.root/'mapping.json').exists():loop.adapt(a.mapping)
                loop.prepare()
            elif a.command=='run':
                require(not (loop.root/'frozen.json').exists(),'Session already frozen; do not use test feedback to continue tuning')
                loop.run()
            elif a.command=='doctor':
                import xgboost as xgb
                print('XGBoost:',xgb.__version__,'device:',loop.c['device'],'free disk GiB:',round(shutil.disk_usage(loop.root).free/1024**3,2))
                print('Policy:',policy(loop.c['domain']))
                if (loop.root/'cache/manifest.json').exists():checked_manifest(loop.c);print('Cached data hashes OK')
                if a.api_check:
                    for role in ('analysis','feature','code'):
                        loop.call(role,dict(task='connection_check',instruction='Return exactly {"ok":true}; do not perform an experiment.'))
                    print('All three DeepSeek role calls returned')
            elif a.command=='export':print('REPORT:',export_report(loop))
            elif a.command=='freeze':print(json.dumps(freeze(loop),indent=2))
            elif a.command=='test':print(json.dumps(final_test(loop),ensure_ascii=False,indent=2))
            elif a.command=='limits':
                update_limits(loop,{k:v for k,v in vars(a).items() if k not in ('command','session') and v is not None})
        except Exception as e:
            loop.s['status']='needs_attention';loop.s['last_error']=str(e);loop.commit();raise


if __name__=='__main__':main()
