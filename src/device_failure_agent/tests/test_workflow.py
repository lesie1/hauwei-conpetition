"""Offline contracts + real CPU XGBoost runs. DeepSeek replies are explicit test doubles."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from common import save,read,sha,DAY
from data_ingest import normalize
from agent import Loop,create_session,receipt,freeze,final_test,export_report,update_limits
from data_ingest import frames,seconds,validate_mapping,ids
from experiment_contract import initial_plans,source_for,training_identity
from feature_programs import validate_source,apply
from sample_builder import targets
from settings import policy
from evaluation import period_cohort,annotate,device_table,metrics

START=1704067200


def fixture(root,domain):
    root.mkdir();data=[];tickets=[]
    for device in range(12):
        day={0:30,1:35,2:70,3:74,4:90,5:94}.get(device)
        rows=[]
        for d in range(100 if day is None else day+1):
            metric=(max(0,8-(day-d)) if day is not None else 0)+device*.03
            rows.append(dict(serial=f'd{device:02d}',date=START+d*DAY,metric=metric,
                failure=int(day is not None and d==day),type='B'))
        frame=pd.DataFrame(rows)
        if domain=='memory':
            (root/'observations').mkdir(exist_ok=True)
            frame[['date','metric']].to_feather(root/'observations'/f'd{device:02d}.feather')
            if day is not None:tickets.append(dict(serial=f'd{device:02d}',when=START+day*DAY,type='B'))
        else:data.append(frame)
    feature=dict(name='metric',column='metric',kind='gauge',unit='count',description='Synthetic measured degradation')
    if domain=='memory':
        pd.DataFrame(tickets).to_csv(root/'failures.csv',index=False)
        mapping=dict(observations=[dict(glob='observations/*.feather',id_column=None,time_column='date',time_unit='s',timezone='UTC',filters={},features=[feature])],
            failure=dict(mode='tickets',glob='failures.csv',id_column='serial',time_column='when',time_unit='s',timezone='UTC',filters={'type':'B'},definition='Test terminal failure ticket'))
    else:
        frame=pd.concat(data);q1=frame.loc[frame.date<START+50*DAY];q2=frame.loc[frame.date>=START+50*DAY].rename(columns={'metric':'metric_new'})
        q1.to_csv(root/'quarter1.csv',index=False);q2.to_parquet(root/'quarter2.parquet',index=False)
        mapping=dict(observations=[dict(glob=g,id_column='serial',time_column='date',time_unit='s',timezone='UTC',filters={},
            features=[dict(feature,column=col)]) for g,col in [('quarter1.csv','metric'),('quarter2.parquet','metric_new')]],
            failure=dict(mode='indicator',column='failure',positive_values=['1'],definition='Test failure indicator'))
    mapping.update(followup=dict(mode='last_observation',basis='Test measurements observed through terminal fault or last record'),description='Synthetic fixture only')
    return mapping


class FakeDeepSeek:
    def __init__(self,mapping):self.mapping=mapping;self.calls=[];self.bad_code=True;self.bad_feature=True;self.fail_analysis_once=False
    def call(self,role,payload,folder):
        self.calls.append((role,payload.get('task')));save(Path(folder)/'usage.json',{'total_tokens':10})
        if role=='code':
            if self.bad_code:self.bad_code=False;return {'source':'def run(ctx):\n    return 123'}
            return {'source':source_for(payload['accepted_plan'])}
        if role=='feature':
            if payload['task']=='adaptation':return {'mapping':self.mapping}
            if self.bad_feature:
                self.bad_feature=False
                return dict(source="def transform(frame):\n    return pd.DataFrame({'fe_rise': frame['metric'].shift(-1)}, index=frame.index)",features=['fe_rise'],reason='Deliberately invalid test response')
            return dict(source="def transform(frame):\n    prior = frame['metric'].rolling('7D', min_periods=1).mean()\n    return pd.DataFrame({'fe_rise': frame['metric'] - prior}, index=frame.index)",features=['fe_rise'],reason='Recent rise relative to trailing mean')
        if self.fail_analysis_once:self.fail_analysis_once=False;raise RuntimeError('Simulated disconnected analysis API')
        t=payload['current'];n=sum(x['status']=='complete' for x in payload['history'])
        common=dict(parent=t['id'],reason='Offline test control')
        if n==1 and not payload['evidence']:return dict(common,action='diagnose',kind='coverage',question='Are faults missing scores?')
        if n==2:return dict(common,action='features',question='Is the metric rising abruptly?',expected='More timely hits',falsification='No validation device-F1 gain')
        return dict(common,action='experiment',changes={'params.learning_rate':.05},hypothesis='A slower rate changes ranking',expected='Validation improvement',falsification='No improvement')


class Contracts(unittest.TestCase):
    def test_lead_bounds_and_no_prediction(self):
        failure=START+20*DAY
        registry=pd.DataFrame([dict(device=i,sn_name=str(i),failure=failure,observed_start=START,observed_end=failure) for i in range(5)])
        times=[failure-900,failure-900-7*DAY,failure-900-7*DAY-1,failure-899]
        meta=pd.DataFrame(dict(device=range(4),time=times,history_rows=1))
        for domain,expected in [('memory',2),('hdd',3)]:
            cohort=period_cohort(registry,[START,START+30*DAY],policy(domain));a=annotate(meta,cohort,900)
            table,_=device_table(a,[.9]*4,cohort);r=metrics(table,.5)
            self.assertEqual(r['TP'],expected);self.assertEqual(r['FN'],5-expected)
            self.assertEqual(r['FP'],4-expected);self.assertEqual(r['no_history_fault_devices'],1)

    def test_terminal_positive_and_censoring(self):
        m=pd.DataFrame(dict(time=[START]*3,failure=[START+3*DAY,-1,-1],
            observed_start=[START]*3,observed_end=[START+3*DAY,START+3*DAY,START+30*DAY]))
        good,y=targets(m,7,START+60*DAY)
        np.testing.assert_array_equal(good,[True,False,True]);np.testing.assert_array_equal(y,[1,0,0])
        good,_=targets(m,7,START+5*DAY);self.assertFalse(good.any())

    def test_feature_causality_and_rejection(self):
        source="def transform(frame):\n    return pd.DataFrame({'fe_m': frame['x'].rolling('2D', min_periods=1).mean()}, index=frame.index)"
        t=np.arange(10)*DAY+START;x=np.arange(10,dtype=np.float32)[:,None]
        a=apply(source,t,x,['x'],['fe_m']);x[5:]=999
        b=apply(source,t,x,['x'],['fe_m']);np.testing.assert_equal(a[:5],b[:5])
        for bad in ["frame['x'].shift(-1)","frame['x'].mean()","frame['x'].rolling(2, center=True).mean()",
                    "frame['x'].replace(method='bfill')","frame['x'].replace(frame['x'])","frame.index.astype('int64')","pd.read_csv('secret')"]:
            with self.assertRaises(ValueError):validate_source("def transform(frame):\n    return "+bad)

    def test_model_identity(self):
        a=initial_plans()[0];b=copy.deepcopy(a);b['score_aggregation']='window_mean'
        self.assertEqual(training_identity(a),training_identity(b))
        b['feature_program']='a'*64;self.assertNotEqual(training_identity(a),training_identity(b))
        b=copy.deepcopy(a);b['base_weight']='device';self.assertNotEqual(training_identity(a),training_identity(b))
        b['label_horizon_days']=15
        with self.assertRaises(ValueError):training_identity(b)

    def test_formats_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=pd.DataFrame(dict(a=[1,2],b=['x','y']))
            data.to_json(root/'a.json',orient='records');data.to_json(root/'a.jsonl',orient='records',lines=True)
            data.to_excel(root/'a.xlsx',index=False)
            (root/'a.log').write_text('1 x\n2 y\n')
            for ext in ('json','jsonl','xlsx'):self.assertEqual(len(next(frames(root/f'a.{ext}'))),2)
            self.assertEqual(len(next(frames(root/'a.log',log_regex=r'(?P<a>\d+) (?P<b>\w+)'))),2)
            self.assertEqual(ids(data,{'id_column':['a','b'],'id_separator':'_'},'a').tolist(),['1_x','2_y'])
            self.assertEqual(int(seconds(pd.Series(['2024-01-01']),'date')[0]),START)


class Integration(unittest.TestCase):
    def exercise(self,domain):
        with tempfile.TemporaryDirectory(prefix='dfa_') as tmp:
            root=Path(tmp);mapping=fixture(root/'data',domain);fake=FakeDeepSeek(mapping)
            cfg=root/'runtime.json';save(cfg,dict(device='cpu',num_threads=1,memory_limit_gb=2,rounds=3,
                eval_every=1,patience=2,max_trials=3,min_free_disk_gb=.01,comparison_bootstraps=10,acceptance='selection_f1'))
            session=create_session(root/'data',root/'work',domain,'Synthetic fixture',cfg,provider=fake)
            loop=Loop(session,fake);normalize(loop.c,mapping,loop.s['dataset']);fake.fail_analysis_once=True
            with self.assertRaisesRegex(RuntimeError,'disconnected'):loop.run()
            saved_baseline=sha(loop.folder(loop.s['trials'][0])/'model.ubj')
            loop=Loop(session,fake);loop.run()
            self.assertEqual(saved_baseline,sha(loop.folder(loop.s['trials'][0])/'model.ubj'))
            self.assertEqual([t['status'] for t in loop.s['trials']],['complete']*3)
            self.assertEqual({r for r,t in fake.calls},{'feature','analysis','code'})
            self.assertEqual(loop.s['feature_jobs'],1)
            feature=loop.s['trials'][2];self.assertNotEqual(feature['recipe']['feature_program'],'base')
            self.assertIn('fe_rise',receipt(loop.folder(feature))['features'])
            self.assertTrue((session/'feature_jobs/job_001/progress.json').exists())
            before=len(fake.calls);models=[sha(loop.folder(t)/'model.ubj') for t in loop.s['trials']]
            Loop(session,fake).run();self.assertEqual(len(fake.calls),before)
            self.assertEqual(models,[sha(loop.folder(t)/'model.ubj') for t in loop.s['trials']])
            # Same training identity: explicit saved-tree reuse, with no fit.
            plan=copy.deepcopy(feature['recipe']);plan['score_aggregation']='window_mean'
            loop.queue(plan,feature['id'],'Saved model policy test');loop.train()
            self.assertEqual(sha(loop.folder(feature)/'model.ubj'),sha(loop.folder(loop.s['trials'][-1])/'model.ubj'))
            self.assertIsNotNone(loop.s['trials'][-1]['model_reused_from'])
            self.assertTrue(export_report(loop).exists())
            # Exercise freezing a generated feature chain too; these toy candidates tie on validation.
            loop.s['best']=feature['id'];loop.commit()
            freeze(loop)
            with self.assertRaisesRegex(RuntimeError,'TEST disabled'):final_test(loop)
            self.assertEqual(loop.s['status'],'frozen')
            self.assertFalse((session/'cache_test').exists())
            self.assertEqual(len(fake.calls),before+1)
            # A different dataset gets a new session and a fresh baseline, only recipe experience travels.
            mapping2=fixture(root/'newdata',domain)
            second=create_session(root/'newdata',root/'work',domain,'Different input',cfg,provider=FakeDeepSeek(mapping2))
            other=Loop(second,FakeDeepSeek(mapping2));self.assertNotEqual(session,second)
            self.assertEqual(other.s['trials'],[]);self.assertIsNotNone(other.s['experience'])
            normalize(other.c,mapping2,other.s['dataset']);other.prepare();other.queue(initial_plans()[0]);other.train()
            self.assertIsNone(receipt(other.folder(other.s['trials'][0]))['model_reused_from'])

    def test_handoff_fallback_scope_and_resource_limits(self):
        with tempfile.TemporaryDirectory(prefix='dfa_contract_') as tmp:
            root=Path(tmp);mapping=fixture(root/'data','hdd');fake=FakeDeepSeek(mapping)
            cfg=root/'runtime.json';save(cfg,dict(device='cpu',min_free_disk_gb=.01))
            session=create_session(root/'data',root/'work','hdd','Test',cfg,provider=fake)
            loop=Loop(session,fake);base=initial_plans()[0]
            # Controller decisions can be checked without another model fit.
            loop.s['trials']=[dict(id='trial_001',recipe=base,status='complete',metrics={'f1':.2})];loop.s['best']='trial_001'
            good=dict(action='experiment',parent='trial_001',changes={'params.max_depth':3},reason='Test',
                hypothesis='Test',expected='Test',falsification='Test')
            self.assertEqual(loop.check_action(good)['recipe']['params']['max_depth'],3)
            with self.assertRaises(ValueError):loop.check_action(dict(good,changes={'maximum_lead_seconds':999}))
            with self.assertRaises(ValueError):loop.check_action(dict(good,parent='from_another_session'))
            self.assertEqual(loop.fallback('bad JSON')['action'],'experiment')
            self.assertEqual(loop.scope()['level'],2)
            self.assertIn('base_weight',loop.scope()['allowed_changes'])
            for i in range(3):loop.s['trials'].append(dict(id=f'control_{i}',recipe=base,status='complete',metrics={'f1':.19}))
            self.assertEqual(loop.scope()['level'],2)
            self.assertNotIn('label_horizon_days',loop.scope()['allowed_changes'])
            before=loop.c['evaluation_protocol'];update_limits(loop,dict(max_agent_calls=200,trial_timeout_seconds=14400))
            self.assertEqual(Loop(session,fake).c['evaluation_protocol'],before)
            with self.assertRaises(ValueError):update_limits(loop,{'maximum_lead_seconds':0})

    def test_reply_format_guard(self):
        from providers import parse_reply
        self.assertEqual(parse_reply('```json\n{"action":"stop","reason":"done"}\n```')['action'],'stop')
        for invalid in ['{"a":1,"a":2}','{"a":NaN}','[]']:
            with self.assertRaises(ValueError):parse_reply(invalid)

    def test_daily_history_and_weighted_training(self):
        with tempfile.TemporaryDirectory(prefix='dfa_history_') as tmp:
            root=Path(tmp);mapping=fixture(root/'data','hdd');fake=FakeDeepSeek(mapping)
            cfg=root/'runtime.json';save(cfg,dict(device='cpu',num_threads=1,memory_limit_gb=2,rounds=2,
                eval_every=1,patience=2,min_free_disk_gb=.01,comparison_bootstraps=5,acceptance='selection_f1'))
            session=create_session(root/'data',root/'work','hdd','Test',cfg,provider=fake)
            loop=Loop(session,fake);normalize(loop.c,mapping,loop.s['dataset']);loop.prepare()
            r=initial_plans()[0];r.update(sample_unit='device_day',feature_set='history',score_aggregation='latest',base_weight='device')
            loop.queue(r);loop.train();first=loop.s['trials'][0]
            result=receipt(loop.folder(first));self.assertEqual(len(result['features']),9)
            self.assertEqual(result['sample_audit']['weight_definition'],'each device total base weight = 1')
            r=copy.deepcopy(r);r.update(feature_set='recent',weight_policy='sqrt_ratio')
            loop.queue(r,first['id'],'Temporal representation/weight integration test');loop.train()
            result=receipt(loop.folder(loop.s['trials'][1]));self.assertEqual(len(result['features']),13)
            self.assertIsNone(result['model_reused_from'])
            self.assertEqual(result['sample_audit']['training_samples'],receipt(loop.folder(first))['sample_audit']['training_samples'])
            loop.diagnostic(dict(parent=first['id'],kind='score_distribution',question='Handle empty FP/FN groups'))
            self.assertTrue((session/'diagnostics/trial_001/score_distribution/evidence.json').exists())

    def test_hdd_mixed_schemas_end_to_end(self):self.exercise('hdd')
    def test_memory_b_end_to_end(self):self.exercise('memory')


if __name__=='__main__':unittest.main()
