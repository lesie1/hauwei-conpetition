"""v3 behavior: recoverable FN signals, exploration limits, causal topology and independent fits."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from common import save,read,sha,digest
from experiment_contract import initial_plans,validate_recipe
from settings import DEFAULTS,policy
import contrast
import raw_events
import raw_spatial_v3 as spatial
import parallel_branch as branch
import rerank
import crossval


class Evidence(unittest.TestCase):
    def test_recall_only_feature_is_not_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for d in range(40):
                group=d//10
                rows.append(dict(device=d,period=0,step=2,positive=group in (0,2),
                    max_score=.9 if group<2 else .1,max_hit_score=(.9 if group==0 else .1) if group in (0,2) else np.nan,
                    timely_prediction_opportunities=1,decision_time=100))
            oof=pd.DataFrame(rows);oof.to_parquet(root/'oof.parquet')
            save(root/'oof_complete.json',dict(best_step=2,cv_metrics={'threshold':.5},oof_sha256=sha(root/'oof.parquet')))
            def snapshot(c,units,names):
                return pd.DataFrame({'signal':np.where(units.device.between(20,29),1.,0.)},index=units.index)
            with patch.object(contrast,'feature_snapshot',snapshot):
                screen=contrast.prescreen(DEFAULTS,root,['signal'])
            self.assertFalse(screen['high_score']['credible_features'])
            self.assertEqual(screen['missed_faults']['credible_features'],['signal'])
            self.assertTrue(contrast.screen_decision(screen,DEFAULTS,{},[])['train'])
            self.assertFalse(contrast.screen_decision(screen,dict(DEFAULTS,prescreen_mode='strict_high'),{},[])['train'])

    def test_exploration_requires_reason_is_bounded_and_resume_idempotent(self):
        screen={'prescreen_passed':False,'high_score':{'credible_features':[]}}
        request={'job':'job_1','target':'interaction','exploration_reason':'conditional pattern hypothesis'}
        c=dict(DEFAULTS,feature_exploration_budget=1)
        self.assertFalse(contrast.screen_decision(screen,c,{},[])['train'])
        self.assertEqual(contrast.screen_decision(screen,c,request,[])['route'],'exploration')
        self.assertTrue(contrast.screen_decision(screen,c,request,['job_1'])['train'])
        self.assertFalse(contrast.screen_decision(screen,c,dict(request,job='job_2'),['job_1'])['train'])

    def test_fn_causes_and_late_alarm_overlap(self):
        u=pd.DataFrame(dict(step=[2]*5,positive=[True]*4+[False],
            max_score=[np.nan,np.nan,.2,.9,.1],max_hit_score=[np.nan,np.nan,.2,.2,np.nan],
            timely_prediction_opportunities=[0,2,2,2,0]))
        c=contrast.error_counts(u,2,.5)
        self.assertEqual((c['FN'],c['FP'],c['TN']),(4,1,1))
        self.assertEqual((c['fn_no_timely_clock'],c['fn_no_timely_score'],c['fn_scored_below_threshold']),(1,1,2))
        self.assertEqual(c['fn_with_outside_window_alarm'],1)

    def test_empty_groups_are_evidence_not_an_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);u=pd.DataFrame(dict(device=[0],period=[0],step=[2],positive=[False],max_score=[.1],
                max_hit_score=[np.nan],timely_prediction_opportunities=[0],decision_time=[1]))
            u.to_parquet(p/'oof.parquet');save(p/'oof_complete.json',dict(best_step=2,cv_metrics={'threshold':.5}))
            ev=contrast.contrast(DEFAULTS,['x'],p)
            self.assertEqual(ev['tp_units'],0);self.assertFalse(ev['credible_features'])


def frame(times,patterns,chips=None):
    n=len(times)
    return pd.DataFrame(dict(LogTime=times,burst_info=patterns,RankId=[0]*n,BankgroupId=[0]*n,
        BankId=[0]*n,RowId=[1]*n,ColumnId=[1]*n,deviceID=chips or [0]*n,error_type_full_name=['READ']*n))


class Spatial(unittest.TestCase):
    def test_chip_identity_short_windows_and_future_prefix(self):
        f=frame([100,200,4000],["{'0':[0]}","{'0':[4]}","{'0':[0,1,2,3] }"],[0,1,0])
        anchors=np.array([200,1000,4000])
        ev=spatial.events(f);x=spatial.features_at(ev,anchors)
        def col(name):return x[:,spatial.FEATURES.index(name)]
        self.assertEqual(col('raw_6h_distinct_cells')[0],1)
        self.assertEqual(col('spatial_15m_distinct_chip_cells')[0],2)
        self.assertEqual(col('raw_15m_events')[1],1) # interval is (anchor-window,anchor]
        before=spatial.features_at(spatial.events(f.iloc[:2]),anchors[:2])
        np.testing.assert_equal(before,x[:2])
        np.testing.assert_equal(x[:,:len(raw_events.FEATURES)],raw_events.features_at(raw_events.events(f),anchors))

    def test_shape_difference_and_missing(self):
        a=spatial.shape("{'0':[0,1], '3':[2,3]}")
        b=spatial.shape("{'0':[0], '1':[1], '2':[2], '3':[3]}")
        self.assertNotEqual(a,b)
        ev=spatial.events(frame([100],['']))
        x=spatial.features_at(ev,[100])
        self.assertTrue(np.isnan(x[0,spatial.FEATURES.index('spatial_15m_dq_run_max_max')]))
        self.assertEqual(x[0,spatial.FEATURES.index('spatial_15m_bit_observed_events')],0)

    def test_cache_representation_roundtrip_and_refuses_wrong_reuse(self):
        from tests.test_rerank import fixture
        from raw_cache import build
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,m=fixture(root,n=6);raw=root/'raw';raw.mkdir()
            frame([1704067200+100],["{'0':[0]}"]).to_csv(raw/'0.csv',index=False)
            out=root/'new_cache'
            r=build(c['active_cache'],raw,out,1,rep='spatial_v3')
            self.assertIn('spatial_15m_distinct_chip_cells',r['features'])
            self.assertEqual(build(c['active_cache'],raw,out,1,rep='spatial_v3')['features'],r['features'])
            with self.assertRaises(ValueError):build(c['active_cache'],raw,out,1,rep='v1')


class Independent(unittest.TestCase):
    def test_no_gate_and_serialization(self):
        from tests.test_rerank import booster
        a,b=booster(1),booster(8);model=branch.Parallel(a,b)
        x=np.random.default_rng(1).random((200,2)).astype(np.float32)
        p,q=a.inplace_predict(x),b.inplace_predict(x)
        self.assertTrue((q>p).any())
        np.testing.assert_equal(model.inplace_predict(x),np.maximum(p,q))
        np.testing.assert_equal(rerank.from_raw(model.save_raw()).inplace_predict(x),model.inplace_predict(x))
        # A low primary score is still promoted; no primary-score gate exists.
        class Scores:
            def __init__(self,v):self.v=v
            def inplace_predict(self,x,iteration_range=(0,0)):return np.full(len(x),self.v)
        np.testing.assert_equal(branch.Parallel(Scores(.1),Scores(.9)).inplace_predict(x),np.full(len(x),.9))

    def test_held_out_poison_does_not_change_auxiliary(self):
        rng=np.random.default_rng(7);x=rng.random((400,2)).astype(np.float32);y=(x[:,0]>.5).astype(np.uint8)
        w=np.ones(400,np.float32);dev=np.repeat(np.arange(40),10);folds=dev%4
        c=dict(DEFAULTS,device='cpu',num_threads=1);recipe=dict(initial_plans()[0],rerank='parallel')
        a=branch.fold_auxiliary(x,y,w,dev,folds,2,recipe,c,4)
        x[folds==2]=9;y[folds==2]=1-y[folds==2];w[folds==2]=1000
        b=branch.fold_auxiliary(x,y,w,dev,folds,2,recipe,c,4)
        self.assertEqual(bytes(a.save_raw()),bytes(b.save_raw()))
        self.assertAlmostEqual(branch.balance(np.array([3.,1.,2.]),np.array([1,1,2]))[:2].sum(),1.)

    def test_parallel_real_fit_cv_reload_memory_and_hdd(self):
        from tests.test_rerank import fixture
        from xgb_runner import run
        from prediction import predict
        from history_store import checked_manifest
        for domain in ('memory','hdd'):
            with self.subTest(domain=domain),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);c,m=fixture(root);c.update(policy(domain));c['domain']=domain
                base=initial_plans()[0];parent=root/'parent';run(c,base,parent)
                crossval.oof_tables(c,base,root/'parent_cv',k=3)
                recipe=dict(base,rerank='parallel');validate_recipe(recipe)
                out=root/'parallel';res=run(c,recipe,out,stage1_model=parent,stage1_cv=root/'parent_cv')
                self.assertTrue(res['sample_audit']['parallel']['all_training_rows'])
                cv=branch.oof_tables(c,recipe,root/'parallel_cv',root/'parent_cv')
                self.assertGreater(cv['fault_units'],0)
                paired=crossval.paired_cv(root/'parallel_cv',root/'parent_cv',20)
                self.assertIn('new_hits',paired)
                model=rerank.load_model(out/'model.ubj');m=checked_manifest(c)
                a,sa=predict(model,dict(c,score_workers=1),recipe,m,'val',res['metrics']['threshold'])
                b,sb=predict(model,dict(c,score_workers=2),recipe,m,'val',res['metrics']['threshold'])
                pd.testing.assert_frame_equal(a,b);self.assertEqual(sa,sb)


class Controller(unittest.TestCase):
    def test_initialize_reuses_active_cache_without_touching_old_session(self):
        from tests.test_v2_loop import fixture
        from search_step import initialize
        from agent import Loop
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old,c=fixture(root)
            before={name:sha(old/name) for name in ('config.json','state.json','code_seal.json')}
            new=root/'new_session'
            initialize(old,new)
            loop=Loop(new)
            self.assertEqual(loop.c['active_cache'],c['active_cache'])
            self.assertEqual(loop.scope()['feature_screen']['mode'],'balanced')
            self.assertTrue(loop.scope()['parallel']['allowed'])
            self.assertEqual(loop.s['trials'],[])
            self.assertEqual(before,{name:sha(old/name) for name in before})

    def test_real_loop_diagnostics_exploration_and_parallel(self):
        from tests.test_v2_loop import fixture,Mock
        from agent import Loop
        with tempfile.TemporaryDirectory() as tmp:
            session,c=fixture(Path(tmp));c.update(max_trials=4,feature_exploration_budget=1)
            save(session/'config.json',c);state=read(session/'state.json');state['config_hash']=digest(c);save(session/'state.json',state)
            class Agent(Mock):
                def call(self,role,payload,folder,**kw):
                    if role!='analysis':return super().call(role,payload,folder,**kw)
                    save(Path(folder)/'usage.json',{'total_tokens':5})
                    n=sum(t['status']=='complete' for t in payload['history'])
                    if n==1 and not payload['evidence']:
                        return dict(action='diagnose',parent='trial_001',kind='error_atlas',question='Both kinds of errors?',reason='Compare mechanisms')
                    if n==1:
                        return dict(action='experiment',parent='trial_001',changes={'base_weight':'device'},reason='Device dominance',hypothesis='Repeated devices dominate',expected='Fewer false alarms',falsification='No paired improvement')
                    if n==2:
                        return dict(action='features',parent='trial_001',hypothesis_id='H1',question='Conditional pattern?',reason='Interaction control',target='interaction',
                            expected='Recover misses',falsification='No gain',exploration_reason='Test whether marginal rejection is too restrictive')
                    return dict(action='experiment',parent='trial_001',changes={'rerank':'parallel'},reason='Complement misses',hypothesis='Another objective finds different faults',expected='New hits exceed added alarms',falsification='No paired gain')
            fake=Agent();loop=Loop(session,fake);loop.run()
            self.assertEqual(len(loop.s['trials']),4)
            self.assertEqual(loop.s['feature_explorations'],['job_001'])
            self.assertEqual(read(session/'feature_jobs/job_001/screen_decision.json')['route'],'exploration')
            self.assertIn('recall',read(session/'diagnostics/trial_001/error_atlas/evidence.json')['contrast'])
            feature_payload=fake.payloads[('feature','features')][0]
            self.assertEqual(feature_payload['target'],'interaction');self.assertTrue(feature_payload['error_evidence'])
            trial=loop.s['trials'][-1]
            self.assertEqual(trial['recipe']['rerank'],'parallel')
            self.assertEqual(trial['status'],'complete')
            self.assertIn(trial['acceptance']['rule'],('cv_paired','screened_out'))
            self.assertTrue((loop.folder(trial)/'model.ubj').read_bytes().startswith(branch.MAGIC))
            self.assertFalse((session/'cache_test').exists())


if __name__=='__main__':unittest.main()
