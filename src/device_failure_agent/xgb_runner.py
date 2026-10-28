"""XGBoost experiments and frozen-model policy controls use one sealed evaluator."""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from common import ROOT, read, save, sha, require, digest
from history_store import checked_manifest, feature_names, cache_path
from sample_builder import build_training
from experiment_contract import validate_source, training_identity
from prediction import predict
from evaluation import select, metrics, lead_summary, budget_curve
from comparison import (seal_evaluation, extended_metrics, operating_points,
                        frozen_operating_points)
from rerank import load_model


def cpu_model(model,threads):
    b=xgb.Booster();b.load_model(model.save_raw(raw_format='ubj'))
    b.set_param(dict(device='cpu',nthread=threads));return b


def reports(out,table,manifest,c,split,threshold,evaluation_id,recipe,frozen_points=None):
    primary=metrics(table,threshold)
    table.to_csv(out/f'devices_{split}.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    table.loc[table.status.str.startswith('unknown_')].to_csv(out/f'unknown_devices_{split}.csv',index=False)
    save(out/f'metrics_{split}.json',primary)
    op=operating_points(table) if split=='val' else frozen_operating_points(table,frozen_points)
    report=dict(primary=primary,additional_metrics=extended_metrics(table,threshold),
        operating_points=op,first_hit_lead_days=lead_summary(table),
        evaluation_id=evaluation_id,evaluation_period=dict(start_inclusive=manifest['windows'][split][0],
            end_exclusive=manifest['windows'][split][1],minimum_lead_seconds=c['minimum_lead_seconds'],maximum_lead_seconds=c['maximum_lead_seconds']),
        cohort_status_counts={str(k):int(v) for k,v in table.status.value_counts().items()},
        coverage_basis=manifest['coverage'],alarm_budgets=budget_curve(table) if split=='val' else [],
        schedule=dict(interval_hours=recipe['prediction_interval_hours'],aggregation=recipe['score_aggregation']),
        comparison_note='Compare only matching evaluation_id. Daily/hourly schedules use actual decision times; neither may remove devices from the denominator.',
        legacy_metric_names='Primary no_history_devices/no_history_fault_devices are retained scorer field names for NO PREDICTION, not proof of absent raw history. See additional_metrics and coverage diagnostics.',
        accuracy_note='Precision is not Accuracy. Late-only alarms count as both FP and FN, so this is not an ordinary four-cell classification confusion matrix.')
    save(out/f'report_{split}.json',report)
    return report


def _run(c,recipe,out,test_model=None,parent_model=None,require_reuse=False,stage1_model=None,stage1_cv=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    final_test=test_model is not None;split='test' if final_test else 'val'
    start=time.monotonic();m=checked_manifest(c,final_test);eid=seal_evaluation(c,m,split)
    from feature_programs import materialize
    c,m=materialize(c,recipe['feature_program'],m,final_test)
    features=feature_names(m['features'],recipe['feature_set'])
    timings={'data_verification':time.monotonic()-start}
    reused=None
    if parent_model and recipe.get('rerank','none')=='none':
        previous=read(Path(parent_model)/'result.json')
        if training_identity(previous['recipe'])==training_identity(recipe):
            require(previous['evaluation_id']==eid,'Frozen-model control evaluator differs')
            require(previous.get('features')==features,'Frozen-model feature order differs')
            reused=previous
    require(not require_reuse or reused is not None,"Required saved model is incompatible; training was NOT started")
    if final_test:
        previous=read(Path(test_model)/'result.json')
        from experiment_contract import with_defaults
        require(with_defaults(previous['recipe'])==with_defaults(recipe) and previous['evaluation_protocol']==c['evaluation_protocol'],'Frozen recipe differs')
        require(previous['features']==features,'Frozen feature order differs')
        require(sha(Path(test_model)/'model.ubj')==previous['model_sha256'],'Frozen model changed')
        model=load_model(Path(test_model)/'model.ubj',c['num_threads'])
        scores_dir=out/'scores_test'
        for p in scores_dir.glob('part-*.parquet'):p.unlink()
        table,stats=predict(model,c,recipe,m,split,previous['metrics']['threshold'],scores_dir)
        report=reports(out,table,m,c,split,previous['metrics']['threshold'],eid,recipe,previous['report']['operating_points'])
        save(out/'complete.json',dict(report=report,test_opened=True));return report
    inference_seconds=0.;training_seconds=0.;feature_seconds=0.
    if reused:
        require(sha(Path(parent_model)/'model.ubj')==reused['model_sha256'],'Parent model changed')
        shutil.copy2(Path(parent_model)/'model.ubj',out/'model.ubj')
        model=load_model(out/'model.ubj',c['num_threads'])
        tick=time.monotonic();table,stats=predict(model,c,recipe,m,'val');inference_seconds+=time.monotonic()-tick
        metric,grid=select(table);best=dict(step=reused['best_step'],metrics=metric,score_sha256=stats['score_sha256'])
        audit=reused['sample_audit'];save(out/'sample_audit.json',audit)
        save(out/'resolved_parameters.json',read(Path(parent_model)/'resolved_parameters.json'))
        save(out/'loss_history.json',[dict(step=best['step'],training_reused=True,
            val_period_unweighted_bce=stats['val_period_unweighted_bce'],**metric)])
        grid.to_csv(out/'threshold_search_val.csv',index=False,float_format='%.17g')
        actual='reused';print('Frozen-model control: no retraining; parent='+str(parent_model),flush=True)
    elif recipe.get('rerank','none') in ('gated','parallel'):
        require(stage1_model and stage1_cv,'Gated rerank needs --stage1-model and --stage1-cv')
        if recipe['rerank']=='parallel':
            from parallel_branch import train_full
        else:
            from rerank import train_full
        best,audit,params,actual,t=train_full(c,recipe,m,out,stage1_model,stage1_cv)
        feature_seconds,training_seconds,inference_seconds=t['feature_and_matrix_seconds'],t['boost_updates_seconds'],t['validation_prediction_seconds']
        save(out/'sample_audit.json',audit);save(out/'resolved_parameters.json',params)
    else:
        tick=time.monotonic()
        x,y,w,features,ratio,audit=build_training(c,recipe,m,out,c['minimum_lead_seconds'],c['memory_limit_gb'])
        save(out/'sample_audit.json',audit)
        dt=xgb.QuantileDMatrix(x,label=y,weight=w,max_bin=recipe['params']['max_bin'],nthread=c['num_threads'])
        del x;gc.collect()
        from scratch import cleanup_matrix
        cleanup_matrix(out)
        feature_seconds=time.monotonic()-tick
        params=dict(recipe['params']);params['eta']=params.pop('learning_rate')
        params.update(objective='binary:logistic',tree_method='hist',device=c['device'],nthread=c['num_threads'],
            subsample=1.,scale_pos_weight=ratio,base_score=.5,seed=recipe['seed'],disable_default_eval_metric=1)
        model=xgb.Booster(params,cache=[dt]);best=None;stale=0;history=[]
        save(out/'resolved_parameters.json',params)
        tick=time.monotonic()
        for step in range(1,c['rounds']+1):
            before=time.monotonic();model.update(dt,step-1);training_seconds+=time.monotonic()-before
            if step!=1 and step%c['eval_every'] and step!=c['rounds']:continue
            before=time.monotonic();cpu=cpu_model(model,c['num_threads'])
            print(f'Step {step}: validating {recipe["prediction_interval_hours"]}h schedule...',flush=True)
            table,stats=predict(cpu,c,recipe,m,'val');metric,grid=select(table)
            inference_seconds+=time.monotonic()-before
            p=np.clip(model.predict(dt).astype(np.float64),1e-12,1-1e-12)
            loss=float((-(y*np.log(p)+(1-y)*np.log1p(-p))*w*np.where(y==1,ratio,1)).sum()/w.sum(dtype=np.float64))
            history.append(dict(step=step,train_weighted_bce=loss,
                val_period_unweighted_bce=stats['val_period_unweighted_bce'],**metric))
            save(out/'loss_history.json',history)
            if best is None or metric['f1']>best['metrics']['f1']+1e-12:
                best=dict(step=step,metrics=metric,score_sha256=stats['score_sha256']);stale=0
                cpu.save_model(out/'model.ubj');grid.to_csv(out/'threshold_search_val.csv',index=False,float_format='%.17g')
            else:stale+=1
            print(f'Step {step}: device F1={metric["f1"]:.6f}; TP={metric["TP"]} FP={metric["FP"]} FN={metric["FN"]}; best={best["step"]}',flush=True)
            if stale>=c['patience']:break
        actual=json.loads(model.save_config())['learner']['generic_param']['device']
        require(not c['device'].startswith('cuda') or actual.startswith('cuda'),'Unexpected CPU fallback')
        del dt;gc.collect()
    model=load_model(out/'model.ubj',c['num_threads'])
    score_dir=out/'scores_val'
    for p in score_dir.glob('part-*.parquet'):p.unlink()
    before=time.monotonic();table,stats=predict(model,c,recipe,m,'val',best['metrics']['threshold'],score_dir)
    inference_seconds+=time.monotonic()-before
    require(stats['score_sha256']==best['score_sha256'],'Reloaded predictions drifted')
    report=reports(out,table,m,c,'val',best['metrics']['threshold'],eid,recipe)
    require(report['primary']==best['metrics'],'Checkpoint and final metrics disagree')
    gains=model.get_score(importance_type='gain')
    pd.DataFrame(dict(feature=features,gain=[gains.get(f'f{i}',0) for i in range(len(features))]))\
        .sort_values('gain',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    timings.update(feature_and_matrix_seconds=feature_seconds,boost_updates_seconds=training_seconds,
                   validation_prediction_seconds=inference_seconds,total_seconds=time.monotonic()-start)
    result=dict(recipe=recipe,evaluation_protocol=c['evaluation_protocol'],evaluation_id=eid,metrics=report['primary'],
        features=features,best_step=best['step'],report=report,sample_audit=audit,timings=timings,
        seconds=time.monotonic()-start,model_sha256=sha(out/'model.ubj'),actual_training_device=actual,
        selection_prediction_device='cpu',test_opened=False,model_reused_from=str(parent_model) if reused else None,
        loss_note='BCE remains sample-wise (events or device-days), not differentiable period F1. Training uses the chosen horizon/weights; VAL BCE is descriptive period-label loss. Model/threshold selection uses fixed period device F1.')
    save(out/'result.json',result)
    save(out/'complete.json',{'files':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*')
        if p.is_file() and p.name not in ('complete.json','execution.log','train_matrix.npy')}})
    (out/'train_matrix.npy').unlink(missing_ok=True)
    return result


def run(c,recipe,out,test_model=None,parent_model=None,require_reuse=False,stage1_model=None,stage1_cv=None):
    from scratch import cleanup_matrix
    try:
        return _run(c,recipe,out,test_model,parent_model,require_reuse,stage1_model,stage1_cv)
    finally:
        try:cleanup_matrix(out)
        except (OSError,ValueError) as exc:print('Scratch cleanup deferred to controller: '+str(exc),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',required=True)
    p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    p.add_argument('--test-model');p.add_argument('--parent-model');p.add_argument('--require-reuse',action='store_true')
    p.add_argument('--stage1-model');p.add_argument('--stage1-cv');a=p.parse_args()
    plan=read(a.plan);r=validate_source(Path(a.source).read_text(encoding='utf-8'),plan)
    run(read(a.config),r,a.out,a.test_model,a.parent_model,a.require_reuse,a.stage1_model,a.stage1_cv)


if __name__=='__main__':main()
