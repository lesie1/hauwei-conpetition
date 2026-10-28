"""Session configuration: task policy belongs to the controller, never a trial."""
from pathlib import Path
from common import require

DEFAULTS = dict(device='cuda:0',num_threads=8,memory_limit_gb=32,max_trials=24,
    max_repairs=2,max_agent_calls=160,max_reported_tokens=1500000,rounds=600,
    eval_every=50,patience=4,trial_timeout_seconds=7200,feature_timeout_seconds=1800,
    deepseek_base_url='https://api.deepseek.com',analysis_model='deepseek-v4-pro',
    code_model='deepseek-v4-pro',feature_model='deepseek-v4-pro',agent_timeout_seconds=600,
    agent_max_tokens=32768,agent_max_tokens_cap=65536,max_truncation_retries=1,
    adaptation_reasoning_effort='low',max_diagnostics_per_trial=2,max_feature_programs=12,
    max_features_per_program=5,max_total_features=2000,max_device_rows=1000000,
    min_free_disk_gb=1,max_json_file_mb=128,comparison_bootstraps=200,
    split_ratios=[.6,.2,.2],coverage_start=None,coverage_end=None,minimum_lead_seconds=900,
    search_policy=dict(min_controls_per_level=3,plateau_controls=2,minimum_f1_gain=1e-6),
    # v2 evidence controls: CV on TRAIN decides acceptance; selection window only screens.
    cv_folds=5,cv_rounds=400,cv_bootstraps=1000,acceptance='cv_paired',
    accept_lower_bound=0.0,screen_margin=0.02,prescreen_features=True,score_workers=1,
    rerank_allowed=False,rerank_min_rows=100,rerank_min_positive_devices=20,
    parallel_allowed=True,device_weight_allowed=True,prescreen_mode='balanced',
    feature_exploration_budget=2,contrast_max_units_per_group=1000)

V2_KEYS=('cv_folds','cv_rounds','cv_bootstraps','acceptance','accept_lower_bound','screen_margin','prescreen_features','score_workers',
         'rerank_allowed','rerank_min_rows','rerank_min_positive_devices',
         'parallel_allowed','device_weight_allowed','prescreen_mode','feature_exploration_budget','contrast_max_units_per_group')


def policy(domain):
    if domain=='memory':
        return dict(evaluation_protocol='memory_15m_7d_device_f1_v2',
                    minimum_lead_seconds=900,maximum_lead_seconds=900+7*86400)
    return dict(evaluation_protocol='period_device_f1_v2',
                minimum_lead_seconds=900,maximum_lead_seconds=None)


def validate_config(c):
    require(c.get('domain') and c.get('data_dir') and c.get('output_root'),'Dataset, domain and session required')
    for k,v in policy(c['domain']).items():require(c.get(k)==v,'Fixed task policy differs: '+k)
    require(c['split_ratios']==[.6,.2,.2],'Chronological 6:2:2 split is fixed for this release')
    for key in ('max_trials','max_repairs','max_agent_calls','max_reported_tokens','rounds','eval_every',
                'patience','trial_timeout_seconds','feature_timeout_seconds','max_device_rows'):
        require(type(c[key]) is int and c[key]>0,'Invalid '+key)
    require(c['memory_limit_gb']>0 and c['max_total_features']>0,'Invalid resource budget')
    require(type(c['agent_max_tokens']) is int and c['agent_max_tokens']>0,'Invalid agent_max_tokens')
    cap=c.get('agent_max_tokens_cap',65536)
    require(type(cap) is int and c['agent_max_tokens']<=cap<=393216,'Invalid agent_max_tokens_cap')
    retries=c.get('max_truncation_retries',1)
    require(type(retries) is int and 0<=retries<=2,'Invalid max_truncation_retries')
    require(c.get('adaptation_reasoning_effort','low') in ('low','high','max'),'Invalid adaptation_reasoning_effort')
    require(c.get('acceptance','selection_f1') in ('selection_f1','cv_paired'),'acceptance must be selection_f1 or cv_paired')
    require(type(c.get('cv_folds',5)) is int and 2<=c.get('cv_folds',5)<=10,'cv_folds must be 2..10')
    require(type(c.get('cv_rounds',400)) is int and c.get('cv_rounds',400)>0,'Invalid cv_rounds')
    require(-1<=c.get('accept_lower_bound',0.)<=1 and 0<=c.get('screen_margin',.02)<=1,'Invalid acceptance bounds')
    require(type(c.get('score_workers',1)) is int and 1<=c.get('score_workers',1)<=128,'score_workers must be 1..128')
    require(c.get('prescreen_mode','balanced') in ('balanced','strict_high'), 'Invalid prescreen_mode')
    require(type(c.get('feature_exploration_budget',2)) is int and 0<=c.get('feature_exploration_budget',2)<=10, 'Invalid exploration budget')
    require(type(c.get('contrast_max_units_per_group',1000)) is int and 10<=c.get('contrast_max_units_per_group',1000)<=10000, 'Invalid diagnostic sample cap')
    for key in ('parallel_allowed','device_weight_allowed','prescreen_features'):
        require(type(c.get(key,True)) is bool, 'Invalid '+key)
    root=Path(c['output_root']).resolve();data=Path(c['data_dir']).resolve()
    require(root!=data and not root.is_relative_to(data),'Session output must be outside the input dataset')
    return c
