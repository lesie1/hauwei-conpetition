"""Independent experimental factors; the evaluator is never a recipe field."""
import ast
import copy
import math
from common import require, digest

HORIZONS = (7,)
BASE = dict(max_depth=4, min_child_weight=1., learning_rate=.05, reg_lambda=1.,
            reg_alpha=0., gamma=0., max_bin=63, colsample_bytree=.7, max_delta_step=0.)
BOUNDS = dict(max_depth=(2,10), min_child_weight=(.01,1000), learning_rate=(.005,.3),
              reg_lambda=(0,1000), reg_alpha=(0,1000), gamma=(0,100), max_bin=(16,256),
              colsample_bytree=(.3,1), max_delta_step=(0,10))
FACTORS = dict(feature_set=('base','history','recent'),
    sample_unit=('event','device_day'), base_weight=('sample','device'),
    label_horizon_days=HORIZONS, weight_policy=('ratio','sqrt_ratio','unit'),
    prediction_interval_hours=(1,24),
    score_aggregation=('latest','window_max','window_mean','window_second','all_history_max'),
    seed=(42,43,44),
    # Two-stage gated reranking (rerank.py); gate is a stage-1 score threshold on TRAIN OOF scores.
    rerank=('none','gated','parallel'), rerank_gate=(0.5,0.7,0.8,0.85,0.9))
# Recipes saved before the rerank factor existed are read as single-stage.
LEGACY_DEFAULTS=dict(rerank='none',rerank_gate=0.5)
MODES = tuple(FACTORS['sample_unit'])
KINDS = ('errors','score_distribution','paired_devices','loss_curve','feature_profiles',
         'coverage','temporal_patterns')


def with_defaults(r):
    """Fill factors added after a recipe was written (keeps old sessions/receipts readable)."""
    if isinstance(r,dict):
        r=dict(r)
        for k,v in LEGACY_DEFAULTS.items():r.setdefault(k,v)
    return r


def validate_recipe(r):
    require(isinstance(r,dict) and set(r)==set(FACTORS)|{'params','feature_program'},
            'Recipe fields: '+','.join(FACTORS)+',params')
    require(r['rerank']=='none' or r['sample_unit']=='event' and r['feature_set']=='base',
            'Composite models support event-row (base) inputs')
    require(r['rerank']=='gated' or float(r['rerank_gate'])==LEGACY_DEFAULTS['rerank_gate'],
            'rerank_gate only applies with rerank=gated; keep 0.5 for single-stage recipes')
    import re
    require(isinstance(r['feature_program'],str) and re.fullmatch(r'base|[0-9a-f]{64}',r['feature_program']), 'Unknown feature program ID')
    for k, allowed in FACTORS.items():
        ok=r[k] in allowed and (type(r[k]) is type(allowed[0]) or k=='rerank_gate' and type(r[k]) in (int,float))
        require(ok, f'{k}: choose {allowed}')
    require(r['sample_unit']!='event' or r['feature_set']=='base',
            'Event training supports base. Use device_day to compare base and history features at identical training times.')
    require(r['feature_set']=='base' or r['score_aggregation']=='latest',
            'History feature models predict directly at the decision time: score_aggregation=latest')
    require(isinstance(r['params'],dict) and set(r['params'])==set(BASE), 'Specify all allowed parameters')
    for k,v in r['params'].items():
        lo,hi=BOUNDS[k]
        require(type(v) in (int,float) and math.isfinite(v) and lo<=v<=hi, f'Invalid {k}')
        if k in ('max_depth','max_bin'): require(type(v) is int, f'{k} must be integer')
    return r


def canonical_recipe(recipe):
    r=copy.deepcopy(validate_recipe(with_defaults(recipe)))
    r['rerank_gate']=float(r['rerank_gate'])
    # JSON 2 and 2.0 describe the same continuous hyperparameter.
    for k,v in r['params'].items():
        if k not in ('max_depth','max_bin'):r['params'][k]=float(v)
    return r


def identity(recipe): return digest(canonical_recipe(recipe))


def initial_plans():
    # Old winning row configuration is retrained under this session's label/split rules.
    row=dict(feature_program='base',feature_set='base',sample_unit='event',base_weight='sample',
        label_horizon_days=7,weight_policy='ratio',prediction_interval_hours=1,
        score_aggregation='window_max',seed=42,rerank='none',rerank_gate=0.5,
        params=dict(BASE,min_child_weight=50.,learning_rate=.1))
    device=dict(feature_program='base',feature_set='history',sample_unit='device_day',base_weight='device',
        label_horizon_days=30,weight_policy='ratio',prediction_interval_hours=24,
        score_aggregation='latest',seed=42,rerank='none',rerank_gate=0.5,params=dict(BASE,colsample_bytree=.5,reg_lambda=2.))
    return [row]


def differences(a,b):
    a,b=with_defaults(a),with_defaults(b);out=[]
    if a['feature_program']!=b['feature_program']:out.append('feature_program')
    for k in FACTORS:
        if a[k]!=b[k]:out.append(k)
    out.extend('params.'+k for k in BASE if a['params'][k]!=b['params'][k])
    return out


def training_identity(recipe):
    r=canonical_recipe(recipe)
    for k in ('prediction_interval_hours','score_aggregation'):r.pop(k)
    return digest(r)


def control_options(parent, tried):
    """Machine generated legal one-factor controls, for the analyst to prioritize."""
    seen={identity(t['recipe']) for t in tried};out=[]
    for key,values in FACTORS.items():
        for value in values:
            if value==parent[key]:continue
            r=copy.deepcopy(parent);r[key]=value
            try:validate_recipe(r)
            except ValueError:continue
            if identity(r) not in seen:
                out.append(dict(change=key,value=value,reuses_model=training_identity(r)==training_identity(parent)))
    return out


def validate_source(source,expected=None,diagnostic=False):
    require(isinstance(source,str) and len(source)<24000,'Candidate source too large')
    tree=ast.parse(source,feature_version=(3,10))
    require(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef),'Need one def run(ctx)')
    fn=tree.body[0]
    require(fn.name=='run' and len(fn.args.args)==1 and fn.args.args[0].arg=='ctx'
        and not fn.decorator_list and not fn.returns and not fn.args.defaults
        and not fn.args.kwonlyargs and not fn.args.posonlyargs and not fn.args.kwarg
        and not fn.args.vararg and not fn.args.args[0].annotation,'Need def run(ctx)')
    require(len(fn.body)==1 and isinstance(fn.body[0],ast.Return),'Return one literal plan')
    call=fn.body[0].value;method='diagnose' if diagnostic else 'run'
    require(isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
        and isinstance(call.func.value,ast.Name) and call.func.value.id=='ctx'
        and call.func.attr==method and len(call.args)==1 and not call.keywords,'Use ctx.'+method+'(literal_plan)')
    r=ast.literal_eval(call.args[0])
    if not diagnostic:r=with_defaults(r)
    if diagnostic:
        require(isinstance(r,dict) and set(r)=={'kind'} and r['kind'] in KINDS,'Unsupported diagnostic kind')
    else:validate_recipe(r)
    if expected is not None:require(r==with_defaults(expected),'Generated plan differs from the accepted experiment')
    return r


def source_for(recipe,diagnostic=False):
    return 'def run(ctx):\n    return ctx.'+('diagnose' if diagnostic else 'run')+'('+repr(recipe)+')\n'
