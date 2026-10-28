"""Frozen calendar-period device F1, with task-specific lead bounds."""
import numpy as np
import pandas as pd
from common import DAY, require

PROTOCOL = "period_device_f1_v2"


def period_cohort(registry, window, config=None):
    """Keep a registry independent of feature availability and alarm decisions."""
    lo, hi = map(int, window)
    c = registry.copy()
    config=config or {}
    c['protocol']=config.get('evaluation_protocol',PROTOCOL)
    c['maximum_lead_seconds']=config.get('maximum_lead_seconds')
    require(not c.device.duplicated().any(), "Duplicate device in cohort")
    failure = c.failure.to_numpy(np.int64)
    start = c.observed_start.to_numpy(np.int64)
    end = c.observed_end.to_numpy(np.int64)
    event = (failure >= lo) & (failure < hi) & (failure >= start) & (failure <= end)
    status = np.full(len(c), "eligible", dtype=object)
    status[(end < hi-1) & ~event] = "unknown_followup"
    status[(start > lo) | (end < lo)] = "unknown_coverage_at_start"
    status[(start > lo) & (end >= start)] = "excluded_late_entry"
    status[(failure >= 0) & (failure < lo)] = "excluded_preexisting_fault"
    c["status"] = status
    c["positive"] = event
    c["period_start"] = lo
    c["period_end_exclusive"] = hi
    c["outcome_known"] = c.status.eq("eligible")
    return c


def annotate(meta, cohort, lead):
    c = cohort.set_index("device")
    require(meta.device.isin(c.index).all(), "Daily record absent from cohort")
    rows = c.loc[meta.device].reset_index(drop=True)
    m = meta.reset_index(drop=True).copy()
    failure = rows.failure.to_numpy()
    t = m.time.to_numpy()
    physical = ((t >= rows.period_start.to_numpy()) & (t < rows.period_end_exclusive.to_numpy())
                & (t >= rows.observed_start.to_numpy()) & (t <= rows.observed_end.to_numpy())
                & ((failure < 0) | (t < failure)))
    m["prediction_eligible"] = physical
    m["primary_eligible"] = physical & rows.outcome_known.to_numpy()
    # Period outcome is used only for descriptive validation BCE, never training labels.
    m["label"] = np.where(rows.outcome_known, rows.positive.astype(np.int8), -1)
    m["timely_opportunity"] = (m.primary_eligible.to_numpy()
                                & rows.positive.to_numpy() & (t <= failure-lead))
    maximum=rows.maximum_lead_seconds.to_numpy(dtype=float)
    m['timely_opportunity'] &= np.isnan(maximum) | (failure-t <= maximum)
    return m


def device_table(meta, scores, cohort):
    s = np.asarray(scores, np.float64)
    require(len(s) == len(meta), "Score length mismatch")
    active = meta.prediction_eligible.to_numpy()
    require(np.isfinite(s[active & (meta.history_rows.to_numpy()>0)]).all(),
            "Missing score for nonempty eligible history")
    require((np.isnan(s) | ((s >= 0) & (s <= 1))).all(), "Invalid risk score")
    m = meta.loc[active].copy()
    m["score"] = s[active]
    m["hit_score"] = np.where(m.timely_opportunity, m.score, np.nan)
    g = m.groupby("device", sort=True)
    stats = g.agg(max_score=("score", "max"), max_hit_score=("hit_score", "max"),
                  daily_predictions=("score", "count"),
                  timely_prediction_opportunities=("timely_opportunity", "sum"))
    # Left join keeps faulty devices with zero scheduled snapshots in the denominator.
    out = cohort.merge(stats, left_on="device", right_index=True, how="left", validate="one_to_one")
    for column in ("daily_predictions", "timely_prediction_opportunities"):
        out[column] = out[column].fillna(0).astype(np.int64)
    return out, m


def classify(table, threshold):
    t = table.copy()
    eligible = t.status.eq("eligible")
    t["alarm"] = t.max_score >= threshold
    t["hit"] = eligible & t.positive & (t.max_hit_score >= threshold)
    t["fp"] = eligible & t.alarm & ~t.hit
    t["fn"] = eligible & t.positive & ~t.hit
    return t


def metrics(table, threshold):
    all_rows = classify(table, threshold)
    t = all_rows.loc[all_rows.status.eq("eligible")]
    unknown = all_rows.loc[all_rows.status.str.startswith("unknown_")]
    tp, fp, fn = (int(t[k].sum()) for k in ("hit", "fp", "fn"))
    protocol=str(table.protocol.iloc[0]) if 'protocol' in table and len(table) else PROTOCOL
    return dict(protocol=protocol, threshold=float(threshold), TP=tp, FP=fp, FN=fn,
        precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        eligible_devices=len(t), eligible_fault_devices=int(t.positive.sum()),
        alarmed_devices=int(t.alarm.sum()),
        no_history_devices=int(t.max_score.isna().sum()),
        no_history_fault_devices=int((t.positive & t.max_score.isna()).sum()),
        fault_without_timely_scheduled_opportunity=int((t.positive & (t.timely_prediction_opportunities==0)).sum()),
        positive_without_scored_positive_day=int((t.positive & t.max_hit_score.isna()).sum()),
        unknown_devices=len(unknown), unknown_alarm_devices=int(unknown.alarm.sum()),
        excluded_late_entry_devices=int(all_rows.status.eq("excluded_late_entry").sum()),
        excluded_preexisting_fault_devices=int(all_rows.status.eq("excluded_preexisting_fault").sum()),
        metric_scope="Local device F1 with frozen task lead interval. Alarmed fault with no in-window hit counts FP and FN. Unknown follow-up excluded and reported separately.")


def select(table):
    t = table.loc[table.status.eq("eligible")]
    alarm = np.sort(t.max_score.dropna().to_numpy())
    hit = np.sort(t.max_hit_score.dropna().to_numpy())
    npos = int(t.positive.sum())
    require(npos>0, "No known fault devices in the evaluation period")
    # An abstain-all threshold avoids unnecessary alarms when no TP is possible.
    thresholds = np.unique(np.r_[alarm, hit, np.nextafter(1., np.inf)])[::-1]
    tp = len(hit)-np.searchsorted(hit, thresholds, side="left")
    n = len(alarm)-np.searchsorted(alarm, thresholds, side="left")
    fp = n-tp
    f1 = 2*tp/(n+npos)
    best = int(np.lexsort((-thresholds, fp, -f1))[0])
    grid = pd.DataFrame(dict(threshold=thresholds, TP=tp, FP=fp, FN=npos-tp,
                            f1=f1, precision=np.divide(tp,n,out=np.zeros_like(f1),where=n>0),
                            recall=tp/npos))
    result = metrics(table, thresholds[best])
    require(abs(result["f1"]-f1[best])<1e-12, "Threshold implementation disagreement")
    return result, grid


def alarm_details(table, daily, threshold):
    t = classify(table, threshold)
    alarms = daily.loc[daily.score >= threshold]
    first = alarms.groupby("device").time.min()
    timely = alarms.loc[alarms.timely_opportunity].groupby("device").time.min()
    t["first_alarm_time"] = t.device.map(first)
    t["first_timely_alarm_time"] = t.device.map(timely)
    t["lead_days"] = np.where(t.hit, (t.failure-t.first_timely_alarm_time)/DAY, np.nan)
    return t


def lead_summary(table):
    a = table.loc[table.hit, "lead_days"].dropna()
    return dict(count=len(a), median=float(a.median()) if len(a) else None,
        p25=float(a.quantile(.25)) if len(a) else None,
        p75=float(a.quantile(.75)) if len(a) else None,
        minimum=float(a.min()) if len(a) else None, maximum=float(a.max()) if len(a) else None,
        buckets={"15_minutes_to_under_1_day":int((a<1).sum()),
                 "1_to_under_7_days":int(((a>=1)&(a<7)).sum()),
                 "7_through_15_days":int(((a>=7)&(a<=15)).sum()),
                 "over_15_days":int((a>15).sum())})


def budget_curve(table):
    finite = table.loc[table.status.eq("eligible"), "max_score"].dropna().sort_values(ascending=False).to_numpy()
    out = []
    for budget in (50,100,200,500,1000):
        if len(finite):
            row = metrics(table, finite[min(budget,len(finite))-1])
            row["requested_alarm_device_budget"] = budget
            out.append(row)
    return out
