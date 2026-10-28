"""Single source for the generated-feature API, validator and agent payloads."""
import numpy as np
import pandas as pd

VERSION='causal_features_v2'
METHODS={'rolling','expanding','ewm','shift','diff','cumsum','cummax','cummin','cumprod',
         'fillna','ffill','clip','abs','where','mask','isna','notna','astype','replace',
         'mean','sum','std','var','min','max','count','median','quantile'}
NP={'log1p','log','exp','sqrt','abs','maximum','minimum','where','isfinite','isnan','clip','sign'}
REDUCE={'mean','sum','std','var','min','max','count','median','quantile'}
HELPERS={'gap_seconds','observation_count'}

class Past:
    @staticmethod
    def gap_seconds(frame):
        # Only current and preceding availability times; first gap is unknown.
        return frame.index.to_series().diff().dt.total_seconds()

    @staticmethod
    def observation_count(frame):
        return pd.Series(np.arange(1,len(frame)+1,dtype=np.float64),index=frame.index)

def contract():
    return dict(version=VERSION,signature='def transform(frame)',
        input='One device, chronological numeric rows. No labels, device IDs or future rows. Select literal column names from profile.feature_names.',
        output='pd.DataFrame of declared unique fe_ columns; preserve rows and index. Original columns are retained automatically.',
        series_methods=sorted(METHODS),numpy_functions=sorted(NP),pandas_functions=['DataFrame'],
        past_helpers={
            'past.gap_seconds(frame)':'Series: current minus previous observation time in seconds; first value NaN.',
            'past.observation_count(frame)':'Series: causal count 1,2,...,n; no measured-value missingness dependence.'},
        rules=[
            'Only assignments to new local names and a final return. No loops, comprehensions, imports or arbitrary functions.',
            'frame.index is permitted ONLY as index=frame.index in pd.DataFrame. For time gaps use past.gap_seconds(frame).',
            'No frame.columns, to_series, dt, iloc, loc, values, pd.Series, pd.to_numeric or apply. Inputs are already numeric.',
            'Use arithmetic operators / + - * instead of .div/.add/.sub/.mul. Use literal column names; never discover columns dynamically.',
            'Reductions must directly follow rolling/expanding/ewm. Use trailing real-time rolling windows; center=True and negative shifts are forbidden.',
            'Missing measurements stay unknown unless the hypothesis explicitly justifies a constant or forward fill. No infinities.',
            'Zero finite values in a bounded sample is a coverage limitation, not evidence that the entire dataset is empty.'
        ],examples=[
            dict(features=['fe_change'],source="def transform(frame):\n    change = frame['metric'].diff(1)\n    return pd.DataFrame({'fe_change': change}, index=frame.index)"),
            dict(features=['fe_gap','fe_count'],source="def transform(frame):\n    gap = past.gap_seconds(frame)\n    count = past.observation_count(frame)\n    return pd.DataFrame({'fe_gap': gap, 'fe_count': count}, index=frame.index)"),
            dict(features=['fe_mean'],source="def transform(frame):\n    avg = frame['metric'].rolling('7D', min_periods=1).mean()\n    return pd.DataFrame({'fe_mean': avg}, index=frame.index)")])
