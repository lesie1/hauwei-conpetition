"""Raw CE event parsing: error-bit maps and DRAM addresses become causal numeric columns.

Each raw event is decoded once (DQ-beat bitmap and rank/bank/row/column/chip
address), then aggregated per device into trailing-window and whole-past
statistics evaluated at the official observation timestamps. Labels, cohorts
and scheduled scoring are untouched. A feature at anchor time a uses only raw
events with time <= a, matching the official trailing-window convention.
All aggregations are vectorized; no per-anchor Python loops.
"""
import ast
import numpy as np
import pandas as pd

VERSION = 'raw_ce_spatial_v1'
DAY = 86400
WINDOWS = {'6h': 6 * 3600, '24h': DAY, '7d': 7 * DAY}
DQ_PER_CHIP = 4   # x4 DDR4: bit b is chip b//4, lane b%4 (checked against deviceID on all raw files)
BEATS = 8
BUS_BITS = 144    # observed range 0..143 (36 chips)
RAW_COLUMNS = ['LogTime', 'burst_info', 'RankId', 'BankgroupId', 'BankId', 'RowId', 'ColumnId',
               'deviceID', 'error_type_full_name']
PATTERN_FIELDS = ['err_bits', 'dq_count', 'beat_count', 'min_dq_interval', 'max_beat_interval',
                  'adjacent_dq_pairs', 'risky']
WINDOW_KEYS = ['events', 'max_err_bits', 'max_dq_count', 'max_beat_count', 'min_dq_interval',
               'max_beat_interval', 'adjacent_dq_events', 'multi_dq_events', 'multi_beat_events',
               'risky_events', 'distinct_cells', 'distinct_rows', 'distinct_cols', 'distinct_banks',
               'distinct_chips']
LIFE_KEYS = ['events', 'distinct_cells', 'distinct_rows', 'distinct_cols', 'distinct_banks',
             'distinct_chips', 'distinct_ranks', 'max_err_bits', 'max_dq_count', 'max_beat_count',
             'risky_events', 'adjacent_dq_events', 'top_cell_share', 'max_cols_per_row',
             'max_rows_per_col', 'days_since_first', 'new_rows_7d', 'new_cols_7d',
             'new_patterns_7d', 'scrub_share']
FEATURES = [f'raw_{w}_{k}' for w in WINDOWS for k in WINDOW_KEYS] + [f'raw_life_{k}' for k in LIFE_KEYS]

DESCRIPTIONS = {
    'events': 'raw CE events', 'max_err_bits': 'maximum error bits in one CE',
    'max_dq_count': 'maximum distinct DQ lanes in one CE', 'max_beat_count': 'maximum distinct beats in one CE',
    'min_dq_interval': 'minimum lane gap among multi-lane CEs (NaN if none)',
    'max_beat_interval': 'maximum beat gap in one CE',
    'adjacent_dq_events': 'CEs with adjacent erroneous DQ lanes in one beat',
    'multi_dq_events': 'CEs spanning more than one DQ lane', 'multi_beat_events': 'CEs spanning more than one beat',
    'risky_events': 'CEs not confined to one half of the 8x4 bitmap (coarse public correctability rule)',
    'distinct_cells': 'distinct (rank,bankgroup,bank,row,column) cells',
    'distinct_rows': 'distinct (rank,bankgroup,bank,row) rows', 'distinct_cols': 'distinct (rank,bankgroup,bank,column) columns',
    'distinct_banks': 'distinct (rank,bankgroup,bank) banks', 'distinct_chips': 'distinct (rank,chip) DRAM chips',
    'distinct_ranks': 'distinct ranks', 'top_cell_share': 'share of all past CEs at the single most frequent cell',
    'max_cols_per_row': 'largest number of distinct columns seen in one row',
    'max_rows_per_col': 'largest number of distinct rows seen in one column',
    'days_since_first': 'days since the first raw CE', 'new_rows_7d': 'rows first seen within the last 7 days',
    'new_cols_7d': 'columns first seen within the last 7 days',
    'new_patterns_7d': 'error-bit patterns first seen within the last 7 days',
    'scrub_share': 'share of past CEs found by patrol scrub'}


def decode_bits(text):
    """burst_info "{beat: [bit,...]}" -> 8x4 bool map, or None when empty/missing."""
    if not isinstance(text, str) or not text.strip():
        return None
    value = ast.literal_eval(text)
    if not isinstance(value, dict):
        raise ValueError('burst_info must be a dict of beat -> bit list')
    grid = np.zeros((BEATS, DQ_PER_CHIP), bool)
    for beat, bits in value.items():
        beat = int(beat)
        if not 0 <= beat < BEATS:
            raise ValueError(f'burst_info beat outside 0..7: {beat}')
        for bit in bits:
            bit = int(bit)
            # Up to 36 x4 chips (two 72-bit halves); in the data bit//4 always equals deviceID.
            if not 0 <= bit < BUS_BITS:
                raise ValueError(f'burst_info bit outside 0..{BUS_BITS - 1}: {bit}')
            grid[beat, bit % DQ_PER_CHIP] = True
    return grid


def pattern_stats(text):
    grid = decode_bits(text)
    if grid is None or not grid.any():
        return [np.nan] * len(PATTERN_FIELDS)
    dq = np.flatnonzero(grid.any(0))
    beat = np.flatnonzero(grid.any(1))
    dq_gap = np.diff(dq)
    beat_gap = np.diff(beat)
    # Public guidance: errors confined to one half (beats 0-3 / 4-7, or lanes 0-1 / 2-3) are
    # correctable by design. Spanning both halves on both axes is marked risky.
    risky = grid[:4].any() and grid[4:].any() and grid[:, :2].any() and grid[:, 2:].any()
    return [float(grid.sum()), float(len(dq)), float(len(beat)),
            float(dq_gap.min()) if len(dq_gap) else np.nan,
            float(beat_gap.max()) if len(beat_gap) else 0.,
            float((grid[:, 1:] & grid[:, :-1]).sum()), float(risky)]


def events(frame):
    """Raw device CSV frame -> time-sorted per-event numeric table."""
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError('Raw CE columns absent: ' + ','.join(sorted(missing)))
    f = frame[RAW_COLUMNS].copy()
    f['LogTime'] = pd.to_numeric(f['LogTime'], errors='raise').astype(np.int64)
    f = f.sort_values('LogTime', kind='stable').reset_index(drop=True)
    text = f['burst_info'].where(f['burst_info'].notna(), '').astype(str)
    codes, uniques = pd.factorize(text)
    table = np.array([pattern_stats(u) for u in uniques], np.float64).reshape(len(uniques), len(PATTERN_FIELDS))
    out = pd.DataFrame(table[codes], columns=PATTERN_FIELDS)
    out['pattern'] = codes.astype(np.int64)
    out['scrub'] = f['error_type_full_name'].astype(str).str.contains('SCRUB', regex=False).to_numpy(np.float64)
    out['time'] = f['LogTime'].to_numpy()
    for c in ('RankId', 'BankgroupId', 'BankId', 'RowId', 'ColumnId', 'deviceID'):
        out[c] = pd.to_numeric(f[c], errors='coerce').fillna(-1).astype(np.int64).to_numpy()
    return out


def _codes(*columns):
    return pd.MultiIndex.from_arrays(columns).factorize()[0].astype(np.int64)


def range_reduce(values, lo, hi, op):
    """op over values[lo:hi] for each query via a sparse table; NaN where empty or all-NaN."""
    fill = -np.inf if op is np.maximum else np.inf
    v = np.where(np.isfinite(values), values, fill).astype(np.float64)
    levels = [v]
    while (1 << len(levels)) <= len(v):
        half = 1 << (len(levels) - 1)
        prev = levels[-1]
        levels.append(op(prev[:-half], prev[half:]))
    out = np.full(len(lo), fill)
    length = hi - lo
    ok = length > 0
    if ok.any():
        k = np.floor(np.log2(length[ok])).astype(np.int64)
        a, b = lo[ok], hi[ok]
        res = np.empty(len(k))
        for level in np.unique(k):
            s = k == level
            res[s] = op(levels[level][a[s]], levels[level][b[s] - (1 << level)])
        out[ok] = res
    return np.where(np.isfinite(out), out, np.nan)


def window_distinct(codes, lo, hi):
    """Distinct codes in codes[lo_q:hi_q] for monotone non-decreasing lo, hi (vectorized).

    Event i counts for query q iff lo_q <= i < hi_q and its previous same-code
    event lies before lo_q. Monotone bounds make each event's valid queries one
    contiguous range, accumulated with a difference array.
    """
    n, q = len(codes), len(lo)
    if not n or not q:
        return np.zeros(q)
    if (np.diff(lo) < 0).any() or (np.diff(hi) < 0).any():
        raise ValueError('window bounds must be monotone')
    order = np.argsort(codes, kind='stable')
    prev = np.full(n, -1, np.int64)
    same = codes[order[1:]] == codes[order[:-1]]
    prev[order[1:][same]] = order[:-1][same]
    i = np.arange(n)
    start = np.maximum(np.searchsorted(hi, i, side='right'), np.searchsorted(lo, prev, side='right'))
    end = np.searchsorted(lo, i, side='right') - 1
    use = start <= end
    diff = np.bincount(start[use], minlength=q + 1)[:q + 1].astype(np.int64)
    diff -= np.bincount(end[use] + 1, minlength=q + 1)[:q + 1]
    return np.cumsum(diff)[:q].astype(np.float64)


def _first_mark(codes):
    mark = np.zeros(len(codes))
    mark[np.unique(codes, return_index=True)[1]] = 1.
    return mark


def features_at(ev, anchors):
    """(len(anchors), len(FEATURES)) float32; anchors sorted; uses events with time <= anchor."""
    anchors = np.asarray(anchors, np.int64)
    out = np.full((len(anchors), len(FEATURES)), np.nan, np.float32)
    if not len(anchors) or ev is None or not len(ev):
        return out
    if (np.diff(anchors) < 0).any():
        raise ValueError('anchors must be sorted')
    t = ev['time'].to_numpy(np.int64)
    if (np.diff(t) < 0).any():
        raise ValueError('raw events must be time-sorted')
    rank, bg, bank = ev.RankId.to_numpy(), ev.BankgroupId.to_numpy(), ev.BankId.to_numpy()
    row, col, chip = ev.RowId.to_numpy(), ev.ColumnId.to_numpy(), ev.deviceID.to_numpy()
    code = dict(cells=_codes(rank, bg, bank, row, col), rows=_codes(rank, bg, bank, row),
                cols=_codes(rank, bg, bank, col), banks=_codes(rank, bg, bank),
                chips=_codes(rank, chip), ranks=_codes(rank))
    num = {k: ev[k].to_numpy(np.float64) for k in PATTERN_FIELDS + ['scrub']}
    flag = dict(adjacent_dq_events=(num['adjacent_dq_pairs'] > 0).astype(float),
                multi_dq_events=(num['dq_count'] > 1).astype(float),
                multi_beat_events=(num['beat_count'] > 1).astype(float),
                risky_events=np.nan_to_num(num['risky']))
    idx = {name: i for i, name in enumerate(FEATURES)}
    hi = np.searchsorted(t, anchors, side='right')

    def csum(v, a, b):
        c = np.r_[0., np.cumsum(v)]
        return c[b] - c[a]

    for w, width in WINDOWS.items():
        lo = np.searchsorted(t, anchors - width, side='right')
        p = f'raw_{w}_'
        out[:, idx[p + 'events']] = hi - lo
        out[:, idx[p + 'max_err_bits']] = range_reduce(num['err_bits'], lo, hi, np.maximum)
        out[:, idx[p + 'max_dq_count']] = range_reduce(num['dq_count'], lo, hi, np.maximum)
        out[:, idx[p + 'max_beat_count']] = range_reduce(num['beat_count'], lo, hi, np.maximum)
        out[:, idx[p + 'min_dq_interval']] = range_reduce(num['min_dq_interval'], lo, hi, np.minimum)
        out[:, idx[p + 'max_beat_interval']] = range_reduce(num['max_beat_interval'], lo, hi, np.maximum)
        for key, v in flag.items():
            out[:, idx[p + key]] = csum(v, lo, hi)
        for key in ('cells', 'rows', 'cols', 'banks', 'chips'):
            out[:, idx[p + 'distinct_' + key]] = window_distinct(code[key], lo, hi)
    last = hi - 1
    has = last >= 0
    k = np.where(has, last, 0)

    def put(name, running):
        out[:, idx['raw_life_' + name]] = np.where(has, np.asarray(running, np.float64)[k], np.nan)

    n = np.arange(1, len(t) + 1, dtype=np.float64)
    put('events', n)
    for key in ('cells', 'rows', 'cols', 'banks', 'chips', 'ranks'):
        put('distinct_' + key, np.cumsum(_first_mark(code[key])))
    for key in ('err_bits', 'dq_count', 'beat_count'):
        put('max_' + key, np.fmax.accumulate(num[key]))
    for key in ('risky_events', 'adjacent_dq_events'):
        put(key, np.cumsum(flag[key]))
    count = pd.Series(code['cells']).groupby(code['cells']).cumcount().to_numpy() + 1
    put('top_cell_share', np.maximum.accumulate(count) / n)
    for name, parent, child in (('max_cols_per_row', code['rows'], col), ('max_rows_per_col', code['cols'], row)):
        fan = pd.Series(_first_mark(_codes(parent, child))).groupby(parent).cumsum().to_numpy()
        put(name, np.maximum.accumulate(fan))
    put('scrub_share', np.cumsum(num['scrub']) / n)
    out[:, idx['raw_life_days_since_first']] = np.where(has, (anchors - t[0]) / DAY, np.nan)
    lo7 = np.searchsorted(t, anchors - 7 * DAY, side='right')
    for name, c in (('new_rows_7d', code['rows']), ('new_cols_7d', code['cols']),
                    ('new_patterns_7d', ev['pattern'].to_numpy(np.int64))):
        out[:, idx['raw_life_' + name]] = np.where(has, csum(_first_mark(c), lo7, hi), np.nan)
    return out


def mapping_features():
    """Mapping entries describing every raw feature (for the agent and the cache manifest)."""
    entries = []
    for name in FEATURES:
        if name.startswith('raw_life_'):
            desc = 'whole past: ' + DESCRIPTIONS[name[len('raw_life_'):]]
        else:
            w, key = name[4:].split('_', 1)
            desc = f'trailing {w}: {DESCRIPTIONS[key]}'
        entries.append(dict(name=name, column=name, kind='numeric_unknown', unit=None,
                            description=desc + '; parsed from raw burst_info and DRAM address fields'))
    return entries
