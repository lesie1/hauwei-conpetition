"""Vectorized raw-event features must equal a naive per-anchor recomputation."""
import unittest
import numpy as np
import pandas as pd
from raw_events import (FEATURES, WINDOWS, DAY, decode_bits, pattern_stats, events, features_at,
                        range_reduce, window_distinct, mapping_features)


def fake_raw(n, seed):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 30 * DAY, n)) + 1_700_000_000
    t[5:9] = t[5]   # duplicate timestamps
    rows = []
    for i in range(n):
        chip = int(rng.integers(0, 3))
        beats = {str(b): [] for b in range(8)}
        for _ in range(int(rng.integers(1, 4))):
            beats[str(int(rng.integers(0, 8)))].append(chip * 4 + int(rng.integers(0, 4)))
        rows.append(dict(LogTime=int(t[i]), burst_info=str(beats) if i % 17 else np.nan,
                         RankId=int(rng.integers(0, 2)), BankgroupId=int(rng.integers(0, 2)),
                         BankId=int(rng.integers(0, 3)), RowId=int(rng.integers(0, 5)),
                         ColumnId=int(rng.integers(0, 4)), deviceID=float(chip),
                         error_type_full_name='CE.SCRUB' if i % 7 == 0 else 'CE.READ'))
    frame = pd.DataFrame(rows).sample(frac=1, random_state=seed)   # unsorted like the source
    return frame


def naive(ev, a):
    """Direct recomputation for one anchor from the events with time <= a."""
    past = ev[ev.time <= a]
    out = {}
    key = lambda d, cols: set(map(tuple, d[cols].to_numpy()))
    for w, width in WINDOWS.items():
        d = past[past.time > a - width]
        p = f'raw_{w}_'
        out[p + 'events'] = len(d)
        for name, col, fn in (('max_err_bits', 'err_bits', np.nanmax), ('max_dq_count', 'dq_count', np.nanmax),
                              ('max_beat_count', 'beat_count', np.nanmax), ('min_dq_interval', 'min_dq_interval', np.nanmin),
                              ('max_beat_interval', 'max_beat_interval', np.nanmax)):
            v = d[col].to_numpy(float)
            out[p + name] = fn(v) if np.isfinite(v).any() else np.nan
        out[p + 'adjacent_dq_events'] = int((d.adjacent_dq_pairs > 0).sum())
        out[p + 'multi_dq_events'] = int((d.dq_count > 1).sum())
        out[p + 'multi_beat_events'] = int((d.beat_count > 1).sum())
        out[p + 'risky_events'] = float(d.risky.fillna(0).sum())
        out[p + 'distinct_cells'] = len(key(d, ['RankId', 'BankgroupId', 'BankId', 'RowId', 'ColumnId']))
        out[p + 'distinct_rows'] = len(key(d, ['RankId', 'BankgroupId', 'BankId', 'RowId']))
        out[p + 'distinct_cols'] = len(key(d, ['RankId', 'BankgroupId', 'BankId', 'ColumnId']))
        out[p + 'distinct_banks'] = len(key(d, ['RankId', 'BankgroupId', 'BankId']))
        out[p + 'distinct_chips'] = len(key(d, ['RankId', 'deviceID']))
    if not len(past):
        return {k: (np.nan if k.startswith('raw_life_') else out.get(k)) for k in FEATURES}
    L = 'raw_life_'
    out[L + 'events'] = len(past)
    out[L + 'distinct_cells'] = len(key(past, ['RankId', 'BankgroupId', 'BankId', 'RowId', 'ColumnId']))
    out[L + 'distinct_rows'] = len(key(past, ['RankId', 'BankgroupId', 'BankId', 'RowId']))
    out[L + 'distinct_cols'] = len(key(past, ['RankId', 'BankgroupId', 'BankId', 'ColumnId']))
    out[L + 'distinct_banks'] = len(key(past, ['RankId', 'BankgroupId', 'BankId']))
    out[L + 'distinct_chips'] = len(key(past, ['RankId', 'deviceID']))
    out[L + 'distinct_ranks'] = past.RankId.nunique()
    for col in ('err_bits', 'dq_count', 'beat_count'):
        v = past[col].to_numpy(float)
        out[L + 'max_' + col] = np.nanmax(v) if np.isfinite(v).any() else np.nan
    out[L + 'risky_events'] = float(past.risky.fillna(0).sum())
    out[L + 'adjacent_dq_events'] = int((past.adjacent_dq_pairs > 0).sum())
    cells = past.groupby(['RankId', 'BankgroupId', 'BankId', 'RowId', 'ColumnId']).size()
    out[L + 'top_cell_share'] = cells.max() / len(past)
    out[L + 'max_cols_per_row'] = past.groupby(['RankId', 'BankgroupId', 'BankId', 'RowId']).ColumnId.nunique().max()
    out[L + 'max_rows_per_col'] = past.groupby(['RankId', 'BankgroupId', 'BankId', 'ColumnId']).RowId.nunique().max()
    out[L + 'days_since_first'] = (a - past.time.min()) / DAY
    for name, cols in (('new_rows_7d', ['RankId', 'BankgroupId', 'BankId', 'RowId']),
                       ('new_cols_7d', ['RankId', 'BankgroupId', 'BankId', 'ColumnId']), ('new_patterns_7d', ['pattern'])):
        firsts = past.drop_duplicates(cols)   # first occurrence in time order
        out[L + name] = int((firsts.time > a - 7 * DAY).sum())
    out[L + 'scrub_share'] = past.scrub.mean()
    return out


class RawEvents(unittest.TestCase):
    def test_decode_and_pattern(self):
        self.assertIsNone(decode_bits(np.nan))
        g = decode_bits("{'0': [], '6': [39], '7': []}")
        self.assertEqual(int(g.sum()), 1)
        self.assertTrue(g[6, 39 % 4])
        s = pattern_stats("{'0': [36, 38], '5': [37]}")
        # chip 9 lanes: beat 0 -> lanes 0,2; beat 5 -> lane 1. Three bits, three lanes, two beats.
        self.assertEqual(s[:3], [3., 3., 2.])
        self.assertEqual(s[3], 1.)       # min lane gap among {0,1,2}
        self.assertEqual(s[4], 5.)       # beat gap 0->5
        self.assertEqual(s[5], 0.)       # no two adjacent lanes inside one beat
        self.assertEqual(s[6], 1.)       # spans beat halves and lane halves
        with self.assertRaises(ValueError):
            decode_bits("{'9': [1]}")
        # Bits 72..143 exist (36-chip DIMMs); lane is still bit % 4.
        g = decode_bits("{'2': [76], '3': [78]}")
        self.assertTrue(g[2, 0] and g[3, 2] and int(g.sum()) == 2)
        with self.assertRaises(ValueError):
            decode_bits("{'0': [144]}")

    def test_range_and_distinct_primitives(self):
        rng = np.random.default_rng(1)
        v = rng.normal(size=200); v[rng.random(200) < .3] = np.nan
        codes = rng.integers(0, 9, 200)
        hi = np.sort(rng.integers(0, 201, 80)); lo = np.minimum(hi, np.sort(rng.integers(0, 201, 80)))
        lo = np.maximum.accumulate(lo)
        for (a, b), mx, mn, dc in zip(zip(lo, hi), range_reduce(v, lo, hi, np.maximum),
                                      range_reduce(v, lo, hi, np.minimum), window_distinct(codes, lo, hi)):
            seg = v[a:b]; fin = seg[np.isfinite(seg)]
            if len(fin):
                self.assertAlmostEqual(mx, fin.max()); self.assertAlmostEqual(mn, fin.min())
            else:
                self.assertTrue(np.isnan(mx) and np.isnan(mn))
            self.assertEqual(dc, len(set(codes[a:b])))

    def test_features_match_naive_and_are_causal(self):
        for seed in (0, 3):
            ev = events(fake_raw(300, seed))
            self.assertTrue((np.diff(ev.time) >= 0).all())
            t = ev.time.to_numpy()
            anchors = np.unique(np.r_[t[::7], t[0] - 10, t[-1] + 3 * DAY, t[5]])
            got = features_at(ev, anchors)
            self.assertEqual(got.shape, (len(anchors), len(FEATURES)))
            for i, a in enumerate(anchors):
                want = naive(ev, a)
                for j, name in enumerate(FEATURES):
                    w, g = want[name], got[i, j]
                    if w is None or (isinstance(w, float) and np.isnan(w)):
                        self.assertTrue(np.isnan(g), (seed, a, name, g))
                    else:
                        self.assertAlmostEqual(float(g), float(w), places=4, msg=(seed, a, name))
            # Appending future events must not change earlier anchors.
            cut = anchors[len(anchors) // 2]
            early = features_at(ev[ev.time <= cut].reset_index(drop=True), anchors[anchors <= cut])
            np.testing.assert_allclose(early, got[anchors <= cut], equal_nan=True, rtol=1e-6)

    def test_empty_inputs_and_mapping(self):
        ev = events(fake_raw(20, 5))
        self.assertTrue(np.isnan(features_at(ev.iloc[:0], [1, 2])).all())
        self.assertEqual(features_at(ev, []).shape, (0, len(FEATURES)))
        names = [m['name'] for m in mapping_features()]
        self.assertEqual(names, FEATURES)
        self.assertEqual(len(set(names)), len(names))


if __name__ == '__main__':
    unittest.main()
