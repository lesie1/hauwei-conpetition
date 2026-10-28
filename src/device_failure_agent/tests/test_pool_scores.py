"""Vectorized pool_scores must equal the original per-window loop exactly."""
import unittest
import numpy as np
from prediction import pool_scores


def reference(times, values, points, step, kind):
    p = np.full(len(points), np.nan); support = np.zeros(len(points), np.int64)
    end = np.searchsorted(times, points, side='right' if kind in ('latest', 'all_history_max') else 'left')
    if kind in ('latest', 'all_history_max'):
        use = end > 0
        v = values if kind == 'latest' else np.maximum.accumulate(values)
        p[use] = v[end[use] - 1]; support = end
        return p, support
    start = np.searchsorted(times, points - step, side='left'); support = end - start
    for i, (a, b) in enumerate(zip(start, end)):
        if a == b: continue
        v = values[a:b]
        if kind == 'window_max': p[i] = v.max()
        elif kind == 'window_mean': p[i] = v.mean(dtype=np.float64)
        elif kind == 'window_second' and len(v) >= 2: p[i] = np.partition(v, -2)[-2]
    return p, support


class Pool(unittest.TestCase):
    def test_equal_to_reference(self):
        rng = np.random.default_rng(7)
        for trial in range(300):
            n = int(rng.integers(0, 60))
            times = np.sort(rng.integers(0, 20000, n)).astype(np.int64)
            if n > 3 and trial % 3 == 0:
                times[1:3] = times[1]          # ties
            values = rng.random(n)
            if trial % 4 == 0 and n:
                values[rng.integers(0, n)] = values.max()   # duplicate maxima
            points = np.sort(rng.integers(-500, 21000, int(rng.integers(0, 40)))).astype(np.int64)
            step = int(rng.choice([1, 300, 3600, 7200]))
            for kind in ('window_max', 'window_mean', 'window_second', 'latest', 'all_history_max'):
                a, sa = pool_scores(times, values, points, step, kind)
                b, sb = reference(times, values, points, step, kind)
                np.testing.assert_array_equal(sa, sb, err_msg=kind)
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=0, equal_nan=True, err_msg=kind)


if __name__ == '__main__':
    unittest.main()
