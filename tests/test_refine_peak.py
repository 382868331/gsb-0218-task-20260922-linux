"""Tests for refine_peak (off-grid single-sinusoid frequency refinement).

Independent references in this file (a hand-written 3x3 weighted
normal-equations solver, a dense grid scan and finite differences) never
call periodogram.refine_peak to produce expected values.
"""

import math
import unittest

from periodogram import (
    EPS,
    PeriodogramError,
    RefinedPeak,
    periodogram,
    refine_peak,
)

TWO_PI = 2.0 * math.pi


def lcg_series(seed, n, lo, hi):
    """Deterministic LCG -> n sorted pseudo-irregular times in [lo, hi]."""
    state = seed
    out = []
    for _ in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(lo + (hi - lo) * (state / (1 << 64)))
    return sorted(out)


def reference_fit(times, values, weights, freq):
    """Independent weighted 3x3 normal-equations solve (Gaussian elimination).

    Returns (c, a, b, sse).  Never calls the module under test.
    """
    n = len(times)
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for i in range(n):
        row = [1.0, math.cos(TWO_PI * freq * times[i]),
               math.sin(TWO_PI * freq * times[i])]
        w = weights[i]
        for j in range(3):
            atb[j] += w * row[j] * values[i]
            for k in range(3):
                ata[j][k] += w * row[j] * row[k]
    # Gaussian elimination with partial pivoting
    m = [ata[j] + [atb[j]] for j in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        d = m[col][col]
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col] / d
            for k in range(col, 4):
                m[r][k] -= factor * m[col][k]
    x = [m[j][3] / m[j][j] for j in range(3)]
    sse = 0.0
    for i in range(n):
        resid = values[i] - x[0] - x[1] * math.cos(TWO_PI * freq * times[i]) \
            - x[2] * math.sin(TWO_PI * freq * times[i])
        sse += weights[i] * resid * resid
    return x[0], x[1], x[2], sse


def reference_sse(times, values, weights, freq):
    return reference_fit(times, values, weights, freq)[3]


class RefinePeakRecoveryTest(unittest.TestCase):
    """Known off-grid frequency on irregularly sampled data."""

    @classmethod
    def setUpClass(cls):
        cls.times = lcg_series(20260926, 60, 0.0, 75.0)
        cls.f0 = 1.7135  # deliberately off any reasonable grid
        cls.amp = 1.3
        cls.phase = 0.4
        cls.offset = 0.3
        a = cls.amp * math.cos(cls.phase)
        b = cls.amp * math.sin(cls.phase)
        # tiny deterministic noise so the problem is not exactly interpolatory
        noise = lcg_series(99, 60, -0.02, 0.02)
        cls.values = [
            cls.offset + a * math.cos(TWO_PI * cls.f0 * t)
            + b * math.sin(TWO_PI * cls.f0 * t) + noise[i]
            for i, t in enumerate(cls.times)
        ]
        cls.weights = [1.0 + 0.5 * ((i * 7) % 3) for i in range(60)]
        cls.result = refine_peak(cls.times, cls.values, cls.weights, 0.4, 3.0)

    def test_frequency_close_to_truth(self):
        # noisy data: the LS optimum is statistically offset from f0;
        # the bound reflects the noise level, not a hidden defect
        self.assertAlmostEqual(self.result.frequency, self.f0, delta=3e-4)

    def test_amplitude_phase_constant(self):
        self.assertAlmostEqual(self.result.amplitude, self.amp, delta=2e-2)
        # phase is origin-dependent; compare model coefficients instead
        c, a, b, _ = reference_fit(
            self.times, self.values, self.weights, self.result.frequency
        )
        self.assertAlmostEqual(self.result.constant, c, places=8)
        self.assertAlmostEqual(self.result.amplitude, math.hypot(a, b), places=8)
        self.assertAlmostEqual(self.result.phase, math.atan2(b, a), places=8)

    def test_result_type_and_fields(self):
        r = self.result
        self.assertIsInstance(r, RefinedPeak)
        self.assertEqual(r.stop_reason, "converged")
        self.assertEqual(r.rank, 3)
        self.assertFalse(r.boundary_lower)
        self.assertGreater(r.n_evaluations, 0)
        self.assertGreaterEqual(r.n_candidates, 1)
        self.assertLess(r.sse, r.sse_low)
        self.assertLess(r.sse, r.sse_high)

    def test_sse_matches_independent_reference(self):
        _, _, _, sse_ref = reference_fit(
            self.times, self.values, self.weights, self.result.frequency
        )
        self.assertAlmostEqual(self.result.sse, sse_ref, delta=1e-10 + 1e-10 * sse_ref)

    def test_not_worse_than_dense_scan(self):
        # independent dense scan (step 1e-4) + local enumeration around the
        # refined frequency: the refined SSE must beat every dense grid point
        f_lo, f_hi = self.result.frequency - 0.05, self.result.frequency + 0.05
        steps = 1000
        for k in range(steps + 1):
            f = f_lo + (f_hi - f_lo) * k / steps
            sse = reference_sse(self.times, self.values, self.weights, f)
            self.assertLessEqual(
                self.result.sse, sse + 1e-12,
                f"refined SSE worse than dense-scan point f={f}",
            )

    def test_stationary_by_finite_difference(self):
        # central finite difference of the objective at the refined frequency
        f = self.result.frequency
        width = 3.0 - 0.4
        h = 1e-5 * width
        d1 = (reference_sse(self.times, self.values, self.weights, f + h)
              - reference_sse(self.times, self.values, self.weights, f - h)) / (2 * h)
        d2 = (reference_sse(self.times, self.values, self.weights, f + h)
              - 2 * self.result.sse
              + reference_sse(self.times, self.values, self.weights, f - h)) / (h * h)
        self.assertGreater(d2, 0.0)  # a minimum, not a maximum
        # distance to stationarity: |d1| ~ d2 * |f_error|, |f_error| <= xtol
        self.assertLessEqual(abs(d1), d2 * 1e-8 * width + 1e-8)


class TwoPeakTest(unittest.TestCase):
    """Interval holding two local peaks (1/day aliases of one signal)."""

    @classmethod
    def setUpClass(cls):
        # nightly cadence with jitter -> aliases at f0 +- 1/day
        cls.times = []
        state = 555
        for night in range(40):
            for _ in range(3):
                state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
                cls.times.append(night + 0.05 * (state / (1 << 64)))
        cls.f0 = 1.7135
        cls.values = [0.2 + 1.1 * math.cos(TWO_PI * cls.f0 * t)
                      + 0.6 * math.sin(TWO_PI * cls.f0 * t) for t in cls.times]
        cls.result = refine_peak(cls.times, cls.values, None, 0.4, 3.0)

    def test_multiple_candidates_found(self):
        self.assertGreaterEqual(self.result.n_candidates, 2)

    def test_best_is_true_peak_not_alias(self):
        self.assertAlmostEqual(self.result.frequency, self.f0, places=6)
        self.assertGreater(self.result.power, 0.999)

    def test_alias_is_also_local_optimum(self):
        # the 0.7135/day alias must appear as a local SSE minimum too:
        # refine the sub-interval around it independently
        alias = refine_peak(self.times, self.values, None, 0.5, 0.9)
        self.assertAlmostEqual(alias.frequency, self.f0 - 1.0, delta=5e-4)
        self.assertLess(alias.power, self.result.power)


class RankDeficiencyTest(unittest.TestCase):
    """Rank-deficient positions must not be accepted as peaks."""

    def test_rank_deficient_grid_minimum_rejected(self):
        # sin(2*pi*1*t) vanishes at every half-integer time -> rank 2 at f=1
        times = [0.0, 0.5, 1.0, 1.5, 2.0]
        values = [math.cos(TWO_PI * t) for t in times]  # exact f=1 signal
        # sanity: the module's own scan confirms rank 2 at f=1
        scan = periodogram(times, values, None, [1.0])
        self.assertEqual(scan.ranks[0], 2)
        result = refine_peak(times, values, None, 0.5, 1.5, segments=4)
        self.assertEqual(result.stop_reason, "boundary_minimum")
        self.assertNotAlmostEqual(result.frequency, 1.0, places=9)
        self.assertEqual(result.rank, 3)

    def test_constant_data_has_no_interior_peak(self):
        times = lcg_series(5, 20, 0.0, 10.0)
        values = [2.5] * 20
        result = refine_peak(times, values, None, 0.2, 2.0)
        self.assertEqual(result.stop_reason, "boundary_minimum")
        self.assertEqual(result.sse, 0.0)
        self.assertEqual(result.power, 0.0)


class BudgetTest(unittest.TestCase):
    def test_budget_exhaustion_is_reported_and_respected(self):
        times = lcg_series(77, 50, 0.0, 60.0)
        f0 = 1.31
        values = [math.cos(TWO_PI * f0 * t) for t in times]
        result = refine_peak(times, values, None, 0.2, 3.5,
                             segments=64, eval_budget=80)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertLessEqual(result.n_evaluations, 80)

    def test_budget_exhaustion_distinct_from_convergence(self):
        times = lcg_series(77, 50, 0.0, 60.0)
        f0 = 1.31
        values = [math.cos(TWO_PI * f0 * t) for t in times]
        full = refine_peak(times, values, None, 1.0, 1.6, segments=16)
        self.assertEqual(full.stop_reason, "converged")
        self.assertAlmostEqual(full.frequency, f0, places=6)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.times = [0.0, 1.3, 2.7, 4.1]
        self.values = [0.0, 1.0, 0.0, 1.0]

    def test_rejects_inverted_interval(self):
        with self.assertRaises(PeriodogramError):
            refine_peak(self.times, self.values, None, 2.0, 1.0)

    def test_rejects_equal_endpoints(self):
        with self.assertRaises(PeriodogramError):
            refine_peak(self.times, self.values, None, 1.0, 1.0)

    def test_rejects_nonpositive_low(self):
        with self.assertRaises(PeriodogramError):
            refine_peak(self.times, self.values, None, 0.0, 1.0)

    def test_rejects_nan_endpoint(self):
        with self.assertRaises(PeriodogramError):
            refine_peak(self.times, self.values, None, float("nan"), 1.0)

    def test_rejects_bool_segments(self):
        with self.assertRaises(PeriodogramError):
            refine_peak(self.times, self.values, None, 0.5, 2.0, segments=True)

    def test_rejects_bool_budget(self):
        with self.assertRaises(PeriodogramError):
            refine_peak(self.times, self.values, None, 0.5, 2.0, eval_budget=False)

    def test_rejects_nan_in_values(self):
        with self.assertRaises(PeriodogramError) as ctx:
            refine_peak(self.times, [0.0, float("nan"), 1.0, 2.0], None, 0.5, 2.0)
        self.assertIn("values[1]", str(ctx.exception))

    def test_rejects_bad_weights(self):
        with self.assertRaises(PeriodogramError):
            refine_peak(self.times, self.values, [1.0, 0.0, 1.0, 1.0], 0.5, 2.0)


if __name__ == "__main__":
    unittest.main()
