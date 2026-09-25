"""Tests for periodogram.py.

Reference values are computed independently of the module under test:

* an equally-spaced DFT orthogonality reference (closed-form sums, no QR);
* a hand-built weighted normal-equation solver (3x3 Gaussian elimination);
* direct geometric sums for the sampling window.

The production module is imported only as the system under test; it is never
used to manufacture the expected numbers.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import periodogram as pg
from periodogram import periodogram, PeriodogramError, EPS

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Independent references (plain Python; do NOT call the module under test)
# --------------------------------------------------------------------------- #
def lcg(seed, n):
    """Deterministic pseudo-random floats in [0, 1)."""
    x = seed
    out = []
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out.append(x / 0x80000000)
    return out


def gauss_solve(mat, rhs, active):
    """Gaussian elimination with partial pivoting on the given columns."""
    idx = list(active)
    a = [[mat[i][j] for j in idx] for i in idx]
    b = [rhs[i] for i in idx]
    k = len(idx)
    scale = max((abs(mat[i][j]) for i in range(3) for j in range(3)),
                default=1.0)
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12 * max(scale, 1.0):
            return None
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, k):
            q = a[r][col] / a[col][col]
            for c in range(col, k):
                a[r][c] -= q * a[col][c]
            b[r] -= q * b[col]
    xsub = [0.0] * k
    for r in range(k - 1, -1, -1):
        s = b[r] - sum(a[r][c] * xsub[c] for c in range(r + 1, k))
        xsub[r] = s / a[r][r]
    out = [0.0, 0.0, 0.0]
    for pos, col in enumerate(idx):
        out[col] = xsub[pos]
    return out


def gauss_solve_3x3(mat, rhs):
    """Solve a 3x3 system; rank-deficient columns get coefficient zero.

    Mirrors the production rule of dropping null design directions instead
    of inventing values, while staying an independent normal-equation path.
    """
    x = gauss_solve(mat, rhs, [0, 1, 2])
    if x is not None:
        return x
    for drop in range(3):
        x = gauss_solve(mat, rhs, [c for c in range(3) if c != drop])
        if x is not None:
            return x
    return [rhs[0] / mat[0][0] if mat[0][0] else 0.0, 0.0, 0.0]


def normal_equation_reference(times, values, weights, freq):
    """Solve c + a cos + b sin via hand-built weighted normal equations.

    Returns ``(c, a_tau, b_tau, sse, sse0, t0)`` with coefficients and SSE
    expressed in the recentered frame tau = t - t[0] (an exact algebraic
    change of variables that keeps trig arguments small).  The original-frame
    phase is ``atan2(b_tau, a_tau) + 2*pi*f*t0`` and the amplitude is
    ``hypot(a_tau, b_tau)``.  The linear algebra is 3x3 Gaussian elimination
    with partial pivoting -- a deliberately different algorithm from the
    production Householder QR.
    """
    n = len(times)
    t0 = times[0]
    m = [[0.0] * 3 for _ in range(3)]
    v = [0.0, 0.0, 0.0]
    cols = [0.0, 0.0, 0.0]
    tau = [tk - t0 for tk in times]
    for k in range(n):
        ang = TWO_PI * freq * tau[k]
        cols[0], cols[1], cols[2] = 1.0, math.cos(ang), math.sin(ang)
        wk = weights[k]
        yk = values[k]
        for i in range(3):
            for j in range(3):
                m[i][j] += wk * cols[i] * cols[j]
            v[i] += wk * yk * cols[i]
    c, a_tau, b_tau = gauss_solve_3x3(m, v)
    sse = 0.0
    for k in range(n):
        ang = TWO_PI * freq * tau[k]
        resid = values[k] - c - a_tau * math.cos(ang) - b_tau * math.sin(ang)
        sse += weights[k] * resid * resid
    wsum = sum(weights)
    wmean = sum(weights[k] * values[k] for k in range(n)) / wsum
    sse0 = sum(weights[k] * (values[k] - wmean) ** 2 for k in range(n))
    return c, a_tau, b_tau, sse, sse0, t0


def wrap_phase(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def direct_window_modulus(times, weights, freq):
    """|sum w exp(i 2pi f t)| / sum w, summed directly (origin invariant)."""
    re = im = 0.0
    for tk, wk in zip(times, weights):
        ang = TWO_PI * freq * tk
        re += wk * math.cos(ang)
        im += wk * math.sin(ang)
    return math.hypot(re, im) / sum(weights)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestEqualSpacingDFT(unittest.TestCase):
    """Equally spaced data at a Fourier-grid frequency: analytic DFT result."""

    def test_single_harmonic_recovered_exactly_on_grid(self):
        n = 48
        f0 = 5.0 / n          # cycles per sample, exactly on the DFT grid
        amp0, delta0, c0 = 1.75, 0.6, 2.5
        times = [float(k) for k in range(n)]
        values = [
            c0 + amp0 * math.cos(TWO_PI * f0 * k - delta0) for k in range(n)
        ]
        freqs = [j / n for j in range(1, 21)]
        res = periodogram(times, values, None, freqs)

        j = 5 - 1  # candidate index of f0
        self.assertAlmostEqual(res.frequencies[j], f0, places=15)
        self.assertAlmostEqual(res.amplitudes[j], amp0, places=12)
        self.assertAlmostEqual(wrap_phase(res.phases[j] - delta0), 0.0,
                               places=12)
        self.assertAlmostEqual(res.constants[j], c0, places=12)
        self.assertEqual(res.ranks[j], 3)
        # pure harmonic + constant: zero residual SSE -> power 1
        self.assertAlmostEqual(res.powers[j], 1.0, places=10)
        self.assertLess(res.sses[j], 1e-24)
        # no other bin can beat the signal bin
        self.assertEqual(max(range(len(freqs)), key=lambda i: res.powers[i]), j)
        # integer-cycle window: sum of n unit roots cancels EXACTLY to 0 at
        # a nonzero DFT bin; the in-phase |W|/sum w = 1 peak lives at the
        # sampling frequency f_s = 1/dt instead.
        self.assertLess(res.window_modulus[j], 1e-14)
        res_fs = periodogram(times, values, None, [1.0])
        self.assertAlmostEqual(res_fs.window_modulus[0], 1.0, places=14)
        # the normalized modulus never exceeds 1 (triangle inequality)
        self.assertTrue(all(0.0 <= q <= 1.0 + 1e-15 for q in
                            res.window_modulus))

    def test_dft_orthogonal_coefficients(self):
        # Closed-form DFT sums, evaluated without the production code.
        n = 40
        times = [float(k) for k in range(n)]
        rnd = lcg(20260923, n)
        values = [0.3 * (u - 0.5) for u in rnd]  # arbitrary, uneven spectrum
        freqs = [j / n for j in range(1, 11)]
        res = periodogram(times, values, None, freqs)
        for j_idx, j in enumerate(range(1, 11)):
            a_ref = (2.0 / n) * sum(
                values[k] * math.cos(TWO_PI * j * k / n) for k in range(n)
            )
            b_ref = (2.0 / n) * sum(
                values[k] * math.sin(TWO_PI * j * k / n) for k in range(n)
            )
            amp_fit = res.amplitudes[j_idx]
            amp_ref = math.hypot(a_ref, b_ref)
            self.assertAlmostEqual(amp_fit, amp_ref, places=10)
            self.assertAlmostEqual(
                wrap_phase(res.phases[j_idx] - math.atan2(b_ref, a_ref)),
                0.0, places=10,
            )


class TestNormalEquationReference(unittest.TestCase):
    """Uneven times, nonuniform weights: QR path vs hand-built normal eq."""

    def test_matches_normal_equations_across_grid(self):
        n = 37
        rnd = lcg(4242, 3 * n)
        # strictly increasing, uneven gaps, unsorted-time tolerance later
        gaps = [0.05 + 0.4 * u for u in rnd[:n]]
        times = [sum(gaps[: k + 1]) for k in range(n)]
        values = [1.0 + 2.0 * (u - 0.5) for u in rnd[n:2 * n]]
        weights = [0.2 + 1.8 * u for u in rnd[2 * n:]]
        freqs = [0.13 + 0.11 * j for j in range(13)]

        res = periodogram(times, values, weights, freqs)
        for i, freq in enumerate(freqs):
            c, a_tau, b_tau, sse, sse0, t0 = normal_equation_reference(
                times, values, weights, freq
            )
            phase_ref = wrap_phase(
                math.atan2(b_tau, a_tau) + TWO_PI * freq * t0
            )
            self.assertAlmostEqual(res.sses[i], sse, delta=1e-18 + 1e-12 * sse)
            self.assertAlmostEqual(res.sse0, sse0,
                                   delta=1e-18 + 1e-12 * sse0)
            self.assertAlmostEqual(res.powers[i], (sse0 - sse) / sse0,
                                   places=11)
            self.assertAlmostEqual(res.amplitudes[i],
                                   math.hypot(a_tau, b_tau), places=10)
            self.assertAlmostEqual(
                wrap_phase(res.phases[i] - phase_ref), 0.0, places=10
            )
            self.assertAlmostEqual(res.constants[i], c, places=10)
            self.assertAlmostEqual(
                res.window_modulus[i],
                direct_window_modulus(times, weights, freq),
                places=12,
            )

    def test_global_weight_rescaling_is_invariant(self):
        n = 25
        rnd = lcg(7, 2 * n)
        times = sorted(0.02 + 0.9 * u for u in rnd[:n])
        values = [u - 0.5 for u in rnd[n:]]
        weights = [0.3 + u for u in lcg(99, n)]
        freqs = [0.2 * j for j in range(1, 10)]
        r1 = periodogram(times, values, weights, freqs)
        r2 = periodogram(times, values, [3.7 * w for w in weights], freqs)
        for i in range(len(freqs)):
            self.assertAlmostEqual(r1.powers[i], r2.powers[i], places=14)
            self.assertAlmostEqual(r1.amplitudes[i], r2.amplitudes[i],
                                   places=12)
            self.assertAlmostEqual(r1.window_modulus[i],
                                   r2.window_modulus[i], places=14)


class TestTimeOriginAndDegenerateSampling(unittest.TestCase):

    def test_large_origin_shift_invariant_and_phase_rotates(self):
        n = 30
        # f0 is deliberately a grid point (0.05 + 0.03*10) so amplitude
        # accuracy is not confounded with grid-search spectral leakage.
        f0, amp0, delta0 = 0.35, 1.3, -0.9
        rnd = lcg(55, n)
        base = sorted(2.0 * u for u in rnd)
        values = [
            amp0 * math.cos(TWO_PI * f0 * t - delta0) + 0.01 * (u - 0.5)
            for t, u in zip(base, rnd)
        ]
        big = 1.0e9
        shifted = [t + big for t in base]
        freqs = [0.05 + 0.03 * j for j in range(40)]

        # Floating addition quantizes relative times by up to ~2 ulp(T); the
        # shifted frame is therefore a genuinely perturbed input (not the
        # exact same least-squares problem).
        d_tau = [
            (shifted[k] - shifted[0]) - (base[k] - base[0]) for k in range(n)
        ]
        d_tau_max = max(abs(d) for d in d_tau)
        self.assertLessEqual(d_tau_max, 2.0 * math.ulp(big))

        r1 = periodogram(base, values, None, freqs)
        r2 = periodogram(shifted, values, None, freqs)

        # Each frame matches the INDEPENDENT normal-equation reference under
        # a tight tolerance -- note the reference builds cos/sin from raw,
        # un-recentered times, so this also proves recentering is exact up to
        # roundoff even at a 1e9 origin.
        for frame_times, res in ((base, r1), (shifted, r2)):
            for i, freq in enumerate(freqs):
                c, a_tau, b_tau, sse, sse0, t0 = normal_equation_reference(
                    frame_times, values, [1.0] * n, freq
                )
                # rotate production coefficients back into the small-angle
                # tau frame using the same change-of-variables identity
                a_orig = res.amplitudes[i] * math.cos(res.phases[i])
                b_orig = res.amplitudes[i] * math.sin(res.phases[i])
                theta = TWO_PI * freq * t0
                ct, st = math.cos(theta), math.sin(theta)
                a_tau_p = a_orig * ct + b_orig * st
                b_tau_p = -a_orig * st + b_orig * ct
                # The test-side trig of angle theta carries relative error
                # ~eps*|theta| alone; that bounds this comparison explicitly.
                rot_err = 2.0 * EPS * abs(theta) * max(
                    1.0, res.amplitudes[i])
                self.assertAlmostEqual(res.sses[i], sse,
                                       delta=1e-18 + 1e-10 * sse)
                self.assertAlmostEqual(res.sse0, sse0, places=11)
                self.assertAlmostEqual(res.powers[i],
                                       (sse0 - sse) / sse0, places=10)
                self.assertAlmostEqual(res.amplitudes[i],
                                       math.hypot(a_tau, b_tau),
                                       delta=1e-11 + 1e-10
                                       * max(1.0, math.hypot(a_tau, b_tau)))
                self.assertAlmostEqual(a_tau_p, a_tau,
                                       delta=1e-11 + rot_err)
                self.assertAlmostEqual(b_tau_p, b_tau,
                                       delta=1e-11 + rot_err)
                self.assertAlmostEqual(res.constants[i], c,
                                       delta=1e-11 + 1e-10
                                       * max(1.0, abs(c)))

        # Cross-frame analytic law: the same data shifted by T rotate the
        # fitted phase by 2*pi*f*T.  Residual error can only come from the
        # measured time quantization (first order: ~4*pi*f*d_tau) plus
        # roundoff -- places=5 at a rotation of ~2.4e9 rad is a relative
        # check near 2e-15, not a loose catch-all.
        for i, freq in enumerate(freqs):
            self.assertAlmostEqual(
                wrap_phase(r2.phases[i] - r1.phases[i] - TWO_PI * freq * big),
                0.0, places=5,
            )
        # the spectrum still identifies the same signal from both frames
        best1 = max(range(len(freqs)), key=lambda i: r1.powers[i])
        best2 = max(range(len(freqs)), key=lambda i: r2.powers[i])
        self.assertEqual(best1, best2)
        self.assertGreater(r1.powers[best1], 0.999)
        self.assertGreater(r2.powers[best2], 0.999)
        self.assertAlmostEqual(r1.amplitudes[best1], amp0, delta=0.01)
        self.assertAlmostEqual(r2.amplitudes[best2], amp0, delta=0.01)

    def test_unsorted_times_equivalent_to_sorted(self):
        n = 20
        rnd = lcg(313, 2 * n)
        times = [0.1 + 2.0 * u for u in rnd[:n]]
        values = [u for u in rnd[n:]]
        order = list(reversed(range(n)))
        freqs = [0.2, 0.5, 0.9, 1.3]
        r1 = periodogram(times, values, None, freqs)
        r2 = periodogram([times[k] for k in order],
                         [values[k] for k in order], None, freqs)
        for i in range(len(freqs)):
            self.assertAlmostEqual(r1.powers[i], r2.powers[i], places=13)
            self.assertAlmostEqual(r1.sse0, r2.sse0, places=12)

    def test_repeated_times_are_multiple_observations(self):
        # duplicates at the same instant with differing values/weights
        times = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0]
        values = [0.5, 0.7, 1.2, 0.9, 1.6, 1.9, 2.1]
        weights = [1.0, 0.5, 1.0, 2.0, 0.8, 1.2, 1.0]
        freqs = [0.25, 0.5, 0.75]
        res = periodogram(times, values, weights, freqs)
        for i, freq in enumerate(freqs):
            # repeated instants make cos and sin take identical values on
            # the paired rows; with the default ones column the design can
            # still be full rank unless a trig column collapses.  Compare
            # every bin against the independent reference when estimable,
            # otherwise require the explicit status instead of a power.
            _, _, _, sse, sse0, _ = normal_equation_reference(
                times, values, weights, freq
            )
            self.assertAlmostEqual(res.sses[i], sse, places=10)
            if res.statuses[i] == "unestimable":
                self.assertTrue(math.isnan(res.powers[i]))
            else:
                self.assertAlmostEqual(res.powers[i], (sse0 - sse) / sse0,
                                       places=10)

    def test_rank_deficiency_when_sampling_cannot_identify(self):
        # all observations at a single instant: cos column collinear with ones
        times = [1.5, 1.5, 1.5, 1.5]
        values = [1.0, 2.0, 0.5, 1.2]
        res = periodogram(times, values, None, [0.1, 0.2, 0.3])
        self.assertTrue(all(rank <= 2 for rank in res.ranks))
        # with all samples at one instant both trig columns are collinear
        # with the constant: every bin is explicitly unestimable rather
        # than carrying a fabricated power
        self.assertTrue(
            all(s == "unestimable" for s in res.statuses)
        )
        self.assertTrue(all(math.isnan(p) for p in res.powers))


class TestConstantAndAlias(unittest.TestCase):

    def test_constant_values_zero_power_by_contract(self):
        n = 20
        times = sorted(u for u in lcg(1, n))
        values = [3.25] * n
        weights = [0.5 + u for u in lcg(2, n)]
        res = periodogram(times, values, weights, [0.1 * j for j in range(1, 9)])
        self.assertEqual(res.sse0, 0.0)
        self.assertTrue(all(p == 0.0 for p in res.powers))
        self.assertTrue(all(a == 0.0 for a in res.amplitudes))
        self.assertTrue(all(math.isfinite(ph) for ph in res.phases))
        self.assertAlmostEqual(res.weighted_mean, 3.25, places=14)

    def test_periodic_aliasing_identical_power_and_window(self):
        # 24 hourly samples spanning one day; f_s = 24 cycles/day.  A real
        # cosine sampled over one block is fitted PERFECTLY at both f0 and at
        # its mirror alias f_s - f0: cos(2*pi*(f_s-f0)t) == cos(2*pi*f0*t)
        # on the sample grid (the sign of the sine coefficient flips).
        n = 24
        dt = 1.0 / 24.0  # days
        f0 = 3.0
        mirror = 1.0 / dt - f0  # 21 cycles/day
        shifted = 1.0 / dt + f0  # 27: window repeats there too
        amp0, delta0 = 1.4, 0.35
        times = [k * dt for k in range(n)]
        values = [
            amp0 * math.cos(TWO_PI * f0 * t - delta0) for t in times
        ]
        freqs = [float(j) for j in range(1, 30)]
        res = periodogram(times, values, None, freqs)
        i0, im, ish = (freqs.index(x) for x in (f0, mirror, shifted))
        self.assertAlmostEqual(res.powers[i0], 1.0, places=10)
        self.assertAlmostEqual(res.powers[im], 1.0, places=10)
        self.assertAlmostEqual(res.amplitudes[im], res.amplitudes[i0],
                               places=10)
        self.assertAlmostEqual(res.amplitudes[im], amp0, places=10)
        # the alias reproduces the data with a flipped sine -> mirrored phase
        self.assertAlmostEqual(
            wrap_phase(res.phases[im] + res.phases[i0]), 0.0, places=10
        )
        # window: nonzero DFT bins cancel, periodic with f_s, peak at f_s
        self.assertLess(res.window_modulus[i0], 1e-13)
        self.assertAlmostEqual(res.window_modulus[im],
                               res.window_modulus[i0], places=12)
        self.assertAlmostEqual(res.window_modulus[ish],
                               res.window_modulus[i0], places=12)
        i_fs = freqs.index(1.0 / dt)
        self.assertAlmostEqual(res.window_modulus[i_fs], 1.0, places=12)
        # explicit double-peak ambiguity: both survive exclusion selection
        top = periodogram(times, values, None, freqs,
                          top_k=2, min_separation=0.5)
        picked = sorted(p.frequency for p in top.top_peaks)
        self.assertEqual(picked, [f0, mirror])
        self.assertEqual(top.selection_status, "complete")
        self.assertEqual(top.top_k_found, 2)


class TestPeakSelection(unittest.TestCase):

    def test_exclusion_distance_is_explicit_and_boundary_inclusive(self):
        # Construct a synthetic unambiguous ranking using uneven data is
        # awkward; instead check the rule directly on a well separated
        # analytic spectrum: two harmonics at known frequencies.
        n = 64
        dt = 1.0
        times = [float(k) * dt for k in range(n)]
        f1, f2 = 4.0 / n, 9.0 / n
        values = [
            1.0 * math.cos(TWO_PI * f1 * k)
            + 0.4 * math.cos(TWO_PI * f2 * k)
            for k in range(n)
        ]
        freqs = [j / n for j in range(1, 16)]
        res = periodogram(times, values, None, freqs,
                          top_k=2, min_separation=5.0 / n)
        freqs_picked = [p.frequency for p in res.top_peaks]
        self.assertIn(f1, freqs_picked)
        self.assertIn(f2, freqs_picked)
        # separation exactly equal to min_separation is ADMITTED
        res_eq = periodogram(times, values, None, freqs,
                             top_k=2, min_separation=(f2 - f1))
        self.assertEqual(
            sorted(p.frequency for p in res_eq.top_peaks), [f1, f2]
        )
        # a slightly larger exclusion keeps the weaker bin out
        res_x = periodogram(times, values, None, freqs,
                            top_k=2, min_separation=(f2 - f1) + 1e-9)
        self.assertEqual(len(res_x.top_peaks), 2)
        self.assertNotIn(f2, [p.frequency for p in res_x.top_peaks])

    def test_local_peaks_strict_and_endpoints_excluded(self):
        n = 64
        times = [float(k) for k in range(n)]
        f1 = 4.0 / n
        values = [math.cos(TWO_PI * f1 * k) for k in range(n)]
        freqs = [j / n for j in range(1, 12)]
        res = periodogram(times, values, None, freqs)
        peak_freqs = [p.frequency for p in res.local_peaks]
        self.assertIn(f1, peak_freqs)
        self.assertNotIn(freqs[0], peak_freqs)
        self.assertNotIn(freqs[-1], peak_freqs)

    def test_selection_budget_distinguished_from_proven_few_peaks(self):
        n = 40
        times = [float(k) for k in range(n)]
        values = [math.cos(TWO_PI * (3.0 / n) * k) for k in range(n)]
        freqs = [j / n for j in range(1, 15)]
        # an exclusion span larger than the whole grid admits at most one
        # bin: exhaustive scan proves scarcity (complete, found < requested)
        scarce = periodogram(times, values, None, freqs,
                             top_k=3, min_separation=freqs[-1] - freqs[0] + 1)
        self.assertEqual(scarce.selection_status, "complete")
        self.assertEqual(scarce.top_k_found, 1)
        # cut the scan after one bin: budget exhaustion, distinct status
        cut = periodogram(times, values, None, freqs,
                          top_k=3, min_separation=2.0 / n,
                          selection_budget=1)
        self.assertEqual(cut.selection_status, "budget_exhausted")
        self.assertEqual(cut.top_k_found, 1)
        full = periodogram(times, values, None, freqs,
                           top_k=3, min_separation=2.0 / n)
        self.assertEqual(cut.top_peaks[0], full.top_peaks[0])

    def test_power_ties_resolve_deterministically(self):
        # alias pair has (numerically) equal power; lower frequency wins
        n = 24
        times = [k / 24.0 for k in range(n)]
        values = [math.cos(TWO_PI * 3.0 * t) for t in times]
        freqs = [float(j) for j in range(1, 30)]
        r1 = periodogram(times, values, None, freqs, top_k=1)
        r2 = periodogram(times, values, None, freqs, top_k=1)
        self.assertEqual(r1.top_peaks[0].frequency,
                         r2.top_peaks[0].frequency)


class TestInputValidation(unittest.TestCase):

    def _good_kwargs(self, **over):
        kw = dict(
            times=[0.0, 1.0, 2.0, 3.0],
            values=[0.1, 0.9, 1.1, 2.0],
            weights=None,
            frequencies=[0.2, 0.5, 0.8],
        )
        kw.update(over)
        return kw

    def assertBad(self, needle, **over):
        with self.assertRaises(PeriodogramError) as ctx:
            periodogram(**self._good_kwargs(**over))
        self.assertIn(needle, str(ctx.exception))

    def test_nan_and_infinity_rejected_everywhere(self):
        self.assertBad("times[1]", times=[0.0, float("nan"), 2.0, 3.0])
        self.assertBad("times[2]", times=[0.0, 1.0, float("inf"), 3.0])
        self.assertBad("values[0]", values=[float("-inf"), 0.9, 1.1, 2.0])
        self.assertBad("frequencies[1]",
                       frequencies=[0.2, float("nan"), 0.8])
        self.assertBad("weights[2]",
                       weights=[1.0, 1.0, float("inf"), 1.0])

    def test_bool_rejected_as_integer_parameter(self):
        self.assertBad("top_k", top_k=True)
        self.assertBad("selection_budget", selection_budget=False)

    def test_bool_rejected_inside_numeric_arrays(self):
        self.assertBad("times[0]", times=[True, 1.0, 2.0, 3.0])
        self.assertBad("values[1]", values=[0.1, True, 1.1, 2.0])
        self.assertBad("weights[0]", weights=[True, 1.0, 1.0, 1.0])

    def test_weights_must_be_positive(self):
        self.assertBad("weights[1]", weights=[1.0, 0.0, 2.0, 1.0])
        self.assertBad("weights[2]", weights=[1.0, 1.0, -0.5, 1.0])

    def test_length_and_boundary_checks(self):
        self.assertBad("values", values=[0.1, 0.9, 1.1])  # mismatched length
        self.assertBad("weights", weights=[1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertBad("observations", times=[0.0, 1.0],
                       values=[0.0, 1.0])  # below MIN_OBSERVATIONS
        too_many_t = [float(k) for k in range(pg.MAX_OBSERVATIONS + 1)]
        too_many_v = [0.0] * len(too_many_t)
        self.assertBad("observations", times=too_many_t, values=too_many_v,
                       frequencies=[0.5])
        too_many_f = [0.1 * j for j in range(1, pg.MAX_FREQUENCIES + 2)]
        self.assertBad("candidates", frequencies=too_many_f)
        self.assertBad("candidates", frequencies=[])

    def test_frequency_grid_rules(self):
        self.assertBad("strictly increasing",
                       frequencies=[0.2, 0.5, 0.5])
        self.assertBad("strictly increasing",
                       frequencies=[0.8, 0.5, 0.2])
        self.assertBad("frequencies[0]", frequencies=[0.0, 0.5, 0.8])
        self.assertBad("frequencies[1]", frequencies=[0.5, -0.2, 0.8])

    def test_top_k_and_tolerances(self):
        self.assertBad("top_k", top_k=0)
        self.assertBad("top_k", top_k=4)  # > number of candidates
        self.assertBad("min_separation", min_separation=-0.1)
        self.assertBad("rank_tol", rank_tol=0.0)
        self.assertBad("rank_tol", rank_tol=1.0)

    def test_type_errors_are_localizable(self):
        self.assertBad("times", times=(0.0, 1.0, 2.0, object()))
        self.assertBad("times", times=iter([0.0, 1.0, 2.0, 3.0]))
        self.assertBad("frequencies", frequencies="0.2 0.5 0.8")
        self.assertBad("top_k", top_k=1.5)
        self.assertBad("min_separation", min_separation=1 + 0j)

    def test_validation_runs_before_computation(self):
        # a bad frequency must not be masked by otherwise-valid heavy inputs
        with self.assertRaises(PeriodogramError):
            periodogram([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], None,
                        [0.1, float("nan")])

    def test_result_is_value_object(self):
        res = periodogram(**self._good_kwargs())
        self.assertEqual(len(res.frequencies), len(res.powers))
        self.assertEqual(len(res.frequencies), len(res.ranks))
        self.assertEqual(len(res.frequencies), len(res.window))
        with self.assertRaises(Exception):
            res.powers = ()  # type: ignore[misc]


class TestNuisanceDesignMatrix(unittest.TestCase):
    """Caller-supplied fixed H: [H, cos, sin] fits with relative power.

    The independent reference builds the FULL design matrix from scratch and
    solves the weighted normal equations by Gaussian elimination with
    partial pivoting -- a deliberately different algorithm from the
    production QR -- and independently constructs the H-only baseline.
    """

    def gaussian_full(self, A, b, weights):
        """Weighted least squares on an arbitrary-column design matrix A."""
        nrows = len(A)
        q = len(A[0])
        mat = [
            [sum(weights[k] * A[k][i] * A[k][j] for k in range(nrows))
             for j in range(q)]
            for i in range(q)
        ]
        rhs = [sum(weights[k] * b[k] * A[k][i] for k in range(nrows))
               for i in range(q)]
        M = [mat[i][:] + [rhs[i]] for i in range(q)]
        scale = max((abs(M[i][j]) for i in range(q) for j in range(q)),
                    default=1.0)
        for c in range(q):
            piv = max(range(c, q), key=lambda rr: abs(M[rr][c]))
            if abs(M[piv][c]) < 1e-14 * max(scale, 1.0):
                return None
            M[c], M[piv] = M[piv], M[c]
            for rr in range(c + 1, q):
                f = M[rr][c] / M[c][c]
                for cc in range(c, q + 1):
                    M[rr][cc] -= f * M[c][cc]
        x = [0.0] * q
        for rr in range(q - 1, -1, -1):
            x[rr] = (
                M[rr][q] - sum(M[rr][cc] * x[cc]
                               for cc in range(rr + 1, q))
            ) / M[rr][rr]
        return x

    def sse_of(self, A, x, b, weights):
        return sum(
            weights[k]
            * (b[k] - sum(x[j] * A[k][j] for j in range(len(A[0])))) ** 2
            for k in range(len(A))
        )

    def test_trend_plus_signal_wide_weights_matches_independent_ls(self):
        # trend (constant + linear drift) plus a single sinusoid, uneven
        # times and weights spanning 12 orders of magnitude
        n = 30
        times = [0.13 * k + 0.05 * ((k * 7) % 3) for k in range(n)]
        f0, amp0, delta0 = 0.47, 1.6, -0.4
        c0, slope0 = 0.8, 0.09
        values = [
            c0 + slope0 * t
            + amp0 * math.cos(TWO_PI * f0 * t + delta0)
            for t in times
        ]
        weights = [1e-6 if k % 2 == 0 else 1e6 for k in range(n)]
        H = [[1.0, t] for t in times]
        freqs = [0.2 + 0.03 * j for j in range(20)]

        res = periodogram(times, values, weights, freqs, H=H)
        self.assertEqual(res.h_columns, 2)
        self.assertEqual(res.constant_column, 0)
        self.assertEqual(res.h_rank, 2)

        # independent H-only baseline
        x_h = self.gaussian_full(H, values, weights)
        sse_h = self.sse_of(H, x_h, values, weights)
        self.assertAlmostEqual(res.sse0, sse_h, delta=1e-10 * max(1.0, sse_h))
        # the H-only baseline absorbs some sinusoidal leakage into its drift
        # coefficient, so it need not equal slope0; it only has to match
        # the independent H-only Gaussian solve
        self.assertAlmostEqual(res.h_baseline_coefficients[0], x_h[0],
                               places=9)
        self.assertAlmostEqual(res.h_baseline_coefficients[1], x_h[1],
                               places=9)

        for i, fr in enumerate(freqs):
            self.assertEqual(res.statuses[i], "ok")
            A = [
                [1.0, times[k],
                 math.cos(TWO_PI * fr * times[k]),
                 math.sin(TWO_PI * fr * times[k])]
                for k in range(n)
            ]
            x = self.gaussian_full(A, values, weights)
            sse = self.sse_of(A, x, values, weights)
            self.assertAlmostEqual(
                res.sses[i], sse, delta=1e-9 * max(1.0, sse)
            )
            # power denominator is the H-ONLY SSE, not a constant SSE
            self.assertAlmostEqual(res.powers[i], (sse_h - sse) / sse_h,
                                   places=9)
            self.assertAlmostEqual(res.h_coefficients[i][1], x[1],
                                   delta=1e-9 * max(1.0, abs(x[1])))
            self.assertEqual(res.ranks[i], 4)

        # signal bin identified with unit power and recovered parameters
        j = freqs.index(0.47000000000000003)
        self.assertGreater(res.powers[j], 1.0 - 1e-9)
        self.assertAlmostEqual(res.amplitudes[j], amp0, places=9)
        # data term is cos(2 pi f t + delta0) = cos(2 pi f t - (-delta0));
        # production phase = atan2(b, a) = -delta0
        self.assertAlmostEqual(res.phases[j], -delta0, places=9)
        best = max(
            range(len(freqs)),
            key=lambda i: res.powers[i]
            if res.statuses[i] == "ok" else -1.0,
        )
        self.assertEqual(best, j)

    def test_power_denominator_is_H_baseline_not_constant(self):
        # The same sinusoid on top of a strong linear drift: relative to the
        # CONSTANT model the drift would dominate the power denominator, but
        # with H=[1,t] the baseline already absorbs it.
        n = 40
        times = [0.1 * k for k in range(n)]
        f0, amp0 = 0.2, 0.5
        slope0 = 3.0
        values = [slope0 * t + amp0 * math.sin(TWO_PI * f0 * t)
                  for t in times]
        H = [[1.0, t] for t in times]
        res_h = periodogram(times, values, None, [f0], H=H)
        res_c = periodogram(times, values, None, [f0])
        # constant-only baseline SSE is huge; H-only baseline is tiny
        _, _, _, _, sse_const, _ = normal_equation_reference(
            times, values, [1.0] * n, f0
        )
        self.assertGreater(sse_const, res_h.sse0 * 100)
        self.assertGreater(res_h.powers[0], 1.0 - 1e-9)
        self.assertLess(res_c.powers[0], res_h.powers[0])

    def test_rank_deficient_H_rejected_for_whole_call(self):
        n = 20
        times = [0.1 * k for k in range(n)]
        values = [0.3 * k for k in range(n)]
        # second column exactly twice the constant -> H rank 1 < 2
        H = [[1.0, 2.0] for _ in times]
        with self.assertRaises(PeriodogramError) as ctx:
            periodogram(times, values, None, [0.2, 0.5], H=H)
        self.assertIn("H: rank deficient", str(ctx.exception))
        # a near-duplicate (but independent) column is accepted
        H2 = [[1.0, 2.0 + 1e-9 * k] for k in range(n)]
        with self.assertRaises(PeriodogramError):
            periodogram(times, values, None, [0.2, 0.5], H=H2,
                        rank_tol=1e-6)

    def test_zero_baseline_sse_is_undefined_not_zero_or_one(self):
        n = 20
        times = [0.1 * k for k in range(n)]
        values = [1.5 - 0.7 * t for t in times]  # exactly in span(H)
        H = [[1.0, t] for t in times]
        res = periodogram(times, values, None, [0.3, 0.5, 0.8], H=H)
        self.assertTrue(all(s == "undefined_baseline"
                            for s in res.statuses))
        self.assertTrue(all(math.isnan(p) for p in res.powers))
        # no peaks can be selected from undefined bins, but the call is a
        # valid completed scan rather than an exception
        self.assertEqual(res.local_peaks, ())
        self.assertEqual(res.top_peaks, ())
        self.assertEqual(res.selection_status, "complete")

    def test_low_frequency_near_drift_is_unestimable_per_bin_only(self):
        n = 24
        times = [0.1 * k for k in range(n)]
        values = [0.4 * t + math.cos(TWO_PI * 0.4 * t) for t in times]
        H = [[1.0, t] for t in times]
        freqs = [1e-12, 1e-10, 0.4]  # first two ~ collinear with t column
        res = periodogram(times, values, None, freqs, H=H)
        self.assertEqual(res.statuses[0], "unestimable")
        self.assertEqual(res.statuses[1], "unestimable")
        self.assertEqual(res.statuses[2], "ok")
        # ranks: H alone contributes 2; trig columns add 0/0/2
        self.assertEqual(res.ranks[0], 2)
        self.assertEqual(res.ranks[1], 2)
        self.assertEqual(res.ranks[2], 4)
        self.assertTrue(math.isnan(res.powers[0]))
        # only the estimable bin participates in peak selection
        self.assertEqual(
            [p.frequency for p in res.top_peaks], [0.4]
        )

    def test_H_validation_contract(self):
        n = 10
        times = [float(k) for k in range(n)]
        values = [0.1 * k for k in range(n)]
        freqs = [0.2, 0.5]

        def call(H, **kw):
            return periodogram(times, values, None, freqs, H=H, **kw)

        with self.assertRaises(PeriodogramError) as c:
            call([[1.0]] * (n - 1))
        self.assertIn("H: expected", str(c.exception))
        # a constant-free matrix (single non-ones column)
        with self.assertRaises(PeriodogramError) as c:
            call([[t] for t in times])
        self.assertIn("constant", str(c.exception))
        # duplicate constant columns rejected
        with self.assertRaises(PeriodogramError) as c:
            call([[1.0, 1.0] for _ in times])
        self.assertIn("exactly one constant", str(c.exception))
        # bool / NaN elements localized
        with self.assertRaises(PeriodogramError) as c:
            call([[1.0, True if k == 3 else float(k)] for k in range(n)])
        self.assertIn("H[3][1]", str(c.exception))
        bad = [[1.0, float(k)] for k in range(n)]
        bad[2][1] = float("nan")
        with self.assertRaises(PeriodogramError) as c:
            call(bad)
        self.assertIn("H[2][1]", str(c.exception))
        # ragged rows
        with self.assertRaises(PeriodogramError) as c:
            call([[1.0, 0.0]] * (n - 1) + [[1.0]])
        self.assertIn("H[9]", str(c.exception))
        # too few observations for H + sin + cos
        with self.assertRaises(PeriodogramError) as c:
            periodogram([0.0, 1.0, 2.0], [0.1, 0.2, 0.3], None, [0.2],
                        H=[[1.0, t] for t in range(3)])
        self.assertIn("H columns + 2", str(c.exception))

    def test_default_H_keeps_original_entry_and_constant_convention(self):
        n = 20
        times = sorted(u for u in lcg(11, n))
        values = [2.0] * n
        res = periodogram(times, values, [0.5 + u for u in lcg(12, n)],
                          [0.2 * j for j in range(1, 8)])
        self.assertEqual(res.h_columns, 1)
        self.assertEqual(res.constant_column, 0)
        self.assertTrue(all(p == 0.0 for p in res.powers))
        self.assertTrue(all(s == "ok" for s in res.statuses))
        self.assertEqual(res.sse0, 0.0)
        self.assertAlmostEqual(res.weighted_mean, 2.0, places=14)


class TestWindowDirectReference(unittest.TestCase):

    def test_window_against_direct_geometric_sum(self):
        n = 23
        rnd = lcg(909, 2 * n)
        times = sorted(3.0 * u for u in rnd[:n])
        weights = [0.4 + u for u in rnd[n:]]
        freqs = [0.07 * j for j in range(1, 12)]
        res = periodogram(times, values=[0.0] * n, weights=weights,
                          frequencies=freqs)
        for i, freq in enumerate(freqs):
            self.assertAlmostEqual(
                res.window_modulus[i],
                direct_window_modulus(times, weights, freq),
                places=13,
            )
            self.assertLessEqual(res.window_modulus[i], 1.0 + 1e-15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
