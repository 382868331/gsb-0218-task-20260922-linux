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
from periodogram import (
    periodogram,
    joint_fit,
    PeriodogramError,
    EPS,
    MAX_JOINT_GRID,
)

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


def joint_normal_equation_reference(times, values, weights, freqs):
    """Independent joint fit via hand-built weighted normal equations.

    Model columns are [1, cos f1, sin f1, (cos f2, sin f2)]; the system has
    3 or 5 rows.  Gaussian elimination with partial pivoting -- a different
    algorithm from the production Householder QR -- and columns whose pivot
    is tiny get coefficient zero, mirroring the production null-direction
    rule without sharing any code with it.  Returns
    ``(coef, sse, sse0, t0)`` with coefficients in the tau = t - t[0] frame.
    """
    n = len(times)
    t0 = times[0]
    tau = [tk - t0 for tk in times]
    k = len(freqs)
    p = 1 + 2 * k

    def basis(ti):
        g = [1.0]
        for fr in freqs:
            ang = TWO_PI * fr * ti
            g.append(math.cos(ang))
            g.append(math.sin(ang))
        return g

    mat = [[0.0] * p for _ in range(p)]
    rhs = [0.0] * p
    for idx in range(n):
        g = basis(tau[idx])
        wk = weights[idx]
        yk = values[idx]
        for a in range(p):
            for b in range(p):
                mat[a][b] += wk * g[a] * g[b]
            rhs[a] += wk * yk * g[a]

    # Gaussian elimination with partial pivoting; record skipped columns.
    a = [row[:] for row in mat]
    bvec = rhs[:]
    skipped = set()
    scale = max((abs(mat[i][j]) for i in range(p) for j in range(p)),
                default=1.0)
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12 * max(scale, 1.0):
            skipped.add(col)
            continue
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            bvec[col], bvec[piv] = bvec[piv], bvec[col]
        for r in range(col + 1, p):
            if a[col][col] == 0.0:
                continue
            q = a[r][col] / a[col][col]
            for c in range(col, p):
                a[r][c] -= q * a[col][c]
            bvec[r] -= q * bvec[col]
    x = [0.0] * p
    for r in range(p - 1, -1, -1):
        if r in skipped:
            x[r] = 0.0
            continue
        s = bvec[r] - sum(a[r][c] * x[c] for c in range(r + 1, p))
        x[r] = s / a[r][r] if a[r][r] != 0.0 else 0.0

    sse = 0.0
    for idx in range(n):
        g = basis(tau[idx])
        resid = values[idx] - sum(x[c] * g[c] for c in range(p))
        sse += weights[idx] * resid * resid
    wsum = sum(weights)
    wmean = sum(weights[idx] * values[idx] for idx in range(n)) / wsum
    sse0 = sum(weights[idx] * (values[idx] - wmean) ** 2
               for idx in range(n))
    return x, sse, sse0, t0


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
            _, _, _, sse, sse0, _ = normal_equation_reference(
                times, values, weights, freq
            )
            self.assertAlmostEqual(res.sses[i], sse, places=10)
            self.assertAlmostEqual(res.powers[i], (sse0 - sse) / sse0,
                                   places=10)

    def test_rank_deficiency_when_sampling_cannot_identify(self):
        # all observations at a single instant: cos column collinear with ones
        times = [1.5, 1.5, 1.5, 1.5]
        values = [1.0, 2.0, 0.5, 1.2]
        res = periodogram(times, values, None, [0.1, 0.2, 0.3])
        self.assertTrue(all(rank <= 2 for rank in res.ranks))
        # constant model is the best achievable; power is 0
        for p in res.powers:
            self.assertAlmostEqual(p, 0.0, places=12)


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


# --------------------------------------------------------------------------- #
# joint_fit: exhaustive enumeration of one or two frequencies
# --------------------------------------------------------------------------- #
def _joint_tau_coefficients(combo, t0):
    """Rotate a production combination's coefficients into the tau frame."""
    out = [combo.constant]
    for freq, amp, phase in zip(combo.frequencies, combo.amplitudes,
                                combo.phases):
        a_orig = amp * math.cos(phase)
        b_orig = amp * math.sin(phase)
        theta = TWO_PI * freq * t0
        ct, st = math.cos(theta), math.sin(theta)
        out.append(a_orig * ct + b_orig * st)
        out.append(-a_orig * st + b_orig * ct)
    return out


class TestJointFitAnalytic(unittest.TestCase):
    """DFT-grid analytic recovery for joint one/two-frequency models."""

    def test_two_harmonics_pair_is_exact_and_singles_leave_energy(self):
        # Two on-grid tones: DFT orthogonality gives closed-form results.
        n = 64
        f1, f2 = 4.0 / n, 11.0 / n
        a1, d1 = 1.3, 0.6          # y = a1 cos(2pi f1 t - d1) + ...
        a2, d2 = 0.7, -1.1
        times = [float(k) for k in range(n)]
        values = [
            a1 * math.cos(TWO_PI * f1 * k - d1)
            + a2 * math.cos(TWO_PI * f2 * k - d2)
            for k in range(n)
        ]
        # f1, f2 plus off-signal grid points, supplied in ascending order
        grid = [3.0 / n, f1, f2, 20.0 / n]
        res = joint_fit(times, values, None, grid)

        best = res.best
        self.assertEqual(best.frequencies, (f1, f2))
        self.assertEqual(best.indices, (1, 2))
        self.assertLess(best.sse, 1e-24)
        self.assertAlmostEqual(best.power, 1.0, places=10)
        self.assertEqual(best.rank, 5)
        # closed-form DFT coefficients (independent sums, no production code)
        self.assertAlmostEqual(best.constant, 0.0, places=12)
        self.assertAlmostEqual(best.amplitudes[0], a1, places=11)
        self.assertAlmostEqual(best.amplitudes[1], a2, places=11)
        self.assertAlmostEqual(wrap_phase(best.phases[0] - d1), 0.0,
                               places=11)
        self.assertAlmostEqual(wrap_phase(best.phases[1] - d2), 0.0,
                               places=11)

        # hand calculation: each single leaves the other tone's orthogonal
        # energy exactly, SSE = n/2 * A**2 (unit weights)
        by_key = {c.frequencies: c for c in res.combinations}
        self.assertAlmostEqual(by_key[(f1,)].sse, 0.5 * n * a2 ** 2,
                               delta=1e-9)
        self.assertAlmostEqual(by_key[(f2,)].sse, 0.5 * n * a1 ** 2,
                               delta=1e-9)
        # every listed SSE is a legitimate model residual (power in [0,1])
        for c in res.combinations:
            self.assertGreaterEqual(c.sse, -1e-18)
            self.assertTrue(-1e-12 <= c.power <= 1.0 + 1e-12,
                            (c.frequencies, c.power))

    def test_single_tone_exact_models_resolve_to_lowest_tuple(self):
        # One on-grid tone: several models reach machine-noise SSE (the exact
        # single (f1,), its sampling alias (1 - f1,), and pairs containing
        # f1).  Every minimum-SSE model contains f1, and the complete table is
        # ordered by the documented (sse, frequency-tuple) key.  The exact
        # lexicographic/prefix tie-break itself is checked deterministically
        # on bitwise-constant data in TestJointFitDegenerateDesigns.
        n = 48
        f1 = 5.0 / n
        amp, delta = 1.1, 0.3
        times = [float(k) for k in range(n)]
        values = [2.0 + amp * math.cos(TWO_PI * f1 * k - delta)
                  for k in range(n)]
        grid = sorted({f1, 1.0 / n, 6.0 / n, 1.0 - f1})
        res = joint_fit(times, values, None, grid)

        self.assertLess(res.best.sse, 1e-24)
        # the exact fit can be the f1 single, its sampling alias, or any
        # pair spanning the same column space; it must not be an unrelated bin
        self.assertTrue(
            f1 in res.best.frequencies
            or res.best.frequencies == (1.0 - f1,)
        )
        single = next(c for c in res.combinations if c.frequencies == (f1,))
        self.assertAlmostEqual(single.amplitudes[0], amp, places=11)
        self.assertAlmostEqual(wrap_phase(single.phases[0] - delta), 0.0,
                               places=11)
        self.assertAlmostEqual(single.constant, 2.0, places=11)
        # the complete table is ordered by the documented key
        keys = [(c.sse, c.frequencies) for c in res.combinations]
        self.assertEqual(keys, sorted(keys))


class TestJointFitNormalEquationReference(unittest.TestCase):
    """Uneven times/weights: 5-column QR vs independent normal equations."""

    def test_all_pairs_match_independent_reference(self):
        n = 31
        rnd = lcg(7071, 3 * n)
        times = [0.05 + 1.3 * u for u in rnd[:n]]
        values = [0.5 + 2.0 * (u - 0.5) for u in rnd[n:2 * n]]
        weights = [0.25 + 1.5 * u for u in rnd[2 * n:]]
        grid = [0.17 + 0.13 * j for j in range(5)]
        res = joint_fit(times, values, weights, grid)

        for c in res.combinations:
            xref, sse_ref, sse0_ref, t0 = joint_normal_equation_reference(
                times, values, weights, list(c.frequencies)
            )
            self.assertAlmostEqual(c.sse, sse_ref,
                                   delta=1e-17 + 1e-11 * max(1.0, sse_ref))
            self.assertAlmostEqual(res.sse0, sse0_ref, places=11)
            self.assertAlmostEqual(c.power, (sse0_ref - c.sse) / sse0_ref,
                                   places=10)
            xprod = _joint_tau_coefficients(c, t0)
            for col in range(len(xref)):
                self.assertAlmostEqual(
                    xprod[col], xref[col],
                    delta=1e-10 + 1e-9 * max(1.0, abs(xref[col])),
                )

    def test_close_frequencies_resolved_against_reference(self):
        # Beat period 1/0.03 ~ 33; the 40-time-unit span covers > 1 beat, so
        # the pair is resolvable but the 5-column design is genuinely
        # ill-conditioned (this is the close-frequency stress case).
        n = 40
        rnd = lcg(808, 3 * n)
        times = sorted(40.0 * u for u in rnd[:n])
        f1, f2, a1, a2, d1, d2 = 0.20, 0.23, 1.2, 0.8, 0.5, -0.9
        values = [
            a1 * math.cos(TWO_PI * f1 * t - d1)
            + a2 * math.cos(TWO_PI * f2 * t - d2)
            + 0.01 * 2.0 * (u - 0.5)
            for t, u in zip(times, rnd[n:2 * n])
        ]
        weights = [0.4 + 1.2 * u for u in rnd[2 * n:]]
        grid = [0.14, f1, f2, 0.29]
        res = joint_fit(times, values, weights, grid)

        self.assertEqual(res.best.frequencies, (f1, f2))
        self.assertEqual(res.best.rank, 5)
        xref, sse_ref, _, t0 = joint_normal_equation_reference(
            times, values, weights, [f1, f2]
        )
        # Observed agreement is ~3e-15 relative on SSE and ~1e-16 on the
        # coefficients; the bounds below leave headroom for the 5x5 normal
        # equations' own squared-condition-number error, the same error-budget
        # logic as the single-frequency suite (not a loose catch-all).
        self.assertAlmostEqual(res.best.sse, sse_ref,
                               delta=1e-17 + 1e-12 * max(1.0, sse_ref))
        xprod = _joint_tau_coefficients(res.best, t0)
        for col in range(5):
            self.assertAlmostEqual(xprod[col], xref[col], places=9)
        # the pair crushes either single despite f2-f1 = only 0.03
        singles = {c.frequencies[0]: c.sse
                   for c in res.combinations if len(c.frequencies) == 1}
        self.assertLess(res.best.sse, 1e-3 * singles[f1])

    def test_global_weight_rescaling_invariant(self):
        n = 20
        rnd = lcg(13, 2 * n)
        times = [0.1 + 2.0 * u for u in rnd[:n]]
        values = [u - 0.5 for u in rnd[n:]]
        weights = [0.3 + u for u in lcg(14, n)]
        grid = [0.2, 0.5, 0.9, 1.3]
        r1 = joint_fit(times, values, weights, grid)
        r2 = joint_fit(times, values, [4.2 * w for w in weights], grid)
        self.assertEqual(r1.best.frequencies, r2.best.frequencies)
        for c1, c2 in zip(r1.combinations, r2.combinations):
            self.assertEqual(c1.frequencies, c2.frequencies)
            self.assertAlmostEqual(c1.sse / r1.sse0, c2.sse / r2.sse0,
                                   places=13)
            for a, b in zip(c1.amplitudes, c2.amplitudes):
                self.assertAlmostEqual(a, b, places=12)


class TestJointFitDegenerateDesigns(unittest.TestCase):

    def test_alias_pair_columns_collapse_rank_drops_no_raise(self):
        # 24 hourly samples: at f=3 and mirror f=21, cos columns coincide and
        # sin columns are exactly opposite on the sample grid.  The joint
        # design has rank 3 (constant + one cos + one sin direction); the
        # duplicate columns must be dropped, and the pair ties its prefix
        # single (3.0,), which therefore wins.
        n = 24
        dt = 1.0 / 24.0
        f0, mirror = 3.0, 1.0 / dt - 3.0
        amp, delta = 1.4, 0.35
        times = [k * dt for k in range(n)]
        values = [amp * math.cos(TWO_PI * f0 * t - delta) for t in times]
        grid = [1.0, f0, 10.0, mirror, 25.0]
        res = joint_fit(times, values, None, grid)

        by_key = {c.frequencies: c for c in res.combinations}
        pair = by_key[(f0, mirror)]
        self.assertEqual(pair.rank, 3)
        single = by_key[(f0,)]
        self.assertAlmostEqual(pair.sse, single.sse, places=12)
        self.assertLess(single.sse, 1e-24)
        # every machine-exact model spans the same column space: the two
        # alias singles and any pair containing f0; the deterministic
        # ordering among them is covered by the constant-data tie test
        exact = {c.frequencies for c in res.combinations if c.sse < 1e-20}
        self.assertIn(res.best.frequencies, exact)
        self.assertTrue(
            f0 in res.best.frequencies or res.best.frequencies == (mirror,)
        )
        # the null (second) direction is reported as zero amplitude rather
        # than invented, while the first keeps the true analytic amplitude
        self.assertAlmostEqual(single.amplitudes[0], amp, places=10)
        self.assertAlmostEqual(pair.amplitudes[0], amp, places=9)
        self.assertAlmostEqual(pair.amplitudes[1], 0.0, places=9)
        # a model with no alias component leaves substantial energy
        self.assertGreater(by_key[(10.0,)].sse, 1.0)

    def test_more_columns_than_rows_is_rank_deficient_not_an_error(self):
        # n = 4 observations, pair model = 5 columns: rank can never exceed
        # 4.  Must fit and rank cleanly instead of dividing by a zero pivot.
        times = [0.0, 0.3, 1.1, 2.0]
        values = [0.4, 1.1, 0.2, 1.6]
        weights = [1.0, 0.7, 1.3, 1.0]
        grid = [0.25, 0.6, 0.95]
        res = joint_fit(times, values, weights, grid)
        self.assertEqual(len(res.combinations), 3 + 3)
        for c in res.combinations:
            self.assertLessEqual(c.rank, 4)
            self.assertTrue(math.isfinite(c.sse))
            self.assertTrue(all(math.isfinite(v) for v in c.amplitudes))
            self.assertTrue(all(math.isfinite(v) for v in c.phases))
            self.assertGreaterEqual(c.sse, -1e-18)
        best_ref_x, best_ref_sse, _, _ = joint_normal_equation_reference(
            times, values, weights, list(res.best.frequencies)
        )
        self.assertAlmostEqual(res.best.sse, best_ref_sse,
                               delta=1e-14 + 1e-9 * max(1.0, best_ref_sse))

    def test_all_times_equal_every_model_reduces_to_constant(self):
        times = [1.5, 1.5, 1.5, 1.5, 1.5]
        values = [1.0, 2.0, 0.5, 1.2, 1.7]
        res = joint_fit(times, values, None, [0.1, 0.2, 0.3])
        for c in res.combinations:
            self.assertLessEqual(c.rank, 2)
            self.assertAlmostEqual(c.power, 0.0, places=12)
            self.assertTrue(all(a == 0.0 for a in c.amplitudes))

    def test_constant_data_exact_zero_convention(self):
        n = 15
        times = sorted(u for u in lcg(3, n))
        values = [-2.75] * n
        weights = [0.6 + u for u in lcg(4, n)]
        res = joint_fit(times, values, weights, [0.2, 0.5, 0.9])
        self.assertEqual(res.sse0, 0.0)
        # exact all-zero-SSE tie: lexicographic tuple order, and the one-tuple
        # is a prefix of any pair starting at the same frequency -> (0.2,) wins
        self.assertEqual(res.best.frequencies, (0.2,))
        for c in res.combinations:
            self.assertEqual(c.sse, 0.0)
            self.assertEqual(c.power, 0.0)
            self.assertEqual(c.constant, -2.75)
            self.assertTrue(all(a == 0.0 for a in c.amplitudes))
        ordered = [c.frequencies for c in res.combinations]
        self.assertEqual(
            ordered,
            sorted(ordered),
        )


class TestJointFitEnumeration(unittest.TestCase):

    def test_grid_of_twelve_enumerates_78_models(self):
        n = 30
        times = sorted(3.0 * u for u in lcg(5, n))
        values = [u - 0.5 for u in lcg(6, n)]
        grid = [0.1 * j for j in range(1, MAX_JOINT_GRID + 1)]
        res = joint_fit(times, values, None, grid)
        m = MAX_JOINT_GRID
        self.assertEqual(len(res.combinations), m + m * (m - 1) // 2)
        keys = {c.indices for c in res.combinations}
        self.assertEqual(len(keys), len(res.combinations))  # no duplicates
        for i in range(m):
            self.assertIn((i,), keys)
            for j in range(i + 1, m):
                self.assertIn((i, j), keys)
        ordered = sorted(res.combinations,
                         key=lambda c: (c.sse, c.frequencies))
        self.assertEqual(tuple(ordered), res.combinations)
        self.assertIs(res.best, res.combinations[0])
        self.assertEqual(res.frequencies, tuple(grid))

    def test_singles_agree_with_periodogram_scan(self):
        # Supplementary round-trip check between the two public entries.
        n = 25
        rnd = lcg(21, 2 * n)
        times = [0.1 + 1.5 * u for u in rnd[:n]]
        values = [u - 0.5 for u in rnd[n:]]
        grid = [0.2, 0.55, 0.9, 1.25]
        res = joint_fit(times, values, None, grid)
        pg_res = periodogram(times, values, None, grid)
        by_key = {c.frequencies: c for c in res.combinations
                  if len(c.frequencies) == 1}
        for i, freq in enumerate(grid):
            c = by_key[(freq,)]
            self.assertAlmostEqual(c.sse, pg_res.sses[i], places=12)
            self.assertAlmostEqual(c.amplitudes[0], pg_res.amplitudes[i],
                                   places=12)
            self.assertAlmostEqual(c.constant, pg_res.constants[i], places=12)
        self.assertAlmostEqual(res.sse0, pg_res.sse0, places=12)

    def test_inputs_are_not_mutated(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        values = [0.5, 1.2, 0.1, 1.9, 1.0]
        weights = [1.0, 0.5, 2.0, 1.0, 0.8]
        grid = [0.3, 0.7, 1.1]
        t_copy, v_copy, w_copy, g_copy = (
            list(times), list(values), list(weights), list(grid))
        joint_fit(times, values, weights, grid)
        self.assertEqual(times, t_copy)
        self.assertEqual(values, v_copy)
        self.assertEqual(weights, w_copy)
        self.assertEqual(grid, g_copy)


class TestJointFitValidation(unittest.TestCase):

    def _good_kwargs(self, **over):
        kw = dict(
            times=[0.0, 0.4, 1.1, 1.8, 2.6],
            values=[0.1, 0.9, 1.1, 2.0, 1.3],
            weights=None,
            frequencies=[0.2, 0.5, 0.8],
        )
        kw.update(over)
        return kw

    def assertBad(self, needle, **over):
        with self.assertRaises(PeriodogramError) as ctx:
            joint_fit(**self._good_kwargs(**over))
        self.assertIn(needle, str(ctx.exception))

    def test_grid_size_limits(self):
        n = 30
        times = [float(k) * 0.1 for k in range(n)]
        values = [0.0] * n
        grid12 = [0.1 * j for j in range(1, MAX_JOINT_GRID + 1)]
        joint_fit(times, values, None, grid12)  # boundary accepted
        grid13 = [0.1 * j for j in range(1, MAX_JOINT_GRID + 2)]
        with self.assertRaises(PeriodogramError) as ctx:
            joint_fit(times, values, None, grid13)
        self.assertIn(str(MAX_JOINT_GRID), str(ctx.exception))
        self.assertBad("candidates", frequencies=[])
        # one candidate is legal (singles only)
        one = joint_fit(**self._good_kwargs(frequencies=[0.4]))
        self.assertEqual(len(one.combinations), 1)
        self.assertEqual(one.best.frequencies, (0.4,))

    def test_grid_order_and_sign(self):
        self.assertBad("strictly increasing",
                       frequencies=[0.2, 0.5, 0.5])
        self.assertBad("strictly increasing",
                       frequencies=[0.8, 0.5, 0.2])
        self.assertBad("frequencies[0]", frequencies=[0.0, 0.5, 0.8])
        self.assertBad("frequencies[2]", frequencies=[0.2, 0.5, -0.1])

    def test_nan_inf_bool_and_friends(self):
        self.assertBad("times[1]",
                       times=[0.0, float("nan"), 1.0, 2.0, 3.0])
        self.assertBad("values[0]",
                       values=[float("inf"), 0.9, 1.1, 2.0, 1.0])
        self.assertBad("weights[2]",
                       weights=[1.0, 1.0, -0.5, 1.0, 1.0])
        self.assertBad("weights[1]",
                       weights=[1.0, 0.0, 1.0, 1.0, 1.0])
        self.assertBad("frequencies[1]",
                       frequencies=[0.2, float("nan"), 0.8])
        self.assertBad("times[0]",
                       times=[True, 1.0, 2.0, 3.0, 4.0])
        self.assertBad("times", times=iter([0.0, 1.0, 2.0, 3.0, 4.0]))
        self.assertBad("frequencies", frequencies="0.2 0.5 0.8")

    def test_length_and_observation_bounds(self):
        self.assertBad("values",
                       values=[0.1, 0.9, 1.1, 2.0])  # length mismatch
        self.assertBad("weights",
                       weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertBad("observations",
                       times=[0.0, 1.0], values=[0.0, 1.0],
                       frequencies=[0.5])
        too_many_t = [float(k) for k in range(pg.MAX_OBSERVATIONS + 1)]
        self.assertBad("observations",
                       times=too_many_t, values=[0.0] * len(too_many_t),
                       frequencies=[0.5])

    def test_rank_tol_bounds_and_bool(self):
        self.assertBad("rank_tol", rank_tol=0.0)
        self.assertBad("rank_tol", rank_tol=1.0)
        self.assertBad("rank_tol", rank_tol=True)
        res = joint_fit(**self._good_kwargs(rank_tol=1e-8))
        self.assertEqual(res.rank_tol, 1e-8)

    def test_no_partial_state_on_validation_failure(self):
        # pure function: a bad call raises and a subsequent good call on the
        # same objects is unaffected (nothing is cached/mutated globally)
        times = [0.0, 0.4, 1.1, 1.8, 2.6]
        values = [0.1, 0.9, 1.1, 2.0, 1.3]
        with self.assertRaises(PeriodogramError):
            joint_fit(times, values, None, [])
        r1 = joint_fit(times, values, None, [0.2, 0.5, 0.8])
        r2 = joint_fit(times, values, None, [0.2, 0.5, 0.8])
        self.assertEqual(
            tuple((c.indices, c.sse) for c in r1.combinations),
            tuple((c.indices, c.sse) for c in r2.combinations),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
