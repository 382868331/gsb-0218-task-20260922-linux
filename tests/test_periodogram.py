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
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import periodogram as pg
from periodogram import (
    periodogram,
    joint_periodogram_fit,
    PeriodogramError,
    EPS,
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


def joint_normal_equation_reference(times, values, weights, freq_subset):
    """Weighted LS for c + sum(a_j cos f_j + b_j sin f_j), independent path.

    A hand-built (1+2k)x(1+2k) weighted normal-equation system solved by
    Gaussian elimination with partial pivoting -- a deliberately different
    algorithm from the production Householder QR.  Coefficients are solved
    in the tau = t - t[0] frame (small angles) and returned as
    ``(c, [(freq, a_tau, b_tau), ...], sse, sse0)``; the original-frame
    coefficients follow from the same closed-form rotation identity the
    production code uses.  The caller picks the reference subset itself.
    """
    n = len(times)
    t0 = times[0]
    tau = [tk - t0 for tk in times]
    k = len(freq_subset)
    ncols = 1 + 2 * k
    mat = [[0.0] * ncols for _ in range(ncols)]
    rhs = [0.0] * ncols
    for idx in range(n):
        g = [1.0]
        for fr in freq_subset:
            ang = TWO_PI * fr * tau[idx]
            g.append(math.cos(ang))
            g.append(math.sin(ang))
        wk = weights[idx]
        yk = values[idx]
        for i in range(ncols):
            for j in range(ncols):
                mat[i][j] += wk * g[i] * g[j]
            rhs[i] += wk * yk * g[i]
    a = [row[:] for row in mat]
    b = rhs[:]
    for col in range(ncols):
        piv = max(range(col, ncols), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) <= 1e-14:
            raise AssertionError("independent reference hit a singular matrix")
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, ncols):
            q = a[r][col] / a[col][col]
            for c in range(col, ncols):
                a[r][c] -= q * a[col][c]
            b[r] -= q * b[col]
    x = [0.0] * ncols
    for r in range(ncols - 1, -1, -1):
        x[r] = (b[r] - sum(a[r][c] * x[c] for c in range(r + 1, ncols))) \
            / a[r][r]
    sse = 0.0
    for idx in range(n):
        fit = x[0]
        for j, fr in enumerate(freq_subset):
            ang = TWO_PI * fr * tau[idx]
            fit += x[2 * j + 1] * math.cos(ang) + x[2 * j + 2] * math.sin(ang)
        d = values[idx] - fit
        sse += weights[idx] * d * d
    wsum = sum(weights)
    wmean = sum(weights[i] * values[i] for i in range(n)) / wsum
    sse0 = sum(weights[i] * (values[i] - wmean) ** 2 for i in range(n))
    comps = [
        (freq_subset[j], x[2 * j + 1], x[2 * j + 2]) for j in range(k)
    ]
    return x[0], comps, sse, sse0, t0


def enumerate_reference_winner(times, values, weights, grid, max_order=2):
    """Brute-force every subset with the independent reference.

    Returns ``(best_subset_tuple, best_sse)`` using the SAME selection rule
    as production (SSE ascending, then frequency lexicographic).  Used only
    on small grids (n <= ~40, m <= 8).
    """
    best_key = None
    best_sse = None
    orders = range(1, min(max_order, len(grid)) + 1)
    for k in orders:
        for subset in combinations(grid, k):
            _, _, sse, _, _ = joint_normal_equation_reference(
                times, values, weights, subset
            )
            key = (sse, subset)
            if best_key is None or key < best_key:
                best_key = key
                best_sse = sse
    return best_key[1], best_sse


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
# Joint 1-/2-frequency enumeration
# --------------------------------------------------------------------------- #
class TestJointFitNormal(unittest.TestCase):
    """Two injected harmonics recovered jointly from a finite small grid."""

    def test_two_grid_harmonics_recovered_exactly_dft(self):
        # Equally spaced data; BOTH injected cycles per sample are DFT bins,
        # so closed-form orthogonality (no production code) gives the exact
        # expected joint coefficients.
        n = 60
        times = [float(k) for k in range(n)]
        j1, j2 = 5, 9
        f1, f2 = j1 / n, j2 / n
        a1, b1, a2, b2, c0 = 1.30, -0.45, 0.60, 0.85, 2.0
        values = [
            c0 + a1 * math.cos(TWO_PI * f1 * k) + b1 * math.sin(TWO_PI * f1 * k)
            + a2 * math.cos(TWO_PI * f2 * k) + b2 * math.sin(TWO_PI * f2 * k)
            for k in range(n)
        ]
        grid = [j / n for j in (1, 3, 5, 7, 9, 11, 13)]
        res = joint_periodogram_fit(times, values, None, grid)

        self.assertEqual(res.order, 2)
        self.assertEqual(res.frequencies, (f1, f2))
        self.assertEqual(res.models_evaluated, 7 + 7 * 6 // 2)
        self.assertEqual(res.rank, 5)
        self.assertLess(res.sse, 1e-26)
        self.assertAlmostEqual(res.constant, c0, places=12)
        self.assertAlmostEqual(res.power, 1.0, places=10)
        # closed-form DFT coefficients (summed directly, no module under test)
        expected = {
            f1: (a1, b1, math.hypot(a1, b1), math.atan2(b1, a1)),
            f2: (a2, b2, math.hypot(a2, b2), math.atan2(b2, a2)),
        }
        for comp in res.components:
            ae, be, ampe, phasee = expected[comp.frequency]
            self.assertAlmostEqual(comp.cos_coefficient, ae, places=11)
            self.assertAlmostEqual(comp.sin_coefficient, be, places=11)
            self.assertAlmostEqual(comp.amplitude, ampe, places=11)
            self.assertAlmostEqual(wrap_phase(comp.phase - phasee), 0.0,
                                   places=11)

    def test_uneven_weighted_joint_fit_matches_normal_equations(self):
        n = 33
        rnd = lcg(20260924, 3 * n)
        gaps = [0.05 + 0.4 * u for u in rnd[:n]]
        times = [sum(gaps[: k + 1]) for k in range(n)]
        f1, f2 = 0.17, 0.41
        values = [
            0.8
            + 1.1 * math.cos(TWO_PI * f1 * t - 0.6)
            + 0.5 * math.cos(TWO_PI * f2 * t + 1.3)
            + 0.02 * 2.0 * (u - 0.5)
            for t, u in zip(times, rnd[n:2 * n])
        ]
        weights = [0.2 + 1.8 * u for u in rnd[2 * n:]]
        grid = [0.05, 0.17, 0.23, 0.41, 0.55, 0.72]

        res = joint_periodogram_fit(times, values, weights, grid)

        # independent brute-force enumeration of every subset (Gaussian
        # elimination on the weighted normal equations -- not the QR core)
        ref_subset, ref_sse = enumerate_reference_winner(
            times, values, weights, grid
        )
        self.assertEqual(res.frequencies, tuple(ref_subset))
        self.assertEqual(res.order, 2)
        self.assertAlmostEqual(res.sse, ref_sse, delta=1e-18 + 1e-11 * ref_sse)

        c_ref, comps_ref, sse_ref, sse0_ref, t0 = \
            joint_normal_equation_reference(
                times, values, weights, list(res.frequencies)
            )
        self.assertAlmostEqual(res.sse, sse_ref,
                               delta=1e-18 + 1e-11 * sse_ref)
        self.assertAlmostEqual(res.sse0, sse0_ref,
                               delta=1e-18 + 1e-12 * sse0_ref)
        self.assertAlmostEqual(res.power, (sse0_ref - sse_ref) / sse0_ref,
                               places=10)
        self.assertAlmostEqual(res.constant, c_ref, places=10)
        for comp, (freq, a_tau, b_tau) in zip(res.components, comps_ref):
            theta = TWO_PI * freq * t0
            ct, st = math.cos(theta), math.sin(theta)
            a_ref = a_tau * ct - b_tau * st
            b_ref = a_tau * st + b_tau * ct
            self.assertAlmostEqual(comp.cos_coefficient, a_ref, places=10)
            self.assertAlmostEqual(comp.sin_coefficient, b_ref, places=10)
            self.assertAlmostEqual(comp.amplitude, math.hypot(a_ref, b_ref),
                                   delta=1e-11 + 1e-10
                                   * max(1.0, math.hypot(a_ref, b_ref)))
            self.assertAlmostEqual(
                wrap_phase(comp.phase - math.atan2(b_ref, a_ref)), 0.0,
                places=10,
            )

    def test_global_weight_rescaling_invariant(self):
        n = 25
        rnd = lcg(7, 2 * n)
        # baseline ~6 time units resolves the injected pair well (a short
        # baseline picks an ill-conditioned pair, which would only measure
        # roundoff amplification, not scale invariance)
        times = sorted(6.0 * u for u in rnd[:n])
        values = [1.0 * math.cos(TWO_PI * 0.2 * t)
                  + 0.4 * math.cos(TWO_PI * 0.47 * t) + (u - 0.5)
                  for t, u in zip(times, rnd[n:])]
        weights = [0.3 + u for u in lcg(99, n)]
        grid = [0.1, 0.2, 0.33, 0.47, 0.6]
        r1 = joint_periodogram_fit(times, values, weights, grid)
        r2 = joint_periodogram_fit(times, values, [3.7 * w for w in weights],
                                   grid)
        self.assertEqual(r1.frequencies, r2.frequencies)
        self.assertEqual(r1.frequencies, (0.2, 0.47))
        self.assertAlmostEqual(r1.sse / r2.sse, 1.0 / 3.7, places=12)
        for c1, c2 in zip(r1.components, r2.components):
            self.assertAlmostEqual(c1.amplitude, c2.amplitude, places=12)
            self.assertAlmostEqual(c1.phase, c2.phase, places=12)

    def test_max_frequencies_one_scans_singles_only(self):
        n = 40
        rnd = lcg(505, 2 * n)
        times = sorted(2.5 * u for u in rnd[:n])
        values = [math.cos(TWO_PI * 0.3 * t) + 0.5 * math.cos(TWO_PI * 0.5 * t)
                  + 0.01 * (u - 0.5)
                  for t, u in zip(times, rnd[n:])]
        grid = [0.1, 0.3, 0.5, 0.7]
        res = joint_periodogram_fit(times, values, None, grid,
                                    max_frequencies=1)
        self.assertEqual(res.order, 1)
        self.assertEqual(res.models_evaluated, 4)
        # supplementary round-trip consistency with the single-frequency entry
        single = periodogram(times, values, None, grid)
        best = min(range(len(grid)), key=lambda i: single.sses[i])
        self.assertEqual(res.frequencies[0], grid[best])
        self.assertAlmostEqual(res.sse, single.sses[best], places=12)

    def test_single_grid_point(self):
        times = [0.0, 0.4, 1.1, 1.7, 2.3]
        values = [0.3, -0.8, 1.2, -0.2, 0.9]
        res = joint_periodogram_fit(times, values, None, [0.25])
        self.assertEqual(res.frequencies, (0.25,))
        self.assertEqual(res.order, 1)
        self.assertEqual(res.models_evaluated, 1)
        c_ref, comps_ref, sse_ref, _, _ = \
            joint_normal_equation_reference(times, values, [1.0] * 5, [0.25])
        self.assertAlmostEqual(res.sse, sse_ref, places=12)
        self.assertAlmostEqual(res.constant, c_ref, places=10)


class TestJointFitTieBreak(unittest.TestCase):
    """Selection is (residual SSE, then frequency lexicographic)."""

    def test_exact_alias_pair_is_rank_deficient_and_loses_to_single(self):
        # 24 hourly samples spanning a day: f=3 and f=27 cycles/day produce
        # BIT-IDENTICAL sample columns (27 = 24 + 3 shifts by exactly one
        # cycle per sample), so the 5-column joint design has rank 3 and the
        # pair cannot reduce SSE below the single-frequency model.  Under the
        # lexicographic tie rule the single at the smaller frequency wins.
        n = 24
        times = [k / 24.0 for k in range(n)]
        f0, f_alias = 3.0, 27.0
        values = [1.4 * math.cos(TWO_PI * f0 * t - 0.35) for t in times]

        res = joint_periodogram_fit(times, values, None, [f0, f_alias])
        self.assertEqual(res.frequencies, (f0,))
        self.assertEqual(res.order, 1)
        self.assertEqual(res.models_evaluated, 3)
        self.assertLessEqual(res.rank, 3)
        # expected SSE from the INDEPENDENT 3x3 reference on the single bin
        _, _, _, sse_ref, _, _ = normal_equation_reference(
            times, values, [1.0] * n, f0
        )
        self.assertLess(sse_ref, 1e-24)
        self.assertAlmostEqual(res.sse, sse_ref, delta=1e-18 + 1e-12 * sse_ref)

        # with an extra non-alias candidate the genuine two-signal pair wins
        res2 = joint_periodogram_fit(times, values, None, [2.0, f0, f_alias])
        self.assertEqual(res2.frequencies, (2.0, f0))
        self.assertEqual(res2.rank, 5)
        self.assertLess(res2.sse, res.sse)

    def test_constant_data_tie_goes_to_smallest_single(self):
        # every subset fits bitwise-constant data with SSE 0: the documented
        # lexicographic convention must select the single smallest frequency
        times = sorted(u for u in lcg(11, 12))
        values = [4.75] * 12
        grid = [0.1 * j for j in range(1, 9)]
        res = joint_periodogram_fit(times, values, [0.5 + u for u in lcg(3, 12)],
                                    grid)
        self.assertEqual(res.frequencies, (grid[0],))
        self.assertEqual(res.order, 1)
        self.assertEqual(res.sse, 0.0)
        self.assertEqual(res.power, 0.0)
        self.assertEqual(res.sse0, 0.0)
        self.assertAlmostEqual(res.constant, 4.75, places=14)
        self.assertTrue(all(c.amplitude == 0.0 for c in res.components))
        self.assertTrue(all(c.cos_coefficient == 0.0
                            and c.sin_coefficient == 0.0
                            for c in res.components))

    def test_choice_is_deterministic_and_components_ascending(self):
        n = 30
        rnd = lcg(808, 2 * n)
        times = sorted(3.0 * u for u in rnd[:n])
        values = [(u - 0.5) + math.cos(TWO_PI * 0.22 * t)
                  for t, u in zip(times, rnd[n:])]
        grid = [0.1, 0.22, 0.4, 0.61, 0.9]
        r1 = joint_periodogram_fit(times, values, None, grid)
        r2 = joint_periodogram_fit(times, values, None, grid)
        self.assertEqual(r1, r2)
        self.assertEqual(list(r1.frequencies), sorted(r1.frequencies))
        self.assertEqual([c.frequency for c in r1.components],
                         list(r1.frequencies))


class TestJointFitCloseAndDegenerate(unittest.TestCase):
    """Close frequencies and degenerate design matrices."""

    def test_close_frequencies_both_recovered(self):
        # frequencies 5% apart in absolute terms but resolvable because the
        # irregular-time baseline (~30 time units) spans many beat periods
        n = 40
        rnd = lcg(77, n)
        times = sorted(30.0 * u for u in rnd)
        f1, f2 = 0.20, 0.21
        values = [
            1.3 * math.cos(TWO_PI * f1 * t - 0.5)
            + 0.6 * math.cos(TWO_PI * f2 * t + 1.2)
            + 0.7
            for t in times
        ]
        grid = [0.18, 0.19, 0.20, 0.21, 0.22, 0.24]
        res = joint_periodogram_fit(times, values, None, grid)
        self.assertEqual(res.frequencies, (f1, f2))
        self.assertEqual(res.rank, 5)
        c_ref, comps_ref, sse_ref, _, t0 = \
            joint_normal_equation_reference(times, values, [1.0] * n,
                                            [f1, f2])
        self.assertAlmostEqual(res.sse, sse_ref, delta=1e-16 + 1e-10 * sse_ref)
        self.assertAlmostEqual(res.constant, c_ref, places=9)
        got = {c.frequency: c for c in res.components}
        for freq, a_tau, b_tau in comps_ref:
            # rotate the independent reference into the original time frame
            # with the same exact change-of-variables identity
            theta = TWO_PI * freq * t0
            ct, st = math.cos(theta), math.sin(theta)
            a_ref = a_tau * ct - b_tau * st
            b_ref = a_tau * st + b_tau * ct
            self.assertAlmostEqual(got[freq].cos_coefficient, a_ref, places=9)
            self.assertAlmostEqual(got[freq].sin_coefficient, b_ref, places=9)
            self.assertAlmostEqual(got[freq].amplitude,
                                   math.hypot(a_ref, b_ref), places=9)
            self.assertAlmostEqual(
                wrap_phase(got[freq].phase - math.atan2(b_ref, a_ref)),
                0.0, places=9,
            )

    def test_too_close_frequencies_on_short_baseline_still_fit(self):
        # the two columns are highly correlated but NOT exactly collinear:
        # QR must keep full rank (the ill conditioning is reported, not
        # hidden) and solve exactly for noiseless data
        times = [0.0, 0.3, 0.7, 1.1, 1.5, 1.9]
        f1, f2 = 0.20, 0.2005
        values = [
            1.2 * math.cos(TWO_PI * f1 * t - 0.5)
            + 0.3 * math.cos(TWO_PI * f2 * t + 0.9)
            for t in times
        ]
        res = joint_periodogram_fit(times, values, None, [f1, f2])
        self.assertEqual(res.frequencies, (f1, f2))
        self.assertEqual(res.rank, 5)
        self.assertLess(res.sse, 1e-20)

    def test_all_times_equal_degenerate_design(self):
        # all observations at one instant: every sinusoid column is a
        # multiple of the constant column -> rank 1, no model beats the
        # weighted mean; every subset SSE equals SSE0, lex rule picks first
        times = [1.5] * 6
        values = [1.0, 2.0, 0.5, 1.2, 0.8, 1.7]
        weights = [1.0, 0.5, 2.0, 1.0, 0.8, 1.2]
        grid = [0.2, 0.35, 0.5]
        res = joint_periodogram_fit(times, values, weights, grid)
        self.assertEqual(res.frequencies, (0.2,))
        self.assertLessEqual(res.rank, 2)
        wmean = sum(w * y for w, y in zip(weights, values)) / sum(weights)
        sse0 = sum(w * (y - wmean) ** 2
                   for w, y in zip(weights, values))
        self.assertAlmostEqual(res.sse0, sse0, places=12)
        self.assertAlmostEqual(res.sse, sse0, delta=1e-14 * sse0)
        self.assertAlmostEqual(res.power, 0.0, places=12)
        self.assertEqual(res.models_evaluated, 3 + 3)

    def test_exact_interpolation_with_five_parameters(self):
        # n = 5 generic irregular instants: the 5-parameter model spans the
        # data exactly (full-rank square system), independently confirmed
        times = [0.0, 0.31, 0.74, 1.33, 2.09]
        values = [0.4, -1.1, 0.9, 0.2, -0.6]
        f1, f2 = 0.22, 0.47
        res = joint_periodogram_fit(times, values, None, [f1, f2])
        self.assertEqual(res.frequencies, (f1, f2))
        self.assertEqual(res.rank, 5)
        self.assertLess(res.sse, 1e-22)
        _, _, sse_ref, _, _ = joint_normal_equation_reference(
            times, values, [1.0] * 5, [f1, f2]
        )
        self.assertAlmostEqual(res.sse, sse_ref, delta=1e-18 + 1e-10 * sse_ref)

    def test_fewer_observations_than_parameters(self):
        # n = 3 < 1 + 2*2 = 5: every pair design is rank deficient.  Null
        # columns get coefficient 0, SSE stays well-defined, and the winner
        # is the single-frequency model with the smallest residual (the
        # minimum-contract n = 3 still permits this documented call).
        times = [0.1, 0.7, 1.3]
        values = [0.4, -0.9, 1.1]
        grid = [0.2, 0.35, 0.5]
        res = joint_periodogram_fit(times, values, None, grid)
        self.assertLessEqual(res.rank, 3)
        # reference: best SSE over the three singles via independent 3x3
        single = periodogram(times, values, None, grid)
        self.assertAlmostEqual(res.sse, min(single.sses), places=12)
        # no pair can beat a single: models were still all enumerated
        self.assertEqual(res.models_evaluated, 3 + 3)


class TestJointFitValidation(unittest.TestCase):

    def _good(self, **over):
        kw = dict(
            times=[0.0, 1.0, 2.0, 3.0, 4.0],
            values=[0.1, 0.9, 1.1, 2.0, 0.4],
            weights=None,
            frequencies=[0.2, 0.5, 0.8],
        )
        kw.update(over)
        return kw

    def assertBad(self, needle, **over):
        with self.assertRaises(PeriodogramError) as ctx:
            joint_periodogram_fit(**self._good(**over))
        self.assertIn(needle, str(ctx.exception))

    def test_max_frequencies_rules(self):
        self.assertBad("max_frequencies", max_frequencies=0)
        self.assertBad("max_frequencies", max_frequencies=3)
        self.assertBad("max_frequencies", max_frequencies=True)
        self.assertBad("max_frequencies", max_frequencies=1.5)
        self.assertBad("max_frequencies", max_frequencies="2")

    def test_grid_size_capped_at_twelve(self):
        self.assertBad("candidates",
                       frequencies=[0.1 * j for j in range(1, 14)])
        self.assertBad("candidates", frequencies=[])
        # exactly 12 is admitted
        res = joint_periodogram_fit(
            **self._good(frequencies=[0.1 * j for j in range(1, 13)])
        )
        self.assertEqual(res.models_evaluated, 12 + 12 * 11 // 2)

    def test_grid_must_be_strictly_increasing_positive(self):
        self.assertBad("strictly increasing",
                       frequencies=[0.2, 0.5, 0.5])
        self.assertBad("strictly increasing",
                       frequencies=[0.8, 0.5, 0.2])
        self.assertBad("frequencies[0]", frequencies=[0.0, 0.5, 0.8])
        self.assertBad("frequencies[1]", frequencies=[0.5, -0.2, 0.8])
        self.assertBad("frequencies[2]",
                       frequencies=[0.2, 0.5, float("nan")])

    def test_array_value_and_weight_checks(self):
        self.assertBad("times[1]",
                       times=[0.0, float("nan"), 2.0, 3.0, 4.0])
        self.assertBad("values[0]",
                       values=[float("inf"), 0.9, 1.1, 2.0, 0.4])
        self.assertBad("weights[1]",
                       weights=[1.0, 0.0, 2.0, 1.0, 1.0])
        self.assertBad("weights[2]",
                       weights=[1.0, 1.0, -0.5, 1.0, 1.0])
        self.assertBad("values", values=[0.1, 0.9, 1.1])
        self.assertBad("observations",
                       times=[0.0, 1.0], values=[0.0, 1.0],
                       frequencies=[0.5])

    def test_rank_tol_range_and_bool_elements(self):
        self.assertBad("rank_tol", rank_tol=0.0)
        self.assertBad("rank_tol", rank_tol=1.0)
        self.assertBad("times[0]", times=[True, 1.0, 2.0, 3.0, 4.0])

    def test_type_errors_localizable(self):
        self.assertBad("times", times=(0.0, 1.0, 2.0, 3.0, object()))
        self.assertBad("frequencies", frequencies="0.2 0.5 0.8")
        # plain integer elements are real numbers and stay accepted
        res = joint_periodogram_fit(
            **self._good(times=[0, 1, 2, 3, 4], weights=[1, 1, 1, 1, 1])
        )
        self.assertEqual(res.order >= 1, True)

    def test_inputs_are_not_mutated(self):
        times = [3.0, 1.0, 0.0, 2.0, 4.0]  # unsorted on purpose
        values = [0.1, 0.9, 1.1, 2.0, 0.4]
        weights = [1.0, 2.0, 1.0, 0.5, 1.5]
        grid = [0.8, 0.2, 0.5]  # invalid order -> must raise, inputs untouched
        t_copy, v_copy, w_copy = (times[:], values[:], weights[:])
        with self.assertRaises(PeriodogramError):
            joint_periodogram_fit(times, values, weights, grid)
        self.assertEqual(times, t_copy)
        self.assertEqual(values, v_copy)
        self.assertEqual(weights, w_copy)
        # a successful call likewise leaves inputs untouched
        good_grid = [0.2, 0.5, 0.8]
        joint_periodogram_fit(times, values, weights, good_grid)
        self.assertEqual(times, t_copy)
        self.assertEqual(values, v_copy)
        self.assertEqual(weights, w_copy)

    def test_result_is_immutable_value_object(self):
        res = joint_periodogram_fit(**self._good())
        with self.assertRaises(Exception):
            res.frequencies = (1.0,)  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main(verbosity=2)
