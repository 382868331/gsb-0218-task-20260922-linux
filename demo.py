"""Fixed-input demonstration of the floating-mean periodogram module.

Run with:  ./.venv/bin/python demo.py

The script performs real computation (no sleeps, no pre-recorded results):

1. normal result  -- a deterministic unevenly sampled light curve with a
   nightly observing cadence produces a genuine double-peak alias
   ambiguity (true f0 and f0 +/- 1/day); the sampling window explains it;
2. known-trend H -- a linear drift baseline and explicit rank/power states;
3. rejection path -- a NaN observation is actually rejected with a
   locatable PeriodogramError (plus a bool-as-int parameter rejection).
"""

import math
import time

from periodogram import PeriodogramError, periodogram

TWO_PI = 2.0 * math.pi


def lcg(seed, count):
    """Deterministic POSIX-style LCG floats in [0, 1); no randomness source."""
    x = seed
    out = []
    for _ in range(count):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out.append(x / 0x80000000)
    return out


def independent_normal_equation_fit(times, values, freq):
    """Tiny independent 3x3 normal-equation reference (not the QR core)."""
    mat = [[0.0] * 3 for _ in range(3)]
    vec = [0.0, 0.0, 0.0]
    for t, y in zip(times, values):
        phi = TWO_PI * freq * t
        g = (1.0, math.cos(phi), math.sin(phi))
        for i in range(3):
            for j in range(3):
                mat[i][j] += g[i] * g[j]
            vec[i] += y * g[i]
    # Gaussian elimination with partial pivoting
    a = [row[:] for row in mat]
    b = vec[:]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, 3):
            q = a[r][col] / a[col][col]
            for k in range(col, 3):
                a[r][k] -= q * a[col][k]
            b[r] -= q * b[col]
    coef = [0.0, 0.0, 0.0]
    for r in range(2, -1, -1):
        coef[r] = (b[r] - sum(a[r][k] * coef[k]
                               for k in range(r + 1, 3))) / a[r][r]
    c, aa, bb = coef
    sse = 0.0
    for t, y in zip(times, values):
        phi = TWO_PI * freq * t
        d = y - c - aa * math.cos(phi) - bb * math.sin(phi)
        sse += d * d
    return math.hypot(aa, bb), sse


def independent_full_matrix_fit(times, values, weights, freq):
    """Independent [1, t, cos, sin] weighted normal-equation fit."""
    rows = []
    for t in times:
        phi = TWO_PI * freq * t
        rows.append((1.0, t, math.cos(phi), math.sin(phi)))
    k = 4
    mat = [[0.0] * k for _ in range(k)]
    vec = [0.0] * k
    for wt, y, g in zip(weights, values, rows):
        for i in range(k):
            vec[i] += wt * y * g[i]
            for j in range(k):
                mat[i][j] += wt * g[i] * g[j]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(mat[r][col]))
        mat[col], mat[piv] = mat[piv], mat[col]
        vec[col], vec[piv] = vec[piv], vec[col]
        for r in range(col + 1, k):
            q = mat[r][col] / mat[col][col]
            for c in range(col, k):
                mat[r][c] -= q * mat[col][c]
            vec[r] -= q * vec[col]
    coef = [0.0] * k
    for r in range(k - 1, -1, -1):
        coef[r] = (vec[r] - sum(mat[r][c] * coef[c]
                                for c in range(r + 1, k))) / mat[r][r]
    sse = 0.0
    for wt, y, g in zip(weights, values, rows):
        d = y - sum(g[i] * coef[i] for i in range(k))
        sse += wt * d * d
    return coef, math.hypot(coef[2], coef[3]), sse


def h_demo():
    n = 16
    times = [0.1 * k + 0.002 * k * k for k in range(n)]
    weights = [10.0 ** (6.0 * k / (n - 1)) for k in range(n)]
    intercept, slope = -1.2, 0.55
    f0, amp, phase = 0.83, 2.0, -0.7
    values = [
        intercept + slope * t
        + amp * math.cos(TWO_PI * f0 * t - phase)
        for t in times
    ]
    H = [[1.0, t] for t in times]
    frequencies = [1.0e-12, 0.83, 1.27]
    result = periodogram(times, values, weights, frequencies, H=H,
                         top_k=1)
    print("-" * 72)
    print("known-trend H=[1,t] demonstration (weights span 10^6):")
    print(f"  baseline: rank={result.h_rank}, SSE_H={result.sse0:.10g}")
    for item in result.frequency_statuses:
        if item.status == "ok":
            print(f"  f={item.frequency:.12g}: status=ok, added_rank="
                  f"{item.added_rank}, P={item.power:.10f}, "
                  f"A={item.amplitude:.10f}")
        else:
            print(f"  f={item.frequency:.12g}: "
                  f"status={item.status}, power={item.power}")
    coef, amp_ref, sse_ref = independent_full_matrix_fit(
        times, values, weights, f0
    )
    peak = result.top_peaks[0]
    print(f"  independent full [H,cos,sin] check at f={f0:.2f}: "
          f"A diff={abs(peak.amplitude - amp_ref):.2e}, "
          f"SSE diff={abs(peak.sse - sse_ref):.2e}")
    print(f"  baseline trend recovered: intercept={coef[0]:.10f}, "
          f"slope={coef[1]:.10f}")

    trend_only = [intercept + slope * t for t in times]
    zero = periodogram(times, trend_only, weights, [f0], H=H)
    print(f"  zero baseline SSE: status={zero.statuses[0]}, "
          f"sse0={zero.sse0}, power={zero.powers[0]} (not divided or faked)")


def build_alias_dataset():
    """25 nights, 3 observations per night; signal at f0 = 1.70 cycles/day.

    Gaps of one observing day create strong +/- 1.0/day aliases.
    """
    rnd = lcg(20260923, 400)
    f_true, amp_true, phase_true = 1.70, 1.50, 0.8
    times, values = [], []
    k = 0
    for night in range(25):
        for _ in range(3):
            t = night + 0.01 + 0.06 * rnd[k]
            k += 1
            times.append(t)
            values.append(
                amp_true * math.cos(TWO_PI * f_true * t - phase_true)
                + 0.05 * 2.0 * (rnd[k] - 0.5)
            )
    return times, values, f_true, amp_true, phase_true


def main():
    print("=" * 72)
    print("Floating-mean periodogram demo (uneven sampling, weighted QR)")
    print("=" * 72)

    times, values, f_true, amp_true, phase_true = build_alias_dataset()
    frequencies = [0.05 + 0.01 * j for j in range(300)]
    print(f"observations         : {len(times)} (25 nights x 3, within-night "
          f"jitter)")
    print(f"candidate frequencies: {len(frequencies)} on "
          f"[{frequencies[0]:.2f}, {frequencies[-1]:.2f}] cycles/day")
    print(f"injected signal      : f={f_true:.2f}/day A={amp_true:.2f} "
          f"phase={phase_true:.2f} rad")

    t_start = time.perf_counter()
    result = periodogram(
        times, values, None, frequencies,
        top_k=4, min_separation=0.5,
    )
    elapsed = time.perf_counter() - t_start

    print("-" * 72)
    print(f"computed in {elapsed:.3f}s; constant-model SSE0={result.sse0:.6f}; "
          f"selection={result.selection_status}")
    print("top peaks with exclusion distance 0.50/day:")
    for p in result.top_peaks:
        print(f"  f = {p.frequency:5.2f}/day   P = {p.power:.6f}   "
              f"A = {p.amplitude:.4f}   phase = {p.phase:+.4f} rad   "
              f"rank = {p.rank}   |W|/sumw = {p.window_modulus:.4f}")

    # independent spot check at the strongest bin
    best = result.top_peaks[0]
    amp_ref, sse_ref = independent_normal_equation_fit(
        times, values, best.frequency
    )
    print("-" * 72)
    print("independent 3x3 normal-equation spot check at the strongest bin:")
    print(f"  amplitude: QR={best.amplitude:.10f}  "
          f"normal-eq={amp_ref:.10f}  diff={abs(best.amplitude-amp_ref):.2e}")
    print(f"  SSE      : QR={best.sse:.10f}  normal-eq={sse_ref:.10f}  "
          f"diff={abs(best.sse - sse_ref):.2e}")

    # explain the ambiguity with the sampling window
    # grid is 0.05 + 0.01*j; locate bins by integer index, never float ==
    j_day = round((1.0 - 0.05) / 0.01)
    j_true = round((f_true - 0.05) / 0.01)
    i_day = j_day
    i_true = j_true
    i_low = j_true - 100
    i_high = j_true + 100
    print("-" * 72)
    print("alias interpretation from the normalized sampling window:")
    print(f"  |W(1.00/day)|/sum w = {result.window_modulus[i_day]:.4f} "
          f"-> near-in-phase nightly cadence")
    print(f"  P(f0={f_true:.2f})={result.powers[i_true]:.6f}   "
          f"P(f0-1={f_true - 1:.2f})={result.powers[i_low]:.6f}   "
          f"P(f0+1={f_true + 1:.2f})={result.powers[i_high]:.6f}")
    ambiguity = (
        result.powers[i_high] > 0.98 * result.powers[i_true]
        and result.powers[i_low] > 0.98 * result.powers[i_true]
    )
    print(f"  double-peak ambiguity present: {ambiguity} "
          f"(f0 and f0 +/- 1/day are nearly equally supported)")

    # ------------------------- known-trend extension -----------------------
    h_demo()

    # ------------------- actually triggered rejection path ------------------
    print("-" * 72)
    print("error-boundary demonstration (real exceptions, actually raised):")
    bad_values = values[:]
    bad_values[7] = float("nan")
    try:
        periodogram(times, bad_values, None, frequencies)
    except PeriodogramError as exc:
        print(f"  NaN rejected        : PeriodogramError -> {exc}")
    else:
        raise SystemExit("FAIL: NaN input was not rejected")

    try:
        periodogram(times, values, None, frequencies, top_k=True)
    except PeriodogramError as exc:
        print(f"  bool int rejected   : PeriodogramError -> {exc}")
    else:
        raise SystemExit("FAIL: bool top_k was not rejected")

    print("=" * 72)
    print("demo finished normally.")


if __name__ == "__main__":
    main()
