"""Fixed-input demonstration of the floating-mean periodogram module.

Run with:  python demo.py

The script performs real computation (no sleeps, no pre-recorded results):

1. normal result  -- a deterministic unevenly sampled light curve with a
   nightly observing cadence produces a genuine double-peak alias
   ambiguity (true f0 and f0 +/- 1/day); the sampling window explains it;
2. joint 1-/2-frequency enumeration -- two injected signals are recovered
   together from a <= 12-point finite grid (residual, then lexicographic
   selection), checked against an independent normal-equation reference,
   with a close-frequency resolvable case and an exact-alias degenerate
   pair;
3. rejection path -- NaN observations and bool-as-int parameters are
   actually rejected with locatable PeriodogramError messages.
"""

import math
import time

from periodogram import PeriodogramError, joint_periodogram_fit, periodogram

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


def independent_joint_normal_equation_fit(times, values, freqs):
    """Independent weighted LS for the joint model (5x5, not the QR core).

    Solves c + a1 cos f1 + b1 sin f1 + a2 cos f2 + b2 sin f2 by Gaussian
    elimination on hand-built normal equations in the recentered tau frame,
    so the trigonometric arguments stay small.  Returns ``(coef, sse)``.
    """
    t0 = times[0]
    tau = [t - t0 for t in times]
    ncols = 1 + 2 * len(freqs)
    mat = [[0.0] * ncols for _ in range(ncols)]
    vec = [0.0] * ncols
    for t, y in zip(tau, values):
        g = [1.0]
        for fr in freqs:
            phi = TWO_PI * fr * t
            g.extend((math.cos(phi), math.sin(phi)))
        for i in range(ncols):
            for j in range(ncols):
                mat[i][j] += g[i] * g[j]
            vec[i] += y * g[i]
    a = [row[:] for row in mat]
    b = vec[:]
    for col in range(ncols):
        piv = max(range(col, ncols), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, ncols):
            q = a[r][col] / a[col][col]
            for k in range(col, ncols):
                a[r][k] -= q * a[col][k]
            b[r] -= q * b[col]
    coef = [0.0] * ncols
    for r in range(ncols - 1, -1, -1):
        coef[r] = (b[r] - sum(a[r][k] * coef[k]
                              for k in range(r + 1, ncols))) / a[r][r]
    sse = 0.0
    for t, y in zip(tau, values):
        fit = coef[0]
        for j, fr in enumerate(freqs):
            phi = TWO_PI * fr * t
            fit += coef[2 * j + 1] * math.cos(phi)
            fit += coef[2 * j + 2] * math.sin(phi)
        d = y - fit
        sse += d * d
    return coef, sse


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

    # --------- joint two-frequency enumeration over a 12-point grid --------
    print("-" * 72)
    print("joint 1-/2-frequency enumeration (finite grid, weighted QR core):")
    n_j = 48
    rnd_j = lcg(424242, 2 * n_j)
    tj = sorted(18.0 * u for u in rnd_j[:n_j])
    g1, g2 = 0.20, 0.45
    yj = [
        1.25 * math.cos(TWO_PI * g1 * t - 0.60)
        + 0.70 * math.cos(TWO_PI * g2 * t + 1.10)
        + 0.30
        + 0.02 * 2.0 * (u - 0.5)
        for t, u in zip(tj, rnd_j[n_j:])
    ]
    jgrid = [0.05, 0.12, 0.20, 0.28, 0.36, 0.45, 0.52, 0.60, 0.68, 0.75,
             0.82, 0.90]
    assert len(jgrid) == 12
    t0 = time.perf_counter()
    jres = joint_periodogram_fit(tj, yj, None, jgrid)
    dt = time.perf_counter() - t0
    print(f"  {n_j} uneven observations; {jres.models_evaluated} subset "
          f"models enumerated in {dt:.3f}s")
    print(f"  injected (model A*cos(2*pi*f*t - delta)): "
          f"f1={g1:.2f} A=1.25 delta=+0.60 | "
          f"f2={g2:.2f} A=0.70 delta=-1.10 | c=0.30")
    print(f"  (fitted phase atan2(b,a) equals +delta)")
    print(f"  selected : order={jres.order} freqs={jres.frequencies} "
          f"rank={jres.rank}")
    for comp in jres.components:
        print(f"      f = {comp.frequency:5.2f}   A = {comp.amplitude:.4f}   "
              f"phase = {comp.phase:+.4f} rad")
    print(f"  constant = {jres.constant:.4f}   joint SSE = {jres.sse:.3e}   "
          f"power = {jres.power:.6f}")

    # independent 5x5 normal-equation check of the winning pair
    coef_ref, sse_ref = independent_joint_normal_equation_fit(
        tj, yj, list(jres.frequencies)
    )
    print("  independent 5x5 normal-equation check at the winning pair:")
    print(f"      SSE: QR={jres.sse:.10e}  normal-eq={sse_ref:.10e}  "
          f"diff={abs(jres.sse - sse_ref):.2e}")
    print(f"      constant: QR={jres.constant:.10f}  "
          f"normal-eq={coef_ref[0]:.10f}")

    # compare against the best single-frequency fit the pair must beat
    jres1 = joint_periodogram_fit(tj, yj, None, jgrid, max_frequencies=1)
    print(f"  best single: f={jres1.frequencies[0]:.2f} "
          f"SSE={jres1.sse:.3e} -> the pair reduces SSE by "
          f"{(1.0 - jres.sse / jres1.sse) * 100:.2f}%")

    # --------------------- boundary: exact alias degeneracy ----------------
    print("-" * 72)
    print("boundary -- exact alias pair on hourly samples (24/day cadence):")
    t_a = [k / 24.0 for k in range(24)]
    y_a = [1.4 * math.cos(TWO_PI * 3.0 * t - 0.35) for t in t_a]
    ra = joint_periodogram_fit(t_a, y_a, None, [3.0, 27.0])
    print(f"  grid (3.0, 27.0): columns are bit-identical on the samples "
          f"(27 = 24 + 3)")
    print(f"  selected freqs={ra.frequencies} order={ra.order} "
          f"rank={ra.rank} SSE={ra.sse:.2e} (lexicographic tie: single wins)")

    # ------------------- boundary: close resolvable frequencies ------------
    print("boundary -- close frequencies 0.20 vs 0.21 on a long baseline:")
    rnd_c = lcg(77, 40)
    t_c = sorted(30.0 * u for u in rnd_c)
    y_c = [
        1.3 * math.cos(TWO_PI * 0.20 * t - 0.5)
        + 0.6 * math.cos(TWO_PI * 0.21 * t + 1.2) + 0.7
        for t in t_c
    ]
    rc = joint_periodogram_fit(t_c, y_c, None, [0.19, 0.20, 0.21, 0.22])
    print(f"  selected freqs={rc.frequencies} rank={rc.rank} "
          f"SSE={rc.sse:.2e} (full rank, both resolved)")
    for comp in rc.components:
        print(f"      f = {comp.frequency:.2f}   A = {comp.amplitude:.4f}   "
              f"phase = {comp.phase:+.4f} rad")

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

    try:
        joint_periodogram_fit(tj, yj, None, jgrid, max_frequencies=3)
    except PeriodogramError as exc:
        print(f"  joint cap rejected  : PeriodogramError -> {exc}")
    else:
        raise SystemExit("FAIL: max_frequencies=3 was not rejected")

    big_grid = [0.1 * j for j in range(1, 14)]  # 13 candidates > cap of 12
    try:
        joint_periodogram_fit(tj, yj, None, big_grid)
    except PeriodogramError as exc:
        print(f"  grid cap rejected   : PeriodogramError -> {exc}")
    else:
        raise SystemExit("FAIL: 13-point joint grid was not rejected")

    print("=" * 72)
    print("demo finished normally.")


if __name__ == "__main__":
    main()
