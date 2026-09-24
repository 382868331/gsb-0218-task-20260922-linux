"""Fixed-input demonstration of the floating-mean periodogram module.

Run with:  python demo.py

The script performs real computation (no sleeps, no pre-recorded results):

1. normal result  -- a deterministic unevenly sampled light curve with a
   nightly observing cadence produces a genuine double-peak alias
   ambiguity (true f0 and f0 +/- 1/day); the sampling window explains it;
2. rejection path -- a NaN observation is actually rejected with a
   locatable PeriodogramError (plus a bool-as-int parameter rejection);
3. joint enumeration -- joint_fit exhaustively fits one/two-frequency
   models on a <= 12-point grid, recovers an injected two-tone signal
   (checked against an independent 5x5 normal-equation fit), shows a
   rank-deficient alias pair, and actually raises on a 13-point grid.
"""

import math
import time

from periodogram import PeriodogramError, joint_fit, periodogram

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


def independent_joint_fit(times, values, freqs):
    """Independent (3 or 5)-column weighted normal-equation joint fit.

    Deliberately a different algorithm from the module's Householder QR;
    used only as a live spot check in the joint-enumeration section.
    """
    p = 1 + 2 * len(freqs)

    def basis(t):
        g = [1.0]
        for fr in freqs:
            phi = TWO_PI * fr * t
            g.extend((math.cos(phi), math.sin(phi)))
        return g

    mat = [[0.0] * p for _ in range(p)]
    vec = [0.0] * p
    for t, y in zip(times, values):
        g = basis(t)
        for i in range(p):
            for j in range(p):
                mat[i][j] += g[i] * g[j]
            vec[i] += y * g[i]
    a = [row[:] for row in mat]
    b = vec[:]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, p):
            q = a[r][col] / a[col][col]
            for k in range(col, p):
                a[r][k] -= q * a[col][k]
            b[r] -= q * b[col]
    coef = [0.0] * p
    for r in range(p - 1, -1, -1):
        coef[r] = (b[r] - sum(a[r][k] * coef[k]
                              for k in range(r + 1, p))) / a[r][r]
    sse = 0.0
    for t, y in zip(times, values):
        g = basis(t)
        sse += (y - sum(coef[c] * g[c] for c in range(p))) ** 2
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

    # ==================== joint enumeration (one/two freqs) =================
    print("=" * 72)
    print("joint_fit: exhaustive one/two-frequency enumeration (<= 12 grid)")
    print("=" * 72)

    rnd_j = lcg(424242, 600)
    n_j = 40
    jtimes = sorted(40.0 * rnd_j[k] for k in range(n_j))
    f1j, f2j, a1j, a2j, d1j, d2j = 0.20, 0.37, 1.2, 0.8, 0.5, -0.9
    jvalues = [
        a1j * math.cos(TWO_PI * f1j * t - d1j)
        + a2j * math.cos(TWO_PI * f2j * t - d2j)
        + 0.02 * 2.0 * (rnd_j[n_j + k] - 0.5)
        for k, t in enumerate(jtimes)
    ]
    jgrid = [0.11, 0.14, 0.17, 0.20, 0.26, 0.31, 0.37, 0.43]
    print(f"observations         : {n_j} uneven samples over "
          f"{jtimes[-1] - jtimes[0]:.1f} time units")
    print(f"injected tones       : f={f1j:.2f} (A={a1j}) and "
          f"f={f2j:.2f} (A={a2j})")
    print(f"candidate grid       : {jgrid}")

    t0j = time.perf_counter()
    jres = joint_fit(jtimes, jvalues, None, jgrid)
    dtj = time.perf_counter() - t0j
    n_single = len(jgrid)
    n_pairs = len(jgrid) * (len(jgrid) - 1) // 2
    print(f"enumerated {len(jres.combinations)} models "
          f"({n_single} singles + {n_pairs} pairs) in {dtj:.4f}s; "
          f"SSE0={jres.sse0:.5f}")
    print("five smallest-SSE models:")
    for c in jres.combinations[:5]:
        fs = ", ".join(f"{x:.2f}" for x in c.frequencies)
        amps = ", ".join(f"{x:.3f}" for x in c.amplitudes)
        print(f"  ({fs})  rank={c.rank}  SSE={c.sse:.6f}  "
              f"power={c.power:.5f}  amplitudes=({amps})")
    best = jres.best
    print(f"-> selected best: {tuple(round(x, 2) for x in best.frequencies)} "
          f"(lowest SSE, frequency-tuple tie-break)")
    assert best.frequencies == (f1j, f2j), "injected pair must win"
    assert abs(best.amplitudes[0] - a1j) < 0.01
    assert abs(best.amplitudes[1] - a2j) < 0.01

    # independent 5x5 normal-equation spot check of the winning pair
    coef_ref, sse_ref = independent_joint_fit(
        jtimes, jvalues, list(best.frequencies)
    )
    amp_ref = (math.hypot(coef_ref[1], coef_ref[2]),
               math.hypot(coef_ref[3], coef_ref[4]))
    print("-" * 72)
    print("independent 5x5 normal-equation spot check of the winning pair:")
    print(f"  SSE       : QR={best.sse:.10f}  normal-eq={sse_ref:.10f}  "
          f"diff={abs(best.sse - sse_ref):.2e}")
    for pos in range(2):
        print(f"  tone {pos + 1} amp: QR={best.amplitudes[pos]:.10f}  "
              f"normal-eq={amp_ref[pos]:.10f}  "
              f"diff={abs(best.amplitudes[pos] - amp_ref[pos]):.2e}")

    # ---- boundary 1: close but resolvable frequencies (ill-conditioned) ----
    print("-" * 72)
    print("boundary: close frequencies df=0.03 (beat ~ 33 units < 40 span)")
    ctimes = jtimes
    cvalues = [
        a1j * math.cos(TWO_PI * 0.20 * t - d1j)
        + a2j * math.cos(TWO_PI * 0.23 * t - d2j)
        + 0.01 * 2.0 * (rnd_j[200 + k] - 0.5)
        for k, t in enumerate(ctimes)
    ]
    cres = joint_fit(ctimes, cvalues, None, [0.14, 0.17, 0.20, 0.23, 0.29])
    csingle = min(
        (c for c in cres.combinations if len(c.frequencies) == 1),
        key=lambda c: c.sse,
    )
    print(f"  best single SSE={csingle.sse:.5f} ({csingle.frequencies}); "
          f"best pair SSE={cres.best.sse:.6f} "
          f"{tuple(round(x, 2) for x in cres.best.frequencies)} rank="
          f"{cres.best.rank}")
    print(f"  pair/single SSE ratio = {cres.best.sse / csingle.sse:.2e} "
          f"-> pair resolved despite the 0.03 separation")

    # ---- boundary 2: alias pair whose design columns collapse --------------
    print("-" * 72)
    print("boundary: sampling alias pair (24 hourly samples, f=3 & f=21/day)")
    n_a = 24
    dt_a = 1.0 / 24.0
    f0a, mirror_a = 3.0, 1.0 / dt_a - 3.0
    atimes = [k * dt_a for k in range(n_a)]
    avalues = [1.4 * math.cos(TWO_PI * f0a * t - 0.35) for t in atimes]
    agrid = [1.0, f0a, 10.0, mirror_a, 25.0]
    ares = joint_fit(atimes, avalues, None, agrid)
    apair = next(c for c in ares.combinations
                 if c.frequencies == (f0a, mirror_a))
    asingle = next(c for c in ares.combinations
                   if c.frequencies == (f0a,))
    print(f"  single ({f0a:.0f})      : rank={asingle.rank} "
          f"SSE={asingle.sse:.2e} amplitude={asingle.amplitudes[0]:.4f}")
    print(f"  pair   ({f0a:.0f},{mirror_a:.0f})  : rank={apair.rank} "
          f"(columns coincide on the grid -> null direction dropped, "
          f"no error)")
    print(f"  pair SSE={apair.sse:.2e} ties the single; second amplitude="
          f"{apair.amplitudes[1]:.2e} (reported as zero, not invented)")
    # n=4 observations vs 5 model columns: rank can never exceed 4
    tiny = joint_fit([0.0, 0.3, 1.1, 2.0], [0.4, 1.1, 0.2, 1.6],
                     [1.0, 0.7, 1.3, 1.0], [0.25, 0.6, 0.95])
    print(f"  4 observations / 5 columns: best rank={tiny.best.rank} "
          f"(<= n), fit still well-defined, SSE={tiny.best.sse:.4f}")

    # ------- actually triggered rejections for the joint entry -------------
    print("-" * 72)
    print("joint_fit error boundaries (real exceptions, actually raised):")
    try:
        joint_fit(jtimes, jvalues, None,
                  [0.1 * k for k in range(1, 14)])  # 13 > 12
    except PeriodogramError as exc:
        print(f"  13-point grid rejected: PeriodogramError -> {exc}")
    else:
        raise SystemExit("FAIL: 13-point grid was not rejected")

    bad_grid = [0.2, 0.5, 0.5]
    try:
        joint_fit(jtimes, jvalues, None, bad_grid)
    except PeriodogramError as exc:
        print(f"  non-strict grid rejected: PeriodogramError -> {exc}")
    else:
        raise SystemExit("FAIL: duplicated grid frequency was not rejected")

    print("=" * 72)
    print("demo finished normally.")


if __name__ == "__main__":
    main()
