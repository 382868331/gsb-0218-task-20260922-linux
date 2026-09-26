"""Offline demo of refine_peak: off-grid frequency refinement.

Deterministic (LCG-generated) nightly-cadence observations with an injected
f = 1.7135/day sinusoid.  The 1/day cadence creates alias peaks at
f0 - 1 and f0 + 1, so the search interval holds several local optima.
Everything is computed live in well under 90 seconds; no network, accounts,
keys, sleeps or prerecorded output.
"""

import math
import time

from periodogram import refine_peak

TWO_PI = 2.0 * math.pi


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield state / (1 << 64)


def main():
    t_start = time.perf_counter()

    gen = lcg(20260926)
    times = []
    for night in range(35):
        for _ in range(3):  # 3 jittered visits per night, 1/day cadence
            times.append(night + 0.04 * next(gen))

    f0, amp, phase, offset = 1.7135, 1.25, 0.6, 0.4
    a0, b0 = amp * math.cos(phase), amp * math.sin(phase)
    values = [
        offset + a0 * math.cos(TWO_PI * f0 * t) + b0 * math.sin(TWO_PI * f0 * t)
        + 0.03 * (next(gen) - 0.5)
        for t in times
    ]

    f_lo, f_hi = 0.4, 3.0
    res = refine_peak(times, values, None, f_lo, f_hi)

    print("refine_peak demo: single sinusoid, irregular nightly sampling")
    print(f"  injected: f = {f0}/day, amplitude = {amp}, phase = {phase}")
    print(f"  interval [{f_lo}, {f_hi}] held {res.n_candidates} local SSE minima"
          f" (aliases at f0-1, f0, f0+1)")
    print(f"  refined frequency : {res.frequency:.10f}/day"
          f"  (error {abs(res.frequency - f0):.2e})")
    print(f"  amplitude / phase : {res.amplitude:.6f} / {res.phase:+.6f} rad")
    print(f"  constant / power  : {res.constant:.6f} / {res.power:.6f}")
    print(f"  residual SSE      : {res.sse:.6e}   (rank {res.rank})")
    print(f"  endpoint SSE      : low {res.sse_low:.4f} vs high {res.sse_high:.4f}"
          f"  -> boundary lower: {res.boundary_lower}")
    print(f"  stop reason       : {res.stop_reason}"
          f"  ({res.n_evaluations} objective evaluations)")

    # independent cross-check: dense scan with a hand-rolled normal-equations
    # solver (does not call refine_peak) must not beat the refined SSE
    def dense_sse(freq):
        ata = [[0.0] * 3 for _ in range(3)]
        atb = [0.0] * 3
        for t, y in zip(times, values):
            row = [1.0, math.cos(TWO_PI * freq * t), math.sin(TWO_PI * freq * t)]
            for j in range(3):
                atb[j] += row[j] * y
                for k in range(3):
                    ata[j][k] += row[j] * row[k]
        m = [ata[j] + [atb[j]] for j in range(3)]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
            m[col], m[piv] = m[piv], m[col]
            for r in range(3):
                if r != col:
                    factor = m[r][col] / m[col][col]
                    for k in range(col, 4):
                        m[r][k] -= factor * m[col][k]
        x = [m[j][3] / m[j][j] for j in range(3)]
        return sum(
            (y - x[0] - x[1] * math.cos(TWO_PI * freq * t)
             - x[2] * math.sin(TWO_PI * freq * t)) ** 2
            for t, y in zip(times, values)
        )

    best_grid = min(
        (f_lo + (f_hi - f_lo) * k / 2600 for k in range(2601)),
        key=dense_sse,
    )
    print(f"  independent dense scan (2601 pts): best grid f = {best_grid:.6f},"
          f" SSE {dense_sse(best_grid):.6e} >= refined {res.sse:.6e}:"
          f" {dense_sse(best_grid) >= res.sse - 1e-12}")

    # a deliberately starved budget reports honestly instead of pretending
    starved = refine_peak(times, values, None, f_lo, f_hi, eval_budget=90)
    print(f"  starved budget=90 : stop={starved.stop_reason},"
          f" best-so-far f = {starved.frequency:.6f}"
          f" (evaluations {starved.n_evaluations} <= 90)")

    elapsed = time.perf_counter() - t_start
    print(f"  elapsed: {elapsed:.2f}s (budget 90s, fully offline)")


if __name__ == "__main__":
    main()

