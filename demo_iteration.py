"""Offline demonstration of continuous one-sine peak refinement.

Run with: python demo_iteration.py

The data, coarse grid, independent dense enumeration and finite-difference
check are all generated in this process.  There are no network calls,
credentials, sleeps, files or pre-recorded results.
"""

import math
import time

from periodogram import (
    PeriodogramError,
    periodogram,
    refine_peak,
    single_sine_derivative,
)

TWO_PI = 2.0 * math.pi


def lcg(seed, count):
    x = seed
    out = []
    for _ in range(count):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out.append(x / 0x80000000)
    return out


def independent_fit_sse(times, values, freq):
    """Small independent 3x3 normal-equation reference for one frequency."""
    mat = [[0.0] * 3 for _ in range(3)]
    vec = [0.0, 0.0, 0.0]
    for t, y in zip(times, values):
        angle = TWO_PI * freq * t
        g = (1.0, math.cos(angle), math.sin(angle))
        for i in range(3):
            for j in range(3):
                mat[i][j] += g[i] * g[j]
            vec[i] += y * g[i]
    a = [row[:] for row in mat]
    b = vec[:]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        divisor = a[col][col]
        for r in range(col + 1, 3):
            factor = a[r][col] / divisor
            for c in range(col, 3):
                a[r][c] -= factor * a[col][c]
            b[r] -= factor * b[col]
    coef = [0.0, 0.0, 0.0]
    for r in range(2, -1, -1):
        coef[r] = (b[r] - sum(
            a[r][c] * coef[c] for c in range(r + 1, 3)
        )) / a[r][r]
    constant, alpha, beta = coef
    sse = 0.0
    for t, y in zip(times, values):
        angle = TWO_PI * freq * t
        residual = y - constant - alpha * math.cos(angle) - beta * math.sin(angle)
        sse += residual * residual
    return sse


def independent_local_minimum(times, values, center, width=0.012, points=2401):
    best_sse = math.inf
    best_f = center
    for i in range(points):
        freq = center - width + 2.0 * width * i / (points - 1)
        sse = independent_fit_sse(times, values, freq)
        if sse < best_sse:
            best_sse, best_f = sse, freq
    return best_f, best_sse


def main():
    print("=" * 72)
    print("Continuous single-sine peak refinement demo")
    print("=" * 72)

    n = 45
    random_values = lcg(202, 2 * n)
    times = sorted(8.0 * value for value in random_values[:n])
    f1, f2 = 0.71325, 0.94761
    values = [
        1.15 * math.cos(TWO_PI * f1 * t - 0.3)
        + 0.90 * math.cos(TWO_PI * f2 * t + 0.7)
        for t in times
    ]
    f_lo, f_hi = 0.55, 1.15
    coarse_grid = [f_lo + 0.02 * i for i in range(31)]

    started = time.perf_counter()
    coarse = periodogram(times, values, None, coarse_grid)
    refined = refine_peak(
        times, values, None, f_lo, f_hi,
        evaluation_budget=260, scan_points=61,
    )
    elapsed = time.perf_counter() - started

    coarse_peak = max(
        coarse.top_peaks, key=lambda peak: peak.power
    )
    print(f"observations      : {n} irregular times over [{times[0]:.3f}, "
          f"{times[-1]:.3f}]")
    print(f"search interval   : [{f_lo:.3f}, {f_hi:.3f}] cycles/time")
    print(f"coarse grid       : {len(coarse_grid)} points, step 0.020; "
          f"best={coarse_peak.frequency:.6f}")
    print(f"refined segments  : {len(refined.candidates)}; evaluations="
          f"{refined.evaluations}/{refined.evaluation_budget}; "
          f"stop={refined.stopping_reason}")
    print(f"elapsed           : {elapsed:.3f}s")
    print("-" * 72)
    for candidate in refined.candidates:
        print(
            f"  f={candidate.frequency:.9f}  SSE={candidate.sse:.10f}  "
            f"A={candidate.amplitude:.8f}  phase={candidate.phase:+.8f}  "
            f"rank={candidate.rank}  stop={candidate.stopping_reason}"
        )

    best = refined.best
    independent_f, independent_sse = independent_local_minimum(
        times, values, best.frequency
    )
    print("-" * 72)
    print("independent local enumeration around the returned best peak:")
    print(f"  refined    f={best.frequency:.9f}  SSE={best.sse:.12f}")
    print(f"  enumerated f={independent_f:.9f}  SSE={independent_sse:.12f}")
    print(f"  frequency difference = {abs(best.frequency-independent_f):.3e}; "
          f"SSE difference = {abs(best.sse-independent_sse):.3e}")
    print(f"  endpoint preference   : {refined.endpoints.preferred} "
          f"(left SSE={refined.endpoints.left.sse:.9f}, "
          f"right SSE={refined.endpoints.right.sse:.9f})")

    derivative_info = single_sine_derivative(times, values, None, best.frequency)
    h = 1.0e-6
    finite_difference = (
        independent_fit_sse(times, values, best.frequency + h)
        - independent_fit_sse(times, values, best.frequency - h)
    ) / (2.0 * h)
    print("-" * 72)
    print("analytic profiled-SSE derivative vs independent finite difference:")
    print(f"  analytic        = {derivative_info.derivative: .6e}")
    print(f"  finite diff     = {finite_difference: .6e}")

    print("-" * 72)
    print("actually triggered invalid-input boundary:")
    try:
        refine_peak(times, values, None, f_hi, f_lo)
    except PeriodogramError as exc:
        print(f"  PeriodogramError -> {exc}")
    else:
        raise SystemExit("FAIL: inverted interval was not rejected")

    print("=" * 72)
    print("demo finished normally.")


if __name__ == "__main__":
    main()
