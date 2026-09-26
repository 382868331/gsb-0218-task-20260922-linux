"""Floating-mean periodogram for unevenly sampled time series.

For every candidate frequency ``f`` the model

    y_k ~= c + a*cos(2*pi*f*t_k) + b*sin(2*pi*f*t_k)

is fitted by *weighted Householder QR* (every observation is one matrix row;
observations are never placed on a uniform grid, zero-padded or FFT-ed).  The
power is the fractional weighted-SSE reduction relative to the constant-only
model, ``P = (SSE0 - SSE(f)) / SSE0``.

The sampling window ``W(f) = sum_k w_k * exp(i*2*pi*f*t_k)`` is also computed;
its modulus normalized by ``sum w_k`` explains alias structure.

Standard library only (Python 3.14.7).  See README.md for the full contract,
numerical error budget and known limitations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

__all__ = [
    "periodogram",
    "refine_peak",
    "PeriodogramResult",
    "RefinedPeak",
    "FrequencyPeak",
    "PeriodogramError",
    "EPS",
    "MAX_OBSERVATIONS",
    "MAX_FREQUENCIES",
    "MIN_OBSERVATIONS",
]

__version__ = "1.1.0"

# machine epsilon of binary64 (Python floats are IEEE-754 binary64)
EPS = 2.220446049250313e-16

MAX_OBSERVATIONS = 200
MAX_FREQUENCIES = 300
MIN_OBSERVATIONS = 3  # the model has three parameters (c, a, b)

_TWO_PI = 2.0 * math.pi


class PeriodogramError(ValueError):
    """Invalid input or contract violation.

    The message always names the offending parameter (and, for array inputs,
    the element index), so callers can locate the problem.  Nothing is
    silently corrected.
    """


@dataclass(frozen=True)
class FrequencyPeak:
    """One candidate-frequency bin of the spectrum."""

    index: int
    frequency: float
    power: float          # (SSE0 - SSE) / SSE0
    amplitude: float      # hypot(a, b) of c + a*cos + b*sin
    phase: float          # atan2(b, a) in radians, wrapped to (-pi, pi]
    rank: int             # numerical rank of the weighted 3-column design
    sse: float            # weighted residual SSE at this frequency
    window_modulus: float  # |sum w exp(i2pi f t)| / sum w


@dataclass(frozen=True)
class RefinedPeak:
    """Result of :func:`refine_peak` (immutable).

    `frequency` is the refined (generally off-grid) minimizer of the
    eliminated-coefficient residual SSE over the requested interval, chosen
    among the locally refined candidates; the linear coefficients
    (c, a, b) at that frequency are reported as `constant`,
    `amplitude = hypot(a, b)` and `phase = atan2(b, a)` in the original
    time frame.  `sse_low`/`sse_high` are the residual SSE values at the
    interval endpoints and `boundary_lower` flags that an endpoint beats
    the refined interior minimum (the interval then likely does not bracket
    the global peak).
    """

    frequency: float
    power: float          # (SSE0 - SSE) / SSE0 at the refined frequency
    amplitude: float
    phase: float
    constant: float
    sse: float
    rank: int             # numerical rank of the design at the solution
    sse_low: float        # residual SSE at f_low
    sse_high: float       # residual SSE at f_high
    boundary_lower: bool  # an endpoint SSE is below the refined minimum
    n_evaluations: int    # objective evaluations actually spent
    n_candidates: int     # interior candidates found by the coarse scan
    # "converged": every candidate was refined to the requested tolerance;
    # "budget_exhausted": eval_budget ran out, the result is the best seen
    #             so far and is NOT guaranteed globally optimal;
    # "boundary_minimum": no full-rank interior candidate exists; the best
    #             evaluated full-rank grid point (usually an endpoint) is
    #             reported instead of a refined peak.
    stop_reason: str


@dataclass(frozen=True)
class PeriodogramResult:
    """Full result of :func:`periodogram` (immutable)."""

    frequencies: tuple[float, ...]
    powers: tuple[float, ...]
    amplitudes: tuple[float, ...]
    phases: tuple[float, ...]
    ranks: tuple[int, ...]
    constants: tuple[float, ...]  # fitted c at every frequency
    sses: tuple[float, ...]       # weighted SSE at every frequency
    sse0: float                   # weighted SSE of the constant model
    weighted_mean: float          # constant-model fit
    # complex sampling window; its complex phase is referenced to t0 =
    # times[0] (internal time recentering), while |window|/sum(w) -- the
    # origin-invariant quantity used for alias analysis -- is window_modulus.
    window: tuple[complex, ...]
    window_modulus: tuple[float, ...]
    local_peaks: tuple[FrequencyPeak, ...]  # strict local power maxima
    top_peaks: tuple[FrequencyPeak, ...]    # greedy exclusion top-K
    top_k_requested: int
    top_k_found: int
    # "complete": every candidate was examined; if top_k_found < requested,
    #             no further bin can be admitted under the (deterministic)
    #             greedy exclusion rule -- an exhaustive-scan proof.
    # "budget_exhausted": selection_budget ran out before the scan finished;
    #             nothing is claimed about unexamined bins.
    selection_status: str
    min_separation: float
    selection_budget: int


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def _int_param(name: str, value: object, *, minimum: int | None = None) -> int:
    # bool is a subclass of int but is never a legitimate integer parameter
    if isinstance(value, bool) or not isinstance(value, int):
        raise PeriodogramError(
            f"{name}: expected an int, got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise PeriodogramError(f"{name}: expected >= {minimum}, got {value}")
    return value


def _float_param(name: str, value: object, *, lower: float, upper: float | None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PeriodogramError(
            f"{name}: expected a real number, got {type(value).__name__}"
        )
    x = float(value)
    if math.isnan(x) or math.isinf(x):
        raise PeriodogramError(f"{name}: NaN/Infinity is not allowed, got {value!r}")
    if x < lower or (upper is not None and x > upper):
        hi = "" if upper is None else f", <= {upper}"
        raise PeriodogramError(f"{name}: expected >= {lower}{hi}, got {value!r}")
    return x


def _real_array(name: str, obj: object, *, strictly_positive: bool = False) -> list[float]:
    if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes, bytearray)):
        raise PeriodogramError(
            f"{name}: expected a list or tuple of real numbers, "
            f"got {type(obj).__name__}"
        )
    out: list[float] = []
    for i, item in enumerate(obj):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise PeriodogramError(
                f"{name}[{i}]: expected a real number, got {type(item).__name__}"
            )
        x = float(item)
        if math.isnan(x):
            raise PeriodogramError(f"{name}[{i}]: NaN is not allowed")
        if math.isinf(x):
            raise PeriodogramError(f"{name}[{i}]: Infinity is not allowed")
        if strictly_positive and x <= 0.0:
            raise PeriodogramError(
                f"{name}[{i}]: expected a strictly positive value, got {x}"
            )
        out.append(x)
    return out


# --------------------------------------------------------------------------- #
# weighted Householder QR (m x n, n <= 3), no external numeric libraries
# --------------------------------------------------------------------------- #
def _weighted_qr(
    rows: list[list[float]], values: list[float], sqrt_w: list[float], ncols: int
) -> tuple[list[list[float]], list[float], list[float]]:
    """Return (R, qty, pivots) of the weighted least-squares problem.

    R is upper triangular (below diagonal zeroed), ``qty = Q.T * W**0.5 * y``.
    The residual norm squared is ``sum(qty[ncols:]**2)``; ``pivots[j]`` is the
    j-th diagonal of R with its Householder sign.
    """
    m = len(rows)
    r = [[rows[i][j] * sqrt_w[i] for j in range(ncols)] for i in range(m)]
    b = [values[i] * sqrt_w[i] for i in range(m)]
    pivots: list[float] = []

    for j in range(ncols):
        # scaled 2-norm of the trailing column (overflow resistant)
        scale = 0.0
        for i in range(j, m):
            a = abs(r[i][j])
            if a > scale:
                scale = a
        if scale == 0.0:
            pivots.append(0.0)
            continue  # exactly zero column in the reduced submatrix
        norm = 0.0
        for i in range(j, m):
            q = r[i][j] / scale
            norm += q * q
        norm = scale * math.sqrt(norm)

        x0 = r[j][j]
        alpha = -norm if x0 >= 0.0 else norm
        v0 = x0 - alpha
        vtv = v0 * v0
        for i in range(j + 1, m):
            vtv += r[i][j] * r[i][j]
        beta = 2.0 / vtv

        for k in range(j + 1, ncols):
            s = v0 * r[j][k]
            for i in range(j + 1, m):
                s += r[i][j] * r[i][k]
            s *= beta
            r[j][k] -= s * v0
            for i in range(j + 1, m):
                r[i][k] -= s * r[i][j]

        s = v0 * b[j]
        for i in range(j + 1, m):
            s += r[i][j] * b[i]
        s *= beta
        b[j] -= s * v0
        for i in range(j + 1, m):
            b[i] -= s * r[i][j]

        r[j][j] = alpha
        for i in range(j + 1, m):
            r[i][j] = 0.0
        pivots.append(alpha)

    return r, b, pivots


def _solve_triangular(
    r: list[list[float]], b: list[float], pivots: list[float], rank_tol: float
) -> tuple[list[float], int]:
    n = len(pivots)
    scale = max(abs(p) for p in pivots)
    threshold = rank_tol * scale
    x = [0.0] * n
    rank = 0
    for j in range(n - 1, -1, -1):
        if abs(pivots[j]) <= threshold:
            x[j] = 0.0  # null-direction coefficient is defined as 0
            continue
        rank += 1
        s = b[j]
        for k in range(j + 1, n):
            s -= r[j][k] * x[k]
        x[j] = s / pivots[j]
    return x, rank


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def periodogram(
    times,
    values,
    weights,
    frequencies,
    *,
    top_k: int | None = None,
    min_separation: float = 0.0,
    rank_tol: float | None = None,
    selection_budget: int | None = None,
) -> PeriodogramResult:
    """Floating-mean weighted periodogram on a fixed candidate frequency grid.

    Parameters
    ----------
    times, values:
        Lists/tuples of finite real numbers, equal length, 3 <= n <= 200.
        Times need not be sorted and may repeat; they are never gridded.
    weights:
        Per-observation strictly positive finite weights, or ``None`` for
        unit weights.  A global weight rescaling changes nothing.
    frequencies:
        Strictly increasing, strictly positive finite candidate frequencies,
        length 1..300 (cycles per time unit).
    top_k:
        Positive int, <= len(frequencies).  Peaks are chosen greedily by
        descending power; a bin is excluded once a chosen bin lies at
        ``|df| < min_separation`` (``|df| == min_separation`` is admitted).
        ``None`` (the default) means ``min(5, len(frequencies))``; an
        explicitly out-of-range value is always rejected.
    min_separation:
        Nonnegative finite exclusion distance in frequency units.
    rank_tol:
        Pivot threshold for numerical rank, ``|R_jj| > rank_tol*max|R_ii|``.
        Default ``n * EPS``.  Override must lie in (0, 1).
    selection_budget:
        Maximum number of candidate bins examined while selecting top-K.
        ``None`` (= number of frequencies) gives an exhaustive scan.  A cut
        short scan is reported as ``selection_status="budget_exhausted"`` and
        is distinct from a complete scan that provably yields fewer picks.

    Raises
    ------
    PeriodogramError
        For every contract violation; the message names the parameter and
        element index.  No computation starts before all inputs validate.
    """
    if top_k is not None:
        top_k = _int_param("top_k", top_k, minimum=1)
    min_separation = _float_param(
        "min_separation", min_separation, lower=0.0, upper=None
    )
    if rank_tol is None:
        rank_tol = None  # resolved once n is known
    else:
        rank_tol = _float_param("rank_tol", rank_tol, lower=EPS, upper=1.0 - EPS)
    if selection_budget is not None:
        selection_budget = _int_param(
            "selection_budget", selection_budget, minimum=1
        )

    t = _real_array("times", times)
    y = _real_array("values", values)
    if weights is None:
        w = [1.0] * len(t)
    else:
        w = _real_array("weights", weights, strictly_positive=True)
    f = _real_array("frequencies", frequencies, strictly_positive=True)

    n, m = len(t), len(f)
    if not (MIN_OBSERVATIONS <= n <= MAX_OBSERVATIONS):
        raise PeriodogramError(
            f"times: expected between {MIN_OBSERVATIONS} and {MAX_OBSERVATIONS} "
            f"observations, got {n}"
        )
    if len(y) != n:
        raise PeriodogramError(
            f"values: length {len(y)} does not match times length {n}"
        )
    if len(w) != n:
        raise PeriodogramError(
            f"weights: length {len(w)} does not match times length {n}"
        )
    if not (1 <= m <= MAX_FREQUENCIES):
        raise PeriodogramError(
            f"frequencies: expected between 1 and {MAX_FREQUENCIES} candidates, "
            f"got {m}"
        )
    for i in range(1, m):
        if not (f[i] > f[i - 1]):
            raise PeriodogramError(
                f"frequencies[{i}]: candidates must be strictly increasing; "
                f"{f[i]} follows {f[i - 1]}"
            )
    if top_k is None:
        top_k = min(5, m)
    elif top_k > m:
        raise PeriodogramError(
            f"top_k: {top_k} exceeds number of candidate frequencies {m}"
        )
    if rank_tol is None:
        rank_tol = n * EPS
    if selection_budget is None:
        selection_budget = m

    # Time-origin recentering for numerical stability.  The model is fitted in
    # tau = t - t0 (t0 = times[0]); a, b are rotated back to the original time
    # frame before amplitude/phase are reported.  Power/amplitude and the
    # window modulus are exactly origin-invariant.
    t0 = t[0]
    tau = [tk - t0 for tk in t]
    sqrt_w = [math.sqrt(wk) for wk in w]
    weights_sum = sum(w)

    # ----- constant model: weighted QR with a single ones column -----------
    ones = [[1.0] for _ in range(n)]
    r0, b0, p0 = _weighted_qr(ones, y, sqrt_w, 1)
    constant_data = all(v == y[0] for v in y)
    if constant_data:
        weighted_mean = float(y[0])
    else:
        weighted_mean = b0[0] / p0[0]
    # Contract: a bitwise-constant value series has zero variance; its power
    # is defined as 0.0 at every frequency (documented convention, not a
    # silent roundoff patch).
    sse0 = 0.0 if constant_data else sum(z * z for z in b0[1:])

    # ------------------------- per-frequency scan --------------------------
    powers: list[float] = [0.0] * m
    amplitudes: list[float] = [0.0] * m
    phases: list[float] = [0.0] * m
    ranks: list[int] = [0] * m
    constants: list[float] = [0.0] * m
    sses: list[float] = [0.0] * m
    window: list[complex] = [0.0j] * m
    window_mod: list[float] = [0.0] * m
    rows: list[list[float]] = [[1.0, 0.0, 0.0] for _ in range(n)]

    for fi, freq in enumerate(f):
        for i in range(n):
            angle = _TWO_PI * freq * tau[i]
            rows[i][1] = math.cos(angle)
            rows[i][2] = math.sin(angle)

        r, qb, pivots = _weighted_qr(rows, y, sqrt_w, 3)
        coef, rank = _solve_triangular(r, qb, pivots, rank_tol)
        if constant_data:
            # Documented exact convention for bitwise-constant data:
            # c = y0, a = b = 0, SSE = 0 at every frequency.
            c_fit, a_tau, b_tau = float(y[0]), 0.0, 0.0
            sse = 0.0
        else:
            c_fit, a_tau, b_tau = coef[0], coef[1], coef[2]
            # Evaluate the model SSE directly.  Reading qb[3:] would be wrong
            # when a column was skipped as numerically zero: its residual
            # energy then lives in qb[1:]/qb[2:].  The fitted values are the
            # orthogonal projection for any rank, so this is the unambiguous
            # model residual (also stable at n <= 200).
            sse = 0.0
            for i in range(n):
                resid = y[i] - c_fit - a_tau * rows[i][1] - b_tau * rows[i][2]
                sse += w[i] * resid * resid

        # rotate (a, b) from the tau frame back into the original t frame
        theta = _TWO_PI * freq * t0
        ct, st = math.cos(theta), math.sin(theta)
        a_fit = a_tau * ct - b_tau * st
        b_fit = a_tau * st + b_tau * ct

        # sampling window in the tau frame (modulus is origin-invariant)
        wre = 0.0
        wim = 0.0
        for i in range(n):
            wre += w[i] * rows[i][1]
            wim += w[i] * rows[i][2]
        wmod = math.hypot(wre, wim) / weights_sum

        powers[fi] = 0.0 if constant_data else (sse0 - sse) / sse0
        amplitudes[fi] = math.hypot(a_fit, b_fit)
        ph = math.atan2(b_fit, a_fit)
        phases[fi] = ph
        ranks[fi] = rank
        constants[fi] = c_fit
        sses[fi] = sse
        window[fi] = complex(wre, wim)
        window_mod[fi] = wmod

    # ---------------- strict local maxima (endpoints never peak) -----------
    local_peaks: list[FrequencyPeak] = []
    for i in range(1, m - 1):
        if powers[i] > powers[i - 1] and powers[i] > powers[i + 1]:
            local_peaks.append(_peak(i, f, powers, amplitudes, phases,
                                     ranks, sses, window_mod))

    # --- greedy top-K with explicit frequency exclusion (NMS by power) -----
    order = sorted(range(m), key=lambda i: (-powers[i], f[i]))
    chosen: list[int] = []
    budget = selection_budget
    status = "complete"
    for i in order:
        if budget <= 0:
            status = "budget_exhausted"
            break
        budget -= 1
        if all(abs(f[i] - f[j]) >= min_separation for j in chosen):
            chosen.append(i)
            if len(chosen) == top_k:
                break
    top_peaks = [
        _peak(i, f, powers, amplitudes, phases, ranks, sses, window_mod)
        for i in chosen
    ]

    return PeriodogramResult(
        frequencies=tuple(f),
        powers=tuple(powers),
        amplitudes=tuple(amplitudes),
        phases=tuple(phases),
        ranks=tuple(ranks),
        constants=tuple(constants),
        sses=tuple(sses),
        sse0=sse0,
        weighted_mean=weighted_mean,
        window=tuple(window),
        window_modulus=tuple(window_mod),
        local_peaks=tuple(local_peaks),
        top_peaks=tuple(top_peaks),
        top_k_requested=top_k,
        top_k_found=len(top_peaks),
        selection_status=status,
        min_separation=min_separation,
        selection_budget=selection_budget,
    )


def _peak(i, f, powers, amplitudes, phases, ranks, sses, window_mod) -> FrequencyPeak:
    return FrequencyPeak(
        index=i,
        frequency=f[i],
        power=powers[i],
        amplitude=amplitudes[i],
        phase=phases[i],
        rank=ranks[i],
        sse=sses[i],
        window_modulus=window_mod[i],
    )


# --------------------------------------------------------------------------- #
# off-grid peak refinement
# --------------------------------------------------------------------------- #
def _fit_model(
    freq: float,
    tau: list[float],
    y: list[float],
    w: list[float],
    sqrt_w: list[float],
    rank_tol: float,
    constant_data: bool,
    y0: float,
) -> tuple[float, float, float, int, float]:
    """Fit c + a*cos + b*sin at one frequency; return (c, a_tau, b_tau, rank, sse).

    This is the same weighted Householder QR fit used by :func:`periodogram`;
    the coefficients are in the recentered tau = t - times[0] frame.
    """
    rows = [[1.0, 0.0, 0.0] for _ in range(len(tau))]
    for i in range(len(tau)):
        angle = _TWO_PI * freq * tau[i]
        rows[i][1] = math.cos(angle)
        rows[i][2] = math.sin(angle)
    r, qb, pivots = _weighted_qr(rows, y, sqrt_w, 3)
    coef, rank = _solve_triangular(r, qb, pivots, rank_tol)
    if constant_data:
        return float(y0), 0.0, 0.0, rank, 0.0
    c_fit, a_tau, b_tau = coef[0], coef[1], coef[2]
    sse = 0.0
    for i in range(len(tau)):
        resid = y[i] - c_fit - a_tau * rows[i][1] - b_tau * rows[i][2]
        sse += w[i] * resid * resid
    return c_fit, a_tau, b_tau, rank, sse


def refine_peak(
    times,
    values,
    weights,
    f_low,
    f_high,
    *,
    segments: int = 64,
    eval_budget: int | None = None,
    rank_tol: float | None = None,
    xtol: float | None = None,
) -> RefinedPeak:
    """Refine the best single-sinusoid frequency inside [f_low, f_high].

    The objective is the residual SSE of the floating-mean model
    ``y = c + a*cos(2*pi*f*t) + b*sin(2*pi*f*t)`` with the linear
    coefficients (c, a, b) eliminated by the same weighted QR fit as
    :func:`periodogram` -- the grid maximum of the power is *not* treated
    as the continuous optimum.

    Procedure: the interval is scanned on a uniform ``segments``-cell grid;
    every strict interior SSE minimum with a full-rank design becomes a
    candidate and is refined by golden-section search inside its two
    neighbouring cells.  The best refined candidate wins.  Because the
    interval may hold several peaks and ``eval_budget`` is finite, an
    exhausted budget does not guarantee the global optimum (reported as
    ``stop_reason="budget_exhausted"``).  Rank-deficient positions are
    never accepted as peaks.  Only a single-sinusoid model is fitted; no
    multi-frequency model is introduced.

    Parameters
    ----------
    times, values, weights:
        Same contract as :func:`periodogram`.
    f_low, f_high:
        Finite interval endpoints with ``0 < f_low < f_high``.
    segments:
        Number of coarse scan cells (int >= 4, bool rejected).
    eval_budget:
        Maximum number of objective (SSE) evaluations, including the
        coarse scan.  ``None`` defaults to ``8 * segments + 64``.
    rank_tol:
        Numerical rank threshold, same meaning as in :func:`periodogram`.
    xtol:
        Frequency tolerance for the golden-section refinement; default
        ``1e-10 * (f_high - f_low)``.
    """
    segments = _int_param("segments", segments, minimum=4)
    if eval_budget is not None:
        eval_budget = _int_param("eval_budget", eval_budget, minimum=1)
    if rank_tol is not None:
        rank_tol = _float_param("rank_tol", rank_tol, lower=EPS, upper=1.0 - EPS)
    if xtol is not None:
        xtol = _float_param("xtol", xtol, lower=EPS, upper=None)
    flo = _float_param("f_low", f_low, lower=EPS, upper=None)
    fhi = _float_param("f_high", f_high, lower=EPS, upper=None)
    if not fhi > flo:
        raise PeriodogramError(
            f"f_high: expected > f_low ({flo}), got {fhi}"
        )

    t = _real_array("times", times)
    y = _real_array("values", values)
    if weights is None:
        w = [1.0] * len(t)
    else:
        w = _real_array("weights", weights, strictly_positive=True)
    n = len(t)
    if not (MIN_OBSERVATIONS <= n <= MAX_OBSERVATIONS):
        raise PeriodogramError(
            f"times: expected between {MIN_OBSERVATIONS} and {MAX_OBSERVATIONS} "
            f"observations, got {n}"
        )
    if len(y) != n:
        raise PeriodogramError(
            f"values: length {len(y)} does not match times length {n}"
        )
    if len(w) != n:
        raise PeriodogramError(
            f"weights: length {len(w)} does not match times length {n}"
        )
    if rank_tol is None:
        rank_tol = n * EPS
    if eval_budget is None:
        eval_budget = 16 * segments + 64
    if xtol is None:
        xtol = 1e-10 * (fhi - flo)

    t0 = t[0]
    tau = [tk - t0 for tk in t]
    sqrt_w = [math.sqrt(wk) for wk in w]
    constant_data = all(v == y[0] for v in y)

    ones = [[1.0] for _ in range(n)]
    r0, b0, p0 = _weighted_qr(ones, y, sqrt_w, 1)
    sse0 = 0.0 if constant_data else sum(z * z for z in b0[1:])

    spent = [0]  # mutable evaluation counter shared by the objective

    def evaluate(freq: float) -> tuple[float, float, float, int, float]:
        spent[0] += 1
        return _fit_model(freq, tau, y, w, sqrt_w, rank_tol, constant_data, y[0])

    # ------------------------- coarse segmented scan -----------------------
    step = (fhi - flo) / segments
    grid_f = [flo + i * step for i in range(segments + 1)]
    grid_sse: list[float] = [0.0] * (segments + 1)
    grid_rank: list[int] = [0] * (segments + 1)
    for i, gf in enumerate(grid_f):
        if spent[0] >= eval_budget:
            # Budget cannot even cover the coarse scan: report the best
            # grid point evaluated so far, honestly flagged.
            return _refined_result(
                grid_f, grid_sse, grid_rank, i, flo, fhi,
                tau, y, w, sqrt_w, rank_tol, constant_data,
                sse0, t0, spent, 0, "budget_exhausted",
            )
        _, _, _, rk, sse = evaluate(gf)
        grid_sse[i] = sse
        grid_rank[i] = rk

    sse_low, sse_high = grid_sse[0], grid_sse[segments]

    # -------- interior candidates: strict local SSE minima, full rank ------
    candidates: list[int] = []
    for i in range(1, segments):
        if grid_rank[i] < 3:
            continue  # rank-deficient positions are never good peaks
        if (
            grid_sse[i] <= grid_sse[i - 1]
            and grid_sse[i] <= grid_sse[i + 1]
            and (grid_sse[i] < grid_sse[i - 1] or grid_sse[i] < grid_sse[i + 1])
        ):
            candidates.append(i)

    # refine the most promising candidates first: with a finite budget the
    # best-so-far result is then meaningful even if the budget runs out
    candidates.sort(key=lambda i: grid_sse[i])

    if not candidates:
        return _refined_result(
            grid_f, grid_sse, grid_rank, segments + 1, flo, fhi,
            tau, y, w, sqrt_w, rank_tol, constant_data,
            sse0, t0, spent, 0, "boundary_minimum",
        )

    # ---------------- golden-section refinement per candidate --------------
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    stop = "converged"
    best_freq = 0.0
    best_sse = math.inf

    for ci in candidates:
        lo = grid_f[ci - 1]
        hi = grid_f[ci + 1]
        c = hi - invphi * (hi - lo)
        d = lo + invphi * (hi - lo)
        if spent[0] >= eval_budget:
            stop = "budget_exhausted"
            break
        fc = evaluate(c)[4]
        if spent[0] >= eval_budget:
            stop = "budget_exhausted"
            break
        fd = evaluate(d)[4]
        while hi - lo > xtol:
            if spent[0] >= eval_budget:
                stop = "budget_exhausted"
                break
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - invphi * (hi - lo)
                fc = evaluate(c)[4]
            else:
                lo, c, fc = c, d, fd
                d = lo + invphi * (hi - lo)
                fd = evaluate(d)[4]
        fstar = 0.5 * (lo + hi)
        # final full-rank check at the refined point (coefficients needed)
        if spent[0] >= eval_budget:
            stop = "budget_exhausted"
            # reuse the last interior evaluation nearest the midpoint
            _, _, _, rk, sse = _fit_model(
                fstar, tau, y, w, sqrt_w, rank_tol, constant_data, y[0]
            )
        else:
            _, _, _, rk, sse = evaluate(fstar)
        if rk < 3:
            continue  # degenerate design at the refined point: not a peak
        if sse < best_sse:
            best_sse = sse
            best_freq = fstar
        if stop == "budget_exhausted":
            break

    if best_sse is math.inf:
        # every refined candidate landed on a rank-deficient point
        return _refined_result(
            grid_f, grid_sse, grid_rank, segments + 1, flo, fhi,
            tau, y, w, sqrt_w, rank_tol, constant_data,
            sse0, t0, spent, len(candidates), "boundary_minimum",
        )

    c_fit, a_tau, b_tau, rk, sse = _fit_model(
        best_freq, tau, y, w, sqrt_w, rank_tol, constant_data, y[0]
    )
    theta = _TWO_PI * best_freq * t0
    ct, st = math.cos(theta), math.sin(theta)
    a_fit = a_tau * ct - b_tau * st
    b_fit = a_tau * st + b_tau * ct
    return RefinedPeak(
        frequency=best_freq,
        power=0.0 if constant_data else (sse0 - sse) / sse0,
        amplitude=math.hypot(a_fit, b_fit),
        phase=math.atan2(b_fit, a_fit),
        constant=c_fit,
        sse=sse,
        rank=rk,
        sse_low=sse_low,
        sse_high=sse_high,
        boundary_lower=min(sse_low, sse_high) < sse,
        n_evaluations=spent[0],
        n_candidates=len(candidates),
        stop_reason=stop,
    )


def _refined_result(
    grid_f, grid_sse, grid_rank, n_valid, flo, fhi,
    tau, y, w, sqrt_w, rank_tol, constant_data,
    sse0, t0, spent, n_candidates, stop_reason,
) -> RefinedPeak:
    """Fallback result when no full-rank interior candidate is available.

    Picks the best evaluated grid point (usually an endpoint); with
    stop_reason="boundary_minimum" the reported frequency is the better
    interval endpoint by construction.
    """
    best_i = min(
        (i for i in range(n_valid) if grid_rank[i] >= 3),
        key=lambda i: grid_sse[i],
        default=None,
    )
    if best_i is None:
        # no full-rank grid point at all: take the raw SSE minimum
        best_i = min(range(n_valid), key=lambda i: grid_sse[i])
    freq = grid_f[best_i]
    c_fit, a_tau, b_tau, rk, sse = _fit_model(
        freq, tau, y, w, sqrt_w, rank_tol, constant_data, y[0]
    )
    theta = _TWO_PI * freq * t0
    ct, st = math.cos(theta), math.sin(theta)
    a_fit = a_tau * ct - b_tau * st
    b_fit = a_tau * st + b_tau * ct
    # endpoint SSEs by direct evaluation (not charged to the scan budget)
    sse_low = _fit_model(flo, tau, y, w, sqrt_w, rank_tol, constant_data, y[0])[4]
    sse_high = _fit_model(fhi, tau, y, w, sqrt_w, rank_tol, constant_data, y[0])[4]
    return RefinedPeak(
        frequency=freq,
        power=0.0 if constant_data else (sse0 - sse) / sse0,
        amplitude=math.hypot(a_fit, b_fit),
        phase=math.atan2(b_fit, a_fit),
        constant=c_fit,
        sse=sse,
        rank=rk,
        sse_low=sse_low,
        sse_high=sse_high,
        boundary_lower=min(sse_low, sse_high) <= sse,
        n_evaluations=spent[0],
        n_candidates=n_candidates,
        stop_reason=stop_reason,
    )
