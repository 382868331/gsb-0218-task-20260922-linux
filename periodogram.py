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
    "PeriodogramResult",
    "FrequencyPeak",
    "PeriodogramError",
    "EPS",
    "MAX_OBSERVATIONS",
    "MAX_FREQUENCIES",
    "MIN_OBSERVATIONS",
]

__version__ = "1.0.0"

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
