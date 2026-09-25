"""Floating-mean periodogram for unevenly sampled time series.

For every candidate frequency ``f`` the model

    y_k ~= H_k @ beta + a*cos(2*pi*f*t_k) + b*sin(2*pi*f*t_k)

is fitted by *weighted Householder QR* (every observation is one matrix row;
observations are never placed on a uniform grid, zero-padded or FFT-ed).  The
power is the fractional weighted-SSE reduction relative to the fixed
nuisance-model fit (``H``; by default a constant-only model),
    ``P = (SSE_H - SSE(f)) / SSE_H``.

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
    "FrequencyStatus",
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
MIN_OBSERVATIONS = 3  # H=[1] plus the two frequency parameters

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
    power: float          # (SSE_H - SSE) / SSE_H when power is defined
    amplitude: float      # hypot(a, b) added to the fixed H model
    phase: float          # atan2(b, a) in radians, wrapped to (-pi, pi]
    rank: int             # numerical rank of the weighted [H, cos, sin] design
    sse: float            # weighted residual SSE at this frequency
    window_modulus: float  # |sum w exp(i2pi f t)| / sum w


@dataclass(frozen=True)
class FrequencyStatus:
    """Estimation status for one candidate frequency."""

    index: int
    frequency: float
    status: str
    rank: int
    added_rank: int
    power: float
    amplitude: float
    phase: float
    sse: float


@dataclass(frozen=True)
class PeriodogramResult:
    """Full result of :func:`periodogram` (immutable)."""

    frequencies: tuple[float, ...]
    powers: tuple[float, ...]
    amplitudes: tuple[float, ...]
    phases: tuple[float, ...]
    statuses: tuple[str, ...]
    frequency_statuses: tuple[FrequencyStatus, ...]
    ranks: tuple[int, ...]
    added_ranks: tuple[int, ...]
    constants: tuple[float, ...]  # fitted constant-column coefficient
    sses: tuple[float, ...]       # weighted SSE at every frequency
    sse0: float                   # weighted SSE of the H-only baseline model
    weighted_mean: float          # weighted sample mean (baseline fit if H=[1])
    h_rank: int
    constant_column: int
    baseline_coefficients: tuple[float, ...]
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


def _design_matrix(name: str, obj: object, n: int) -> tuple[list[list[float]], int]:
    if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes, bytearray)):
        raise PeriodogramError(
            f"{name}: expected a list/tuple of real-number rows, "
            f"got {type(obj).__name__}"
        )
    if len(obj) != n:
        raise PeriodogramError(
            f"{name}: expected {n} rows matching times, got {len(obj)}"
        )
    rows: list[list[float]] = []
    p = -1
    constant_column = -1
    for i, row in enumerate(obj):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise PeriodogramError(
                f"{name}[{i}]: expected a list/tuple of real numbers, "
                f"got {type(row).__name__}"
            )
        if p < 0:
            p = len(row)
            if not 1 <= p <= n:
                raise PeriodogramError(
                    f"{name}: expected between 1 and {n} columns, got {p}"
                )
        elif len(row) != p:
            raise PeriodogramError(
                f"{name}[{i}]: expected {p} columns, got {len(row)}"
            )
        parsed = _real_array(f"{name}[{i}]", row)
        rows.append(parsed)
    for j in range(p):
        if all(rows[i][j] == 1.0 for i in range(n)):
            if constant_column >= 0:
                break
            else:
                constant_column = j
    if constant_column < 0:
        raise PeriodogramError(f"{name}: a constant column of ones is required")
    return rows, constant_column


# --------------------------------------------------------------------------- #
# weighted Householder QR, no external numeric libraries
# --------------------------------------------------------------------------- #
def _weighted_qr(
    rows: list[list[float]], values: list[float], sqrt_w: list[float], ncols: int
) -> tuple[list[list[float]], list[float], list[float]]:
    """Return (R, qty, pivots) of the weighted least-squares problem.

    R is upper triangular (below diagonal zeroed), ``qty = Q.T * W**0.5 * y``.
    The residual norm squared is ``sum(qty[ncols:]**2)``; ``pivots[j]`` is the
    j-th diagonal of R with its Householder sign.
    """
    r, b, pivots, _ = _weighted_qr_factor(rows, values, sqrt_w, ncols)
    return r, b, pivots


def _weighted_qr_factor(
    rows: list[list[float]], values: list[float], sqrt_w: list[float], ncols: int
) -> tuple[list[list[float]], list[float], list[float], list[tuple[int, float, list[float], float]]]:
    m = len(rows)
    r = [[rows[i][j] * sqrt_w[i] for j in range(ncols)] for i in range(m)]
    b = [values[i] * sqrt_w[i] for i in range(m)]
    pivots: list[float] = []
    reflectors: list[tuple[int, float, list[float], float]] = []

    for j in range(ncols):
        # scaled 2-norm of the trailing column (overflow resistant)
        scale = 0.0
        for i in range(j, m):
            a = abs(r[i][j])
            if a > scale:
                scale = a
        if scale == 0.0:
            pivots.append(0.0)
            reflectors.append((j, 0.0, [], 0.0))
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
        vtail = [r[i][j] for i in range(j + 1, m)]
        reflectors.append((j, v0, vtail, beta))

        for k in range(j + 1, ncols):
            s = v0 * r[j][k]
            for i in range(j + 1, m):
                s += r[i][j] * r[i][k]
            s *= beta
            r[j][k] -= s * v0
            for i in range(j + 1, m):
                r[i][k] -= s * r[i][j]

        s = v0 * b[j]
        for value, bvalue in zip(vtail, b[j + 1:]):
            s += value * bvalue
        s *= beta
        b[j] -= s * v0
        for offset, value in enumerate(vtail):
            b[j + 1 + offset] -= s * value

        r[j][j] = alpha
        for i in range(j + 1, m):
            r[i][j] = 0.0
        pivots.append(alpha)

    return r, b, pivots, reflectors


def _apply_reflectors(
    r: list[list[float]], start_col: int, ncols: int,
    reflectors: list[tuple[int, float, list[float], float]],
) -> None:
    for j, v0, vtail, beta in reflectors:
        for k in range(start_col, ncols):
            s = v0 * r[j][k]
            for offset, value in enumerate(vtail):
                s += value * r[j + 1 + offset][k]
            s *= beta
            r[j][k] -= s * v0
            for offset, value in enumerate(vtail):
                r[j + 1 + offset][k] -= s * value


def _qr_append(
    r: list[list[float]], b: list[float], ncols: int, start: int,
) -> list[float]:
    m = len(r)
    appended_pivots: list[float] = []
    for j in range(start, ncols):
        scale = 0.0
        for i in range(j, m):
            a = abs(r[i][j])
            if a > scale:
                scale = a
        if scale == 0.0:
            appended_pivots.append(0.0)
            continue
        norm = 0.0
        for i in range(j, m):
            q = r[i][j] / scale
            norm += q * q
        norm = scale * math.sqrt(norm)
        x0 = r[j][j]
        alpha = -norm if x0 >= 0.0 else norm
        v0 = x0 - alpha
        vtv = v0 * v0
        v = [r[i][j] for i in range(j + 1, m)]
        vtv += sum(x * x for x in v)
        beta = 2.0 / vtv
        for k in range(j + 1, ncols):
            s = v0 * r[j][k] + sum(x * r[j + 1 + idx][k] for idx, x in enumerate(v))
            s *= beta
            r[j][k] -= s * v0
            for idx, x in enumerate(v):
                r[j + 1 + idx][k] -= s * x
        s = v0 * b[j] + sum(x * b[j + 1 + idx] for idx, x in enumerate(v))
        s *= beta
        b[j] -= s * v0
        for idx, x in enumerate(v):
            b[j + 1 + idx] -= s * x
        r[j][j] = alpha
        for i in range(j + 1, m):
            r[i][j] = 0.0
        appended_pivots.append(alpha)
    return appended_pivots


def _solve_triangular(
    r: list[list[float]], b: list[float], pivots: list[float], rank_tol: float,
    threshold_scale: float | None = None,
) -> tuple[list[float], int]:
    n = len(pivots)
    scale = max(abs(p) for p in pivots) if threshold_scale is None else threshold_scale
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
    H=None,
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
    H:
        Optional fixed nuisance design with one row per observation.  It must
        contain a bitwise constant ones column and may also contain drift
        columns such as ``t``.  With ``H=None`` the old constant-only model
        is used.  Each fit adds the two frequency columns to this H.

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

    if H is None:
        h_rows = [[1.0] for _ in range(n)]
        constant_column = 0
    else:
        h_rows, constant_column = _design_matrix("H", H, n)
    p = len(h_rows[0])
    if n < p + 2:
        raise PeriodogramError(
            f"H: {p} columns leave {n - p} degrees of freedom; at least 2 "
            f"are required for the sin/cos pair"
        )

    # Time-origin recentering for numerical stability.  The model is fitted in
    # tau = t - t0 (t0 = times[0]); a, b are rotated back to the original time
    # frame before amplitude/phase are reported.  Power/amplitude and the
    # window modulus are exactly origin-invariant.
    t0 = t[0]
    tau = [tk - t0 for tk in t]
    sqrt_w = [math.sqrt(wk) for wk in w]
    weights_sum = sum(w)

    # ----- fixed nuisance baseline H, QR factorized only once --------------
    rh, bh, h_pivots, h_reflectors = _weighted_qr_factor(h_rows, y, sqrt_w, p)
    h_scale = max(abs(z) for z in h_pivots)
    h_rank = sum(1 for z in h_pivots if abs(z) > rank_tol * h_scale)
    if h_rank != p:
        null = [j for j, z in enumerate(h_pivots)
                if abs(z) <= rank_tol * h_scale]
        raise PeriodogramError(
            f"H: weighted design is rank deficient (rank {h_rank} < {p}); "
            f"null pivot column indices {null}"
        )
    baseline_coef, _ = _solve_triangular(rh, bh, h_pivots, rank_tol, h_scale)
    design_scale = max(
        (abs(h_rows[i][j]) * max(1.0, abs(baseline_coef[j]))
         for i in range(n) for j in range(p)),
        default=1.0,
    )
    baseline_zero_tol = n * EPS * max(1.0, design_scale)
    baseline_residuals_exact_zero = True
    for i in range(n):
        fitted = sum(h_rows[i][j] * baseline_coef[j] for j in range(p))
        if abs(y[i] - fitted) > baseline_zero_tol:
            baseline_residuals_exact_zero = False
            break
    if baseline_residuals_exact_zero:
        sse0 = 0.0
    else:
        sse = 0.0
        for i in range(n):
            fitted = sum(h_rows[i][j] * baseline_coef[j] for j in range(p))
            d = y[i] - fitted
            sse += w[i] * d * d
        sse0 = sse
    weighted_mean = sum(w[i] * y[i] for i in range(n)) / weights_sum

    # ------------------------- per-frequency scan --------------------------
    powers: list[float] = [0.0] * m
    amplitudes: list[float] = [0.0] * m
    phases: list[float] = [0.0] * m
    statuses: list[str] = ["ok"] * m
    ranks: list[int] = [0] * m
    added_ranks: list[int] = [0] * m
    constants: list[float] = [0.0] * m
    sses: list[float] = [0.0] * m
    window: list[complex] = [0.0j] * m
    window_mod: list[float] = [0.0] * m
    for fi, freq in enumerate(f):
        r = [[rh[i][j] for j in range(p)] + [0.0, 0.0] for i in range(n)]
        qb = bh[:]
        trig = []
        for i in range(n):
            angle = _TWO_PI * freq * tau[i]
            cs, sn = math.cos(angle), math.sin(angle)
            trig.append((cs, sn))
            r[i][p] = cs * sqrt_w[i]
            r[i][p + 1] = sn * sqrt_w[i]
        _apply_reflectors(r, p, p + 2, h_reflectors)
        appended_pivots = _qr_append(r, qb, p + 2, p)
        pivots = h_pivots + appended_pivots
        full_scale = max(abs(z) for z in pivots)
        added_rank = sum(
            1 for z in appended_pivots if abs(z) > rank_tol * full_scale
        )
        rank = h_rank + added_rank
        added_ranks[fi] = added_rank
        ranks[fi] = rank

        wre = sum(w[i] * trig[i][0] for i in range(n))
        wim = sum(w[i] * trig[i][1] for i in range(n))
        wmod = math.hypot(wre, wim) / weights_sum
        window[fi] = complex(wre, wim)
        window_mod[fi] = wmod

        if added_rank < 2:
            statuses[fi] = "frequency_unestimable"
            nan = float("nan")
            powers[fi] = amplitudes[fi] = phases[fi] = nan
            constants[fi] = sses[fi] = nan
            continue

        coef, _ = _solve_triangular(r, qb, pivots, rank_tol, full_scale)
        a_tau, b_tau = coef[p], coef[p + 1]
        sse = 0.0
        for i in range(n):
            fitted = sum(h_rows[i][j] * coef[j] for j in range(p))
            cs, sn = trig[i]
            resid = y[i] - fitted - a_tau * cs - b_tau * sn
            sse += w[i] * resid * resid
        sses[fi] = sse
        constants[fi] = coef[constant_column]

        theta = _TWO_PI * freq * t0
        ct, st = math.cos(theta), math.sin(theta)
        a_fit = a_tau * ct - b_tau * st
        b_fit = a_tau * st + b_tau * ct
        amplitudes[fi] = math.hypot(a_fit, b_fit)
        phases[fi] = math.atan2(b_fit, a_fit)
        if sse0 == 0.0:
            statuses[fi] = "zero_baseline_sse"
            powers[fi] = float("nan")
        else:
            powers[fi] = (sse0 - sse) / sse0

    # ---------------- strict local maxima (endpoints never peak) -----------
    local_peaks: list[FrequencyPeak] = []
    for i in range(1, m - 1):
        if (
            statuses[i] == "ok"
            and powers[i] > powers[i - 1]
            and powers[i] > powers[i + 1]
        ):
            local_peaks.append(_peak(i, f, powers, amplitudes, phases,
                                     ranks, sses, window_mod))

    # --- greedy top-K with explicit frequency exclusion (NMS by power) -----
    eligible = [i for i in range(m) if statuses[i] == "ok"]
    order = sorted(eligible, key=lambda i: (-powers[i], f[i]))
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
    frequency_statuses = tuple(
        FrequencyStatus(
            index=i,
            frequency=f[i],
            status=statuses[i],
            rank=ranks[i],
            added_rank=added_ranks[i],
            power=powers[i],
            amplitude=amplitudes[i],
            phase=phases[i],
            sse=sses[i],
        )
        for i in range(m)
    )

    return PeriodogramResult(
        frequencies=tuple(f),
        powers=tuple(powers),
        amplitudes=tuple(amplitudes),
        phases=tuple(phases),
        statuses=tuple(statuses),
        frequency_statuses=frequency_statuses,
        ranks=tuple(ranks),
        added_ranks=tuple(added_ranks),
        constants=tuple(constants),
        sses=tuple(sses),
        sse0=sse0,
        weighted_mean=weighted_mean,
        h_rank=h_rank,
        constant_column=constant_column,
        baseline_coefficients=tuple(baseline_coef),
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
