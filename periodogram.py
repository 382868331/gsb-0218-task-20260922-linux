"""Floating-mean periodogram for unevenly sampled time series.

For every candidate frequency ``f`` the model

    y_k ~= h_k . gamma  +  a*cos(2*pi*f*t_k) + b*sin(2*pi*f*t_k)

is fitted by *weighted Householder QR* (every observation is one matrix row;
observations are never placed on a uniform grid, zero-padded or FFT-ed).
``h_k`` is a caller-supplied fixed nuisance/design row ``H`` that MUST
contain a constant (all-ones) column and may carry, for example, a linear
drift column.  When ``H`` is omitted it defaults to a single ones column,
recovering the classic floating-mean model ``c + a*cos + b*sin``.

The nuisance block is QR-factored first (weighted QR); the frequency columns
are assessed against that basis at every candidate.  The power is the
fractional weighted-SSE reduction relative to the **H-only** model,
``P = (SSE_H - SSE(f)) / SSE_H`` (never relative to a constant-only model
when ``H`` carries more columns).

Per-bin statuses:

* ``"ok"``: the augmented design ``[H, cos, sin]`` has full numerical rank
  and ``SSE_H > 0``, so the power is a finite number;
* ``"unestimable"``: after projection onto the weighted H basis one or both
  frequency directions lose rank (e.g. a very low frequency nearly
  collinear with a drift column); that bin alone is marked, the scan
  continues;
* ``"undefined_baseline"``: ``SSE_H == 0`` (the H-only model already
  interpolates the data exactly); the relative reduction is 0/0 and is
  reported as NaN with an explicit status, never as 0 or 1.

A rank-deficient ``H`` itself is rejected for the whole call.

The sampling window ``W(f) = sum_k w_k * exp(i*2*pi*f*t_k)`` is also
computed; its modulus normalized by ``sum w_k`` explains alias structure.

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
    "MAX_H_COLUMNS",
    "MIN_OBSERVATIONS",
]

__version__ = "2.0.0"

# machine epsilon of binary64 (Python floats are IEEE-754 binary64)
EPS = 2.220446049250313e-16

MAX_OBSERVATIONS = 200
MAX_FREQUENCIES = 300
MAX_H_COLUMNS = 50
MIN_OBSERVATIONS = 3  # the model has three parameters (c, a, b)

# per-frequency-bin statuses
STATUS_OK = "ok"
STATUS_UNESTIMABLE = "unestimable"
STATUS_UNDEFINED_BASELINE = "undefined_baseline"

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
    status: str = STATUS_OK  # "ok" (peaks are only built from ok bins)


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
    sse0: float                   # weighted SSE of the baseline (H-only) model
    weighted_mean: float          # constant-column fit of the H-only model;
                                  # for the default ones-column H this is the
                                  # weighted mean, kept for compatibility
    # complex sampling window; its complex phase is referenced to t0 =
    # times[0] (internal time recentering), while |window|/sum(w) -- the
    # origin-invariant quantity used for alias analysis -- is window_modulus.
    window: tuple[complex, ...]
    window_modulus: tuple[float, ...]
    # per-bin status: STATUS_OK / STATUS_UNESTIMABLE /
    # STATUS_UNDEFINED_BASELINE (length = number of candidate frequencies)
    statuses: tuple[str, ...]
    # fitted H coefficients at every candidate frequency (p coefficients
    # per bin); the constant lives at index constant_column within each row
    h_coefficients: tuple[tuple[float, ...], ...]
    # fitted coefficients and residual SSE of the H-only baseline model
    h_baseline_coefficients: tuple[float, ...]
    h_rank: int                  # numerical rank of the weighted H block
    h_columns: int               # number of columns p of H (>= 1)
    constant_column: int         # index of the mandatory all-ones column
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
    """Validate a caller-supplied fixed design matrix H.

    Contract: a non-empty sequence (rows) of equal-length non-empty
    sequences (columns), 1..MAX_H_COLUMNS columns, every entry a finite
    real number (bools rejected), exactly one column bitwise equal to one
    (the mandatory constant column).  Returns (column-major entries,
    constant-column index).
    """
    if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes, bytearray)):
        raise PeriodogramError(
            f"{name}: expected a list/tuple of {n} rows (sequences of "
            f"finite real numbers), got {type(obj).__name__}"
        )
    if len(obj) != n:
        raise PeriodogramError(
            f"{name}: expected {n} rows (one per observation), got {len(obj)}"
        )
    if len(obj) == 0:
        raise PeriodogramError(f"{name}: must contain at least one row")
    p: int | None = None
    for i, row in enumerate(obj):
        if not isinstance(row, Sequence) or isinstance(
            row, (str, bytes, bytearray)
        ):
            raise PeriodogramError(
                f"{name}[{i}]: expected a row (list/tuple of finite real "
                f"numbers), got {type(row).__name__}"
            )
        if p is None:
            p = len(row)
            if not (1 <= p <= MAX_H_COLUMNS):
                raise PeriodogramError(
                    f"{name}[{i}]: expected between 1 and {MAX_H_COLUMNS} "
                    f"columns, got {p}"
                )
        elif len(row) != p:
            raise PeriodogramError(
                f"{name}[{i}]: row has {len(row)} entries but earlier rows "
                f"have {p}"
            )
    assert p is not None
    # validate every entry and locate the mandatory constant column
    cols = [[0.0] * n for _ in range(p)]
    const_index = -1
    for j in range(p):
        all_one = True
        for i in range(n):
            item = obj[i][j]
            if isinstance(item, bool) or not isinstance(item, Real):
                raise PeriodogramError(
                    f"{name}[{i}][{j}]: expected a real number, got "
                    f"{type(item).__name__}"
                )
            x = float(item)
            if math.isnan(x) or math.isinf(x):
                raise PeriodogramError(
                    f"{name}[{i}][{j}]: NaN/Infinity is not allowed, got {item!r}"
                )
            cols[j][i] = x
            if x != 1.0:
                all_one = False
        if all_one:
            if const_index != -1:
                raise PeriodogramError(
                    f"{name}: exactly one constant (all-ones) column is "
                    f"allowed; found columns {const_index} and {j}"
                )
            const_index = j
    if const_index == -1:
        raise PeriodogramError(
            f"{name}: a constant (all-ones) column is mandatory; no column "
            f"of {p} is identically 1.0"
        )
    return cols, const_index


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


def _solve_triangular_coef(
    r: list[list[float]], b: list[float], pivots: list[float],
    threshold: float,
) -> list[float]:
    """Back-substitution with null directions set to zero (minimum-norm)."""
    n = len(pivots)
    x = [0.0] * n
    for j in range(n - 1, -1, -1):
        if abs(pivots[j]) <= threshold:
            x[j] = 0.0
            continue
        s = b[j]
        for k in range(j + 1, n):
            s -= r[j][k] * x[k]
        x[j] = s / pivots[j]
    return x


def _solve_full(
    r: list[list[float]], b: list[float], pivots: list[float]
) -> list[float]:
    """Plain back-substitution for a known full-rank triangular system."""
    n = len(pivots)
    x = [0.0] * n
    for j in range(n - 1, -1, -1):
        s = b[j]
        for k in range(j + 1, n):
            s -= r[j][k] * x[k]
        x[j] = s / pivots[j]
    return x


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def periodogram(
    times,
    values,
    weights,
    frequencies,
    *,
    H=None,
    top_k: int | None = None,
    min_separation: float = 0.0,
    rank_tol: float | None = None,
    selection_budget: int | None = None,
) -> PeriodogramResult:
    r"""Floating-mean weighted periodogram on a fixed candidate frequency grid.

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
    H:
        Caller-supplied fixed design matrix with one row per observation
        (``len(H) == len(times)``) and 1..MAX_H_COLUMNS columns.  It MUST
        contain exactly one constant (all-ones) column and may additionally
        carry, e.g., a linear drift column [1, t_k] or any other fixed
        nuisance regressors.  Every entry must be a finite real number
        (``bool`` is rejected).  ``None`` (the default) uses a single ones
        column, i.e. the original floating-constant model.

        The whole call is rejected if ``H`` itself is rank deficient at
        ``rank_tol``.  At a candidate frequency where the cos/sin columns
        lose rank after projection onto the weighted H space, that bin alone
        is marked ``statuses[i] == 'unestimable'``; the scan continues.
        When the H-only baseline SSE is zero the power is 0/0 and the bin is
        marked ``'undefined_baseline'`` with ``powers[i] == nan``.
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

    # ---------------- caller-supplied fixed design matrix H ---------------
    if H is None:
        # default: the classic floating-constant model (single ones column)
        h_cols = [[1.0] * n]
        const_col = 0
        default_h = True
    else:
        h_cols, const_col = _design_matrix("H", H, n)
        default_h = False
    p = len(h_cols)
    if n < p + 2:
        # cos and sin nominally need two residual degrees of freedom beyond
        # the fixed nuisance block at every candidate
        raise PeriodogramError(
            f"times: need at least H columns + 2 = {p + 2} observations to "
            f"fit H plus sin/cos, got {n}"
        )

    # Time-origin recentering for numerical stability.  The model is fitted in
    # tau = t - t0 (t0 = times[0]); a, b are rotated back to the original time
    # frame before amplitude/phase are reported.  Power/amplitude and the
    # window modulus are exactly origin-invariant.
    t0 = t[0]
    tau = [tk - t0 for tk in t]
    sqrt_w = [math.sqrt(wk) for wk in w]
    weights_sum = sum(w)

    # ----- baseline: weighted QR of the fixed nuisance block H -------------
    h_rows = [[h_cols[j][i] for j in range(p)] for i in range(n)]
    r_h, qb_h, piv_h = _weighted_qr(h_rows, y, sqrt_w, p)
    h_scale = max((abs(q) for q in piv_h), default=0.0)
    h_threshold = rank_tol * h_scale
    h_rank = sum(1 for q in piv_h if abs(q) > h_threshold)
    if h_rank < p:
        # a deficient nuisance basis is a caller error for the whole call;
        # per-frequency deficiency (of the projected sin/cos) is handled per
        # bin instead
        bad = [j for j in range(p) if abs(piv_h[j]) <= h_threshold]
        raise PeriodogramError(
            f"H: rank deficient design matrix: numerical rank {h_rank} < "
            f"{p} columns (zero-pivot column indices {bad}); remove the "
            f"redundant column(s)"
        )

    # baseline coefficients and residual SSE from the H-only fit.  The
    # baseline SSE is computed from fitted values: when the block is full
    # rank (guaranteed above) qb_h[p:] is already the residual, but the
    # direct evaluation is the unambiguous convention used at every bin.
    gamma0 = _solve_full(r_h, qb_h, piv_h)
    sse0 = 0.0
    for i in range(n):
        resid = y[i] - sum(gamma0[j] * h_rows[i][j] for j in range(p))
        sse0 += w[i] * resid * resid
    sse0 = max(0.0, sse0)
    weighted_mean = gamma0[const_col]
    # The H-only baseline SSE is mathematically zero exactly when y lies in
    # the column space of H.  Detect it from the QR residual (orthogonal,
    # numerically reliable): a response whose residual norm is at/below
    # n*eps of its own weighted norm has been interpolated to floating-point
    # precision.  Such data make the relative power 0/0 -- undefined, never
    # reported as 0 or 1.  A genuinely small-but-nonzero residual above that
    # relative noise floor is still processed normally.
    qb_norm2 = sum(q * q for q in qb_h)
    qb_resid2 = sum(qb_h[i] * qb_h[i] for i in range(p, n))
    baseline_zero = sse0 == 0.0 or (
        qb_norm2 > 0.0 and qb_resid2 <= n * EPS * qb_norm2
    )

    # Backward-compatible exact convention for the DEFAULT H (ones column)
    # and bitwise-constant data: SSE0 = 0, power 0 at every frequency.
    constant_data = default_h and all(v == y[0] for v in y)
    if constant_data:
        sse0 = 0.0
        weighted_mean = float(y[0])
        gamma0 = [weighted_mean]

    # ------------------------- per-frequency scan --------------------------
    ncols = p + 2
    powers: list[float] = [float("nan")] * m
    amplitudes: list[float] = [0.0] * m
    phases: list[float] = [0.0] * m
    ranks: list[int] = [0] * m
    constants: list[float] = [0.0] * m
    sses: list[float] = [0.0] * m
    statuses: list[str] = [STATUS_OK] * m
    h_coef: list[tuple[float, ...]] = [(0.0,) * p for _ in range(m)]
    window: list[complex] = [0.0j] * m
    window_mod: list[float] = [0.0] * m
    rows: list[list[float]] = [
        [0.0] * ncols for _ in range(n)
    ]
    for i in range(n):
        for j in range(p):
            rows[i][j] = h_rows[i][j]

    for fi, freq in enumerate(f):
        cos_i = p
        sin_i = p + 1
        for i in range(n):
            angle = _TWO_PI * freq * tau[i]
            rows[i][cos_i] = math.cos(angle)
            rows[i][sin_i] = math.sin(angle)

        r, qb, pivots = _weighted_qr(rows, y, sqrt_w, ncols)
        full_scale = max(abs(q) for q in pivots)
        threshold = rank_tol * full_scale
        rank = sum(1 for q in pivots if abs(q) > threshold)
        # The frequency is estimable iff BOTH projected trig columns add
        # rank on top of the (full-rank) H block.
        freq_estimable = (
            abs(pivots[cos_i]) > threshold
            and abs(pivots[sin_i]) > threshold
        )

        if constant_data:
            # documented exact convention (default H, bitwise-constant data)
            coef = [0.0] * ncols
            coef[const_col] = float(y[0])
            sse = 0.0
            status = STATUS_OK
            power = 0.0
        else:
            coef = _solve_triangular_coef(r, qb, pivots, threshold)
            # SSE from the actual projected fit (unambiguous at any rank)
            sse = 0.0
            for i in range(n):
                fitted = sum(coef[j] * rows[i][j] for j in range(ncols))
                resid = y[i] - fitted
                sse += w[i] * resid * resid
            sse = max(0.0, sse)
            ranks[fi] = rank
            if not freq_estimable:
                status = STATUS_UNESTIMABLE
                power = float("nan")
            elif baseline_zero:
                # relative reduction 0/0: undefined, never reported as 1
                status = STATUS_UNDEFINED_BASELINE
                power = float("nan")
            else:
                status = STATUS_OK
                power = (sse0 - sse) / sse0
                # clamp to the theoretical range against roundoff
                if power < 0.0 and power > -1e-12:
                    power = 0.0
                elif power > 1.0 and power - 1.0 < 1e-12:
                    power = 1.0

        a_tau, b_tau = coef[cos_i], coef[sin_i]
        # rotate (a, b) from the tau frame back into the original t frame
        theta = _TWO_PI * freq * t0
        ct, st = math.cos(theta), math.sin(theta)
        a_fit = a_tau * ct - b_tau * st
        b_fit = a_tau * st + b_tau * ct

        # sampling window in the tau frame (modulus is origin-invariant);
        # it depends only on the sampling times/weights, never on H
        wre = 0.0
        wim = 0.0
        for i in range(n):
            wre += w[i] * rows[i][cos_i]
            wim += w[i] * rows[i][sin_i]
        wmod = math.hypot(wre, wim) / weights_sum

        if constant_data:
            ranks[fi] = p + 2
        powers[fi] = power
        amplitudes[fi] = math.hypot(a_fit, b_fit)
        phases[fi] = math.atan2(b_fit, a_fit)
        constants[fi] = coef[const_col]
        sses[fi] = sse
        statuses[fi] = status
        h_coef[fi] = tuple(coef[j] for j in range(p))
        window[fi] = complex(wre, wim)
        window_mod[fi] = wmod

    # ---------------- strict local maxima (endpoints never peak) -----------
    # bins with an undefined power never participate
    admissible = [
        i for i in range(m) if statuses[i] == STATUS_OK
    ]
    admissible_set = set(admissible)
    local_peaks: list[FrequencyPeak] = []
    for i in range(1, m - 1):
        if (
            i in admissible_set
            and i - 1 in admissible_set
            and i + 1 in admissible_set
            and powers[i] > powers[i - 1]
            and powers[i] > powers[i + 1]
        ):
            local_peaks.append(_peak(i, f, powers, amplitudes, phases,
                                     ranks, sses, window_mod, statuses))

    # --- greedy top-K with explicit frequency exclusion (NMS by power) -----
    order = sorted(admissible, key=lambda i: (-powers[i], f[i]))
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
        _peak(i, f, powers, amplitudes, phases, ranks, sses, window_mod,
              statuses)
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
        statuses=tuple(statuses),
        h_coefficients=tuple(h_coef),
        h_baseline_coefficients=tuple(gamma0),
        h_rank=p,
        h_columns=p,
        constant_column=const_col,
        local_peaks=tuple(local_peaks),
        top_peaks=tuple(top_peaks),
        top_k_requested=top_k,
        top_k_found=len(top_peaks),
        selection_status=status,
        min_separation=min_separation,
        selection_budget=selection_budget,
    )


def _peak(i, f, powers, amplitudes, phases, ranks, sses, window_mod,
          statuses) -> FrequencyPeak:
    return FrequencyPeak(
        index=i,
        frequency=f[i],
        power=powers[i],
        amplitude=amplitudes[i],
        phase=phases[i],
        rank=ranks[i],
        sse=sses[i],
        window_modulus=window_mod[i],
        status=statuses[i],
    )
