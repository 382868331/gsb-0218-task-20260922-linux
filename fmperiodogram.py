"""fmperiodogram — 非均匀采样时间序列的浮动均值谱估计 (floating-mean periodogram).

只用 Python 3.14 标准库实现。核心入口: :func:`periodogram`。

对每个候选正频率 f 拟合加权最小二乘模型

    v_i = c + a*cos(2*pi*f*t_i) + b*sin(2*pi*f*t_i) + e_i

功率定义为相对常量模型 (仅加权均值) 的 SSE 减少比例:

    P(f) = (SSE_const - SSE_full(f)) / SSE_const

采样窗口 W(f) = sum_i w_i * exp(i*2*pi*f*(t_i-t0)) 的归一模
|W(f)|/sum(w) 用于解释别名: 若两个峰频之差处窗口模接近 1, 二者在
当前采样下几乎不可区分。

设计要点见 README.md。本模块不做任何均匀重采样/补零。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "MAX_OBSERVATIONS",
    "MAX_FREQUENCIES",
    "DEFAULT_ALIAS_THRESHOLD",
    "FrequencyPoint",
    "AliasLink",
    "PeriodogramResult",
    "PeriodogramError",
    "PeriodogramInputError",
    "periodogram",
]

MAX_OBSERVATIONS = 200
MAX_FREQUENCIES = 300
MIN_OBSERVATIONS = 3
# 秩判据: rank = #{k: |R_kk| >= RCOND * max_j |R_jj|} (见 README)。
_RCOND = 1e-12
# 功率相对舍入钳制宽度; 超出则抛 FloatingPointError (见 README 误差口径)。
_POWER_CLAMP = 1e-10
DEFAULT_ALIAS_THRESHOLD = 0.8
_TWO_PI = 2.0 * math.pi


class PeriodogramError(ValueError):
    """本库所有值类错误的基类 (同时是 ValueError)。"""


class PeriodogramInputError(PeriodogramError):
    """输入不满足契约; 消息中带数组名与下标, 可定位。"""


@dataclass(frozen=True)
class FrequencyPoint:
    """单个候选频率上的谱估计结果。

    frequency        候选频率 (与输入同序, 严格递增)
    power            常量模型 SSE 减少比例, 约定落在 [0, 1]
    amplitude        幅度 A = hypot(a, b)
    phase            相位 phi (弧度, (-pi, pi]); 模型为
                     c + A*cos(2*pi*f*(t-phase_epoch) + phi)
    sse              该频率三参数模型的加权残差平方和
    rank             加权设计矩阵的 QR 对角秩估计 (1..3)
    window_modulus   |sum w exp(i2pi f (t-t0))| / sum(w), 范围 [0, 1]
    local_peak       是否为严格局部最大功率峰 (两端点不计)
    """

    frequency: float
    power: float
    amplitude: float
    phase: float
    sse: float
    rank: int
    window_modulus: float
    local_peak: bool


@dataclass(frozen=True)
class AliasLink:
    """采样窗口解释的别名链接。

    两个局部峰频 f1=peak_frequency, f2=alias_frequency 在当前采样下
    不可区分有两种机制:
      * 平移型: exp(i2π f2 t_k) ≈ exp(i2π f1 t_k), 即
        |W(f2-f1)|/W0 >= 阈值 (beat_frequency=|f2-f1|);
      * 镜像型: exp(i2π f2 t_k) ≈ exp(-i2π f1 t_k), 即
        cos(2π f2 t_k) ≈ cos(2π f1 t_k), 对应
        |W(f2+f1)|/W0 >= 阈值 (beat_frequency=f2+f1)。
        均匀采样的 Nyquist 别名 (f2 = fs - f1) 正是此类。
    window_modulus 为命中的那一个; beat_kind 标识机制。
    """

    peak_frequency: float
    alias_frequency: float
    difference_frequency: float
    beat_frequency: float
    beat_kind: str
    window_modulus: float
    alias_power: float


@dataclass(frozen=True)
class PeriodogramResult:
    """periodogram 的固定返回结构 (不可变)。"""

    points: tuple[FrequencyPoint, ...]
    peaks: tuple[FrequencyPoint, ...]
    top: tuple[FrequencyPoint, ...]
    aliases: tuple[AliasLink, ...]
    constant: float
    sse_constant: float
    total_weight: float
    phase_epoch: float
    constant_series: bool
    top_k: int
    min_separation: float
    alias_threshold: float


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def _check_int(value: int, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是 int, 实际为 {type(value).__name__}")
    if value < minimum or value > maximum:
        raise PeriodogramInputError(
            f"{name}={value} 超出允许范围 [{minimum}, {maximum}]"
        )
    return value


def _check_real_number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数 (int/float), 实际为 {type(value).__name__}")
    fv = float(value)
    if not math.isfinite(fv):
        raise PeriodogramInputError(f"{name} 必须有限, 不能为 NaN/Infinity")
    if positive and fv <= 0.0:
        raise PeriodogramInputError(f"{name} 必须为正数, 实际为 {fv}")
    return fv


def _check_real_array(
    seq: Sequence[float],
    name: str,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
) -> list[float]:
    if not isinstance(seq, (list, tuple)):
        raise TypeError(
            f"{name} 必须是 list/tuple, 实际为 {type(seq).__name__}"
        )
    n = len(seq)
    if min_len is not None and n < min_len:
        raise PeriodogramInputError(
            f"{name} 长度 {n} 小于最小值 {min_len}"
        )
    if max_len is not None and n > max_len:
        raise PeriodogramInputError(
            f"{name} 长度 {n} 超过上限 {max_len}"
        )
    out: list[float] = []
    for i, x in enumerate(seq):
        # bool 是 int 子类; 按契约显式拒绝, 不静默当 0/1。
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError(
                f"{name}[{i}] 必须是实数 (int/float), 实际为 {type(x).__name__}"
            )
        fx = float(x)
        if not math.isfinite(fx):
            raise PeriodogramInputError(f"{name}[{i}] 为 NaN/Infinity, 必须有限")
        out.append(fx)
    return out


def _check_frequencies(
    seq: Sequence[float],
) -> list[float]:
    freqs = _check_real_array(
        seq, "frequencies", min_len=1, max_len=MAX_FREQUENCIES
    )
    prev = -0.0  # 第一个频率必须 > 0
    for i, f in enumerate(freqs):
        if f <= 0.0:
            raise PeriodogramInputError(
                f"frequencies[{i}]={f!r} 不是正频率 (必须 f > 0)"
            )
        if f <= prev:
            raise PeriodogramInputError(
                f"frequencies 必须严格递增且无重复: "
                f"frequencies[{i}]={f!r} <= frequencies[{i - 1}]={prev!r}"
            )
        prev = f
    return freqs


# --------------------------------------------------------------------------- #
# 加权 Householder QR (3 列), 无第三方依赖
# --------------------------------------------------------------------------- #
def _weighted_qr_fit3(
    sw: list[float],
    y: list[float],
    c0: list[float],
    c1: list[float],
    c2: list[float],
) -> tuple[list[float], list[float]]:
    """加权最小二乘: A 的行已乘 sqrt(w)。Householder 正交三角化。

    返回 (系数 [x0,x1,x2], R 对角绝对值列表)。秩亏列对应系数取 0。
    """
    m = len(sw)
    R = [[sw[i] * c0[i], sw[i] * c1[i], sw[i] * c2[i]] for i in range(m)]
    z = [sw[i] * y[i] for i in range(m)]

    ncol = 3
    for k in range(ncol):
        # 列范数 (k 行以下), math.hypot 抗溢出; m <= 200。
        nrm = math.hypot(*(R[i][k] for i in range(k, m)))
        if nrm == 0.0:
            continue
        alpha = -math.copysign(nrm, R[k][k])
        v = [0.0] * m
        v[k] = R[k][k] - alpha
        for i in range(k + 1, m):
            v[i] = R[i][k]
        vnorm2 = math.fsum(v[i] * v[i] for i in range(k, m))
        if vnorm2 == 0.0:
            continue
        beta = 2.0 / vnorm2
        for j in range(k, ncol):
            s = math.fsum(v[i] * R[i][j] for i in range(k, m))
            scale = beta * s
            for i in range(k, m):
                R[i][j] -= scale * v[i]
        s = math.fsum(v[i] * z[i] for i in range(k, m))
        scale = beta * s
        for i in range(k, m):
            z[i] -= scale * v[i]

    diag = [abs(R[k][k]) for k in range(ncol)]
    tol = _RCOND * max(diag) if diag else 0.0
    x = [0.0, 0.0, 0.0]
    for k in range(ncol - 1, -1, -1):
        if diag[k] <= tol:
            x[k] = 0.0
            continue
        tail = math.fsum(R[k][j] * x[j] for j in range(k + 1, ncol))
        x[k] = (z[k] - tail) / R[k][k]
    return x, diag


# --------------------------------------------------------------------------- #
# 核心入口
# --------------------------------------------------------------------------- #
def periodogram(
    times: Sequence[float],
    values: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    frequencies: Optional[Sequence[float]] = None,
    *,
    top_k: int = 5,
    min_separation: Optional[float] = None,
    alias_threshold: float = DEFAULT_ALIAS_THRESHOLD,
) -> PeriodogramResult:
    """计算浮动均值周期图。

    参数
    ----
    times, values:
        等长 list/tuple, 长度 3..200, 实数有限。times 可乱序、可重复,
        不要求等间隔; 绝不做均匀化重采样。
    weights:
        None 表示全 1 权重; 否则每个权重必须为严格正有限数。
    frequencies:
        1..300 个严格递增正频率 (Hz 或 1/时间单位)。重复、非正、乱序
        均抛 PeriodogramInputError, 不静默重排。
    top_k:
        正整数, 排斥选择返回的峰数上限。
    min_separation:
        频率排斥距离 (与频率同单位); 已入选峰的 |Δf| <= 该值的候选被排斥。
        None 时取候选网格平均间距 (f_max-f_min)/(n-1), n==1 时取 +inf。
    alias_threshold:
        归一窗口模阈值, [0, 1] 内实数; 窗口模 >= 该值才报告别名链接。

    异常
    ----
    TypeError: 类型/布尔参数非法。
    PeriodogramInputError: 形状、取值、范围、排序等非法, 消息带定位信息。
    FloatingPointError: 功率显著越出 [-1e-10, 1+1e-10] (不应发生;
        发生说明数值路径有缺陷而非数据问题)。
    """
    # ---- 全部校验先于任何结果分配; 函数无状态、无部分变更。 ----
    t = _check_real_array(times, "times", MIN_OBSERVATIONS, MAX_OBSERVATIONS)
    v = _check_real_array(values, "values", MIN_OBSERVATIONS, MAX_OBSERVATIONS)
    if len(v) != len(t):
        raise PeriodogramInputError(
            f"values 长度 {len(v)} 与 times 长度 {len(t)} 不一致"
        )
    if weights is None:
        w = [1.0] * len(t)
    else:
        w = _check_real_array(weights, "weights", MIN_OBSERVATIONS, MAX_OBSERVATIONS)
        if len(w) != len(t):
            raise PeriodogramInputError(
                f"weights 长度 {len(w)} 与 times 长度 {len(t)} 不一致"
            )
        for i, wi in enumerate(w):
            if wi <= 0.0:
                raise PeriodogramInputError(
                    f"weights[{i}]={wi!r} 必须严格为正 (不静默丢弃零权重点)"
                )
    if frequencies is None:
        raise PeriodogramInputError("frequencies 为必选参数 (1..300 个正频率)")
    fr = _check_frequencies(frequencies)

    _check_int(top_k, "top_k", minimum=1, maximum=MAX_FREQUENCIES)
    if min_separation is None:
        sep = math.inf if len(fr) == 1 else (fr[-1] - fr[0]) / (len(fr) - 1)
    else:
        sep = _check_real_number(min_separation, "min_separation", positive=True)
    if isinstance(alias_threshold, bool) or not isinstance(
        alias_threshold, (int, float)
    ):
        raise TypeError(
            "alias_threshold 必须是实数, 实际为 "
            f"{type(alias_threshold).__name__}"
        )
    alias_threshold = float(alias_threshold)
    if not 0.0 <= alias_threshold <= 1.0:
        raise PeriodogramInputError(
            f"alias_threshold={alias_threshold} 必须落在 [0, 1]"
        )

    n = len(t)
    nf = len(fr)

    # ---- 常量模型 (加权均值) ----
    wsum = math.fsum(w)
    wv = math.fsum(w[i] * v[i] for i in range(n))
    cbar = wv / wsum
    sse0 = math.fsum(w[i] * (v[i] - cbar) ** 2 for i in range(n))
    constant_series = sse0 == 0.0

    # 相位参考历元取第一个观测, 内部一律用 t' = t - t0 计算,
    # 从而时间原点大幅平移不影响功率/幅度 (见 README 误差口径)。
    t0 = t[0]
    tt = [t[i] - t0 for i in range(n)]
    sw = [math.sqrt(w[i]) for i in range(n)]

    points: list[FrequencyPoint] = []

    for f in fr:
        ang = [_TWO_PI * f * tt[i] for i in range(n)]
        c1 = [math.cos(x) for x in ang]
        c2 = [math.sin(x) for x in ang]

        coef, diag = _weighted_qr_fit3(sw, v, [1.0] * n, c1, c2)
        c0, a, b = coef
        rank = sum(1 for d in diag if d >= _RCOND * max(diag))

        pred = [
            c0 + a * c1[i] + b * c2[i] for i in range(n)
        ]
        sse_f = math.fsum(w[i] * (v[i] - pred[i]) ** 2 for i in range(n))

        # 采样窗口 (用 t'; 模对原点不变)。
        wr = math.fsum(w[i] * c1[i] for i in range(n))
        wi_ = math.fsum(w[i] * c2[i] for i in range(n))
        wmod = math.hypot(wr, wi_) / wsum
        if wmod > 1.0 and wmod - 1.0 <= 1e-12:
            wmod = 1.0

        if constant_series:
            power = 0.0
        else:
            power = (sse0 - sse_f) / sse0
            if -_POWER_CLAMP <= power < 0.0:
                power = 0.0
            elif 1.0 < power <= 1.0 + _POWER_CLAMP:
                power = 1.0
            elif not (-_POWER_CLAMP <= power <= 1.0 + _POWER_CLAMP):
                raise FloatingPointError(
                    f"频率 {f} 处功率 {power!r} 越出 [0,1]±{_POWER_CLAMP}; "
                    "加权 LS 理论上保证 SSE_full <= SSE_const"
                )

        # v_i = c + A cos(x + phi): a=A cos phi, b=-A sin phi。
        # 常量序列 (SSE_const==0) 时 a,b 在数学精确解中为 0, QR 只可能
        # 留下 ~eps 的数值噪声; 按退化约定给零幅度/零相位 (README 有述)。
        if constant_series:
            amp = 0.0
            phase = 0.0
        else:
            amp = math.hypot(a, b)
            phase = math.atan2(-b, a)

        points.append(
            FrequencyPoint(
                frequency=f,
                power=power,
                amplitude=amp,
                phase=phase,
                sse=sse_f,
                rank=rank,
                window_modulus=wmod,
                local_peak=False,
            )
        )

    # ---- 严格局部峰 (端点不计; 平台不标记) ----
    peak_flags = [False] * nf
    for i in range(1, nf - 1):
        if points[i].power > points[i - 1].power and points[i].power > points[i + 1].power:
            peak_flags[i] = True
    points = [
        FrequencyPoint(
            frequency=p.frequency,
            power=p.power,
            amplitude=p.amplitude,
            phase=p.phase,
            sse=p.sse,
            rank=p.rank,
            window_modulus=p.window_modulus,
            local_peak=peak_flags[i],
        )
        for i, p in enumerate(points)
    ]
    peaks = [p for p in points if p.local_peak]

    # ---- topK: 功率降序贪心, 平局取低频; 显式频率排斥距离 ----
    ordered = sorted(peaks, key=lambda p: (-p.power, p.frequency))
    chosen: list[FrequencyPoint] = []
    for cand in ordered:
        if all(abs(cand.frequency - q.frequency) > sep for q in chosen):
            chosen.append(cand)
            if len(chosen) == top_k:
                break
    chosen.sort(key=lambda p: p.frequency)

    # ---- 别名: top 峰 vs 其它局部峰 ----
    # 采样窗口 W(g) = Σ w_k exp(i2π g t'_k), 归一模在 g 处接近 1 表示
    # exp(i2π g t'_k) ≈ 1 在所有采样时刻成立。
    #   差频 g=|f2-f1|: 相位平移型歧义 (e^{i2π f2 t}≈e^{i2π f1 t});
    #   和频 g=f2+f1 : 共轭镜像型歧义 (e^{-i2π f1 t}≈e^{i2π f2 t},
    #                  即余弦 Nyquist 折叠 f2 = fs - f1)。
    aliases: list[AliasLink] = []

    def _window_mod(gfreq: float) -> float:
        re_ = math.fsum(
            w[i] * math.cos(_TWO_PI * gfreq * tt[i]) for i in range(n)
        )
        im_ = math.fsum(
            w[i] * math.sin(_TWO_PI * gfreq * tt[i]) for i in range(n)
        )
        return math.hypot(re_, im_) / wsum

    # chosen 按频率升序遍历; 用无序对去重, 每对峰最多报告一次,
    # 报告时 peak_frequency 取两个 top/局部峰中先被选中 (低频) 的一侧。
    seen_pairs: set[frozenset[float]] = set()
    for src in chosen:
        for tgt in peaks:
            if tgt.frequency == src.frequency:
                continue
            key = frozenset((src.frequency, tgt.frequency))
            if key in seen_pairs:
                continue
            f_hi = max(src.frequency, tgt.frequency)
            f_lo = min(src.frequency, tgt.frequency)
            candidates = (
                (f_hi - f_lo, "difference", f_hi - f_lo),
                (f_hi + f_lo, "sum", f_hi - f_lo),
            )
            best = None
            for gfreq, kind, dfreq in candidates:
                wmod_g = _window_mod(gfreq)
                if wmod_g >= alias_threshold and (
                    best is None or wmod_g > best[0]
                ):
                    best = (wmod_g, kind, dfreq, gfreq)
            if best is not None:
                seen_pairs.add(key)
                wmod_g, kind, dfreq, gfreq = best
                aliases.append(
                    AliasLink(
                        peak_frequency=src.frequency,
                        alias_frequency=tgt.frequency,
                        difference_frequency=dfreq,
                        beat_frequency=gfreq,
                        beat_kind=kind,
                        window_modulus=wmod_g,
                        alias_power=tgt.power,
                    )
                )
    aliases.sort(
        key=lambda a: (a.peak_frequency, a.alias_frequency)
    )

    return PeriodogramResult(
        points=tuple(points),
        peaks=tuple(sorted(peaks, key=lambda p: p.frequency)),
        top=tuple(chosen),
        aliases=tuple(aliases),
        constant=cbar,
        sse_constant=sse0,
        total_weight=wsum,
        phase_epoch=t0,
        constant_series=constant_series,
        top_k=top_k,
        min_separation=sep,
        alias_threshold=alias_threshold,
    )
