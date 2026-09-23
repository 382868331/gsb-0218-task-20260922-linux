#!/usr/bin/env python3
"""demo.py — 浮动均值谱估计的固定输入演示 (确定性, 无外部数据)。

运行: python demo.py   (约 1 秒内; 纯标准库, 离线)

内容:
  1. 正常结果: 均匀采样下真实频率 f=2.0 与其采样别名 f=8.0 形成
     等功率双峰, 采样窗口在差频处归一模 ≈ 1 —— 双峰歧义。
  2. 同一信号改用轻微非均匀采样后, 别名峰功率塌缩, 歧义消失
     (不做任何补零/重采样)。
  3. 边界拒绝: 实际触发一个 PeriodogramInputError (NaN) 和一个
     TypeError (bool 冒充整数参数)。
"""

import math
import traceback

import fmperiodogram as fm

TWO_PI = 2.0 * math.pi


def make_signal(times, freq, amp, phase, const):
    return [const + amp * math.cos(TWO_PI * freq * t + phase) for t in times]


def print_peaks(title, res):
    print(title)
    print(f"  加权均值 c = {res.constant:.12g}  SSE_const = {res.sse_constant:.6g}")
    print("  top 峰 (频率排斥选择):")
    for p in res.top:
        print(
            f"    f={p.frequency:5.2f}  P={p.power:.8f}  "
            f"A={p.amplitude:.6f}  phase={p.phase:+.6f} rad  "
            f"rank={p.rank}  |W|/W0={p.window_modulus:.6f}"
        )
    if res.aliases:
        print("  采样窗口报告的别名链接 (当前采样下不可区分):")
        kind_cn = {"difference": "平移型(差频)", "sum": "镜像型(和频)"}
        for a in res.aliases:
            print(
                f"    {a.peak_frequency:5.2f} <-> {a.alias_frequency:5.2f}  "
                f"{kind_cn[a.beat_kind]} 拍频={a.beat_frequency:6.2f}  "
                f"|W|/W0={a.window_modulus:.8f}  "
                f"别名峰功率={a.alias_power:.8f}"
            )
    else:
        print("  无超过阈值的别名链接。")


def main():
    print("=" * 72)
    print("演示 1 (正常结果): 均匀采样的双峰歧义 (Nyquist 别名)")
    print("=" * 72)
    n, dt = 24, 0.1                 # 采样率 fs = 10
    fs = 1.0 / dt
    f_true, amp, phase, const = 2.0, 1.3, 0.35, -0.4
    t_uniform = [i * dt for i in range(n)]
    v = make_signal(t_uniform, f_true, amp, phase, const)
    freqs = [0.5 + 0.1 * k for k in range(86)]   # 0.5 .. 9.0, 共 86 个

    res = fm.periodogram(
        t_uniform, v, None, freqs, top_k=4, min_separation=0.3
    )
    print(
        f"n={n} 个等间隔观测, Δt={dt}, fs={fs:g}, "
        f"候选正频率 {len(freqs)} 个, 真实信号 f={f_true}"
    )
    print_peaks("", res)
    p_true = next(p for p in res.points if abs(p.frequency - f_true) < 1e-12)
    f_alias = fs - f_true
    p_alias = next(p for p in res.points if abs(p.frequency - f_alias) < 1e-12)
    print(
        f"=> f={f_true} 与 f={f_alias:g} 功率差 "
        f"{abs(p_true.power - p_alias.power):.3e}: 两个峰同样好地拟合数据,"
    )
    print("   仅靠这组等间隔采样无法判定真实频率 (和频 fs 处窗口模≈1)。")

    print()
    print("=" * 72)
    print("演示 2: 轻微非均匀采样破除别名 (无补零、无均匀化)")
    print("=" * 72)
    # 确定性抖动: 0.7 倍步长以内, 固定 LCG, 不引入任何外部数据。
    seed = 987654321
    t_jit = []
    base = 0.0
    for _ in range(n):
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        base += dt
        t_jit.append(base + dt * 0.7 * (seed / 0x7FFFFFFF - 0.5))
    v2 = make_signal(t_jit, f_true, amp, phase, const)
    res2 = fm.periodogram(t_jit, v2, None, freqs, top_k=4, min_separation=0.3)
    print_peaks("", res2)
    p2_true = next(p for p in res2.points if abs(p.frequency - f_true) < 1e-12)
    p2_alias = min(
        (p for p in res2.points if abs(p.frequency - f_alias) <= 0.11),
        key=lambda p: abs(p.frequency - f_alias),
        default=None,
    )
    if p2_alias is not None:
        print(
            f"=> 非均匀采样后 f={f_true} 处 P={p2_true.power:.6f}, "
            f"原别名 f≈{p2_alias.frequency:.1f} 处 P={p2_alias.power:.6f},"
        )
        print("   歧义被采样不规则性消除。")

    print()
    print("=" * 72)
    print("演示 3 (异常边界): 非法输入被明确拒绝, 不静默纠正")
    print("=" * 72)
    # 3a. 数值输入含 NaN —— PeriodogramInputError, 消息可定位到下标。
    bad_times = [0.0, 0.1, float("nan"), 0.3]
    bad_values = [1.0, 0.5, -0.2, 0.8]
    try:
        fm.periodogram(bad_times, bad_values, None, [1.0, 2.0])
    except fm.PeriodogramInputError as exc:
        print(f"[已拒绝] {type(exc).__name__}: {exc}")
    else:  # pragma: no cover - demo 自检
        raise AssertionError("NaN 输入应当被拒绝")

    # 3b. bool 冒充整数参数 —— TypeError, 不把 True 当 1。
    try:
        fm.periodogram(t_uniform[:5], v[:5], None, [1.0, 2.0], top_k=True)
    except TypeError as exc:
        print(f"[已拒绝] {type(exc).__name__}: {exc}")
    else:  # pragma: no cover - demo 自检
        raise AssertionError("bool 参数应当被拒绝")

    # 3c. 非正权重同样定位到具体下标。
    try:
        fm.periodogram([0.0, 0.1, 0.2], [1.0, 0.0, -1.0],
                       [1.0, -0.5, 1.0], [1.0])
    except fm.PeriodogramInputError as exc:
        print(f"[已拒绝] {type(exc).__name__}: {exc}")

    print()
    print("演示完成: 以上数值均为本进程实时计算, 无预录结果、无网络访问。")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # pragma: no cover - demo 失败时保留完整堆栈
        traceback.print_exc()
        raise
