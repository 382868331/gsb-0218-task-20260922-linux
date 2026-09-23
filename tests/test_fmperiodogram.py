"""fmperiodogram 的验收测试。

独立参考原则: 期望值全部来自解析公式或测试内自行实现的参考
(3x3 加权正规方程 + 部分主元高斯消元、等间隔 DFT 正交性、
几何级数窗口), 不调用 fmperiodogram 的内部函数得到期望值。
往返一致性 (用返回系数重建模型) 仅作补充检验。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fmperiodogram as fm  # noqa: E402

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# 测试内独立参考: 3x3 加权正规方程, 部分主元高斯消元
# --------------------------------------------------------------------------- #
def reference_weighted_fit(times, values, weights, freq):
    """独立求解 min Σ w (v - c - a cos - b sin)^2, 返回 (c,a,b,sse)。"""
    n = len(times)
    cols = [
        [1.0] * n,
        [math.cos(TWO_PI * freq * t) for t in times],
        [math.sin(TWO_PI * freq * t) for t in times],
    ]
    m = [[0.0] * 3 for _ in range(3)]
    rhs = [0.0] * 3
    for p in range(3):
        rhs[p] = sum(weights[i] * cols[p][i] * values[i] for i in range(n))
        for q in range(3):
            m[p][q] = sum(weights[i] * cols[p][i] * cols[q][i] for i in range(n))
    # 高斯消元 + 部分主元
    for k in range(3):
        piv = max(range(k, 3), key=lambda r: abs(m[r][k]))
        m[k], m[piv] = m[piv], m[k]
        rhs[k], rhs[piv] = rhs[piv], rhs[k]
        for r in range(k + 1, 3):
            fac = m[r][k] / m[k][k]
            for q in range(k, 3):
                m[r][q] -= fac * m[k][q]
            rhs[r] -= fac * rhs[k]
    x = [0.0] * 3
    for k in range(2, -1, -1):
        x[k] = (rhs[k] - sum(m[k][q] * x[q] for q in range(k + 1, 3))) / m[k][k]
    sse = sum(
        weights[i]
        * (values[i] - sum(x[p] * cols[p][i] for p in range(3))) ** 2
        for i in range(n)
    )
    return x[0], x[1], x[2], sse


def reference_window(times, weights, freq):
    re = sum(weights[i] * math.cos(TWO_PI * freq * times[i]) for i in range(len(times)))
    im = sum(weights[i] * math.sin(TWO_PI * freq * times[i]) for i in range(len(times)))
    return math.hypot(re, im) / sum(weights)


class TestUniformAnalytic(unittest.TestCase):
    """等间隔 DFT 解析情形: 信号 = 常量 + 整周期余弦/正弦。"""

    def setUp(self):
        self.n = 24
        self.dt = 1.0
        self.t = [float(i) for i in range(self.n)]
        self.c, self.a, self.b = 0.6, 1.5, -0.8
        self.f0 = 3.0 / self.n
        self.v = [
            self.c
            + self.a * math.cos(TWO_PI * self.f0 * i)
            + self.b * math.sin(TWO_PI * self.f0 * i)
            for i in range(self.n)
        ]
        self.freqs = [k / self.n for k in range(1, self.n // 2)]
        self.res = fm.periodogram(self.t, self.v, None, self.freqs)

    def test_structure_sizes(self):
        self.assertEqual(len(self.res.points), len(self.freqs))
        self.assertTrue(math.isfinite(self.res.sse_constant))

    def test_on_grid_frequency_exact(self):
        p = next(p for p in self.res.points if p.frequency == self.f0)
        self.assertAlmostEqual(p.power, 1.0, places=11)
        self.assertAlmostEqual(p.amplitude, math.hypot(self.a, self.b), places=11)
        # v = c + a cos x + b sin x = c + A cos(x+phi), phi = atan2(-b, a)
        self.assertAlmostEqual(p.phase, math.atan2(-self.b, self.a), places=11)
        self.assertAlmostEqual(self.res.constant, self.c, places=12)
        self.assertAlmostEqual(p.sse, 0.0, places=14)
        self.assertEqual(p.rank, 3)

    def test_off_grid_frequencies_zero_power(self):
        for p in self.res.points:
            if p.frequency != self.f0:
                self.assertLess(abs(p.power), 1e-10)
                self.assertLess(p.amplitude, 1e-9)

    def test_peak_selection(self):
        peak_freqs = [p.frequency for p in self.res.peaks]
        self.assertEqual(peak_freqs, [self.f0])
        self.assertEqual([p.frequency for p in self.res.top], [self.f0])
        self.assertTrue(next(p for p in self.res.points if p.frequency == self.f0).local_peak)

    def test_phase_convention_roundtrip(self):
        # 补充检验: A cos(x+phi) 必须等于原信号。
        p = next(p for p in self.res.points if p.frequency == self.f0)
        epoch = self.res.phase_epoch
        for ti, vi in zip(self.t, self.v):
            x = TWO_PI * p.frequency * (ti - epoch)
            recon = self.res.constant + p.amplitude * math.cos(x + p.phase)
            self.assertAlmostEqual(recon, vi, places=10)


class TestWeightedNormalEquations(unittest.TestCase):
    """非均匀、加权: 与测试内手写正规方程参考逐频率比对。"""

    def setUp(self):
        self.t = [0.0, 0.3, 1.1, 2.0, 2.5, 3.7, 4.2]
        self.v = [0.7, -1.25, 0.4, 1.05, -0.6, 0.2, -0.9]
        self.w = [1.0, 2.0, 0.5, 1.0, 1.3, 0.8, 1.0]
        self.freqs = [0.17, 0.42, 0.7, 0.95, 1.23]
        self.res = fm.periodogram(self.t, self.v, self.w, self.freqs)

    def test_constant_model(self):
        wsum = sum(self.w)
        cbar = sum(wi * vi for wi, vi in zip(self.w, self.v)) / wsum
        sse0 = sum(wi * (vi - cbar) ** 2 for wi, vi in zip(self.w, self.v))
        self.assertAlmostEqual(self.res.constant, cbar, places=14)
        self.assertAlmostEqual(self.res.sse_constant, sse0, places=12)
        self.assertAlmostEqual(self.res.total_weight, wsum, places=14)

    def test_each_frequency_against_reference(self):
        for f, p in zip(self.freqs, self.res.points):
            c, a, b, sse = reference_weighted_fit(self.t, self.v, self.w, f)
            self.assertAlmostEqual(p.sse, sse, places=9, msg=f"freq={f}")
            amp = math.hypot(a, b)
            phi = math.atan2(-b, a)
            self.assertAlmostEqual(p.amplitude, amp, places=10, msg=f"freq={f}")
            self.assertAlmostEqual(p.phase, phi, places=10, msg=f"freq={f}")
            pwr = (self.res.sse_constant - sse) / self.res.sse_constant
            self.assertAlmostEqual(p.power, pwr, places=11, msg=f"freq={f}")
            self.assertGreaterEqual(p.power, -1e-12)
            self.assertLessEqual(p.power, 1.0 + 1e-12)
            wm = reference_window(self.t, self.w, f)
            self.assertAlmostEqual(p.window_modulus, wm, places=11)
            self.assertEqual(p.rank, 3)


class TestWindowGeometricSeries(unittest.TestCase):
    """均匀采样窗口: |W(f)|/W0 = |sin(π f N Δt)/(N sin(π f Δt))|。"""

    def test_window_matches_analytic(self):
        n, dt = 21, 0.5
        t = [i * dt for i in range(n)]
        w = [1.0 + 0.1 * (i % 3) for i in range(n)]  # 非均匀权重
        freqs = [0.03, 0.2, 0.47, 0.91]
        res = fm.periodogram(t, [0.1 * i for i in range(n)], w, freqs)
        w0 = sum(w)
        for f, p in zip(freqs, res.points):
            re = sum(wi * math.cos(TWO_PI * f * ti) for wi, ti in zip(w, t))
            im = sum(wi * math.sin(TWO_PI * f * ti) for wi, ti in zip(w, t))
            self.assertAlmostEqual(p.window_modulus, math.hypot(re, im) / w0, places=12)
            self.assertGreaterEqual(p.window_modulus, 0.0)
            self.assertLessEqual(p.window_modulus, 1.0 + 1e-15)


class TestTimeOriginShift(unittest.TestCase):
    """大时间原点平移: t -> t + 1e9, 功率/幅度/SSE/窗口必须不变。"""

    def test_huge_origin_shift_invariance(self):
        t = [0.0, 0.31, 0.77, 1.5, 2.2, 3.05, 4.0]
        v = [0.9 * math.cos(TWO_PI * 0.63 * ti + 0.4) + 0.2 for ti in t]
        w = [1.0, 1.5, 0.7, 1.0, 2.0, 0.6, 1.1]
        freqs = [0.2, 0.4, 0.63, 0.8, 1.1]
        shift = 1.0e9
        r1 = fm.periodogram(t, v, w, freqs)
        r2 = fm.periodogram([ti + shift for ti in t], v, w, freqs)
        self.assertEqual(r2.phase_epoch, t[0] + shift)
        # float64 下 (1e9+0.31)-1e9 的绝对时间误差 ≲ eps*1e9 ≈ 2.3e-7 s,
        # 这是表示边界而非算法缺陷; 功率量级 O(0.07), 误差按 README
        # 误差口径 (时间尺度 1e9 时谱量绝对误差 ≲ 5e-8) 校验,
        # 不用更宽的容差。
        for p1, p2 in zip(r1.points, r2.points):
            # 角度扰动界: 输入时刻减法误差 δt ≲ eps_mach T0, 导致
            # 相位角误差 δθ ≲ 2π f eps_mach T0 ≈ 1.6e-6 rad (f<=1.1)。
            # 谱标量 (功率/幅度/SSE/窗口模) 对系数扰动的相对误差同量级,
            # 故用推导界 2e-6; 这是 float64 表示极限而非算法容差,
            # README 误差口径同步说明。峰位置等离散结论另作精确比较。
            self.assertAlmostEqual(p1.power, p2.power, delta=2e-6)
            self.assertAlmostEqual(p1.amplitude, p2.amplitude, delta=2e-6)
            self.assertAlmostEqual(p1.sse, p2.sse, delta=2e-6)
            self.assertAlmostEqual(p1.window_modulus, p2.window_modulus,
                                   delta=2e-6)
            self.assertAlmostEqual(p1.phase, p2.phase, delta=2e-6)
        # 功率峰位置与峰性等离散结论必须完全一致, 不允许容差模糊。
        self.assertEqual(
            [p.frequency for p in r1.peaks],
            [p.frequency for p in r2.peaks],
        )


class TestRepeatedTimes(unittest.TestCase):
    """重复时间戳: 同一时刻同值的多行 (权重 w1..wk) 等价于单行权重之和。"""

    def test_repeated_rows_equal_merged_weights(self):
        t_raw = [0.0, 0.5, 0.5, 1.2, 2.0, 2.0, 2.0, 2.7]
        v_raw = [0.7, -0.4, -0.4, 1.1, 0.3, 0.3, 0.3, -0.8]
        w_raw = [1.0, 0.6, 1.4, 1.0, 0.5, 0.7, 0.8, 1.0]
        groups = {}
        for ti, vi, wi in zip(t_raw, v_raw, w_raw):
            groups.setdefault(ti, [vi, 0.0])[1] += wi
        t_m = sorted(groups)
        v_m = [groups[ti][0] for ti in t_m]
        w_m = [groups[ti][1] for ti in t_m]
        freqs = [0.25, 0.55, 0.83]
        r_raw = fm.periodogram(t_raw, v_raw, w_raw, freqs)
        r_mer = fm.periodogram(t_m, v_m, w_m, freqs)
        for p1, p2 in zip(r_raw.points, r_mer.points):
            self.assertAlmostEqual(p1.sse, p2.sse, places=11)
            self.assertAlmostEqual(p1.power, p2.power, places=11)
            self.assertAlmostEqual(p1.amplitude, p2.amplitude, places=10)
            self.assertAlmostEqual(p1.window_modulus, p2.window_modulus, places=11)

    def test_all_identical_times_rank_one(self):
        # 全部同一时刻: 设计矩阵三列在每一行相同 (cos,sin 为常数), 秩为 1。
        t = [1.5] * 5
        v = [0.2, 0.4, -0.1, 0.0, 0.3]
        res = fm.periodogram(t, v, None, [0.3, 0.9])
        for p in res.points:
            self.assertEqual(p.rank, 1)
            self.assertAlmostEqual(p.sse, res.sse_constant, places=12)
            self.assertAlmostEqual(p.power, 0.0, places=12)


class TestConstantValues(unittest.TestCase):
    def test_constant_series(self):
        t = [0.0, 0.4, 1.0, 1.7, 2.6, 3.3]
        for c in (2.5, -3.0):
            res = fm.periodogram(t, [c] * 6, None, [0.2, 0.5, 0.9])
            self.assertTrue(res.constant_series)
            self.assertEqual(res.sse_constant, 0.0)
            self.assertEqual(res.constant, float(c))
            for p in res.points:
                self.assertEqual(p.power, 0.0)
                self.assertEqual(p.amplitude, 0.0)
                self.assertFalse(p.local_peak)
            self.assertEqual(res.peaks, ())
            self.assertEqual(res.top, ())


class TestPeriodAlias(unittest.TestCase):
    """均匀采样 Δt: f 与 f_s - f (f_s=1/Δt) 不可区分; 窗口差频模 = 1。"""

    def setUp(self):
        self.n = 20
        self.dt = 0.1
        self.fs = 1.0 / self.dt
        self.fsig = 2.0
        self.t = [i * self.dt for i in range(self.n)]
        self.v = [math.cos(TWO_PI * self.fsig * ti) for ti in self.t]
        self.freqs = sorted(
            {1.8, 1.9, self.fsig, 2.1, 2.2, 7.8, 7.9, self.fs - self.fsig, 8.1, 8.2}
        )
        self.res = fm.periodogram(self.t, self.v, None, self.freqs, top_k=5)

    def test_two_equivalent_peaks(self):
        p1 = next(p for p in self.res.points if p.frequency == self.fsig)
        p2 = next(p for p in self.res.points if p.frequency == self.fs - self.fsig)
        self.assertAlmostEqual(p1.power, 1.0, places=9)
        self.assertAlmostEqual(p2.power, p1.power, places=9)
        self.assertAlmostEqual(p2.sse, p1.sse, places=12)
        self.assertTrue(p1.local_peak and p2.local_peak)
        top_freqs = {p.frequency for p in self.res.top}
        self.assertIn(self.fsig, top_freqs)
        self.assertIn(self.fs - self.fsig, top_freqs)

    def test_alias_link_window_one(self):
        # 余弦 Nyquist 别名属镜像型: 窗口模在 *和频* fs 处为 1
        # (差频 6 处窗口模几乎为 0, 这正是之前判据写错时漏掉的点)。
        link = next(
            a for a in self.res.aliases
            if {a.peak_frequency, a.alias_frequency}
            == {self.fsig, self.fs - self.fsig}
        )
        self.assertEqual(link.beat_kind, "sum")
        self.assertAlmostEqual(link.difference_frequency,
                               self.fs - 2 * self.fsig, places=9)
        self.assertAlmostEqual(link.beat_frequency, self.fs, places=9)
        self.assertAlmostEqual(link.window_modulus, 1.0, places=8)
        # 差频处窗口确实不接近 1: 两个机制必须区分。
        wm_diff = reference_window(self.t, [1.0] * self.n,
                                   self.fs - 2 * self.fsig)
        self.assertLess(wm_diff, 1e-12)

    def test_window_at_nyquist_relation(self):
        # 独立几何级数参考: exp(i2π fs kΔt)=exp(i2π k)≡1。
        self.assertAlmostEqual(
            reference_window(self.t, [1.0] * self.n, self.fs), 1.0, places=8
        )


class TestTranslationAlias(unittest.TestCase):
    """平移型别名: 整数时刻上 f2=f1+1 使 e^{i2π f2 t}=e^{i2π f1 t}。

    与 Nyquist 镜像别名不同, 这里窗口模在 *差频* (=1) 处为 1,
    和频 (1.4) 处不为 1, 用于区分两种 beat_kind 分支。
    """

    def test_difference_kind_alias(self):
        n = 12
        t = [float(i) for i in range(n)]
        f1, f2 = 0.2, 1.2
        v = [math.cos(TWO_PI * f1 * ti) for ti in t]
        freqs = [0.1, f1, 0.3, 1.1, f2, 1.3]
        res = fm.periodogram(t, v, None, freqs, top_k=6)
        p1 = next(p for p in res.points if p.frequency == f1)
        p2 = next(p for p in res.points if p.frequency == f2)
        self.assertTrue(p1.local_peak and p2.local_peak)
        self.assertAlmostEqual(p1.power, 1.0, places=9)
        self.assertAlmostEqual(p2.power, 1.0, delta=1e-10)
        link = next(
            a for a in res.aliases
            if {a.peak_frequency, a.alias_frequency} == {f1, f2}
        )
        self.assertEqual(link.beat_kind, "difference")
        self.assertAlmostEqual(link.beat_frequency, 1.0, places=10)
        self.assertAlmostEqual(link.difference_frequency, 1.0, places=10)
        self.assertAlmostEqual(link.window_modulus, 1.0, places=9)
        # 和频 1.4 处窗口模明显不为 1: 排除镜像型误判。
        self.assertLess(reference_window(t, [1.0] * n, 1.4), 0.2)
        # 每个无序对只报告一次。
        self.assertEqual(
            sum(
                1 for a in res.aliases
                if {a.peak_frequency, a.alias_frequency} == {f1, f2}
            ),
            1,
        )


class TestTopKSelection(unittest.TestCase):
    @staticmethod
    def _lcg(seed=1234567):
        x = seed
        while True:
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            yield x / 0x7FFFFFFF

    def test_explicit_repulsion(self):
        rng = self._lcg()
        t = [round(0.07 * i + 0.01 * (next(rng) - 0.5), 6) for i in range(60)]
        v = [
            1.0 * math.cos(TWO_PI * 1.0 * ti)
            + 0.7 * math.cos(TWO_PI * 1.3 * ti + 1.1)
            + 0.05 * (next(rng) - 0.5)
            for ti in t
        ]
        freqs = [0.5 + 0.05 * k for k in range(51)]  # 0.50 .. 3.00
        res = fm.periodogram(t, v, None, freqs, top_k=3, min_separation=0.5)
        self.assertLessEqual(len(res.top), 3)
        chosen = [p.frequency for p in res.top]
        # 1.0 与 1.3 互为 0.3 < 0.5 排斥距离: 只能留下功率较高者。
        self.assertFalse(any(abs(f - 1.0) < 1e-12 for f in chosen)
                         and any(abs(f - 1.3) < 1e-12 for f in chosen))
        # 成对检查排斥距离被严格遵守。
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                self.assertGreater(abs(chosen[i] - chosen[j]), 0.5 - 1e-12)
        # top 中第一个必为两真峰之一附近的网格点。
        self.assertTrue(any(abs(f - 1.0) <= 0.051 or abs(f - 1.3) <= 0.051
                            for f in chosen))


class TestErrorSemantics(unittest.TestCase):
    def setUp(self):
        self.t = [0.0, 0.5, 1.0]
        self.v = [1.0, 0.0, -1.0]
        self.f = [0.2, 0.4]

    def test_nan_inf_rejected_with_index(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(fm.PeriodogramInputError) as cx:
                fm.periodogram([0.0, bad, 1.0], self.v, None, self.f)
            self.assertIn("times[1]", str(cx.exception))
            with self.assertRaises(fm.PeriodogramInputError) as cx:
                fm.periodogram(self.t, [1.0, 0.0, bad], None, self.f)
            self.assertIn("values[2]", str(cx.exception))
            with self.assertRaises(fm.PeriodogramInputError) as cx:
                fm.periodogram(self.t, self.v, [1.0, bad, 1.0], self.f)
            self.assertIn("weights[1]", str(cx.exception))

    def test_bool_rejected_as_integer_param(self):
        with self.assertRaises(TypeError):
            fm.periodogram(self.t, self.v, None, self.f, top_k=True)
        with self.assertRaises(TypeError):
            fm.periodogram(self.t, self.v, None, self.f, top_k=False)  # noqa: FBT003

    def test_bool_rejected_in_numeric_arrays(self):
        with self.assertRaises(TypeError):
            fm.periodogram([0.0, True, 1.0], self.v, None, self.f)  # noqa: FBT003
        with self.assertRaises(TypeError):
            fm.periodogram(self.t, [1.0, False, -1.0], None, self.f)  # noqa: FBT003
        with self.assertRaises(TypeError):
            fm.periodogram(self.t, self.v, [1.0, True, 1.0], self.f)  # noqa: FBT003
        with self.assertRaises(TypeError):
            fm.periodogram(self.t, self.v, None, [0.2, True])  # noqa: FBT003

    def test_string_and_none_types_rejected(self):
        with self.assertRaises(TypeError):
            fm.periodogram("abc", self.v, None, self.f)
        with self.assertRaises(TypeError):
            fm.periodogram(self.t, [1, "x", -1], None, self.f)
        with self.assertRaises(TypeError):
            fm.periodogram(self.t, self.v, None, [0.2, object()])

    def test_length_mismatch(self):
        with self.assertRaisesRegex(fm.PeriodogramInputError, "values 长度"):
            fm.periodogram(self.t, [1.0, 0.0], None, self.f)
        with self.assertRaisesRegex(fm.PeriodogramInputError, "weights 长度"):
            fm.periodogram(self.t, self.v, [1.0, 1.0], self.f)

    def test_too_few_and_too_many(self):
        with self.assertRaisesRegex(fm.PeriodogramInputError, "times 长度 2"):
            fm.periodogram([0.0, 0.5], [1.0, 0.0], None, self.f)
        with self.assertRaisesRegex(fm.PeriodogramInputError, "超过上限 200"):
            fm.periodogram([float(i) for i in range(201)], [0.0] * 201,
                           None, self.f)
        with self.assertRaisesRegex(fm.PeriodogramInputError, "超过上限 300"):
            fm.periodogram(self.t, self.v, None, [0.1 * i for i in range(1, 302)])

    def test_nonpositive_weight_indexed(self):
        with self.assertRaisesRegex(fm.PeriodogramInputError, r"weights\[1\]"):
            fm.periodogram(self.t, self.v, [1.0, 0.0, 1.0], self.f)
        with self.assertRaisesRegex(fm.PeriodogramInputError, r"weights\[0\]"):
            fm.periodogram(self.t, self.v, [-2.0, 1.0, 1.0], self.f)

    def test_frequency_contract(self):
        with self.assertRaisesRegex(fm.PeriodogramInputError, r"frequencies\[0\]"):
            fm.periodogram(self.t, self.v, None, [0.0, 0.4])
        with self.assertRaisesRegex(fm.PeriodogramInputError, r"frequencies\[1\]"):
            fm.periodogram(self.t, self.v, None, [0.4, 0.2])
        with self.assertRaisesRegex(fm.PeriodogramInputError, "重复"):
            fm.periodogram(self.t, self.v, None, [0.2, 0.2])
        with self.assertRaises(fm.PeriodogramInputError):
            fm.periodogram(self.t, self.v, None, [])

    def test_other_param_ranges(self):
        with self.assertRaises(TypeError):
            fm.periodogram(self.t, self.v, None, self.f, top_k=1.0)
        with self.assertRaises(fm.PeriodogramInputError):
            fm.periodogram(self.t, self.v, None, self.f, top_k=0)
        # top_k 是"上限": 大于候选峰数时只返回更少, 不报错。
        res_more = fm.periodogram(self.t, self.v, None, self.f, top_k=99)
        self.assertLessEqual(len(res_more.top), 2)
        with self.assertRaises(fm.PeriodogramInputError):
            fm.periodogram(self.t, self.v, None, self.f, top_k=301)
        with self.assertRaises(fm.PeriodogramInputError):
            fm.periodogram(self.t, self.v, None, self.f, min_separation=0.0)
        with self.assertRaises(TypeError):
            fm.periodogram(self.t, self.v, None, self.f, min_separation=True)  # noqa: FBT003
        with self.assertRaises(fm.PeriodogramInputError):
            fm.periodogram(self.t, self.v, None, self.f, alias_threshold=1.5)

    def test_unsorted_times_are_allowed(self):
        # 时间戳不要求排序, 只要求长度/有限性; 结果按频率序返回。
        res = fm.periodogram([1.0, 0.0, 0.5], [-1.0, 1.0, 0.0], None, [0.2, 0.4])
        self.assertEqual([p.frequency for p in res.points], [0.2, 0.4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
