# 非均匀采样浮动均值谱估计 / Floating-mean periodogram

面向非均匀采样天文时间序列的可复用周期图模块：在每个候选频率上拟合
`y = c + a·cos(2πft) + b·sin(2πft)`（常数均值**浮动**，而非先扣除均值），
采用加权 Householder QR 最小二乘；功率定义为相对常量模型的加权残差平方和
（SSE）减少比例。另计算采样窗口的归一化模用于解释周期别名。

在此基础上，refine_peak(...) 在调用方给定的有限闭频率区间内先做确定性分段，
再对每个局部 SSE 极小值做连续频率黄金分割细化；每个尝试频率仍重新消去常数、
余弦、正弦三个线性系数，粗扫描点本身不会被直接宣布为连续最优。

**不使用 FFT、不把不等间隔观测插值/补零成均匀序列**；每个观测就是设计矩阵的
一行，重复时间、未排序时间均可直接处理。

- Python：**3.14.7**（仅用标准库 `math`、`dataclasses`、`numbers`、`collections.abc`）
- 第三方依赖：**无**；安装 Python 后即可运行
- 完全离线：无网络、账号、密钥、数据库、中间件、云 API、Docker
- 示例数据由仓库内确定性 LCG 代码生成，无外部附件

## 运行

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip       # 可选；requirements.txt 为空
python -m pip install -r requirements.txt # 无第三方包，离线环境也可跳过
python -m unittest discover -s tests -v   # 32 个必要测试
python demo_iteration.py                  # 约 0.2 秒的连续精化真实计算
python demo.py                            # 基线固定网格谱演示，约 0.1 秒
```

全部计算在本机 Linux 原生 Python 3.14.7 验证；venv 仅用于隔离，不访问外部
服务，也不需要账号、密钥或 Docker（见文末“误差口径与验证结果”）。

## 接口

```python
from periodogram import periodogram, PeriodogramError, PeriodogramResult

res = periodogram(times, values, weights, frequencies,
                  *, top_k=None, min_separation=0.0,
                  rank_tol=None, selection_budget=None)
```

| 参数 | 契约 |
|---|---|
| `times` | list/tuple 实数，有限（拒绝 NaN/±Inf），长度 3..200；可不排序、可重复 |
| `values` | 同长度有限实数 |
| `weights` | 同长度**严格正**有限权重；`None` 表示等权。全局权重缩放不改变结果 |
| `frequencies` | 1..300 个候选频率，**严格递增、严格为正**的有限实数（单位：周/时间单位） |
| `top_k` | 正整数且 ≤ 候选数；`None` 表示 `min(5, 候选数)`。**拒绝 bool** |
| `min_separation` | 非负有限频率排斥距离；已选峰与候选 `|Δf| < min_separation` 时排斥，`|Δf| == min_separation` **允许**（边界内闭） |
| `rank_tol` | 数值秩阈值，`|R_jj| > rank_tol·max|R_ii|` 判为非零；默认 `n·eps`；可覆盖范围 `(0,1)` |
| `selection_budget` | top-K 选择时检查的候选数上限；`None` = 全部候选（穷举扫描） |

返回不可变值对象 `PeriodogramResult`（逐频率数组 + 峰列表）：

- `frequencies, powers, amplitudes, phases, ranks, constants, sses`：每个候选
  频率的功率、幅度 `hypot(a,b)`、相位 `atan2(b,a)∈(−π,π]`、设计矩阵数值秩、
  拟合常数 c、加权 SSE。
- `sse0`、`weighted_mean`：常量模型 SSE 与加权均值。
- `window`（复数，τ=t−times[0] 参考系）与 `window_modulus`
  `= |Σ w_k exp(i2πft_k)| / Σw_k ∈ [0,1]`；模与时间原点无关，用于别名解释。
- `local_peaks`：严格局部功率极大（`P[i-1]<P[i]>P[i+1]`，端点永不算峰）。
- `top_peaks`：按功率降序（功率并列时频率升序）贪心选取，施加显式排斥距离；
  元素为 `FrequencyPeak(index, frequency, power, amplitude, phase, rank, sse,
  window_modulus)`。
- `selection_status`：
  - `"complete"`：扫描了全部候选；若 `top_k_found < top_k_requested`，
    这是在确定性贪心规则下**可证明无更多可选峰**的结果；
  - `"budget_exhausted"`：`selection_budget` 用尽、扫描提前终止；
    对未检查候选不作任何断言。**预算耗尽与证明无解严格分开**。

### 功率定义

`P(f) = (SSE0 − SSE(f)) / SSE0`，其中 `SSE0 = min_c Σw(y−c)²`（浮动均值常量
模型），`SSE(f)` 为三参数模型的最小二乘残差。普通条件下 `0 ≤ P ≤ 1`。

## 连续峰细化

调用：

    refine_peak(times, values, weights, frequency_min, frequency_max,
                evaluation_budget=256, scan_points=81, rank_tol=None)

frequency_min/frequency_max 是有限正实数，区间为闭区间且 max > min。
evaluation_budget 为正 int（最小 5，拒绝 bool），统计真实新频率目标求值次数；
scan_points 为不少于 5 的初始分段扫描点数，超过预算时由预算截断。

算法先在区间内做确定性扫描，只把它用于切出多个局部候选；每个候选再在相邻
扫描点构成的括号内做连续黄金分割。目标函数是“线性系数消元后的加权 SSE(f)”，
因此不是取格点最大值，也不新增双频或多频模型。黄金分割不依赖导数；模块另外
提供 single_sine_fit(...) 与 single_sine_derivative(...)，便于单点拟合与
局部导数核对。

RefinePeakResult 返回：

- best：内部候选和两个闭区间端点中 SSE 最低的结果；
- candidates：分段找到并局部细化的候选，含 frequency、amplitude、phase、
  constant、sse、rank、segment_index、evaluations、stopping_reason；
- endpoints：左右端点拟合及 preferred（refined、left、right、tie、none）；
- scan_frequencies、rank_deficient_frequencies：初始扫描频率和秩亏位置；
- evaluations、evaluation_budget 与总 stopping_reason。

总停止原因包括 converged、budget_exhausted、rank_deficient、iteration_limit。
区间可能含多个峰或由采样产生多个旁瓣；预算只保证优先细化扫描发现的候选，预算
不足时不宣称全局最优。设计矩阵秩不足的频率不会成为好峰；候选为满秩三点时才
进入局部优化。

## 错误语义

- 所有数值输入拒绝 NaN / Infinity；整数参数（`top_k`、`selection_budget`）
  拒绝 `bool`（`True/False` 不会被当作 1/0），数组内的布尔元素同样拒绝。
- 非法输入抛 `PeriodogramError`（`ValueError` 子类），消息**点名参数与元素
  下标**（如 `values[7]: NaN is not allowed`、`frequencies[2]: candidates must
  be strictly increasing`），不静默纠正、不截断、不重排用户数据。
- 全部输入校验先于任何计算；函数无任何外部可观察的部分变更（纯函数，
  不写文件、不修改入参），不存在“部分完成”状态。
- 权重必须严格为正（零/负权重直接拒绝，而不是悄悄丢点）。

## 设计取舍

1. **加权 Householder QR（逐行构造 W^½A，3 列）**：不组正规方程，避免条件数
   平方化；n≤200、m≤300 下纯 Python 约 0.1 秒。这是本题核心算法，不调用任何
   第三方谱分析实现。
2. **时间原点重定中心**：内部以 `τ = t − times[0]` 拟合（大原点下保证三角
   参数有界），再用闭式旋转把 (a,b) 转回原时间参考系。功率、幅度、窗口模对
   原点平移在数学上严格不变；`window` 的复相位以 τ 参考系报告。
3. **SSE 直接按拟合模型残差计算**：当某列被数值秩判定跳过（秩亏）时，
   `Qᵀb` 的残差能量仍分布在被跳过列方向，直接取尾部会漏算；用投影拟合值
   计算对任意秩都无歧义。
4. **逐位常量值约定**：当所有 `values` 位级相等时，精确返回 `SSE0=0`、
   每频率 `P=0`、`a=b=0`、`c=该常量`，避免 0/0（这是文档化约定，不是用容差
   掩盖）。非位级常量的近常量数据照常按浮点计算。
5. **无插值/无网格/无 FFT**：候选频率由调用方显式给出；模块不假设均匀采样，
   也不做任何零填充。
6. **采样窗口**：报告 `|W|/Σw` 而非原始复和，天然归一到 `[0,1]` 且对权重
   全局缩放不变；均匀采样在 `f_s=1/Δt` 处模为 1，直接指示别名间隔。

## 已知限制（本题数学子集，不宣称行业标准兼容）

- 仅实现本题契约：单变量时间、固定网格三参数正弦+常量谱，以及连续频率上的
  单正弦细化；标量权重；不做多谐波、双频/多信号联合拟合、误差棒传播、
  显著性/FAP 统计检验、Lomb–Scargle 标准时移或任何行业标准合规声明。
- 连续细化是局部搜索：初始分段扫描可能漏掉窄于扫描间距的峰；多峰、旁瓣或
  预算过小时不保证全局最优，并通过 stopping_reason="budget_exhausted" 明示。
- 固定网格 periodogram(...) 的峰位置仍受候选网格步长限制；离线格点信号会有
  频谱泄漏（幅度被低估），这是物理效应而非数值缺陷。需要亚格点频率时对有限
  区间调用 refine_peak(...)。
- `phases` 在幅度接近 0 的 bin 上无物理意义（atan2(0,0)=0 的延续）。
- 观测时间跨度与周期相比极大时，二进制64浮点本身的时间量化会改变输入问题
   （例如原点平移 1e9 后相对时间误差可达 ~2·ulp(1e9)≈2.4e-7）；模块在
  “给定的浮点输入”上精确求解，但不消除输入表示误差。
- 纯解释器实现，规模上限即接口边界（200×300）；不追求更大规模性能。

## 误差口径（README 明确承诺，不用宽容差掩盖边界）

二进制64 浮点，机器精度 `eps = 2.220446049250313e-16`。测试中的独立参考
（等间隔 DFT 闭式和、3×3 手工加权正规方程高斯消元、窗口直接几何求和）
**不调用被测核心**产生期望值；与被测代码的往返一致仅作补充检查。

- **等间隔、信号落在 DFT 格点**：残差 SSE `< 1e-24`（相对功率 ≈ 1，
  达 `places=10`），幅度/相位/常数与解析 DFT 系数一致到 `places=10..12`。
- **非均匀时间 + 任意正权重**：QR 与独立正规方程参考逐 bin 比较
  `SSE/SSE0`，容差 `1e-18 + 1e-12·量值`（即 ~12 位有效数字）；
  幅度、相位、常数 `places=10`。在最病态 bin（拟合系数 ~50）正规方程自身
  条件数平方化、约有 1e-10 绝对误差，测试显式以
  `1e-11 + 1e-10·max(1,|x|)` 为界并要求 QR 不劣于参考——该容差是正规方程
  的已知误差界，非掩盖边界错误的“大网”。
- **大原点平移（1e9）**：两帧各自对照独立参考（参考也在小角度 τ 帧求解）；
  跨帧相位旋转律 `φ₂−φ₁ = 2πfT` 以 `places=5` 检验——该比较中的旋转角
  高达 ~2.4e9 rad，`places=5` 对应约 2e-15 的**相对**精度。输入时间量化
  （≤2·ulp(T)，测试中先实测上界）作为真实输入扰动单独说明。
- **窗口模**：与直接复数几何求和一致到 `places=12..13`，且 ≤ 1+1e-15。
- **别名/秩亏/重复时间**：镜像别名两 bin 功率差 `places=12`；秩亏采样
  功率精确为 0、秩 ≤2；重复时间与独立正规方程一致到 `places=10`。

## 测试与 demo 内容

`tests/test_periodogram.py`（32 个用例）覆盖：

- 等间隔 DFT 解析恢复（格点纯谐波功率=1、闭式 DFT 系数）；
- 手工加权正规方程独立参考（非均匀时间、非均匀权重、权重全局缩放不变）；
- 大时间原点平移（两帧对独立参考 + 解析相位旋转律）、重复时间、未排序时间、
  采样秩亏；
- 常量值（精确零功率约定）、周期镜像别名（`f0` 与 `f_s−f0` 等权，窗口解释）；
- 局部峰/端点规则、排斥距离边界内闭、并列确定性、预算耗尽 vs 证明无解；
- 非格点频率与不规则时间采样的连续细化、两个局部峰的分段后分别局部细化；
- 独立高密度扫描 + 局部枚举核对、消元目标解析导数与独立有限差分核对；
- 细化中的秩亏拒绝、预算耗尽、闭区间端点比较与新输入校验；
- 全部输入校验（NaN/Inf、bool、长度/上限、频率递增与为正、类型可定位错误）。

当前测试总数为 32 个（基线 25 个 + 连续精化 7 个），统一命令仍为
python -m unittest discover -s tests -v。

python demo_iteration.py（固定输入、确定性、实测约 0.2 秒，远低于 90 秒预算）：

1. 使用 45 个不规则时间点构造两个局部单正弦拟合峰，粗网格步长 0.02；
2. refine_peak(...) 分段后连续细化两个候选，报告振幅、相位、SSE、秩、
   实际评估次数、停止原因与左右端点比较；
3. 用独立 3×3 正规方程在最优点附近做 2401 点局部枚举，并比较消元 SSE；
4. 用独立正规方程中心差分检查 single_sine_derivative(...)；
5. 实际传入倒置区间并捕获 PeriodogramError。

基线 python demo.py（固定输入、确定性、约 0.1 秒）：

1. **正常结果**：25 晚 × 3 次的夜间观测节奏、注入 f=1.70/day 信号；
   真实计算出 `1.70 / 0.70 / 2.70` 三峰歧义（P≈0.999 与 0.989），
   采样窗口在 1.00/day 处 |W|/Σw≈0.995 解释周日别名；主峰再用独立
   3×3 正规方程现场核对（差异 ~1e-16）。
2. **实际触发的拒绝**：NaN 观测与 `top_k=True` 都真正抛出
   `PeriodogramError` 并打印可定位消息（不是预录文本）。

## 文件

```
periodogram.py             # 核心模块（唯一库文件）
demo.py                    # 固定输入演示
tests/test_periodogram.py  # 独立参考 + 32 个 unittest 用例
demo_iteration.py          # 连续峰细化、枚举与有限差分演示
README.md
requirements.txt           # 无第三方依赖（仅注释说明）
```
