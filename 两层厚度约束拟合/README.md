# 两层厚度约束拟合 — 集成进主 GUI

适用场景：相邻两层样品的**总厚度已知**（已用千分尺/SEM/台阶仪等量出），但每一层各自的厚度未知。
通过 3ω 频谱反推两层各自的厚度，**可同时联合拟合传感器参数和其它材料参数（k_y、Rc 等）**。

> 本目录基于主程序 `thermal_measurement_gui2.py` **复制并扩展**而来。
> 主程序源文件**未做任何修改**，所有改动只在本目录的脚本里。

---

## 数学原理

设两相邻层厚度分别为 \(H_1, H_2\)，已知 \(H_\text{total} = H_1 + H_2\)。
引入归一化变量 \(f = H_1/H_\text{total} \in (0,1)\)，则：

\[
H_1 = f \cdot H_\text{total},\qquad H_2 = (1-f)\cdot H_\text{total}
\]

只用 **1 个**自由变量 \(f\) 同时表达两层厚度，**总厚度严格满足**。
该 \(f\) 加进非线性最小二乘 (`scipy.optimize.least_squares`，TRF 信赖域)，
迭代最小化 \(\|V_{3\omega}^\text{exp} - V_{3\omega}^\text{th}\|\) 收敛到最优分配。

若用户在 UI 上额外勾选材料/传感器参数（如 `k_y`、`Rc`、`R`、`dR_dT`...），
拟合维度变为 \(1 + N_\text{extra}\)，**总厚度约束依然严格成立**。

---

## 文件结构

```
两层厚度约束拟合/
├── two_layer_thickness_fitter.py        # 核心拟合器（可独立导入）
├── run_fit_example.py                   # 命令行示例（顶部配置 + 运行）
├── thermal_measurement_gui_thickness.py # 主 GUI（在主程序基础上扩展，新增模式）
└── README.md
```

---

## 启动方式

### 推荐：扩展后的主 GUI

```bash
cd 两层厚度约束拟合
python thermal_measurement_gui_thickness.py
```

启动的窗口和主程序完全一致，但是：
- 标题栏：`3ω热物性测量系统 v1.1 (含两层厚度约束拟合)`
- 「数据拟合」标签页的"拟合模式"下拉里**多了一项**：`两层厚度约束拟合`
- 「数据拟合」内多了一个标签页：**「两层厚度约束」**

### 备用：命令行脚本

```bash
python run_fit_example.py
```
适合批处理、CI 或离线分析（详见脚本顶部 CONFIG）。

---

## GUI 使用步骤（两层厚度约束拟合）

1. **加载数据**（数据拟合页 →「数据加载」）：选择主程序导出的 `.xlsx`/`.csv`，
   自动识别 `V3_Real/V3_Imag` 或 `V3_幅值/V3_相位` 列。

2. **传感器参数**：用预设或手工填 `b/L/R/dR_dT/V_rms/H_e/rho_C_e`。
   - 若想同时拟合传感器参数（例如 `R`、`dR_dT`），在该模式下传感器参数行的
     「拟合」复选框会自动可见，**勾选**对应项即可，可填可选范围。

3. **材料层设置**（GUI 顺序：顶层=0，基底在最后一层）：
   - 给每一层选材料预设、调整热导率/密度/比热容等
   - 想要联合拟合的层参数 (`k_y`、`Rc`...) 也可以勾选「拟合」
   - **重要：不要勾选耦合两层的 `H` 作为拟合参数**（程序会强制阻止）

4. **拟合模式**下拉：选择 `两层厚度约束拟合`，软件会自动跳到「两层厚度约束」标签页。

5. **两层厚度约束**标签页：
   - **顶层索引** / **下层索引**：选要做厚度耦合的相邻两层（下层下拉会自动只显示相邻候选）
   - **H_total (m)**：两层总厚度，例如 4 μm 填 `4e-6`
   - **f 初值 / f 下限 / f 上限**：归一化比例的初始与边界
   - **最大迭代**：least_squares 的最大函数评价次数（默认 400 足够）
   - **「拟合前扫描 cost(f) 曲线」**：勾上后会在 81 个 f 点扫一次目标函数，
     右侧可视化页会**额外画出 cost(f) 曲线**，便于判断解是否唯一

6. **拟合方法**下拉：`Amplitude` / `Phase` / `Real+Imag` / `Amplitude+Phase` /
   `Real Part Only` / `Imaginary Part Only`，按数据信噪比情况选用。

7. **点「开始拟合」** → 自动跳到「结果可视化」页：
   - 上左：V3 幅值 — 实验 vs 初值 vs 拟合
   - 下左：V3 相位 — 同上
   - 右侧：cost(f) 扫描曲线 + 拟合落点 (单峰说明解唯一)
   - 拟合结果同时显示在「拟合结果」表格里：
     - 厚度约束行：`H_total`、`f`、`H_top`、`H_bot`
     - 传感器/材料层联合拟合行（若勾选了）

8. **「导出结果」**：自动保存 JSON，字段包含
   `H_total_m`、`f_fraction`、`H_top_m`、`H_bot_m`、`extra_params`、`cost` 等。

---

## 三种调用方式

### 方式 A：扩展后的主 GUI（最推荐）
见上文。

### 方式 B：作为 Python 模块导入

```python
import numpy as np, pandas as pd
from two_layer_thickness_fitter import TwoLayerThicknessFitter

df = pd.read_excel('measurement.xlsx')
freqs = df['频率 (Hz)'].values
V3 = df['V3_Real'].values + 1j * df['V3_Imag'].values

sensor = {'b': 250e-6, 'L': 8e-3, 'R': 6.225, 'dR_dT': 0.0099,
          'V_rms': 0.264, 'H_e': 100e-9, 'rho_C_e': 2.5116e6}

layers = [
    {'k_y': 1.4,   'rho': 2200, 'cap': 730, 'H': 5e-6, 'psi': 1.0, 'Rc': 1e-8},
    {'k_y': 400.0, 'rho': 8960, 'cap': 385, 'H': 2e-6, 'psi': 1.0, 'Rc': 1e-8},
    {'k_y': 237.0, 'rho': 2900, 'cap': 1000,'H': 2e-6, 'psi': 1.0, 'Rc': 1e-8},
    {'k_y': 140.0, 'rho': 2330, 'cap': 700, 'H': np.inf,'psi':1.0,'Rc':0.0},
]

fitter = TwoLayerThicknessFitter(
    freqs=freqs, V3_exp=V3,
    b=sensor['b'], sensor_params=sensor,
    initial_layers=layers,
    layer_top_idx=1, layer_bot_idx=2,
    H_total=4e-6,
    # 同时联合拟合（可选）
    extra_params_to_fit={
        'layer_1': ['k_y'],          # 顶层 k_y
        'layer_2': ['k_y'],          # 下层 k_y
        'sensor':  ['R', 'dR_dT'],   # 传感器参数
    },
    extra_param_bounds={
        'layer_1': {'k_y': (50, 800)},
        'layer_2': {'k_y': (50, 800)},
        'sensor':  {'R': (1, 20), 'dR_dT': (1e-4, 1e-1)},
    },
    fit_mode='both',
)
r = fitter.fit(max_iter=400, verbose=1)
print(f"H_top = {r.H_top*1e6:.3f} μm, H_bot = {r.H_bot*1e6:.3f} μm")
print(f"Sensor R = {r.extra_params['sensor']['R']:.3f}")
```

### 方式 C：命令行批处理 (`run_fit_example.py`)
在脚本顶部填好 `CONFIG` 后 `python run_fit_example.py`，自动出图 + JSON。

---

## 拟合稳定性建议

1. **先扫描 cost(f)**（默认勾选），单峰下凹 → 解唯一；平坦或多极值 → 数据对该层厚度比不敏感，应：
   - 扩大频率范围（向高频扩展，让热渗透深度跨过这两层）
   - 选 `Amplitude+Phase` 联合拟合而非单一通道
   - 收窄 `f_bounds`（用先验缩小搜索区间）

2. **初值选择**：若大致知道哪层更厚，把 `f 初值` 偏向那一侧（默认 0.5）。

3. **不要把 H 加进 extra 参数**：会破坏总厚度约束；UI 已经强制阻止。

4. **同时拟合传感器参数时注意简并**：3ω 公式里
   \(V_3 \propto V_\text{rms} \cdot \alpha \cdot \Delta T / R\)（粗略），
   `R/V_rms/dR_dT` 等会存在多组组合给出相同 V3 的简并解。
   - 推荐：用先验把传感器参数边界收紧（`±10%`），先固定一项再拟合另一项
   - 或只拟合 `dR_dT`，固定其它

5. **频率筛选**：低频段如有仪器漂移可剔除，仅保留信噪比好的中高频。

---

## 输出 JSON 字段

| 字段 | 含义 |
|---|---|
| `H_total_m` | 输入的总厚度 (m) |
| `f_fraction` | 拟合得到的 `H_top / H_total` |
| `H_top_m`, `H_bot_m` | 反推的上/下层厚度 (m) |
| `extra_params` | 同时拟合的额外参数及结果（含 `sensor` 子字典） |
| `cost`, `n_iter`, `success`, `message` | scipy.optimize 返回的状态信息 |

---

## 与主程序的关系

- **完全独立**：本目录的修改**不动**主目录里的任何源文件
- **数据兼容**：直接读主程序导出的 `.xlsx/.csv`
- **预设复用**：使用同一个 `MaterialDatabase` / `SensorDatabase`（从主目录导入），
  主程序新增材料这里同步可见
- **模型一致**：底层调用 `olson_model26_3.calculate_analytical_Z` 与 `convert_Z_to_V3`，
  物理模型完全相同
- **其它功能保留**：锁相控制、数据采集、灵敏度分析、正向模拟、数据对比、信号校正
  等所有原有标签页都正常可用
