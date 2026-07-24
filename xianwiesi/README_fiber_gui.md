# 纤维丝 3ω 拟合 GUI（导电 / 绝缘可切换）

支持两种物理模型，在 GUI 顶部下拉框中自由切换（拟合页与正向模拟页联动）：

| 模型 | 文献 | 适用对象 |
|------|------|----------|
| **导电丝 (Lu 2001)** | Lu, Yi, Zhang, RSI **72**, 2996 (2001) | 丝本体导电，四线悬空 |
| **绝缘丝镀层 (3ω+辐射)** | Munro / Xing / Ban, IMECE2015-52056 | 不导电丝 + 镀金属加热/测温 |

模型按原文公式实现，**未照搬** `c++/`、`导电/` 下旧代码。

## 文件

| 文件 | 说明 |
|------|------|
| `lu2001_fiber_model.py` | 导电丝：正演、拟合、适用性 |
| `insulating_fiber_model.py` | 绝缘丝镀层：sinh 公式 + 辐射 \(h_r\) |
| `fiber_thermal_gui.py` | PyQt5 GUI（拟合 / 可视化 / 正向模拟） |
| `纤维丝.md` | Lu 2001 全文 |
| `绝缘纤维丝.md` | 绝缘丝 TET/3ω 综述 |

## 启动

```bash
cd xianwiesi
pip install numpy scipy pandas PyQt5 matplotlib openpyxl
python fiber_thermal_gui.py
```

## 数据格式

Excel/CSV，需能识别：

- 频率列：`频率 (Hz)` / `freq` / 第 1 列  
- V3：`V3_Real`+`V3_Imag`，或 `幅值`+`相位`(度)，或第 2/3 列幅值/相位(度)

电压为 **rms**（与锁相一致）；电流填 **I_rms**。

## 模型要点

### 导电丝
- \(V_{3\omega}\propto 1/\sqrt{1+(2\omega\gamma)^2}\)，\(\gamma=L^2/(\pi^2\alpha)\)
- \(R,R'\) 为纤维本体；可选式(24)与辐射表观 \(\kappa_\mathrm{ap}\)

### 绝缘丝镀层
- \(m^2=i2\omega/\alpha+4h_r/(Dk)\)，\(h_r=4\varepsilon\sigma T_0^3\)
- \(V_3\) 用 \(\sinh\) 传递函数（文中式8 按低频极限式9 取 \(I_0^3\)）
- \(R,R'\) 为**镀层**标定值；\(\kappa,C_p\) 为**纤维芯**
- 细长丝建议开启辐射

## 正向模拟

左侧完整参数栏，可「从数据拟合同步」（含模型类型）。  
**导出模拟数据** 格式：

`频率 (Hz), V3_Real, V3_Imag, V3_幅值, V3_相位`

可直接再导入「数据拟合」页。

## 使用流程

1. 选择 **导电丝** 或 **绝缘丝镀层**
2. 填 \(L, D\)（或 \(S\)）、\(R, R', I_\mathrm{rms}\)
3. 填材料初值 \(\kappa, \rho, C_p\)
4. 加载扫频数据 → 拟合 \(\kappa\) / \(C_p\)
5. 可视化对比；可用正向模拟做扫频预估

## 与主程序关系

主程序 `thermal_measurement_gui2.py` 为 **平面多层 Olson**；  
本目录为 **轴向一维纤维** 模型，二者互不替代。
