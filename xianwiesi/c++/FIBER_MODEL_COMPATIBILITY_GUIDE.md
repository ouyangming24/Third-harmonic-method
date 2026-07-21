# FiberThermalFitter 兼容性指南

## 概述

`fiber_model.py` 中的 `FiberThermalFitter` 类已完全改造为与 `olson_model26.py` 中的 `ThermalModelFitter` 类兼容的接口。现在两者可以使用完全相同的调用方式和参数结构。

## ✅ 完成的改造

### 1. 统一的初始化接口

**新接口**（兼容 ThermalModelFitter）：

```python
fitter = FiberThermalFitter(
    freqs,                      # 频率数组
    V3_exp,                     # 实验3ω电压（复数）
    fiber_params,               # 纤维参数字典
    initial_material_props,     # 材料属性初值字典
    params_to_fit={},           # 要拟合的参数（字典格式）
    fit_mode='both',            # 拟合模式
    param_bounds={},            # 参数边界约束
    fit_type='material'         # 拟合类型
)
```

### 2. 三种拟合模式（与 ThermalModelFitter 一致）

#### A. 材料参数拟合 (`fit_type='material'`)

拟合材料的热物性参数，纤维参数保持不变。

```python
fitter = FiberThermalFitter(
    freqs, V3_exp, fiber_params, initial_material_props,
    params_to_fit={'material': ['kappa', 'C_p', 'rho']},
    fit_mode='both',
    param_bounds={'material': {
        'kappa': (0.1, 5000),
        'C_p': (100, 5000),
        'rho': (100, 20000)
    }},
    fit_type='material'
)
result = fitter.fit()  # 返回 material_props (dict)
```

**类比 ThermalModelFitter**：
```python
# olson_model26.py
fitter = ThermalModelFitter(
    freqs, V3_exp, b, sensor_params, initial_layers,
    params_to_fit={'layer_0': ['k_y', 'Rc']},
    fit_type='material'
)
result = fitter.fit()  # 返回 layers (list)
```

#### B. 传感器参数拟合 (`fit_type='sensor'`)

拟合纤维/传感器参数，材料属性保持不变。

```python
fitter = FiberThermalFitter(
    freqs, V3_exp, fiber_params, material_props,
    params_to_fit={'fiber': ['R', 'R_prime', 'I_rms']},
    fit_mode='both',
    param_bounds={'fiber': {
        'R': (4.0, 6.0),
        'R_prime': (0.005, 0.015),
        'I_rms': (5e-3, 15e-3)
    }},
    fit_type='sensor'
)
result = fitter.fit()  # 返回 fiber_params (dict)
```

**类比 ThermalModelFitter**：
```python
# olson_model26.py
fitter = ThermalModelFitter(
    freqs, V3_exp, b, sensor_params, layers,
    params_to_fit={'sensor': ['R', 'dR_dT', 'V_rms']},
    fit_type='sensor'
)
result = fitter.fit()  # 返回 sensor_params (dict)
```

#### C. 联合拟合 (`fit_type='combined'`)

同时拟合材料参数和纤维参数。

```python
fitter = FiberThermalFitter(
    freqs, V3_exp, fiber_params, initial_material_props,
    params_to_fit={
        'material': ['kappa', 'C_p'],
        'fiber': ['R', 'R_prime']
    },
    fit_mode='both',
    fit_type='combined'
)
result = fitter.fit()  # 返回 {'fiber_params': dict, 'material_props': dict}
```

**类比 ThermalModelFitter**：
```python
# olson_model26.py
fitter = ThermalModelFitter(
    freqs, V3_exp, b, sensor_params, layers,
    params_to_fit={
        'layer_0': ['k_y'],
        'sensor': ['R', 'dR_dT']
    },
    fit_type='combined'
)
result = fitter.fit()  # 返回 {'layers': list, 'sensor_params': dict}
```

### 3. 统一的拟合模式

支持与 ThermalModelFitter 完全相同的 5 种拟合模式：

| `fit_mode` | 说明 | 适用场景 |
|------------|------|---------|
| `'both'` | 同时拟合实部和虚部（推荐） | 充分利用复数信息 |
| `'real'` | 仅拟合实部 | 虚部不可靠时 |
| `'imag'` | 仅拟合虚部 | 实部不可靠时 |
| `'amp'` | 仅拟合幅值 | 只关心幅值信息 |
| `'phase'` | 仅拟合相位 | 提取比热容信息 |

### 4. 统一的参数边界机制

与 ThermalModelFitter 相同的边界检测和警告系统：

```python
param_bounds = {
    'material': {
        'kappa': (0.1, 5000),
        'C_p': (100, 5000),
        'rho': (100, 20000)
    },
    'fiber': {
        'R': (4.0, 6.0),
        'R_prime': (0.005, 0.015),
        'L': (7e-3, 9e-3),
        'S': (1e-10, 1e-8),
        'I_rms': (5e-3, 15e-3)
    }
}
```

## 🔄 参数映射对应关系

### fiber_model.py ↔ olson_model26.py

| fiber_model | olson_model | 说明 |
|-------------|-------------|------|
| `fiber_params` | `sensor_params` | 传感器/纤维参数 |
| `material_props` | `layers` | 材料属性 |
| `'fiber'` | `'sensor'` | 参数类型标识 |
| `'material'` | `'layer_N'` | 材料层标识 |

## 📊 完整使用示例

### 示例 1：材料参数拟合（碳纳米管测量）

```python
import numpy as np
from fiber_model import FiberThermalFitter, create_typical_fiber_params

# 实验数据
freqs = np.logspace(0, 2, 20)  # 1-100 Hz
V3_exp = np.load('experimental_V3.npy')  # 加载实验数据

# 纤维参数（已知）
fiber_params = {
    'L': 1e-3,                      # 长度 1mm
    'S': np.pi * (5e-6)**2,         # 横截面积
    'R': 50.0,                      # 电阻 50Ω
    'R_prime': -0.05,               # 温度系数（负）
    'I_rms': 5e-3                   # RMS电流 5mA
}

# 材料属性初值
initial_material_props = {
    'kappa': 2000,   # 热导率初值
    'rho': 1300,     # 密度（假设已知）
    'C_p': 600       # 比热容初值
}

# 创建拟合器
fitter = FiberThermalFitter(
    freqs, V3_exp, fiber_params, initial_material_props,
    params_to_fit={'material': ['kappa', 'C_p']},  # 只拟合热导率和比热
    fit_mode='both',
    param_bounds={'material': {
        'kappa': (1000, 5000),
        'C_p': (400, 1000)
    }},
    fit_type='material'
)

# 执行拟合
result = fitter.fit(max_iter=20, tol=1e-5, zeta=0.5)

print(f"拟合结果:")
print(f"  热导率: {result['kappa']:.2f} W/m·K")
print(f"  比热容: {result['C_p']:.2f} J/kg·K")
```

### 示例 2：传感器参数校准

```python
# 材料属性已知（标准样品）
known_material_props = {
    'kappa': 71.6,    # 铂的热导率
    'rho': 21450,     # 铂的密度
    'C_p': 133        # 铂的比热容
}

# 纤维参数初值（待校准）
fiber_params = {
    'L': 8e-3,
    'S': np.pi * (10e-6)**2,
    'R': 5.2,         # 电阻待校准
    'R_prime': 0.009, # 温度系数待校准
    'I_rms': 10e-3
}

# 创建校准拟合器
fitter = FiberThermalFitter(
    freqs, V3_exp, fiber_params, known_material_props,
    params_to_fit={'fiber': ['R', 'R_prime']},
    fit_mode='both',
    fit_type='sensor'
)

# 执行校准
calibrated_params = fitter.fit(max_iter=20)

print(f"校准后的参数:")
print(f"  电阻 R: {calibrated_params['R']:.4f} Ω")
print(f"  温度系数 R': {calibrated_params['R_prime']:.6f} Ω/K")
```

### 示例 3：联合拟合（未知材料和不确定传感器）

```python
fitter = FiberThermalFitter(
    freqs, V3_exp, fiber_params, initial_material_props,
    params_to_fit={
        'material': ['kappa'],      # 拟合热导率
        'fiber': ['R', 'I_rms']    # 同时校准电阻和电流
    },
    fit_mode='both',
    param_bounds={
        'material': {'kappa': (100, 500)},
        'fiber': {'R': (40, 60), 'I_rms': (3e-3, 7e-3)}
    },
    fit_type='combined'
)

result = fitter.fit(max_iter=20)

# result 包含两部分
print(f"材料参数:")
print(f"  kappa: {result['material_props']['kappa']:.2f} W/m·K")
print(f"纤维参数:")
print(f"  R: {result['fiber_params']['R']:.4f} Ω")
print(f"  I_rms: {result['fiber_params']['I_rms']:.6f} A")
```

## 🔧 向后兼容性

旧的接口仍然支持，会自动转换为新格式：

```python
# 旧接口（仍然有效）
fitter = FiberThermalFitter(
    freqs, V3_exp, fiber_params, initial_material_props,
    params_to_fit=['kappa', 'C_p'],  # 列表格式（旧）
    fit_mode='amplitude'
)

# 内部自动转换为：
# params_to_fit={'material': ['kappa', 'C_p']}
# fit_type='material'
```

## 📈 拟合输出格式

### 材料拟合输出

```
最终拟合的材料属性:
  kappa: 7.4128e+01 (边界: [5.00e+01, 1.00e+02])
  C_p: 1.3159e+02 (边界: [1.00e+02, 2.00e+02])

导出参数:
  热扩散率 alpha: 2.6262e-05 m^2/s
  热时间常数 gamma: 2.4692e-01 s
```

### 传感器拟合输出

```
最终拟合的纤维参数:
  R: 5.0203e+00 (边界: [4.00e+00, 6.00e+00])
  R_prime: 1.0041e-02 (边界: [5.00e-03, 1.50e-02])
```

### 联合拟合输出

```
最终拟合的材料和纤维参数:
--- 材料属性 ---
  kappa: 7.8864e+01 (无约束)

--- 纤维参数 ---
  R: 5.3195e+00 (无约束)
```

## ⚠️ 注意事项

1. **参数命名差异**：
   - `fiber_params` 中的参数：`R`, `R_prime`, `I_rms`, `L`, `S`
   - `sensor_params` 中的参数：`R`, `dR_dT`, `V_rms`, `L`, `H_e`, `rho_C_e`

2. **边界约束格式**：
   - 使用嵌套字典：`{'material': {...}, 'fiber': {...}}`
   - 与 ThermalModelFitter 的 `{'sensor': {...}, 'layer_0': {...}}` 对应

3. **拟合模式建议**：
   - 材料测量：`fit_mode='both'`（充分利用相位信息）
   - 传感器校准：`fit_mode='both'` 或 `'amp'`
   - 联合拟合：`fit_mode='both'`（信息最丰富）

## ✨ 主要改进

1. ✅ 完全兼容 ThermalModelFitter 的接口
2. ✅ 支持传感器参数拟合（纤维参数校准）
3. ✅ 支持联合拟合（材料+传感器）
4. ✅ 统一的参数映射机制
5. ✅ 统一的边界约束和警告系统
6. ✅ 统一的输出格式
7. ✅ 向后兼容旧接口
8. ✅ 完整的测试覆盖

## 🚀 快速开始

```python
from fiber_model import (
    FiberThermalFitter, 
    calculate_fiber_V3omega,
    create_typical_fiber_params,
    create_fiber_material_library
)

# 1. 加载预定义配置
fiber_params = create_typical_fiber_params()['platinum_wire_20um']
materials = create_fiber_material_library()
material_props = materials['platinum']

# 2. 生成/加载实验数据
freqs = np.logspace(0, 2, 20)
V3_exp = np.load('your_data.npy')

# 3. 创建拟合器并拟合
fitter = FiberThermalFitter(
    freqs, V3_exp, fiber_params, material_props,
    params_to_fit={'material': ['kappa', 'C_p']},
    fit_mode='both',
    fit_type='material'
)
result = fitter.fit()

# 4. 使用结果
print(result)
```

## 📞 技术支持

- 参考论文：Lu et al., Rev. Sci. Instrum. 72, 2996 (2001)
- 对比参考：Olson et al., Rev. Sci. Instrum. 76, 053901 (2005)
- 测试文件：运行 `python fiber_model.py` 查看完整示例

---

**版本**: v2.0 - 完全兼容 ThermalModelFitter  
**更新日期**: 2025-11-14



