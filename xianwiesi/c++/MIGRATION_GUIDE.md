# 迁移指南：旧接口 → 新接口

本文档说明如何将基于旧版 `FiberThermalFitter` 的代码迁移到新的兼容接口。

## 📋 主要变化概述

### 1. 参数格式变化

**旧接口：**
```python
params_to_fit = ['kappa', 'C_p']
param_bounds = {
    'kappa': (0.1, 5000),
    'C_p': (100, 5000)
}
```

**新接口：**
```python
params_to_fit = {'material': ['kappa', 'C_p']}
param_bounds = {
    'material': {
        'kappa': (0.1, 5000),
        'C_p': (100, 5000)
    }
}
```

### 2. 拟合模式名称变化

| 旧名称 | 新名称 | 说明 |
|--------|--------|------|
| `'amplitude'` | `'amp'` | 幅值拟合 |
| `'phase'` | `'phase'` | 相位拟合（不变） |
| `'both'` | `'both'` | 实部+虚部拟合（不变） |
| - | `'real'` | 新增：仅实部 |
| - | `'imag'` | 新增：仅虚部 |

### 3. 返回值变化

**旧接口返回字典：**
```python
result = {
    'fitted_params': {'kappa': 71.6, 'C_p': 133},
    'fitted_material_props': {...},
    'V3_fitted': array([...]),
    'gamma': 0.25,
    'alpha': 2.5e-5,
    'rmse': 1.2e-6,
    'iterations': 15,
    'converged': True,
    'validity_check': {...}
}
```

**新接口直接返回 material_props：**
```python
result = {
    'kappa': 71.6,
    'rho': 21450,
    'C_p': 133
}
# 其他信息需要重新计算
from fiber_model import calculate_fiber_V3omega
V3_fitted, gamma, alpha = calculate_fiber_V3omega(freqs, fiber_params, result)
```

## 🔄 代码迁移示例

### 示例 1: 基本拟合代码

**旧代码：**
```python
fitter = FiberThermalFitter(
    freqs=freqs,
    V3_exp=V3_exp,
    fiber_params=fiber_params,
    initial_material_props=initial_guess,
    params_to_fit=['kappa', 'C_p'],
    fit_mode='amplitude',
    param_bounds={'kappa': (0.1, 5000), 'C_p': (100, 5000)}
)

result = fitter.fit(max_iter=30, tol=1e-5, zeta=0.5)

kappa_fitted = result['fitted_params']['kappa']
Cp_fitted = result['fitted_params']['C_p']
V3_fitted = result['V3_fitted']
alpha = result['alpha']
gamma = result['gamma']
rmse = result['rmse']
```

**新代码：**
```python
from fiber_model import calculate_fiber_V3omega

fitter = FiberThermalFitter(
    freqs=freqs,
    V3_exp=V3_exp,
    fiber_params=fiber_params,
    initial_material_props=initial_guess,
    params_to_fit={'material': ['kappa', 'C_p']},  # 字典格式
    fit_mode='amp',  # 新名称
    param_bounds={'material': {'kappa': (0.1, 5000), 'C_p': (100, 5000)}},  # 嵌套字典
    fit_type='material'  # 新增参数
)

result = fitter.fit(max_iter=30, tol=1e-5, zeta=0.5)

# 直接访问
kappa_fitted = result['kappa']
Cp_fitted = result['C_p']

# 重新计算其他量
V3_fitted, gamma, alpha = calculate_fiber_V3omega(freqs, fiber_params, result)
```

### 示例 2: 多模式对比

**旧代码：**
```python
fit_modes = ['amplitude', 'phase', 'both']
for mode in fit_modes:
    fitter = FiberThermalFitter(..., fit_mode=mode)
    result = fitter.fit()
    
    kappa = result['fitted_params']['kappa']
    rmse = result['rmse']
```

**新代码：**
```python
fit_modes = ['amp', 'phase', 'both']  # 修改模式名
for mode in fit_modes:
    fitter = FiberThermalFitter(
        ..., 
        fit_mode=mode,
        params_to_fit={'material': ['kappa', 'C_p']},  # 新格式
        param_bounds={'material': {...}},  # 新格式
        fit_type='material'  # 新参数
    )
    result = fitter.fit()
    
    kappa = result['kappa']  # 直接访问
```

### 示例 3: 结果保存

**旧代码：**
```python
result = fitter.fit()

# 保存参数
for name, value in result['fitted_params'].items():
    save_param(name, value)

# 保存拟合曲线
V3_fitted = result['V3_fitted']
save_curve(V3_fitted)

# 保存导出参数
alpha = result['alpha']
gamma = result['gamma']
```

**新代码：**
```python
from fiber_model import calculate_fiber_V3omega

result = fitter.fit()

# 保存参数（直接访问）
for name in ['kappa', 'C_p', 'rho']:
    if name in result:
        save_param(name, result[name])

# 重新计算拟合曲线
V3_fitted, gamma, alpha = calculate_fiber_V3omega(freqs, fiber_params, result)
save_curve(V3_fitted)

# 保存导出参数
save_derived_param('alpha', alpha)
save_derived_param('gamma', gamma)
```

## ✨ 新功能使用

### 传感器参数拟合（新增）

```python
# 拟合纤维参数（如电阻、温度系数）
fitter = FiberThermalFitter(
    freqs, V3_exp, fiber_params, known_material_props,
    params_to_fit={'fiber': ['R', 'R_prime', 'I_rms']},
    fit_mode='both',
    param_bounds={'fiber': {
        'R': (4.0, 6.0),
        'R_prime': (0.005, 0.015),
        'I_rms': (5e-3, 15e-3)
    }},
    fit_type='sensor'  # 传感器拟合模式
)

fitted_fiber_params = fitter.fit()
# 返回: {'R': 5.02, 'R_prime': 0.01, 'I_rms': 0.01, ...}
```

### 联合拟合（新增）

```python
# 同时拟合材料和纤维参数
fitter = FiberThermalFitter(
    freqs, V3_exp, fiber_params, initial_material_props,
    params_to_fit={
        'material': ['kappa', 'C_p'],
        'fiber': ['R', 'I_rms']
    },
    fit_mode='both',
    fit_type='combined'
)

result = fitter.fit()
# 返回: {'fiber_params': {...}, 'material_props': {...}}
fitted_kappa = result['material_props']['kappa']
fitted_R = result['fiber_params']['R']
```

## 🔧 辅助函数适配

### 函数 1: 参数归一化

**旧代码（fit_experimental_data.py）：**
```python
def normalize_params_to_fit(raw_params):
    # ... 处理逻辑 ...
    return params  # 返回列表 ['kappa', 'C_p']
```

**新代码：**
```python
def normalize_params_to_fit(raw_params):
    # ... 处理逻辑 ...
    return {'material': params}  # 返回字典格式
```

### 函数 2: 边界构建

**旧代码：**
```python
def build_param_bounds(bounds_cfg, params_to_fit):
    bounds = {}
    for param in params_to_fit:  # params_to_fit 是列表
        bounds[param] = ...
    return bounds
```

**新代码：**
```python
def build_param_bounds(bounds_cfg, params_to_fit):
    bounds = {'material': {}}
    for param in params_to_fit['material']:  # 访问字典
        bounds['material'][param] = ...
    return bounds
```

## 📊 数据导出适配

### Excel 导出

**旧代码：**
```python
params_data.append({
    'Kappa': result['fitted_params']['kappa'],
    'C_p': result['fitted_params']['C_p'],
    'Alpha': result['alpha'],
    'Gamma': result['gamma'],
    'RMSE': result['rmse']
})
```

**新代码：**
```python
from fiber_model import calculate_fiber_V3omega

# 重新计算导出参数
_, gamma_val, alpha_val = calculate_fiber_V3omega(freqs, fiber_params, result)

params_data.append({
    'Kappa': result['kappa'],  # 直接访问
    'C_p': result['C_p'],
    'Alpha': alpha_val,
    'Gamma': gamma_val
})
```

### JSON 导出

**旧代码：**
```python
output = {
    'fitted_params': result['fitted_params'],
    'rmse': result['rmse'],
    'iterations': result['iterations'],
    'converged': result['converged'],
    'alpha': result['alpha'],
    'gamma': result['gamma']
}
```

**新代码：**
```python
from fiber_model import calculate_fiber_V3omega

# 重新计算导出参数
_, gamma_val, alpha_val = calculate_fiber_V3omega(freqs, fiber_params, result)

output = {
    'fitted_material_props': result,  # 直接使用
    'alpha': alpha_val,
    'gamma': gamma_val
}
```

## ⚙️ 完整迁移清单

- [x] 更新 `params_to_fit` 为字典格式 `{'material': [...]}`
- [x] 更新 `param_bounds` 为嵌套字典 `{'material': {...}}`
- [x] 修改 `fit_mode` 名称 `'amplitude'` → `'amp'`
- [x] 添加 `fit_type='material'` 参数
- [x] 修改结果访问方式：`result['fitted_params']['kappa']` → `result['kappa']`
- [x] 添加 V3_fitted 重新计算：`calculate_fiber_V3omega(freqs, fiber_params, result)`
- [x] 添加 gamma/alpha 重新计算
- [x] 更新所有辅助函数以处理新格式
- [x] 更新数据导出代码
- [x] 测试向后兼容性（旧的列表格式仍可用）

## 🆕 向后兼容性

新接口**仍然支持**旧的列表格式：

```python
# 这样写仍然有效（自动转换）
fitter = FiberThermalFitter(
    ...,
    params_to_fit=['kappa', 'C_p'],  # 列表格式（旧）
    fit_mode='amplitude'  # 会被接受但推荐用 'amp'
)
# 内部自动转换为：
# params_to_fit={'material': ['kappa', 'C_p']}
# fit_type='material'
```

但**强烈建议**迁移到新格式以获得完整功能。

## 📞 问题排查

### 常见错误 1: KeyError 访问结果

**错误：**
```python
kappa = result['fitted_params']['kappa']  # KeyError: 'fitted_params'
```

**解决：**
```python
kappa = result['kappa']  # 直接访问
```

### 常见错误 2: V3_fitted 不存在

**错误：**
```python
V3_fitted = result['V3_fitted']  # KeyError: 'V3_fitted'
```

**解决：**
```python
from fiber_model import calculate_fiber_V3omega
V3_fitted, _, _ = calculate_fiber_V3omega(freqs, fiber_params, result)
```

### 常见错误 3: 参数格式错误

**错误：**
```python
params_to_fit = ['kappa', 'C_p']
param_bounds = {'kappa': (0.1, 5000)}
# TypeError: string indices must be integers
```

**解决：**
```python
params_to_fit = {'material': ['kappa', 'C_p']}
param_bounds = {'material': {'kappa': (0.1, 5000)}}
```

## 📚 更多资源

- [完整兼容性指南](FIBER_MODEL_COMPATIBILITY_GUIDE.md)
- [新接口文档](fiber_model.py) - 查看 `FiberThermalFitter.__init__` 的文档字符串
- [测试示例](fiber_model.py) - `if __name__ == "__main__"` 部分

---

**版本**: v2.0  
**更新日期**: 2025-11-14  
**兼容性**: 向后兼容旧接口（列表格式），但推荐使用新接口



