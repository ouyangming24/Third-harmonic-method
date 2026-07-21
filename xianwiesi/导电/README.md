# 纤维丝3ω方法热物性测量系统

基于Lu等人论文 "3ω method for specific heat and thermal conductivity measurements" (Rev. Sci. Instrum. 72, 2996 (2001)) 的纤维丝热导率和比热容同时测量系统。

## 功能特点

- **完整的理论模型**: 实现Lu等人的一维热传导方程解析解
- **多种拟合模式**: 支持幅值、相位、复合拟合
- **参数同时拟合**: 可同时测量热导率κ和比热容C_p
- **预配置模式**: 无需交互输入，直接运行
- **批量处理**: 支持多文件自动处理
- **有效性检查**: 自动验证模型适用条件
- **可视化分析**: 完整的结果图表和残差分析

## 文件结构

```
纤维丝3ω系统/
├── fiber_model.py              # 核心理论模型
├── run_fiber_simulation.py     # 模拟数据生成与验证
├── fit_fiber_data.py           # 实验数据拟合（预配置）
├── requirements.txt            # 依赖包列表
└── README.md                   # 说明文档
```

## 快速开始

### 1. 环境配置

```bash
# 安装依赖包
pip install -r requirements.txt
```

### 2. 配置参数

编辑 `fit_fiber_data.py` 文件顶部的配置区域：

```python
# 数据文件配置
DATA_FILE_CONFIG = {
    'file_path': 'your_data_file.xlsx',  # 修改为您的数据文件路径
    'backup_files': [...]  # 备选文件列表
}

# 纤维参数配置
FIBER_CONFIG = {
    'config_name': 'platinum_wire_20um',  # 或选择其他预设配置
    'custom_params': {...}  # 自定义参数
}

# 材料初值配置
MATERIAL_CONFIG = {
    'material_name': 'platinum',  # 或选择其他材料
    'custom_material': {...}  # 自定义材料
}

# 拟合参数配置
FITTING_CONFIG = {
    'params_to_fit': ['kappa', 'C_p'],  # 选择要拟合的参数
    'fit_mode': 'both',  # 拟合模式
    'param_bounds': {...}  # 参数边界
}
```

### 3. 运行拟合

```bash
python fit_fiber_data.py
```

### 4. 运行模拟验证

```bash
python run_fiber_simulation.py
```

## 理论基础

### 核心方程

纤维丝3ω方法的核心方程为：

$$V_{3\omega} = \frac{4 I^3 L R R'}{π^4 κ S \sqrt{1+(2ωγ)^2}}$$

其中：
- $γ = L^2/(π^2 α)$ 是热时间常数
- $α = κ/(ρ C_p)$ 是热扩散率
- $\tan φ = 2ωγ$ 是相位关系

### 有效性条件

1. **条件(10)**: 功率均匀性 $\frac{I_0^2 R' L}{π^2 κ S} \ll 1$
2. **条件(31)**: 辐射损失 $gγ \ll 1$
3. **频率范围**: $2ωγ < 4$ (即 $φ < 76°$)

## 配置选项详解

### 纤维参数预设

| 配置名称 | 描述 | 适用场景 |
|---------|------|----------|
| `platinum_wire_20um` | 20μm铂丝 | 标准参考材料测量 |
| `carbon_nanotube_bundle` | 碳纳米管束 | 高热导率材料 |
| `thin_wire_1um` | 1μm超细丝 | 高阻抗样品 |
| `custom` | 自定义配置 | 特殊样品 |

### 材料预设

| 材料名称 | κ (W/m·K) | ρ (kg/m³) | C_p (J/kg·K) |
|---------|-----------|-----------|--------------|
| `platinum` | 71.6 | 21450 | 133 |
| `carbon_nanotube` | 3000 | 1300 | 700 |
| `copper` | 401 | 8960 | 385 |
| `tungsten` | 173 | 19300 | 132 |
| `nichrome` | 11.3 | 8400 | 450 |

### 拟合参数组合

| 组合 | 参数 | 适用情况 |
|------|------|----------|
| 标准 | `['kappa', 'C_p']` | 同时测量热导率和比热容 |
| 已知比热容 | `['kappa', 'rho']` | 比热容已知，测量热导率和密度 |
| 仅热导率 | `['kappa']` | 密度和比热容已知 |

### 拟合模式

- `'amplitude'`: 仅拟合幅值，适合低噪声数据
- `'phase'`: 仅拟合相位，适合相位精确的数据
- `'both'`: 同时拟合幅值和相位（推荐）

## 数据文件格式

Excel文件应包含以下列：
- 频率列（包含'freq'关键词）
- V3ω实部列（包含'real'和'v3'关键词）
- V3ω虚部列（包含'imag'和'v3'关键词）

示例：
```
Frequency (Hz) | V3_Real (V) | V3_Imag (V)
1.0           | 1.23e-6     | -2.45e-7
2.0           | 1.15e-6     | -4.12e-7
...
```

## 批量处理

启用批量处理模式：

```python
BATCH_CONFIG = {
    'enabled': True,
    'data_directory': './data',
    'file_pattern': '*.xlsx',
    'selected_files': 'all'
}
```

## 结果输出

### 拟合结果文件
- `{原文件名}_fiber_fit_results.xlsx`: 详细拟合结果
- 包含三个工作表：
  - `Fit_Results`: 拟合参数和精度
  - `Fiber_Params`: 纤维配置参数
  - `Validity_Check`: 模型有效性检查

### 批量处理总结
- `batch_processing_summary.xlsx`: 所有文件的拟合结果总结

## 故障排除

### 常见问题

1. **数据加载失败**
   - 检查文件路径是否正确
   - 确认Excel文件列名格式
   - 验证数据是否包含NaN值

2. **拟合不收敛**
   - 调整参数边界范围
   - 检查初值是否合理
   - 降低优化容差或增加迭代次数

3. **模型有效性条件不满足**
   - 调整频率范围
   - 检查纤维参数设置
   - 考虑辐射损失修正

### 调试建议

1. 先运行模拟验证 (`run_fiber_simulation.py`)
2. 检查有效性条件输出
3. 分析残差图判断拟合质量
4. 对比不同拟合模式的结果

## 引用

如果您在研究中使用本系统，请引用原始论文：

L. Lu, W. Yi, and D. L. Zhang, "3ω method for specific heat and thermal conductivity measurements," Rev. Sci. Instrum. 72, 2996 (2001).

## 版本历史

- v1.0: 基础理论模型实现
- v1.1: 添加预配置模式
- v1.2: 支持批量处理
- v1.3: 完善有效性检查和可视化

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue到项目仓库
- 发送邮件到开发者邮箱
