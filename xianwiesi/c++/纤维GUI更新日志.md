# 纤维丝3ω方法GUI - 更新日志

## v1.0.1 (2025-11-14 下午)

### 🐛 Bug修复

#### 修复：数据长度不匹配错误
**问题**：
```
ValueError: x and y must have same first dimension, but have shapes (99,) and (30,)
```

**原因**：
- GUI设置的频率点数与加载的实验数据点数不一致
- 理论曲线使用旧的频率数组，实验数据使用新的频率数组

**修复**：
1. ✅ 在 `plot_V3omega()` 中添加数据长度验证
2. ✅ 自动截取到相同长度
3. ✅ 打印警告信息（便于调试）
4. ✅ 加载数据时重新计算理论曲线（使用实验频率）
5. ✅ 添加 NaN/Inf 数据过滤
6. ✅ 自动按频率排序

**代码位置**：
- `gui_fiber_3omega.py` 第 406-423 行（plot_V3omega）
- `gui_fiber_3omega.py` 第 924-950 行（load_experimental_data）

**影响**：
- 修复后可以正常加载任意点数的实验数据
- 自动过滤无效数据点
- 确保绘图不会出错

---

### ✨ 改进

#### 改进1：数据验证增强
```python
# 数据验证：移除NaN和Inf
valid_mask = (np.isfinite(freqs) & np.isfinite(V3_real) & np.isfinite(V3_imag))

if not np.all(valid_mask):
    n_invalid = np.sum(~valid_mask)
    print(f"警告: 发现 {n_invalid} 个无效数据点（NaN或Inf），已自动过滤")

freqs = freqs[valid_mask]
V3_real = V3_real[valid_mask]
V3_imag = V3_imag[valid_mask]
```

**好处**：
- 自动处理脏数据
- 提高鲁棒性
- 用户无需手动清洗数据

#### 改进2：自动排序
```python
# 排序（按频率）
sort_idx = np.argsort(freqs)
freqs = freqs[sort_idx]
V3_real = V3_real[sort_idx]
V3_imag = V3_imag[sort_idx]
```

**好处**：
- 适应任意顺序的数据文件
- 相位展开（unwrap）更准确

#### 改进3：调试信息
```python
print(f"成功加载 {len(freqs)} 个有效数据点")
print(f"频率范围: {freqs.min():.3f} - {freqs.max():.3f} Hz")
```

**好处**：
- 便于用户确认数据加载正确
- 快速定位问题

#### 改进4：数据不足检查
```python
if len(freqs) < 5:
    QMessageBox.warning(self, "数据不足", 
        f"有效数据点仅 {len(freqs)} 个，需要至少5个点")
    return
```

**好处**：
- 避免拟合数据点过少导致失败
- 提前提示用户

---

## v1.0.0 (2025-11-14 上午)

### 🎉 初始发布

#### 核心功能
- ✅ 完整的PyQt5 GUI界面
- ✅ 材料参数管理（5种预设）
- ✅ 纤维参数管理（3种预设）
- ✅ 频率范围动态设置
- ✅ 实时3ω电压计算
- ✅ 4图可视化（全英文标签）
- ✅ 模型有效性自动检查
- ✅ 三种拟合类型（material/sensor/combined）
- ✅ 五种拟合模式（both/amp/phase/real/imag）
- ✅ 数据导入（Excel/CSV）
- ✅ 数据导出（JSON/CSV/图像）
- ✅ 配置保存/加载
- ✅ 模拟数据生成

#### GUI组件
- ✅ MaterialParamsWidget - 材料参数输入
- ✅ FiberParamsWidget - 纤维参数输入
- ✅ FrequencyRangeWidget - 频率设置
- ✅ ValidityCheckWidget - 有效性检查
- ✅ PlotCanvas - 绘图区域
- ✅ FittingControlWidget - 拟合控制
- ✅ ResultsDisplayWidget - 结果显示
- ✅ FiberThermal3OmegaGUI - 主窗口

#### 文档
- ✅ README_FIBER_GUI.md
- ✅ 纤维GUI使用说明.md
- ✅ 纤维GUI完整功能列表.md
- ✅ 快速参考.txt
- ✅ 示例配置文件×2

#### 特色功能
- ✅ 图表标签全英文（国际标准）
- ✅ 智能参数联动（直径→面积）
- ✅ 实时有效性检查
- ✅ 颜色编码提示
- ✅ 完全兼容 ThermalModelFitter 接口

---

## 版本计划

### v1.1 (计划)
- [ ] 后台线程拟合（避免界面卡顿）
- [ ] 实时拟合进度条
- [ ] 参数不确定度估计
- [ ] 残差图显示
- [ ] 批量文件处理
- [ ] 快捷键支持

### v1.2 (计划)
- [ ] 多样品对比功能
- [ ] 温度依赖性测量
- [ ] 自动报告生成
- [ ] 数据库管理
- [ ] 云端同步

### v2.0 (远期)
- [ ] 3D可视化
- [ ] 机器学习辅助拟合
- [ ] 实时仪器控制
- [ ] Web版本

---

## 升级指南

### 从 v1.0.0 升级到 v1.0.1

**无需特殊操作**：
1. 下载新版本 `gui_fiber_3omega.py`
2. 替换旧文件
3. 重新启动程序

**兼容性**：
- ✅ 配置文件完全兼容
- ✅ 数据格式完全兼容
- ✅ API接口无变化

**新功能**：
- 自动数据验证和清洗
- 更详细的调试信息
- 更好的错误处理

---

## 已知问题

### 当前版本已知问题

**问题1**：拟合时界面会暂时冻结
- **影响**：拟合进行中无法操作
- **原因**：在主线程执行计算
- **临时方案**：等待拟合完成（通常<5秒）
- **计划修复**：v1.1 使用后台线程

**问题2**：大数据文件（>1000点）加载较慢
- **影响**：加载时间>5秒
- **原因**：数据验证和排序
- **临时方案**：耐心等待
- **计划修复**：v1.1 优化算法

**问题3**：无快捷键支持
- **影响**：操作效率
- **临时方案**：使用鼠标点击
- **计划修复**：v1.1 添加快捷键

---

## Bug报告

如发现bug，请提供以下信息：

1. **错误信息**（完整的traceback）
2. **操作步骤**（如何重现错误）
3. **环境信息**：
   - Python版本
   - 操作系统
   - PyQt5版本
4. **数据文件**（如涉及数据加载）
5. **配置文件**（如涉及参数设置）

---

## 更新检查

**当前版本检查**：
```python
# 在 gui_fiber_3omega.py 中查看
# 标题栏显示版本号
```

**获取最新版本**：
- GitHub: <repository_url>
- 或联系开发者

---

## 致谢

感谢以下项目和工具：
- Lu 等人的开创性论文
- cursor/gui_3omega.py 项目（界面参考）
- PyQt5 框架
- Matplotlib 绘图库
- NumPy/SciPy 科学计算库

---

**当前版本**: v1.0.1  
**发布日期**: 2025-11-14  
**状态**: ✅ Stable  
**下一版本**: v1.1 (计划中)



