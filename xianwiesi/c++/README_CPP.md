# 纤维丝3ω方法热物性测量模型 (C++ 版本)

## 概述

这是基于 Python `fiber_model.py` 的完整 C++ 实现，用于纤维丝（悬空线）的热物性测量。

**理论基础**: Lu 等人，Rev. Sci. Instrum. 72, 2996 (2001)

**特性**:
- ✅ 完全兼容 `olson_model26.ThermalModelFitter` 接口
- ✅ 支持材料参数、传感器参数和联合拟合
- ✅ 5种拟合模式：both, real, imag, amp, phase
- ✅ 自动边界约束和收敛检测
- ✅ 高性能 C++ 实现（使用 Eigen 库）

## 文件结构

```
.
├── fiber_model.h           # 头文件（接口定义）
├── fiber_model.cpp         # 实现文件
├── test_fiber_model.cpp    # 测试/示例程序
├── CMakeLists.txt          # CMake 构建配置
└── README_CPP.md           # 本文档
```

## 依赖项

### 必需
- **C++17 或更高版本**
- **Eigen3** (线性代数库，版本 >= 3.3)

### 安装 Eigen3

#### Ubuntu/Debian
```bash
sudo apt-get install libeigen3-dev
```

#### macOS
```bash
brew install eigen
```

#### Windows
下载预编译版本或使用 vcpkg:
```bash
vcpkg install eigen3
```

## 编译和运行

### 方法 1: 使用 CMake（推荐）

```bash
# 创建构建目录
mkdir build
cd build

# 配置
cmake ..

# 编译
cmake --build .

# 运行测试
./test_fiber_model

# 或运行 CTest
ctest --verbose
```

### 方法 2: 手动编译

#### Linux/macOS
```bash
g++ -std=c++17 -O2 -o test_fiber_model \
    fiber_model.cpp test_fiber_model.cpp \
    -I/usr/include/eigen3
    
./test_fiber_model
```

#### Windows (MSVC)
```batch
cl /EHsc /std:c++17 /O2 /I"C:\path\to\eigen3" ^
   fiber_model.cpp test_fiber_model.cpp /Fe:test_fiber_model.exe
   
test_fiber_model.exe
```

## 快速开始

### 示例 1: 基本3ω电压计算

```cpp
#include "fiber_model.h"
using namespace FiberThermal;

int main() {
    // 创建纤维和材料参数
    auto fiber_configs = createTypicalFiberParams();
    auto materials = createFiberMaterialLibrary();
    
    FiberParams fiber = fiber_configs["platinum_wire_20um"];
    MaterialProps material = materials["platinum"];
    
    // 频率数组 (1-100 Hz, 20点)
    std::vector<double> freqs = logspace(1.0, 100.0, 20);
    
    // 计算3ω电压
    V3OmegaResult result = calculateFiberV3Omega(freqs, fiber, material);
    
    std::cout << "热扩散率: " << result.alpha << " m²/s" << std::endl;
    std::cout << "热时间常数: " << result.gamma << " s" << std::endl;
    
    return 0;
}
```

### 示例 2: 材料参数拟合

```cpp
#include "fiber_model.h"
using namespace FiberThermal;

int main() {
    // 准备实验数据
    std::vector<double> freqs = logspace(1.0, 100.0, 20);
    std::vector<Complex> V3_exp = ...; // 您的实验数据
    
    // 纤维参数（已知）
    FiberParams fiber;
    fiber.L = 8e-3;
    fiber.S = 3.14e-10;
    fiber.R = 5.0;
    fiber.R_prime = 0.01;
    fiber.I_rms = 10e-3;
    
    // 材料属性初值
    MaterialProps initial_material(60.0, 21450.0, 140.0);
    
    // 设置参数边界
    ParamBounds bounds;
    bounds.material_bounds["kappa"] = {50.0, 100.0};
    bounds.material_bounds["C_p"] = {100.0, 200.0};
    
    // 创建拟合器
    FiberThermalFitter fitter(
        freqs,
        V3_exp,
        fiber,
        initial_material,
        {"kappa", "C_p"},     // 要拟合的参数
        {},                   // 不拟合纤维参数
        FitMode::BOTH,
        bounds,
        FitType::MATERIAL
    );
    
    // 执行拟合
    fitter.fit(20, 1e-5, 0.5);
    
    // 获取结果
    MaterialProps fitted = fitter.getFittedMaterialProps();
    std::cout << "拟合的热导率: " << fitted.kappa << " W/m·K" << std::endl;
    std::cout << "拟合的比热容: " << fitted.C_p << " J/kg·K" << std::endl;
    std::cout << "收敛: " << (fitter.isConverged() ? "是" : "否") << std::endl;
    
    return 0;
}
```

### 示例 3: 传感器参数拟合

```cpp
// 当材料属性已知，需要校准传感器参数时
FiberThermalFitter fitter(
    freqs,
    V3_exp,
    initial_fiber,
    known_material,          // 已知材料
    {},                      // 不拟合材料参数
    {"R", "R_prime"},        // 拟合纤维参数
    FitMode::BOTH,
    bounds,
    FitType::SENSOR
);

fitter.fit();
FiberParams calibrated = fitter.getFittedFiberParams();
```

### 示例 4: 联合拟合

```cpp
// 同时拟合材料和传感器参数
FiberThermalFitter fitter(
    freqs,
    V3_exp,
    initial_fiber,
    initial_material,
    {"kappa"},               // 拟合热导率
    {"R"},                   // 同时拟合电阻
    FitMode::BOTH,
    bounds,
    FitType::COMBINED
);

fitter.fit();
MaterialProps fitted_mat = fitter.getFittedMaterialProps();
FiberParams fitted_fib = fitter.getFittedFiberParams();
```

## API 参考

### 核心数据结构

#### FiberParams
```cpp
struct FiberParams {
    double L;           // 电压触点间长度 [m]
    double S;           // 横截面积 [m²]
    double R;           // 电阻 [Ω]
    double R_prime;     // 温度系数 dR/dT [Ω/K]
    double I_rms;       // RMS电流 [A]
    double diameter;    // 直径 [m] (可选)
    std::string description;
};
```

#### MaterialProps
```cpp
struct MaterialProps {
    double kappa;       // 热导率 [W/m·K]
    double rho;         // 密度 [kg/m³]
    double C_p;         // 比热容 [J/kg·K]
    std::string description;
};
```

### 核心函数

#### calculateFiberV3Omega
```cpp
V3OmegaResult calculateFiberV3Omega(
    const std::vector<double>& freqs,
    const FiberParams& fiber_params,
    const MaterialProps& material_props
);

// 返回值
struct V3OmegaResult {
    std::vector<Complex> V3_omega; // 3ω电压数组
    double gamma;                   // 热时间常数 [s]
    double alpha;                   // 热扩散率 [m²/s]
};
```

#### checkValidityConditions
```cpp
ValidityCheck checkValidityConditions(
    const FiberParams& fiber_params,
    const MaterialProps& material_props,
    const std::vector<double>& freqs
);

// 检查:
// - 条件(10): 功率均匀性
// - 条件(31): 辐射损失
// - 频率范围合理性
```

### FiberThermalFitter 类

#### 构造函数
```cpp
FiberThermalFitter(
    const std::vector<double>& freqs,
    const std::vector<Complex>& V3_exp,
    const FiberParams& fiber_params,
    const MaterialProps& initial_material_props,
    const std::vector<std::string>& material_params_to_fit = {"kappa", "C_p"},
    const std::vector<std::string>& fiber_params_to_fit = {},
    FitMode fit_mode = FitMode::BOTH,
    const ParamBounds& param_bounds = ParamBounds(),
    FitType fit_type = FitType::MATERIAL
);
```

#### 主要方法
```cpp
// 执行拟合
void fit(int max_iter = 20, double tol = 1e-5, double zeta = 0.5);

// 获取结果
MaterialProps getFittedMaterialProps() const;
FiberParams getFittedFiberParams() const;
std::vector<Complex> calculateFittedV3() const;

// 获取统计信息
int getIterations() const;
bool isConverged() const;
double getFinalEpsilon() const;
double getAlpha() const;
double getGamma() const;
```

### 枚举类型

#### FitMode
```cpp
enum class FitMode {
    BOTH,   // 实部+虚部（推荐）
    REAL,   // 仅实部
    IMAG,   // 仅虚部
    AMP,    // 仅幅值
    PHASE   // 仅相位
};
```

#### FitType
```cpp
enum class FitType {
    MATERIAL,   // 材料参数拟合
    SENSOR,     // 传感器参数拟合
    COMBINED    // 联合拟合
};
```

## 材料库

内置常见材料的热物性参数：

```cpp
auto materials = createFiberMaterialLibrary();

// 可用材料:
materials["platinum"]         // 铂丝
materials["carbon_nanotube"]  // 碳纳米管束
materials["copper"]           // 铜丝
materials["tungsten"]         // 钨丝
materials["nichrome"]         // 镍铬合金丝
```

## 纤维配置库

内置典型纤维参数配置：

```cpp
auto configs = createTypicalFiberParams();

// 可用配置:
configs["platinum_wire_20um"]      // 20μm铂丝
configs["carbon_nanotube_bundle"]  // 碳纳米管束
```

## 性能优化建议

1. **编译优化**: 使用 `-O2` 或 `-O3` 优化级别
2. **并行化**: 灵敏度矩阵计算可以并行化（未实现，可扩展）
3. **缓存**: 对重复计算可以添加缓存机制

## 与 Python 版本的对应关系

| Python | C++ |
|--------|-----|
| `calculate_fiber_V3omega()` | `calculateFiberV3Omega()` |
| `FiberThermalFitter.fit()` | `FiberThermalFitter::fit()` |
| `params_to_fit={'material': [...]}` | `material_params_to_fit` |
| `params_to_fit={'fiber': [...]}` | `fiber_params_to_fit` |
| `fit_mode='both'` | `FitMode::BOTH` |
| `fit_type='material'` | `FitType::MATERIAL` |

## 测试

运行测试程序以验证功能：

```bash
./test_fiber_model
```

测试包括：
1. ✅ 基础3ω电压计算
2. ✅ 材料参数拟合 (fit_type='material')
3. ✅ 传感器参数拟合 (fit_type='sensor')
4. ✅ 联合拟合 (fit_type='combined')

预期输出：
```
纤维丝3ω方法模型 C++ 版本测试
完全兼容 ThermalModelFitter 接口

======================================================================
基础测试：3ω 电压计算
======================================================================
...
所有测试完成！
```

## 故障排除

### 问题 1: 找不到 Eigen3

**解决方案**:
```bash
# 手动指定 Eigen3 路径
cmake -DEigen3_DIR=/path/to/eigen3/cmake ..
```

### 问题 2: C++17 不支持

**解决方案**:
- 确保编译器支持 C++17 (GCC >= 7, Clang >= 5, MSVC >= 2017)
- 或降级到 C++14 并移除部分新特性

### 问题 3: 拟合不收敛

**解决方案**:
- 检查初值是否合理
- 增加最大迭代次数 `max_iter`
- 调整步长因子 `zeta` (0.3 - 0.7)
- 检查数据质量和噪声水平

## 许可证

本代码基于原始 Python 实现，遵循相同的许可协议。

## 作者

- 原始 Python 版本: AI Assistant
- C++ 移植: AI Assistant
- 日期: 2025-11-14

## 参考文献

1. Lu, L., Yi, W., & Zhang, D. L. (2001). 3ω method for specific heat and thermal conductivity measurements. Review of Scientific Instruments, 72(7), 2996-3003.

2. Olson, B. W., Graham, S., & Chen, K. (2005). A practical extension of the 3ω method to multilayer structures. Review of Scientific Instruments, 76(5), 053901.

## 联系方式

如有问题或建议，请参考 Python 版本的文档或提交 issue。

---

**版本**: v1.0  
**状态**: ✅ 完整实现并测试  
**兼容性**: 完全兼容 Python 版本和 ThermalModelFitter 接口



