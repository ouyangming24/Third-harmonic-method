/**
 * @file fiber_model.h
 * @brief 纤维丝3ω方法热物性测量模型 (C++ 实现)
 * 
 * 基于 Lu 等人的论文:
 * "3ω method for specific heat and thermal conductivity measurements"
 * Rev. Sci. Instrum. 72, 2996 (2001)
 * 
 * 完全兼容 olson_model26 的 ThermalModelFitter 接口
 * 
 * @author AI Assistant
 * @date 2025-11-14
 */

#ifndef FIBER_MODEL_H
#define FIBER_MODEL_H

#include <vector>
#include <complex>
#include <string>
#include <map>
#include <memory>
#include <stdexcept>
#include <cmath>

namespace FiberThermal {

// 常量定义
constexpr double PI = 3.14159265358979323846;

/**
 * @brief 复数类型定义
 */
using Complex = std::complex<double>;

/**
 * @brief 纤维参数结构体
 */
struct FiberParams {
    double L;           ///< 电压触点间长度 [m]
    double S;           ///< 横截面积 [m²]
    double R;           ///< 电阻 [Ω]
    double R_prime;     ///< 温度系数 dR/dT [Ω/K]
    double I_rms;       ///< RMS电流 [A]
    double diameter;    ///< 直径 [m] (可选)
    std::string description; ///< 描述
    
    FiberParams() : L(0), S(0), R(0), R_prime(0), I_rms(0), diameter(0) {}
};

/**
 * @brief 材料热物性参数结构体
 */
struct MaterialProps {
    double kappa;       ///< 热导率 [W/m·K]
    double rho;         ///< 密度 [kg/m³]
    double C_p;         ///< 比热容 [J/kg·K]
    std::string description; ///< 描述
    
    MaterialProps() : kappa(0), rho(0), C_p(0) {}
    MaterialProps(double k, double r, double c) : kappa(k), rho(r), C_p(c) {}
};

/**
 * @brief 3ω 电压计算结果
 */
struct V3OmegaResult {
    std::vector<Complex> V3_omega; ///< 复数3ω电压数组
    double gamma;                   ///< 热时间常数 [s]
    double alpha;                   ///< 热扩散率 [m²/s]
};

/**
 * @brief 模型有效性检查结果
 */
struct ValidityCheck {
    double condition_10_value;
    bool condition_10_satisfied;
    double condition_31_value;
    bool condition_31_satisfied;
    double max_2omega_gamma;
    bool frequency_range_ok;
    double gamma;
    double equivalent_diameter;
};

/**
 * @brief 参数类型枚举
 */
enum class ParamType {
    MATERIAL,  ///< 材料参数
    FIBER      ///< 纤维/传感器参数
};

/**
 * @brief 拟合模式枚举
 */
enum class FitMode {
    BOTH,   ///< 实部+虚部
    REAL,   ///< 仅实部
    IMAG,   ///< 仅虚部
    AMP,    ///< 仅幅值
    PHASE   ///< 仅相位
};

/**
 * @brief 拟合类型枚举
 */
enum class FitType {
    MATERIAL,   ///< 材料参数拟合
    SENSOR,     ///< 传感器参数拟合
    COMBINED    ///< 联合拟合
};

/**
 * @brief 参数映射结构
 */
struct ParamMap {
    ParamType type;
    std::string prop_name;
    size_t index;  ///< 在向量中的索引
};

/**
 * @brief 参数边界
 */
struct ParamBounds {
    std::map<std::string, std::pair<double, double>> material_bounds;
    std::map<std::string, std::pair<double, double>> fiber_bounds;
};

/**
 * @brief 计算纤维丝3ω电压
 * 
 * @param freqs 频率数组 [Hz]
 * @param fiber_params 纤维参数
 * @param material_props 材料热物性
 * @return V3OmegaResult 包含V3ω电压、gamma和alpha
 */
V3OmegaResult calculateFiberV3Omega(
    const std::vector<double>& freqs,
    const FiberParams& fiber_params,
    const MaterialProps& material_props
);

/**
 * @brief 计算直流温升
 * 
 * @param fiber_params 纤维参数
 * @param material_props 材料热物性
 * @return double 中心点最大直流温升 [K]
 */
double calculateDCTemperatureRise(
    const FiberParams& fiber_params,
    const MaterialProps& material_props
);

/**
 * @brief 检查模型有效性条件
 * 
 * @param fiber_params 纤维参数
 * @param material_props 材料热物性
 * @param freqs 频率数组
 * @return ValidityCheck 有效性检查结果
 */
ValidityCheck checkValidityConditions(
    const FiberParams& fiber_params,
    const MaterialProps& material_props,
    const std::vector<double>& freqs
);

/**
 * @brief 纤维丝热物性参数拟合器
 * 
 * 完全兼容 ThermalModelFitter 接口
 */
class FiberThermalFitter {
public:
    /**
     * @brief 构造函数
     * 
     * @param freqs 频率数组
     * @param V3_exp 实验3ω电压数据（复数）
     * @param fiber_params 纤维参数
     * @param initial_material_props 材料属性初值
     * @param material_params_to_fit 要拟合的材料参数名称列表
     * @param fiber_params_to_fit 要拟合的纤维参数名称列表
     * @param fit_mode 拟合模式
     * @param param_bounds 参数边界约束
     * @param fit_type 拟合类型
     */
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
    
    /**
     * @brief 执行迭代拟合
     * 
     * @param max_iter 最大迭代次数
     * @param tol 收敛容差
     * @param zeta 步长因子 (0 < zeta < 1)
     */
    void fit(int max_iter = 20, double tol = 1e-5, double zeta = 0.5);
    
    /**
     * @brief 获取拟合后的材料属性
     */
    MaterialProps getFittedMaterialProps() const { return material_props_; }
    
    /**
     * @brief 获取拟合后的纤维参数
     */
    FiberParams getFittedFiberParams() const { return fiber_params_; }
    
    /**
     * @brief 获取最终迭代次数
     */
    int getIterations() const { return iterations_; }
    
    /**
     * @brief 是否收敛
     */
    bool isConverged() const { return converged_; }
    
    /**
     * @brief 获取最终epsilon值
     */
    double getFinalEpsilon() const { return final_epsilon_; }
    
    /**
     * @brief 计算拟合后的V3ω电压
     */
    std::vector<Complex> calculateFittedV3() const;
    
    /**
     * @brief 获取热扩散率
     */
    double getAlpha() const;
    
    /**
     * @brief 获取热时间常数
     */
    double getGamma() const;

private:
    // 数据成员
    std::vector<double> freqs_;
    std::vector<Complex> V3_exp_;
    FiberParams fiber_params_;
    MaterialProps material_props_;
    
    std::vector<std::string> material_params_to_fit_;
    std::vector<std::string> fiber_params_to_fit_;
    
    FitMode fit_mode_;
    ParamBounds param_bounds_;
    FitType fit_type_;
    
    std::vector<ParamMap> param_map_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    
    int iterations_;
    bool converged_;
    double final_epsilon_;
    
    // 私有方法
    void createParamMap();
    void createBoundsVector();
    std::vector<double> getParamsAsVector() const;
    void updateParamsFromVector(const std::vector<double>& param_vector);
    void applyBounds(std::vector<double>& param_vector) const;
    
    std::vector<std::vector<Complex>> calculateSensitivityMatrix() const;
    
    void printValidityCheck() const;
    void printIterationInfo(int iter, double epsilon) const;
    void printFinalResults() const;
    
    double getParamValue(const std::string& name, ParamType type) const;
    void setParamValue(const std::string& name, ParamType type, double value);
};

/**
 * @brief 创建材料库
 */
std::map<std::string, MaterialProps> createFiberMaterialLibrary();

/**
 * @brief 创建典型纤维参数配置
 */
std::map<std::string, FiberParams> createTypicalFiberParams();

/**
 * @brief 字符串转FitMode
 */
FitMode stringToFitMode(const std::string& mode_str);

/**
 * @brief 字符串转FitType
 */
FitType stringToFitType(const std::string& type_str);

/**
 * @brief FitMode转字符串
 */
std::string fitModeToString(FitMode mode);

/**
 * @brief FitType转字符串
 */
std::string fitTypeToString(FitType type);

} // namespace FiberThermal

#endif // FIBER_MODEL_H



