/**
 * @file test_fiber_model.cpp
 * @brief 纤维丝3ω方法模型测试程序
 */

#include "fiber_model.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

using namespace FiberThermal;

// 生成对数空间的频率数组
std::vector<double> logspace(double start, double end, size_t num) {
    std::vector<double> result(num);
    double log_start = std::log10(start);
    double log_end = std::log10(end);
    double step = (log_end - log_start) / (num - 1);
    
    for (size_t i = 0; i < num; ++i) {
        result[i] = std::pow(10.0, log_start + i * step);
    }
    return result;
}

// 添加复数噪声
std::vector<Complex> addNoise(const std::vector<Complex>& signal, double noise_level, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<> dis(0.0, 1.0);
    
    // 计算平均幅值
    double mean_amp = 0.0;
    for (const auto& s : signal) {
        mean_amp += std::abs(s);
    }
    mean_amp /= signal.size();
    
    std::vector<Complex> noisy_signal(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        double noise_real = noise_level * dis(gen) * mean_amp;
        double noise_imag = noise_level * dis(gen) * mean_amp;
        noisy_signal[i] = signal[i] + Complex(noise_real, noise_imag);
    }
    
    return noisy_signal;
}

void printSeparator(const std::string& title = "") {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    if (!title.empty()) {
        std::cout << title << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }
}

void testBasicCalculation() {
    printSeparator("基础测试：3ω 电压计算");
    
    // 创建测试参数
    auto fiber_configs = createTypicalFiberParams();
    auto materials = createFiberMaterialLibrary();
    
    FiberParams fiber = fiber_configs["platinum_wire_20um"];
    MaterialProps material = materials["platinum"];
    
    // 生成频率数组
    std::vector<double> freqs = logspace(1.0, 100.0, 20);
    
    // 计算3ω电压
    V3OmegaResult result = calculateFiberV3Omega(freqs, fiber, material);
    
    std::cout << "\n测试配置:" << std::endl;
    std::cout << "  纤维: " << fiber.description << std::endl;
    std::cout << "  材料: " << material.description << std::endl;
    std::cout << "  频率点数: " << freqs.size() << std::endl;
    
    std::cout << "\n计算结果:" << std::endl;
    std::cout << "  热扩散率 α: " << std::scientific << result.alpha << " m²/s" << std::endl;
    std::cout << "  热时间常数 γ: " << std::scientific << result.gamma << " s" << std::endl;
    
    double min_V3 = std::abs(result.V3_omega[0]);
    double max_V3 = std::abs(result.V3_omega[0]);
    for (const auto& v : result.V3_omega) {
        double amp = std::abs(v);
        min_V3 = std::min(min_V3, amp);
        max_V3 = std::max(max_V3, amp);
    }
    std::cout << "  3ω电压范围: " << min_V3 << " - " << max_V3 << " V" << std::endl;
    
    // 有效性检查
    ValidityCheck validity = checkValidityConditions(fiber, material, freqs);
    std::cout << "\n有效性检查:" << std::endl;
    std::cout << "  condition_10_value: " << validity.condition_10_value 
              << " (" << (validity.condition_10_satisfied ? "OK" : "FAIL") << ")" << std::endl;
    std::cout << "  condition_31_value: " << validity.condition_31_value
              << " (" << (validity.condition_31_satisfied ? "OK" : "FAIL") << ")" << std::endl;
    std::cout << "  max_2omega_gamma: " << std::fixed << validity.max_2omega_gamma
              << " (" << (validity.frequency_range_ok ? "OK" : "FAIL") << ")" << std::endl;
}

void testMaterialFitting() {
    printSeparator("测试 1: 材料参数拟合 (fit_type='material')");
    
    // 准备数据
    auto fiber_configs = createTypicalFiberParams();
    auto materials = createFiberMaterialLibrary();
    
    FiberParams fiber = fiber_configs["platinum_wire_20um"];
    MaterialProps true_material = materials["platinum"];
    
    std::vector<double> freqs = logspace(1.0, 100.0, 20);
    
    // 生成理论数据并添加噪声
    V3OmegaResult true_result = calculateFiberV3Omega(freqs, fiber, true_material);
    std::vector<Complex> V3_exp = addNoise(true_result.V3_omega, 0.02);
    
    std::cout << "\n真实材料属性:" << std::endl;
    std::cout << "  kappa: " << true_material.kappa << " W/m·K" << std::endl;
    std::cout << "  C_p: " << true_material.C_p << " J/kg·K" << std::endl;
    
    // 创建初始猜测（偏离真值）
    MaterialProps initial_guess = true_material;
    initial_guess.kappa *= 1.2;  // 偏离20%
    initial_guess.C_p *= 0.8;
    
    std::cout << "\n初始猜测:" << std::endl;
    std::cout << "  kappa: " << initial_guess.kappa << " W/m·K" << std::endl;
    std::cout << "  C_p: " << initial_guess.C_p << " J/kg·K" << std::endl;
    
    // 设置参数边界
    ParamBounds bounds;
    bounds.material_bounds["kappa"] = {50.0, 100.0};
    bounds.material_bounds["C_p"] = {100.0, 200.0};
    
    // 创建拟合器
    FiberThermalFitter fitter(
        freqs, V3_exp, fiber, initial_guess,
        {"kappa", "C_p"},  // 要拟合的参数
        {},                // 不拟合纤维参数
        FitMode::BOTH,
        bounds,
        FitType::MATERIAL
    );
    
    // 执行拟合
    fitter.fit(10, 1e-5, 0.5);
    
    // 输出结果
    MaterialProps fitted = fitter.getFittedMaterialProps();
    std::cout << "\n拟合误差:" << std::endl;
    std::cout << "  kappa误差: " << std::fixed << std::setprecision(2)
              << std::abs(fitted.kappa - true_material.kappa) / true_material.kappa * 100.0 
              << "%" << std::endl;
    std::cout << "  C_p误差: " << std::fixed << std::setprecision(2)
              << std::abs(fitted.C_p - true_material.C_p) / true_material.C_p * 100.0 
              << "%" << std::endl;
    std::cout << "  收敛状态: " << (fitter.isConverged() ? "是" : "否") << std::endl;
    std::cout << "  迭代次数: " << fitter.getIterations() << std::endl;
}

void testSensorFitting() {
    printSeparator("测试 2: 传感器参数拟合 (fit_type='sensor')");
    
    // 准备数据
    auto fiber_configs = createTypicalFiberParams();
    auto materials = createFiberMaterialLibrary();
    
    FiberParams true_fiber = fiber_configs["platinum_wire_20um"];
    MaterialProps material = materials["platinum"];
    
    std::vector<double> freqs = logspace(1.0, 100.0, 20);
    
    // 生成理论数据并添加噪声
    V3OmegaResult true_result = calculateFiberV3Omega(freqs, true_fiber, material);
    std::vector<Complex> V3_exp = addNoise(true_result.V3_omega, 0.02);
    
    std::cout << "\n真实纤维参数:" << std::endl;
    std::cout << "  R: " << true_fiber.R << " Ω" << std::endl;
    std::cout << "  R_prime: " << true_fiber.R_prime << " Ω/K" << std::endl;
    
    // 创建初始猜测
    FiberParams initial_fiber = true_fiber;
    initial_fiber.R *= 1.05;
    initial_fiber.R_prime *= 0.95;
    
    std::cout << "\n初始猜测:" << std::endl;
    std::cout << "  R: " << initial_fiber.R << " Ω" << std::endl;
    std::cout << "  R_prime: " << initial_fiber.R_prime << " Ω/K" << std::endl;
    
    // 设置参数边界
    ParamBounds bounds;
    bounds.fiber_bounds["R"] = {4.0, 6.0};
    bounds.fiber_bounds["R_prime"] = {0.005, 0.015};
    
    // 创建拟合器
    FiberThermalFitter fitter(
        freqs, V3_exp, initial_fiber, material,
        {},  // 不拟合材料参数
        {"R", "R_prime"},  // 要拟合的纤维参数
        FitMode::BOTH,
        bounds,
        FitType::SENSOR
    );
    
    // 执行拟合
    fitter.fit(10, 1e-5, 0.5);
    
    // 输出结果
    FiberParams fitted_fiber = fitter.getFittedFiberParams();
    std::cout << "\n拟合误差:" << std::endl;
    std::cout << "  R误差: " << std::fixed << std::setprecision(2)
              << std::abs(fitted_fiber.R - true_fiber.R) / true_fiber.R * 100.0 
              << "%" << std::endl;
    std::cout << "  R_prime误差: " << std::fixed << std::setprecision(2)
              << std::abs(fitted_fiber.R_prime - true_fiber.R_prime) / true_fiber.R_prime * 100.0 
              << "%" << std::endl;
    std::cout << "  收敛状态: " << (fitter.isConverged() ? "是" : "否") << std::endl;
}

void testCombinedFitting() {
    printSeparator("测试 3: 联合拟合 (fit_type='combined')");
    
    // 准备数据
    auto fiber_configs = createTypicalFiberParams();
    auto materials = createFiberMaterialLibrary();
    
    FiberParams true_fiber = fiber_configs["platinum_wire_20um"];
    MaterialProps true_material = materials["platinum"];
    
    std::vector<double> freqs = logspace(1.0, 100.0, 20);
    
    // 生成理论数据并添加噪声
    V3OmegaResult true_result = calculateFiberV3Omega(freqs, true_fiber, true_material);
    std::vector<Complex> V3_exp = addNoise(true_result.V3_omega, 0.02);
    
    std::cout << "\n真实值:" << std::endl;
    std::cout << "  kappa: " << true_material.kappa << " W/m·K" << std::endl;
    std::cout << "  R: " << true_fiber.R << " Ω" << std::endl;
    
    // 创建初始猜测
    MaterialProps initial_material = true_material;
    initial_material.kappa *= 1.15;
    
    FiberParams initial_fiber = true_fiber;
    initial_fiber.R *= 1.08;
    
    std::cout << "\n初始猜测:" << std::endl;
    std::cout << "  kappa: " << initial_material.kappa << " W/m·K" << std::endl;
    std::cout << "  R: " << initial_fiber.R << " Ω" << std::endl;
    
    // 创建拟合器
    FiberThermalFitter fitter(
        freqs, V3_exp, initial_fiber, initial_material,
        {"kappa"},  // 拟合kappa
        {"R"},      // 拟合R
        FitMode::BOTH,
        ParamBounds(),
        FitType::COMBINED
    );
    
    // 执行拟合
    fitter.fit(10, 1e-5, 0.5);
    
    // 输出结果
    MaterialProps fitted_material = fitter.getFittedMaterialProps();
    FiberParams fitted_fiber = fitter.getFittedFiberParams();
    
    std::cout << "\n拟合误差:" << std::endl;
    std::cout << "  kappa误差: " << std::fixed << std::setprecision(2)
              << std::abs(fitted_material.kappa - true_material.kappa) / true_material.kappa * 100.0 
              << "%" << std::endl;
    std::cout << "  R误差: " << std::fixed << std::setprecision(2)
              << std::abs(fitted_fiber.R - true_fiber.R) / true_fiber.R * 100.0 
              << "%" << std::endl;
    std::cout << "  收敛状态: " << (fitter.isConverged() ? "是" : "否") << std::endl;
}

int main() {
    std::cout << "纤维丝3ω方法模型 C++ 版本测试" << std::endl;
    std::cout << "完全兼容 ThermalModelFitter 接口" << std::endl;
    
    try {
        testBasicCalculation();
        testMaterialFitting();
        testSensorFitting();
        testCombinedFitting();
        
        printSeparator();
        std::cout << "所有测试完成！" << std::endl;
        printSeparator();
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}



