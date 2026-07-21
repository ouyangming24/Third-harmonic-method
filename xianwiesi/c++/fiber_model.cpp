/**
 * @file fiber_model.cpp
 * @brief 纤维丝3ω方法热物性测量模型实现
 */

#include "fiber_model.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

namespace FiberThermal {

using namespace Eigen;

// ============================================================================
// 核心计算函数
// ============================================================================

V3OmegaResult calculateFiberV3Omega(
    const std::vector<double>& freqs,
    const FiberParams& fiber_params,
    const MaterialProps& material_props
) {
    V3OmegaResult result;
    
    // 计算热扩散率和热时间常数
    result.alpha = material_props.kappa / (material_props.rho * material_props.C_p);
    result.gamma = fiber_params.L * fiber_params.L / (PI * PI * result.alpha);
    
    // 计算3ω电压
    result.V3_omega.reserve(freqs.size());
    
    for (double f : freqs) {
        double omega = 2.0 * PI * f;
        double denominator = std::sqrt(1.0 + std::pow(2.0 * omega * result.gamma, 2));
        
        // 方程(19): V3ω ≈ (4 I³ L R R')/(π⁴ κ S √(1+(2ωγ)²))
        double V3_amplitude = (4.0 * std::pow(fiber_params.I_rms, 3) * fiber_params.L 
                             * fiber_params.R * std::abs(fiber_params.R_prime))
                             / (std::pow(PI, 4) * material_props.kappa * fiber_params.S * denominator);
        
        // 计算相位 - 方程(18)
        double tan_phi = 2.0 * omega * result.gamma;
        double phi = std::atan(tan_phi);
        
        // 根据R_prime的符号确定相位
        double phase_offset;
        if (fiber_params.R_prime < 0) {
            phase_offset = -phi;  // 负温度系数
        } else {
            phase_offset = PI - phi;  // 正温度系数
        }
        
        // 构建复数3ω电压
        Complex V3(V3_amplitude * std::cos(phase_offset), 
                   V3_amplitude * std::sin(phase_offset));
        result.V3_omega.push_back(V3);
    }
    
    return result;
}

double calculateDCTemperatureRise(
    const FiberParams& fiber_params,
    const MaterialProps& material_props
) {
    // Δ₀ = 2I₀²R/(π κS/L)
    // 注意：I_rms = I₀/√2，所以 I₀² = 2*I_rms²
    double Delta_0 = (2.0 * 2.0 * fiber_params.I_rms * fiber_params.I_rms * fiber_params.R)
                   / (PI * material_props.kappa * fiber_params.S / fiber_params.L);
    return Delta_0;
}

ValidityCheck checkValidityConditions(
    const FiberParams& fiber_params,
    const MaterialProps& material_props,
    const std::vector<double>& freqs
) {
    ValidityCheck result;
    
    // 条件(10): 加热功率均匀性条件
    double I0_squared = 2.0 * fiber_params.I_rms * fiber_params.I_rms;
    result.condition_10_value = (I0_squared * std::abs(fiber_params.R_prime) * fiber_params.L)
                               / (PI * PI * material_props.kappa * fiber_params.S);
    result.condition_10_satisfied = (result.condition_10_value < 0.1);
    
    // 条件(31): 辐射热损失条件
    double D = 2.0 * std::sqrt(fiber_params.S / PI);  // 等效直径
    result.equivalent_diameter = D;
    
    double T0 = 300.0;  // 假设室温
    double epsilon = 0.1;  // 金属表面发射率
    double sigma = 5.67e-8;  // Stefan-Boltzmann常数
    
    double alpha = material_props.kappa / (material_props.rho * material_props.C_p);
    result.gamma = fiber_params.L * fiber_params.L / (PI * PI * alpha);
    
    double g = 16.0 * epsilon * sigma * std::pow(T0, 3) 
             / (material_props.rho * material_props.C_p * D);
    result.condition_31_value = g * result.gamma;
    result.condition_31_satisfied = (result.condition_31_value < 0.1);
    
    // 频率范围建议: 2ωγ < 4
    if (!freqs.empty()) {
        double omega_max = 2.0 * PI * (*std::max_element(freqs.begin(), freqs.end()));
        result.max_2omega_gamma = 2.0 * omega_max * result.gamma;
        result.frequency_range_ok = (result.max_2omega_gamma < 4.0);
    } else {
        result.max_2omega_gamma = 0.0;
        result.frequency_range_ok = true;
    }
    
    return result;
}

// ============================================================================
// FiberThermalFitter 实现
// ============================================================================

FiberThermalFitter::FiberThermalFitter(
    const std::vector<double>& freqs,
    const std::vector<Complex>& V3_exp,
    const FiberParams& fiber_params,
    const MaterialProps& initial_material_props,
    const std::vector<std::string>& material_params_to_fit,
    const std::vector<std::string>& fiber_params_to_fit,
    FitMode fit_mode,
    const ParamBounds& param_bounds,
    FitType fit_type
) : freqs_(freqs),
    V3_exp_(V3_exp),
    fiber_params_(fiber_params),
    material_props_(initial_material_props),
    material_params_to_fit_(material_params_to_fit),
    fiber_params_to_fit_(fiber_params_to_fit),
    fit_mode_(fit_mode),
    param_bounds_(param_bounds),
    fit_type_(fit_type),
    iterations_(0),
    converged_(false),
    final_epsilon_(0.0)
{
    if (freqs_.size() != V3_exp_.size()) {
        throw std::invalid_argument("freqs and V3_exp must have the same size");
    }
    
    // 验证拟合类型和参数
    if (fit_type_ == FitType::MATERIAL && material_params_to_fit_.empty()) {
        throw std::invalid_argument("Material fitting requires material parameters");
    }
    if (fit_type_ == FitType::SENSOR && fiber_params_to_fit_.empty()) {
        throw std::invalid_argument("Sensor fitting requires fiber parameters");
    }
    if (fit_type_ == FitType::COMBINED && 
        material_params_to_fit_.empty() && fiber_params_to_fit_.empty()) {
        throw std::invalid_argument("Combined fitting requires at least one parameter");
    }
    
    createParamMap();
    createBoundsVector();
}

void FiberThermalFitter::createParamMap() {
    param_map_.clear();
    size_t index = 0;
    
    // 根据拟合类型创建参数映射
    if (fit_type_ == FitType::SENSOR) {
        // 仅纤维参数
        for (const auto& param : fiber_params_to_fit_) {
            param_map_.push_back({ParamType::FIBER, param, index++});
        }
    } else if (fit_type_ == FitType::COMBINED) {
        // 先材料参数，后纤维参数
        for (const auto& param : material_params_to_fit_) {
            param_map_.push_back({ParamType::MATERIAL, param, index++});
        }
        for (const auto& param : fiber_params_to_fit_) {
            param_map_.push_back({ParamType::FIBER, param, index++});
        }
    } else {
        // 仅材料参数
        for (const auto& param : material_params_to_fit_) {
            param_map_.push_back({ParamType::MATERIAL, param, index++});
        }
    }
}

void FiberThermalFitter::createBoundsVector() {
    lower_bounds_.clear();
    upper_bounds_.clear();
    
    for (const auto& pmap : param_map_) {
        double current_val = getParamValue(pmap.prop_name, pmap.type);
        double lower, upper;
        
        if (pmap.type == ParamType::FIBER) {
            // 纤维参数边界
            auto it = param_bounds_.fiber_bounds.find(pmap.prop_name);
            if (it != param_bounds_.fiber_bounds.end()) {
                lower = it->second.first;
                upper = it->second.second;
            } else {
                // 默认边界
                if (pmap.prop_name == "R") {
                    lower = current_val * 0.5;
                    upper = current_val * 2.0;
                } else if (pmap.prop_name == "R_prime") {
                    if (current_val > 0) {
                        lower = current_val * 0.1;
                        upper = current_val * 10.0;
                    } else {
                        lower = current_val * 10.0;
                        upper = current_val * 0.1;
                    }
                } else if (pmap.prop_name == "I_rms") {
                    lower = current_val * 0.5;
                    upper = current_val * 2.0;
                } else if (pmap.prop_name == "L") {
                    lower = current_val * 0.9;
                    upper = current_val * 1.1;
                } else if (pmap.prop_name == "S") {
                    lower = current_val * 0.5;
                    upper = current_val * 2.0;
                } else {
                    lower = current_val * 0.1;
                    upper = current_val * 10.0;
                }
            }
        } else {
            // 材料参数边界
            auto it = param_bounds_.material_bounds.find(pmap.prop_name);
            if (it != param_bounds_.material_bounds.end()) {
                lower = it->second.first;
                upper = it->second.second;
            } else {
                // 默认边界
                if (pmap.prop_name == "kappa") {
                    lower = 0.1;
                    upper = 5000.0;
                } else if (pmap.prop_name == "C_p") {
                    lower = 100.0;
                    upper = 5000.0;
                } else if (pmap.prop_name == "rho") {
                    lower = 100.0;
                    upper = 20000.0;
                } else {
                    lower = (current_val > 0) ? current_val * 0.01 : 1e-10;
                    upper = (current_val > 0) ? current_val * 100.0 : 1000.0;
                }
            }
        }
        
        lower_bounds_.push_back(lower);
        upper_bounds_.push_back(upper);
    }
}

std::vector<double> FiberThermalFitter::getParamsAsVector() const {
    std::vector<double> vec;
    vec.reserve(param_map_.size());
    
    for (const auto& pmap : param_map_) {
        vec.push_back(getParamValue(pmap.prop_name, pmap.type));
    }
    
    return vec;
}

void FiberThermalFitter::updateParamsFromVector(const std::vector<double>& param_vector) {
    for (size_t i = 0; i < param_map_.size(); ++i) {
        setParamValue(param_map_[i].prop_name, param_map_[i].type, param_vector[i]);
    }
}

void FiberThermalFitter::applyBounds(std::vector<double>& param_vector) const {
    for (size_t i = 0; i < param_vector.size(); ++i) {
        param_vector[i] = std::max(lower_bounds_[i], 
                                   std::min(upper_bounds_[i], param_vector[i]));
    }
}

double FiberThermalFitter::getParamValue(const std::string& name, ParamType type) const {
    if (type == ParamType::FIBER) {
        if (name == "R") return fiber_params_.R;
        else if (name == "R_prime") return fiber_params_.R_prime;
        else if (name == "I_rms") return fiber_params_.I_rms;
        else if (name == "L") return fiber_params_.L;
        else if (name == "S") return fiber_params_.S;
        throw std::invalid_argument("Unknown fiber parameter: " + name);
    } else {
        if (name == "kappa") return material_props_.kappa;
        else if (name == "C_p") return material_props_.C_p;
        else if (name == "rho") return material_props_.rho;
        throw std::invalid_argument("Unknown material parameter: " + name);
    }
}

void FiberThermalFitter::setParamValue(const std::string& name, ParamType type, double value) {
    if (type == ParamType::FIBER) {
        if (name == "R") fiber_params_.R = value;
        else if (name == "R_prime") fiber_params_.R_prime = value;
        else if (name == "I_rms") fiber_params_.I_rms = value;
        else if (name == "L") fiber_params_.L = value;
        else if (name == "S") fiber_params_.S = value;
        else throw std::invalid_argument("Unknown fiber parameter: " + name);
    } else {
        if (name == "kappa") material_props_.kappa = value;
        else if (name == "C_p") material_props_.C_p = value;
        else if (name == "rho") material_props_.rho = value;
        else throw std::invalid_argument("Unknown material parameter: " + name);
    }
}

std::vector<std::vector<Complex>> FiberThermalFitter::calculateSensitivityMatrix() const {
    const size_t n_params = param_map_.size();
    const size_t n_freqs = freqs_.size();
    
    std::vector<std::vector<Complex>> D(n_params, std::vector<Complex>(n_freqs));
    
    // 获取当前参数向量
    std::vector<double> current_params = getParamsAsVector();
    
    // 计算基准V3
    V3OmegaResult base_result = calculateFiberV3Omega(freqs_, fiber_params_, material_props_);
    
    // 保存原始状态
    FiberParams original_fiber = fiber_params_;
    MaterialProps original_material = material_props_;
    
    // 对每个参数计算灵敏度
    for (size_t k = 0; k < n_params; ++k) {
        std::vector<double> params_perturbed = current_params;
        double delta = (params_perturbed[k] != 0.0) ? 1e-6 * std::abs(params_perturbed[k]) : 1e-8;
        params_perturbed[k] += delta;
        
        // 更新参数（修改临时对象）
        FiberThermalFitter* mutable_this = const_cast<FiberThermalFitter*>(this);
        mutable_this->updateParamsFromVector(params_perturbed);
        
        // 计算扰动后的V3
        V3OmegaResult perturbed_result = calculateFiberV3Omega(freqs_, fiber_params_, material_props_);
        
        // 恢复原始状态
        mutable_this->fiber_params_ = original_fiber;
        mutable_this->material_props_ = original_material;
        
        // 计算导数并存储灵敏度 D_k = p_k * dV3/dp_k
        for (size_t i = 0; i < n_freqs; ++i) {
            Complex derivative = (perturbed_result.V3_omega[i] - base_result.V3_omega[i]) / delta;
            D[k][i] = current_params[k] * derivative;
        }
    }
    
    return D;
}

void FiberThermalFitter::fit(int max_iter, double tol, double zeta) {
    std::string fit_target = (fit_type_ == FitType::SENSOR) ? "纤维参数" : "材料热物性";
    std::cout << "开始迭代拟合 (目标: " << fit_target 
              << ", 模式: " << fitModeToString(fit_mode_) << ")..." << std::endl;
    
    printValidityCheck();
    
    std::cout << "\n开始迭代 (最大迭代次数: " << max_iter 
              << ", 容差: " << std::scientific << tol 
              << ", 步长因子: " << std::fixed << zeta << ")" << std::endl;
    
    const size_t n_params = param_map_.size();
    const size_t n_freqs = freqs_.size();
    
    // 迭代主循环
    for (int j = 0; j < max_iter; ++j) {
        // 计算当前理论V3
        V3OmegaResult V3_tilde_result = calculateFiberV3Omega(freqs_, fiber_params_, material_props_);
        const auto& V3_tilde = V3_tilde_result.V3_omega;
        
        // 计算灵敏度矩阵
        auto D_V3_complex = calculateSensitivityMatrix();
        
        // 使用 Eigen 构建方程组
        MatrixXd A = MatrixXd::Zero(n_params, n_params);
        VectorXd B = VectorXd::Zero(n_params);
        
        // 根据拟合模式构建方程组
        if (fit_mode_ == FitMode::BOTH) {
            // 实部和虚部
            for (size_t k = 0; k < n_params; ++k) {
                for (size_t m = 0; m < n_params; ++m) {
                    for (size_t i = 0; i < n_freqs; ++i) {
                        A(k, m) += D_V3_complex[k][i].real() * D_V3_complex[m][i].real()
                                 + D_V3_complex[k][i].imag() * D_V3_complex[m][i].imag();
                    }
                }
                for (size_t i = 0; i < n_freqs; ++i) {
                    double r_err = V3_exp_[i].real() - V3_tilde[i].real();
                    double i_err = V3_exp_[i].imag() - V3_tilde[i].imag();
                    B(k) += D_V3_complex[k][i].real() * r_err
                          + D_V3_complex[k][i].imag() * i_err;
                }
            }
        } else if (fit_mode_ == FitMode::REAL) {
            // 仅实部
            for (size_t k = 0; k < n_params; ++k) {
                for (size_t m = 0; m < n_params; ++m) {
                    for (size_t i = 0; i < n_freqs; ++i) {
                        A(k, m) += D_V3_complex[k][i].real() * D_V3_complex[m][i].real();
                    }
                }
                for (size_t i = 0; i < n_freqs; ++i) {
                    double r_err = V3_exp_[i].real() - V3_tilde[i].real();
                    B(k) += D_V3_complex[k][i].real() * r_err;
                }
            }
        } else if (fit_mode_ == FitMode::IMAG) {
            // 仅虚部
            for (size_t k = 0; k < n_params; ++k) {
                for (size_t m = 0; m < n_params; ++m) {
                    for (size_t i = 0; i < n_freqs; ++i) {
                        A(k, m) += D_V3_complex[k][i].imag() * D_V3_complex[m][i].imag();
                    }
                }
                for (size_t i = 0; i < n_freqs; ++i) {
                    double i_err = V3_exp_[i].imag() - V3_tilde[i].imag();
                    B(k) += D_V3_complex[k][i].imag() * i_err;
                }
            }
        } else if (fit_mode_ == FitMode::AMP) {
            // 仅幅值
            for (size_t k = 0; k < n_params; ++k) {
                for (size_t m = 0; m < n_params; ++m) {
                    for (size_t i = 0; i < n_freqs; ++i) {
                        double V3_amp = std::abs(V3_tilde[i]);
                        Complex conj_V3 = std::conj(V3_tilde[i]);
                        double D_amp_k = (conj_V3 * D_V3_complex[k][i]).real() / V3_amp;
                        double D_amp_m = (conj_V3 * D_V3_complex[m][i]).real() / V3_amp;
                        A(k, m) += D_amp_k * D_amp_m;
                    }
                }
                for (size_t i = 0; i < n_freqs; ++i) {
                    double err = std::abs(V3_exp_[i]) - std::abs(V3_tilde[i]);
                    double V3_amp = std::abs(V3_tilde[i]);
                    Complex conj_V3 = std::conj(V3_tilde[i]);
                    double D_amp_k = (conj_V3 * D_V3_complex[k][i]).real() / V3_amp;
                    B(k) += D_amp_k * err;
                }
            }
        } else if (fit_mode_ == FitMode::PHASE) {
            // 仅相位
            std::vector<double> phase_exp(n_freqs), phase_tilde(n_freqs);
            for (size_t i = 0; i < n_freqs; ++i) {
                phase_exp[i] = std::arg(V3_exp_[i]);
                phase_tilde[i] = std::arg(V3_tilde[i]);
            }
            // Unwrap phase
            for (size_t i = 1; i < n_freqs; ++i) {
                double diff = phase_exp[i] - phase_exp[i-1];
                if (diff > PI) phase_exp[i] -= 2*PI;
                if (diff < -PI) phase_exp[i] += 2*PI;
                
                diff = phase_tilde[i] - phase_tilde[i-1];
                if (diff > PI) phase_tilde[i] -= 2*PI;
                if (diff < -PI) phase_tilde[i] += 2*PI;
            }
            
            for (size_t k = 0; k < n_params; ++k) {
                for (size_t m = 0; m < n_params; ++m) {
                    for (size_t i = 0; i < n_freqs; ++i) {
                        double V3_amp_sq = std::norm(V3_tilde[i]);
                        Complex conj_V3 = std::conj(V3_tilde[i]);
                        double D_phase_k = (conj_V3 * D_V3_complex[k][i]).imag() / V3_amp_sq;
                        double D_phase_m = (conj_V3 * D_V3_complex[m][i]).imag() / V3_amp_sq;
                        A(k, m) += D_phase_k * D_phase_m;
                    }
                }
                for (size_t i = 0; i < n_freqs; ++i) {
                    double err = phase_exp[i] - phase_tilde[i];
                    double V3_amp_sq = std::norm(V3_tilde[i]);
                    Complex conj_V3 = std::conj(V3_tilde[i]);
                    double D_phase_k = (conj_V3 * D_V3_complex[k][i]).imag() / V3_amp_sq;
                    B(k) += D_phase_k * err;
                }
            }
        }
        
        // 添加正则化
        A += MatrixXd::Identity(n_params, n_params) * 1e-12;
        
        // 求解线性系统
        VectorXd c = A.ldlt().solve(B);
        
        // 更新参数
        std::vector<double> current_params = getParamsAsVector();
        std::vector<double> new_params(n_params);
        
        for (size_t k = 0; k < n_params; ++k) {
            new_params[k] = (c(k) * zeta + 1.0) * current_params[k];
        }
        
        // 应用边界约束
        applyBounds(new_params);
        updateParamsFromVector(new_params);
        
        // 计算epsilon
        double epsilon = 0.0;
        for (size_t k = 0; k < n_params; ++k) {
            epsilon += std::abs((new_params[k] - current_params[k]) / current_params[k]);
        }
        
        printIterationInfo(j + 1, epsilon);
        
        // 检查边界
        std::vector<std::string> bounded_params;
        for (size_t k = 0; k < n_params; ++k) {
            if (std::abs(new_params[k] - lower_bounds_[k]) < 1e-6 * std::abs(lower_bounds_[k]) ||
                std::abs(new_params[k] - upper_bounds_[k]) < 1e-6 * std::abs(upper_bounds_[k])) {
                std::string param_name = (param_map_[k].type == ParamType::FIBER ? "fiber." : "material.")
                                       + param_map_[k].prop_name;
                bounded_params.push_back(param_name);
            }
        }
        if (!bounded_params.empty()) {
            std::cout << "  警告: 以下参数触及边界: ";
            for (size_t i = 0; i < bounded_params.size(); ++i) {
                std::cout << bounded_params[i];
                if (i < bounded_params.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        
        // 检查收敛
        iterations_ = j + 1;
        final_epsilon_ = epsilon;
        
        if (epsilon < tol) {
            converged_ = true;
            std::cout << "在 " << (j + 1) << " 次迭代后收敛。" << std::endl;
            break;
        }
    }
    
    if (!converged_) {
        std::cout << "警告: 已达到最大迭代次数但未收敛。" << std::endl;
    }
    
    printFinalResults();
}

std::vector<Complex> FiberThermalFitter::calculateFittedV3() const {
    return calculateFiberV3Omega(freqs_, fiber_params_, material_props_).V3_omega;
}

double FiberThermalFitter::getAlpha() const {
    return calculateFiberV3Omega(freqs_, fiber_params_, material_props_).alpha;
}

double FiberThermalFitter::getGamma() const {
    return calculateFiberV3Omega(freqs_, fiber_params_, material_props_).gamma;
}

void FiberThermalFitter::printValidityCheck() const {
    ValidityCheck validity = checkValidityConditions(fiber_params_, material_props_, freqs_);
    
    std::cout << "\n模型有效性检查:" << std::endl;
    std::cout << "  条件(10) - 功率均匀性: " << std::scientific << validity.condition_10_value
              << " (" << (validity.condition_10_satisfied ? "满足" : "不满足") << ")" << std::endl;
    std::cout << "  条件(31) - 辐射损失: " << std::scientific << validity.condition_31_value
              << " (" << (validity.condition_31_satisfied ? "满足" : "不满足") << ")" << std::endl;
    std::cout << "  频率范围 (2ωγ_max): " << std::fixed << std::setprecision(2) 
              << validity.max_2omega_gamma
              << " (" << (validity.frequency_range_ok ? "合适" : "过高") << ")" << std::endl;
}

void FiberThermalFitter::printIterationInfo(int iter, double epsilon) const {
    std::cout << "迭代 " << std::setw(2) << iter << "/" << iterations_ 
              << ": Epsilon = " << std::scientific << std::setprecision(4) << epsilon << std::endl;
}

void FiberThermalFitter::printFinalResults() const {
    if (fit_type_ == FitType::SENSOR) {
        std::cout << "\n最终拟合的纤维参数:" << std::endl;
        for (const auto& pmap : param_map_) {
            double val = getParamValue(pmap.prop_name, pmap.type);
            std::cout << "  " << pmap.prop_name << ": " << std::scientific << val;
            auto it = param_bounds_.fiber_bounds.find(pmap.prop_name);
            if (it != param_bounds_.fiber_bounds.end()) {
                std::cout << " (边界: [" << it->second.first << ", " << it->second.second << "])";
            }
            std::cout << std::endl;
        }
    } else if (fit_type_ == FitType::COMBINED) {
        std::cout << "\n最终拟合的材料和纤维参数:" << std::endl;
        std::cout << "--- 材料属性 ---" << std::endl;
        for (const auto& pmap : param_map_) {
            if (pmap.type == ParamType::MATERIAL) {
                double val = getParamValue(pmap.prop_name, pmap.type);
                std::cout << "  " << pmap.prop_name << ": " << std::scientific << val;
                auto it = param_bounds_.material_bounds.find(pmap.prop_name);
                if (it != param_bounds_.material_bounds.end()) {
                    std::cout << " (边界: [" << it->second.first << ", " << it->second.second << "])";
                }
                std::cout << std::endl;
            }
        }
        std::cout << "\n--- 纤维参数 ---" << std::endl;
        for (const auto& pmap : param_map_) {
            if (pmap.type == ParamType::FIBER) {
                double val = getParamValue(pmap.prop_name, pmap.type);
                std::cout << "  " << pmap.prop_name << ": " << std::scientific << val;
                auto it = param_bounds_.fiber_bounds.find(pmap.prop_name);
                if (it != param_bounds_.fiber_bounds.end()) {
                    std::cout << " (边界: [" << it->second.first << ", " << it->second.second << "])";
                }
                std::cout << std::endl;
            }
        }
    } else {
        std::cout << "\n最终拟合的材料属性:" << std::endl;
        for (const auto& pmap : param_map_) {
            double val = getParamValue(pmap.prop_name, pmap.type);
            std::cout << "  " << pmap.prop_name << ": " << std::scientific << val;
            auto it = param_bounds_.material_bounds.find(pmap.prop_name);
            if (it != param_bounds_.material_bounds.end()) {
                std::cout << " (边界: [" << it->second.first << ", " << it->second.second << "])";
            }
            std::cout << std::endl;
        }
        std::cout << "\n导出参数:" << std::endl;
        std::cout << "  热扩散率 alpha: " << std::scientific << getAlpha() << " m^2/s" << std::endl;
        std::cout << "  热时间常数 gamma: " << std::scientific << getGamma() << " s" << std::endl;
    }
}

// ============================================================================
// 辅助函数实现
// ============================================================================

std::map<std::string, MaterialProps> createFiberMaterialLibrary() {
    std::map<std::string, MaterialProps> materials;
    
    MaterialProps platinum(71.6, 21450, 133);
    platinum.description = "铂丝 (标准参考材料)";
    materials["platinum"] = platinum;
    
    MaterialProps carbon_nanotube(3000, 1300, 700);
    carbon_nanotube.description = "碳纳米管束 (高热导率)";
    materials["carbon_nanotube"] = carbon_nanotube;
    
    MaterialProps copper(401, 8960, 385);
    copper.description = "铜丝";
    materials["copper"] = copper;
    
    MaterialProps tungsten(173, 19300, 132);
    tungsten.description = "钨丝";
    materials["tungsten"] = tungsten;
    
    MaterialProps nichrome(11.3, 8400, 450);
    nichrome.description = "镍铬合金丝";
    materials["nichrome"] = nichrome;
    
    return materials;
}

std::map<std::string, FiberParams> createTypicalFiberParams() {
    std::map<std::string, FiberParams> configs;
    
    FiberParams platinum_wire;
    platinum_wire.L = 8e-3;
    platinum_wire.diameter = 20e-6;
    platinum_wire.S = PI * std::pow(10e-6, 2);
    platinum_wire.R = 5.0;
    platinum_wire.R_prime = 0.01;
    platinum_wire.I_rms = 10e-3;
    platinum_wire.description = "20μm铂丝 (论文中的配置)";
    configs["platinum_wire_20um"] = platinum_wire;
    
    FiberParams cnt_bundle;
    cnt_bundle.L = 1e-3;
    cnt_bundle.diameter = 10e-6;
    cnt_bundle.S = PI * std::pow(5e-6, 2);
    cnt_bundle.R = 50.0;
    cnt_bundle.R_prime = -0.05;
    cnt_bundle.I_rms = 5e-3;
    cnt_bundle.description = "碳纳米管束 (负温度系数)";
    configs["carbon_nanotube_bundle"] = cnt_bundle;
    
    return configs;
}

FitMode stringToFitMode(const std::string& mode_str) {
    if (mode_str == "both") return FitMode::BOTH;
    if (mode_str == "real") return FitMode::REAL;
    if (mode_str == "imag") return FitMode::IMAG;
    if (mode_str == "amp" || mode_str == "amplitude") return FitMode::AMP;
    if (mode_str == "phase") return FitMode::PHASE;
    throw std::invalid_argument("Unknown fit mode: " + mode_str);
}

FitType stringToFitType(const std::string& type_str) {
    if (type_str == "material") return FitType::MATERIAL;
    if (type_str == "sensor") return FitType::SENSOR;
    if (type_str == "combined") return FitType::COMBINED;
    throw std::invalid_argument("Unknown fit type: " + type_str);
}

std::string fitModeToString(FitMode mode) {
    switch (mode) {
        case FitMode::BOTH: return "both";
        case FitMode::REAL: return "real";
        case FitMode::IMAG: return "imag";
        case FitMode::AMP: return "amp";
        case FitMode::PHASE: return "phase";
        default: return "unknown";
    }
}

std::string fitTypeToString(FitType type) {
    switch (type) {
        case FitType::MATERIAL: return "material";
        case FitType::SENSOR: return "sensor";
        case FitType::COMBINED: return "combined";
        default: return "unknown";
    }
}

} // namespace FiberThermal



