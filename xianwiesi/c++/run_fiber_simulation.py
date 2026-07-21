import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from fiber_model import (
    calculate_fiber_V3omega, 
    FiberThermalFitter,
    create_fiber_material_library,
    create_typical_fiber_params,
    check_validity_conditions
)
from matplotlib import rcParams
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def run_fiber_simulation():
    """
    运行纤维丝3ω方法的模拟验证
    
    生成合成数据，验证拟合算法，并保存结果
    
    【配置说明】
    - 随机种子已固定为42，确保噪声可重复
    - 初值偏离：κ×1.2, Cp×0.8（测试算法收敛性）
    - 参数边界：kappa(0.1,5000), C_p(50,2000)
    - 优化参数：max_iter=30, zeta=0.5
    
    生成的数据可用 fit_fiber_data.py 拟合，结果应完全一致
    """
    print("纤维丝3ω方法模拟验证")
    print("="*60)
    
    # --- 1. 设置实验参数 ---
    # 频率范围：1-100 Hz，适合纤维丝测量
    freqs = np.logspace(0, 3, 30)  # 1 to 100 Hz, 30 points
    
    # --- 2. 选择材料和纤维配置 ---
    materials = create_fiber_material_library()
    fiber_configs = create_typical_fiber_params()
    
    # 可以选择不同的配置进行测试
    test_configurations = [
        {
            'name': '铂丝测试',
            'material': 'platinum',
            'fiber': 'platinum_wire_20um',
            'noise_level': 0.02  # 2% 噪声
        },
        {
            'name': '碳纳米管束测试', 
            'material': 'carbon_nanotube',
            'fiber': 'carbon_nanotube_bundle',
            'noise_level': 0.01  # 5% 噪声
        }
    ]
    
    # 选择当前测试配置
    current_config = test_configurations[0]  # 可以修改索引选择不同配置
    
    print(f"\n当前配置: {current_config['name']}")
    
    # 获取材料和纤维参数
    true_material = materials[current_config['material']].copy()
    fiber_params = fiber_configs[current_config['fiber']].copy()
    noise_level = current_config['noise_level']
    
    print(f"\n真实材料属性:")
    print(f"  热导率 κ: {true_material['kappa']:.2f} W/m·K")
    print(f"  密度 ρ: {true_material['rho']:.0f} kg/m³") 
    print(f"  比热容 C_p: {true_material['C_p']:.0f} J/kg·K")
    
    print(f"\n纤维参数:")
    print(f"  长度 L: {fiber_params['L']*1e3:.1f} mm")
    print(f"  直径: {fiber_params['diameter']*1e6:.1f} μm")
    print(f"  电阻 R: {fiber_params['R']:.1f} Ω")
    print(f"  温度系数 R': {fiber_params['R_prime']:.3f} Ω/K")
    print(f"  RMS电流: {fiber_params['I_rms']*1e3:.1f} mA")
    
    # --- 3. 检查模型有效性条件 ---
    validity = check_validity_conditions(fiber_params, true_material, freqs)
    
    print(f"\n模型有效性检查:")
    print(f"  条件(10) - 功率均匀性: {validity['condition_10_value']:.2e} "
          f"({'✓' if validity['condition_10_satisfied'] else '✗'})")
    print(f"  条件(31) - 辐射损失: {validity['condition_31_value']:.2e} "
          f"({'✓' if validity['condition_31_satisfied'] else '✗'})")
    print(f"  频率范围 (2ωγ_max): {validity['max_2omega_gamma']:.2f} "
          f"({'✓' if validity['frequency_range_ok'] else '✗'})")
    print(f"  热时间常数 γ: {validity['gamma']:.2e} s")
    
    # --- 4. 生成理想的3ω数据 ---
    print(f"\n生成理想3ω数据...")
    V3_ideal, gamma_true, alpha_true = calculate_fiber_V3omega(
        freqs, fiber_params, true_material
    )
    
    print(f"  计算得到的热扩散率: {alpha_true:.2e} m²/s")
    print(f"  计算得到的热时间常数: {gamma_true:.2e} s")
    
    # --- 5. 添加噪声生成实验数据 ---
    print(f"\n添加 {noise_level*100:.1f}% 噪声...")
    
    # 固定随机种子以确保可重复性（与fit_fiber_data.py结果一致）
    np.random.seed(42)
    
    # 添加复数噪声
    noise_real = noise_level * np.random.randn(len(freqs)) * np.mean(np.abs(V3_ideal))
    noise_imag = noise_level * np.random.randn(len(freqs)) * np.mean(np.abs(V3_ideal))
    noise = noise_real + 1j * noise_imag
    
    V3_noisy = V3_ideal + noise
    
    print(f"  信号幅值范围: {np.min(np.abs(V3_ideal)):.2e} - {np.max(np.abs(V3_ideal)):.2e} V")
    print(f"  噪声RMS: {np.sqrt(np.mean(np.abs(noise)**2)):.2e} V")
    
    # --- 6. 设置拟合参数 ---
    # 创建初始猜测（偏离真值）
    initial_guess = true_material.copy()
    initial_guess['kappa'] *= 1.2  # 热导率初值偏离20%
    initial_guess['C_p'] *= 0.8    # 比热容初值偏离20%
    
    print(f"\n初始猜测:")
    print(f"  热导率 κ: {initial_guess['kappa']:.2f} W/m·K (真值: {true_material['kappa']:.2f})")
    print(f"  比热容 C_p: {initial_guess['C_p']:.0f} J/kg·K (真值: {true_material['C_p']:.0f})")
    
    # --- 7. 执行不同拟合模式的测试 ---
    fit_modes = ['amp', 'phase', 'both']  # 使用新的模式名称
    fit_results = {}
    
    for fit_mode in fit_modes:
        print(f"\n{'='*20} {fit_mode.upper()} 拟合模式 {'='*20}")
        
        # 设置参数边界（使用新的字典格式）
        param_bounds = {
            'material': {
                'kappa': (0.1, 5000),      # 热导率边界
                'C_p': (50, 2000),         # 比热容边界
            }
        }
        
        # 创建拟合器（使用新接口）
        fitter = FiberThermalFitter(
            freqs=freqs,
            V3_exp=V3_noisy,
            fiber_params=fiber_params,
            initial_material_props=initial_guess,
            params_to_fit={'material': ['kappa', 'C_p']},  # 新的字典格式
            fit_mode=fit_mode,
            param_bounds=param_bounds,
            fit_type='material'  # 明确指定拟合类型
        )
        
        # 执行拟合（使用迭代方法）
        result = fitter.fit(max_iter=30, tol=1e-5, zeta=0.5)
        fit_results[fit_mode] = result
        
        # 计算误差（新接口直接返回 material_props）
        kappa_error = abs(result['kappa'] - true_material['kappa']) / true_material['kappa'] * 100
        Cp_error = abs(result['C_p'] - true_material['C_p']) / true_material['C_p'] * 100
        
        print(f"\n拟合误差:")
        print(f"  热导率误差: {kappa_error:.2f}%")
        print(f"  比热容误差: {Cp_error:.2f}%")
    
    # --- 8. 可视化结果 ---
    print(f"\n生成可视化结果...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'纤维丝3ω方法拟合结果 - {current_config["name"]}', fontsize=14)
    
    # 选择最佳拟合结果进行显示（通常是'both'模式）
    best_result = fit_results['both']
    # 重新计算拟合的V3（新接口返回material_props）
    V3_fitted, _, _ = calculate_fiber_V3omega(freqs, fiber_params, best_result)
    
    # 子图1: 幅值对比
    ax1 = axes[0, 0]
    ax1.loglog(freqs, np.abs(V3_noisy) * 1e6, 'ro', label='实验数据 (含噪声)', markersize=6)
    ax1.loglog(freqs, np.abs(V3_ideal) * 1e6, 'b-', label='理想数据', linewidth=2)
    ax1.loglog(freqs, np.abs(V3_fitted) * 1e6, 'g--', label='拟合结果', linewidth=2)
    ax1.set_xlabel('频率 [Hz]')
    ax1.set_ylabel('|V₃ω| [μV]')
    ax1.set_title('3ω电压幅值')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    
    # 子图2: 相位对比
    ax2 = axes[0, 1]
    phase_noisy = np.unwrap(np.angle(V3_noisy)) * 180/np.pi
    phase_ideal = np.unwrap(np.angle(V3_ideal)) * 180/np.pi
    phase_fitted = np.unwrap(np.angle(V3_fitted)) * 180/np.pi
    
    ax2.semilogx(freqs, phase_noisy, 'ro', label='实验数据', markersize=6)
    ax2.semilogx(freqs, phase_ideal, 'b-', label='理想数据', linewidth=2)
    ax2.semilogx(freqs, phase_fitted, 'g--', label='拟合结果', linewidth=2)
    ax2.set_xlabel('频率 [Hz]')
    ax2.set_ylabel('相位 [度]')
    ax2.set_title('3ω电压相位')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    
    # 子图3: 不同拟合模式的参数对比
    ax3 = axes[1, 0]
    modes = list(fit_results.keys())
    kappa_values = [fit_results[mode]['kappa'] for mode in modes]  # 新接口直接访问
    kappa_errors = [abs(k - true_material['kappa'])/true_material['kappa']*100 for k in kappa_values]
    
    bars = ax3.bar(modes, kappa_errors, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax3.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5%误差线')
    ax3.set_ylabel('热导率拟合误差 [%]')
    ax3.set_title('不同拟合模式的精度对比')
    ax3.legend()
    
    # 在柱子上标注数值
    for bar, error in zip(bars, kappa_errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.1f}%', ha='center', va='bottom')
    
    # 子图4: 频率响应理论曲线
    ax4 = axes[1, 1]
    
    # 计算理论频率响应（无噪声）
    omega = 2 * np.pi * freqs
    # 重新计算gamma（新接口不直接返回）
    _, gamma, _ = calculate_fiber_V3omega(freqs, fiber_params, best_result)
    
    # 归一化幅值 (相对于低频值)
    normalized_amplitude = 1 / np.sqrt(1 + (2 * omega * gamma)**2)
    
    # 相位 (弧度转度)
    theoretical_phase = np.arctan(2 * omega * gamma) * 180/np.pi
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.semilogx(freqs, normalized_amplitude, 'b-', linewidth=2, label='归一化幅值')
    line2 = ax4_twin.semilogx(freqs, theoretical_phase, 'r-', linewidth=2, label='相位')
    
    ax4.set_xlabel('频率 [Hz]')
    ax4.set_ylabel('归一化幅值', color='b')
    ax4_twin.set_ylabel('相位 [度]', color='r')
    ax4.set_title(f'理论频率响应 (γ={gamma:.2e}s)')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right')
    
    ax4.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # --- 9. 保存数据 ---
    save_simulation_data(freqs, V3_noisy, current_config, fit_results)
    
    # --- 10. 输出总结 ---
    print(f"\n{'='*60}")
    print(f"模拟验证总结:")
    print(f"  测试配置: {current_config['name']}")
    print(f"  噪声水平: {noise_level*100:.1f}%")
    print(f"  最佳拟合模式: both")
    print(f"  最终热导率: {best_result['kappa']:.2f} W/m·K "
          f"(误差: {abs(best_result['kappa'] - true_material['kappa'])/true_material['kappa']*100:.1f}%)")
    print(f"  最终比热容: {best_result['C_p']:.0f} J/kg·K "
          f"(误差: {abs(best_result['C_p'] - true_material['C_p'])/true_material['C_p']*100:.1f}%)")
    # 重新计算导出参数
    _, gamma_fitted, alpha_fitted = calculate_fiber_V3omega(freqs, fiber_params, best_result)
    print(f"  热扩散率: {alpha_fitted:.2e} m^2/s")
    print(f"  热时间常数: {gamma_fitted:.2e} s")
    
    return fit_results

def save_simulation_data(freqs, V3_data, config, fit_results):
    """保存模拟数据到Excel文件"""
    # 准备数据
    data = {
        'Frequency (Hz)': freqs,
        'V3_Real (V)': V3_data.real,
        'V3_Imag (V)': V3_data.imag, 
        'V3_Magnitude (V)': np.abs(V3_data),
        'V3_Phase (deg)': np.angle(V3_data) * 180/np.pi
    }
    
    # 添加拟合结果（新接口需要重新计算V3）
    best_result = fit_results['both']
    fiber_configs = create_typical_fiber_params()
    fiber_params = fiber_configs[config['fiber']].copy()
    V3_fitted, _, _ = calculate_fiber_V3omega(freqs, fiber_params, best_result)
    
    data.update({
        'V3_Fitted_Real (V)': V3_fitted.real,
        'V3_Fitted_Imag (V)': V3_fitted.imag,
        'V3_Fitted_Magnitude (V)': np.abs(V3_fitted),
        'V3_Fitted_Phase (deg)': np.angle(V3_fitted) * 180/np.pi
    })
    
    df = pd.DataFrame(data)
    
    # 生成文件名
    filename = f"fiber_3omega_simulation_{config['name'].replace(' ', '_').lower()}.xlsx"
    
    # 创建Excel写入器
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 写入数据表
        df.to_excel(writer, sheet_name='Data', index=False)
        
        # 写入参数表（新接口直接访问属性）
        params_data = []
        for mode, result in fit_results.items():
            # 重新计算导出参数
            _, gamma_val, alpha_val = calculate_fiber_V3omega(freqs, fiber_params, result)
            params_data.append({
                'Fit_Mode': mode,
                'Kappa (W/m·K)': result['kappa'],
                'C_p (J/kg·K)': result['C_p'],
                'Alpha (m^2/s)': alpha_val,
                'Gamma (s)': gamma_val
            })
        
        df_params = pd.DataFrame(params_data)
        df_params.to_excel(writer, sheet_name='Fit_Results', index=False)
        
        # 写入配置信息
        config_data = pd.DataFrame([
            {'Parameter': 'Material', 'Value': config['material']},
            {'Parameter': 'Fiber_Config', 'Value': config['fiber']},
            {'Parameter': 'Noise_Level', 'Value': config['noise_level']},
            {'Parameter': 'Test_Name', 'Value': config['name']}
        ])
        config_data.to_excel(writer, sheet_name='Config', index=False)
    
    print(f"\n模拟数据已保存到: {filename}")

def run_parameter_sensitivity_analysis():
    """运行参数敏感性分析"""
    print("\n" + "="*60)
    print("参数敏感性分析")
    print("="*60)
    
    # 基础配置
    freqs = np.logspace(0, 2, 20)
    materials = create_fiber_material_library()
    fiber_configs = create_typical_fiber_params()
    
    base_material = materials['platinum']
    base_fiber = fiber_configs['platinum_wire_20um']
    
    # 参数变化范围
    kappa_range = np.linspace(20, 150, 10)  # 热导率范围
    Cp_range = np.linspace(80, 200, 10)     # 比热容范围
    
    results = {
        'kappa_sensitivity': [],
        'Cp_sensitivity': []
    }
    
    # 热导率敏感性
    print("分析热导率敏感性...")
    base_V3, _, _ = calculate_fiber_V3omega(freqs, base_fiber, base_material)
    
    for kappa in kappa_range:
        test_material = base_material.copy()
        test_material['kappa'] = kappa
        
        V3_test, _, _ = calculate_fiber_V3omega(freqs, base_fiber, test_material)
        
        # 计算相对变化
        rel_change = np.mean(np.abs(V3_test - base_V3) / np.abs(base_V3))
        results['kappa_sensitivity'].append(rel_change)
    
    # 比热容敏感性
    print("分析比热容敏感性...")
    for Cp in Cp_range:
        test_material = base_material.copy()
        test_material['C_p'] = Cp
        
        V3_test, _, _ = calculate_fiber_V3omega(freqs, base_fiber, test_material)
        
        # 计算相对变化
        rel_change = np.mean(np.abs(V3_test - base_V3) / np.abs(base_V3))
        results['Cp_sensitivity'].append(rel_change)
    
    # 可视化敏感性分析
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(kappa_range, np.array(results['kappa_sensitivity']) * 100, 'b-o')
    ax1.set_xlabel('热导率 κ [W/m·K]')
    ax1.set_ylabel('V₃ω相对变化 [%]')
    ax1.set_title('热导率敏感性')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(Cp_range, np.array(results['Cp_sensitivity']) * 100, 'r-o')
    ax2.set_xlabel('比热容 C_p [J/kg·K]')
    ax2.set_ylabel('V₃ω相对变化 [%]')
    ax2.set_title('比热容敏感性')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("敏感性分析完成!")

if __name__ == "__main__":
    # 运行主要模拟
    results = run_fiber_simulation()
    
    # 可选：运行敏感性分析
    run_sensitivity = input("\n是否运行参数敏感性分析? (y/n): ").lower().strip()
    if run_sensitivity == 'y':
        run_parameter_sensitivity_analysis()
    
    print("\n纤维丝3ω方法模拟完成!")
