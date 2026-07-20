import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from olson_model26_3 import calculate_analytical_Z, convert_Z_to_V3
import copy


def run_sensitivity_analysis():
    """
    Performs a sensitivity analysis on various model parameters to see their effect
    on the V3 omega signal (amplitude and phase).
    """
    # --- 1. Define Baseline Model (Layers and Sensor) ---
    freqs = np.logspace(np.log10(4), np.log10(4000), 50)

    # Baseline sensor parameters
    baseline_sensor_params_d1 = {
        'b': 300e-6, 'L': 7e-3, 'R': 6.74, 'dR_dT': 0.014391,
        'V_rms': 0.4, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    }
    baseline_sensor_params_d2 = {
        'b': 250e-6, 'L': 6e-3, 'R': 6.74, 'dR_dT': 0.014391,
        'V_rms': 0.4, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    }
    baseline_sensor_params_d3 = {
        'b': 200e-6, 'L': 5e-3, 'R': 6.74, 'dR_dT': 0.014391,
        'V_rms': 0.4, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    }
    baseline_sensor_params_d4 = {
        'b': 150e-6, 'L': 4e-3, 'R': 6.74, 'dR_dT': 0.014391,
        'V_rms': 0.4, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    }

    # 小传感器0.1MM
    baseline_sensor_params_x1 = {
        'b': 50e-6, 'L': 2e-3, 'R': 6.74, 'dR_dT': 0.014391,
        'V_rms': 0.4, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    }
    baseline_sensor_params_x2 = {
        'b': 20e-6, 'L': 2e-3, 'R': 6.74, 'dR_dT': 0.014391,
        'V_rms': 0.4, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    }
    baseline_sensor_params_x3 = {
        'b': 5e-6, 'L': 2e-3, 'R': 6.74, 'dR_dT': 0.014391,
        'V_rms': 0.4, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    }
    baseline_sensor_params_x4 = {
        'b': 4e-6, 'L': 2e-3, 'R': 6.74, 'dR_dT': 0.014391,
        'V_rms': 0.4, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    }

    # base_sensor_params = {
    #     'b': 50e-6, 'L': 2e-3, 'R': 9.032, 'dR_dT': 0.0153,
    #     'V_rms': 0.264, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    # }#金刚x1


    # base_sensor_params =   {
    #     'b': 200e-6, 'L': 5e-3, 'R': 6.684, 'dR_dT': 0.012,
    #     'V_rms': 0.265*0.89, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    # }#d3
    #
    # base_sensor_params =   {
    #     'b': 250e-6, 'L': 6e-3, 'R': 7.4, 'dR_dT': 0.0126,
    #     'V_rms': 0.263*0.89, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    # }#pi-d2

    # base_sensor_params = {
    #     'b': 300e-6, 'L': 9e-3, 'R':5.834, 'dR_dT': 0.0128,
    #     'V_rms': 0.256, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    # }#pi-d4

    # base_sensor_params = {
    #     'b': 200e-6, 'L': 7e-3, 'R': 0.538, 'dR_dT': 0.000794,
    #     'V_rms': 0.076, 'H_e': 1000e-9, 'rho_C_e': 2.5116e6
    # } #pi-d1

    # base_sensor_params =  {
    #     'b': 50e-6, 'L': 2e-3, 'R': 6.16, 'dR_dT': 0.0158,
    #     'V_rms': 0.4, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    # }

    # base_sensor_params =  {
    #     'b': 400e-6, 'L': 14e-3, 'R': 4.6, 'dR_dT': 0.012,
    #     'V_rms': 0.2, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    # }

    base_sensor_params =   {
        'b': 50e-6, 'L': 2e-3, 'R': 6.74, 'dR_dT': 0.014391,
        'V_rms': 0.4, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    }

    # 修改材料定义：使用 rho 和 cap 代替 alpha_y
    # alpha_y = k_y / (rho * cap)
    Al2O3 = {'k_y': 35.0, 'rho': 3900, 'cap': 880, 'H': 4e-6, 'psi': 1.0, 'Rc': 4E-7}
    sio2 = {'k_y': 1.4, 'rho': 2200, 'cap': 730, 'H': 1.47e-3, 'psi': 1.0, 'Rc': 2.1e-6}
    si = {'k_y': 120, 'rho': 2329, 'cap': 700, 'H': 100e-6, 'psi': 1.0, 'Rc': 1E-8}
    Ag_grease = {'k_y': 130, 'rho': 1500, 'cap': 1000, 'H': 40e-6, 'psi': 1.0, 'Rc': 1e-5}
    diamond = {'k_y': 2000, 'rho': 3510, 'cap': 500, 'H': 1000e-6, 'psi': 1.0, 'Rc': 1e-8}
    diamond_senor = {'k_y': 2000, 'rho': 3500, 'cap': 500, 'H': 1.0e-3, 'psi': 1.0, 'Rc': 1e-8}
    air = {'k_y': 0.02, 'rho': 1.2, 'cap': 1000, 'H': 10e-6, 'psi': 1.0, 'Rc': 1e-6}
    silicone_grease = {'k_y': 18, 'rho': 3000, 'cap': 1250, 'H': 50e-6, 'psi': 1.0, 'Rc': 4e-8}
    glass = {'k_y': 1.4, 'rho': 2200, 'cap': 800, 'H': 0.3e-6, 'psi': 1.0, 'Rc': 6e-4}
    zeolite = {'k_y': 1.07, 'rho': 2000, 'cap': 1250, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8}
    steel = {'k_y': 40, 'rho': 7850, 'cap': 500, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8}
    MgO = {'k_y': 50, 'rho': 3580, 'cap': 1000, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8}
    P_si = {'k_y': 120, 'rho': 2329, 'cap': 700, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8}
    pi = {'k_y': 0.3, 'rho': 1420, 'cap': 1090, 'H': 25e-6, 'psi': 1.0, 'Rc': 3.88e-6}
    sensor_pi = {'k_y': 0.72, 'rho': 1900, 'cap': 500, 'H': 8e-6, 'psi': 1.0, 'Rc': 1e-8}
    JueYuan = {'k_y': 40, 'rho': 3965, 'cap': 1000, 'H': 1e-6, 'psi': 1.0, 'Rc': 6.9574e-07}
    MXene = {'k_y': 0.5, 'rho': 1000, 'cap': 2300, 'H': 0.02e-6, 'psi': 1.0, 'Rc': 1e-6}
    Al= {'k_y': 237, 'rho': 2700, 'cap': 900, 'H': 7500e-6, 'psi': 1.0, 'Rc': 2.2E-6}
    Cu= {'k_y': 400, 'rho': 8960, 'cap': 385, 'H': 4e-3, 'psi': 1.0, 'Rc': 6.8e-7}
    yunmu= {'k_y': 0.55, 'rho': 2800, 'cap':880, 'H': 0.1e-3, 'psi': 1.0, 'Rc': 5e-6}
    CNO={'k_y': 4, 'rho': 469, 'cap': 1660, 'H': 4.2e-6, 'psi': 1.0, 'Rc': 1e-7}
    COF = {'k_y': 0.6, 'rho': 0.5, 'cap': 1E6, 'H': 20e-9, 'psi': 20.0, 'Rc': 1e-8}
    cu_pi = {'k_y': 1.4, 'rho': 1.85e6, 'cap': 1, 'H': 37.5e-6, 'psi': 1, 'Rc': 2.88e-5}
    siN = {'k_y': 1, 'rho': 300, 'cap': 3200, 'H': 1.5e-6, 'psi': 20.0, 'Rc': 1e-8}
    siC = {'k_y': 150, 'rho': 3100, 'cap': 710, 'H': 6e-7, 'psi': 1.0, 'Rc': 1e-8}

    DRAM2 = {'k_y': 120, 'rho': 2329, 'cap': 700, 'H': 30e-7, 'psi': 1.0, 'Rc': 1E-8}
    sample2 = {'k_y': 6, 'rho': 2270, 'cap': 1000, 'H': 8e-6, 'psi': 1.0, 'Rc': 1E-7}
    DRAM1 = {'k_y': 120, 'rho': 2329, 'cap': 700, 'H': 1e-6, 'psi': 1.0, 'Rc': 1E-7}
    sample1 = {'k_y': 6, 'rho': 2280, 'cap': 1000, 'H': 4e-6, 'psi': 1.0, 'Rc': 1E-7}
    logic = {'k_y': 120, 'rho': 2329, 'cap': 700, 'H': 100e-6, 'psi': 1.0, 'Rc': 1E-7}

    AlN = {'k_y': 4, 'rho': 2410, 'cap': 1000, 'H': 400e-9, 'psi': 1.0, 'Rc': 1E-7}
    juzhi = {'k_y': 0.2, 'rho': 1600, 'cap': 1000, 'H': 4.5e-6, 'psi': 1.0, 'Rc': 2.88e-6}

    # Baseline layer properties
    # base_layers = [ si,siN,cu_pi,pi]
    # base_layers = [ DRAM2,sample2,DRAM1,sample1,logic]
    # base_layers = [ logic,sample1,DRAM1,sample2,DRAM2]
    # base_layers = [si,AlN]
    # base_layers = [sio2,Al,diamond]
    #base_layers = [sio2, juzhi]
    #base_layers = [sio2, Al, diamond]
    base_layers = [Al2O3,CNO,diamond]

    # --- 2. Calculate Baseline V3 Signal ---------------------------------------------------------------------------------------
    Z_base = calculate_analytical_Z(freqs, base_sensor_params['b'], base_layers)
    V3_base = convert_Z_to_V3(Z_base, freqs, base_sensor_params['b'], base_sensor_params)
    V3_base_amp = np.abs(V3_base)
    V3_base_phase = np.unwrap(np.angle(V3_base))

    # --- 3. Define Parameters to Analyze and Perturbation Factor ---------------------------------------------------------------------------------------
    # Format: ('Description', 'type'={'layer' or 'sensor'}, 'location'=[layer_idx, 'prop_name'] or ['prop_name'])
    # 增加更多参数以展示同时拟合的必要性
    params_to_analyze = [
        # 材料参数
        #("1 Rc",  'layer', [1, 'Rc']),
        ("1 k_y", 'layer', [1, 'k_y']),  # 比热容的影响
        ("0 k_y", 'layer', [0, 'k_y']),  # 比热容的影响
        #("0 cap", 'layer', [1, 'cap']),  # 比热容的影响
        #("0 Rc",  'layer', [1, 'Rc']),  # 比热容的影响
        # ("4 k_y", 'layer', [4, 'k_y']),  # 比热容的影响
        # ("4 Rc", 'layer', [4, 'Rc']),  # 比热容的影响
        ("2 k_y", 'layer', [2, 'k_y']),  # 比热容的影响
        #("2 Rc", 'layer', [2, 'Rc']),  # 比热容的影响
        #("2 H", 'layer', [2, 'H']),  # 比热容的影响
        # ("3 k_y", 'layer', [3, 'k_y']),  # 比热容的影响
        # ("3 Rc", 'layer', [3, 'Rc']),  # 比热容的影响


        #("c cap", 'layer', [0, 'cap']),  # 比热容的影响

        # 传感器参数
        #("Sensor b", 'sensor', ['b']),
        #("Sensor dR/dT", 'sensor', ['dR_dT']),  # 关键参数：温度系数
        #("Sensor R", 'sensor', ['R']),  # 电阻的影响
        # ("Sensor V_rms", 'sensor', ['V_rms']),  # 驱动电压
        # ("Sensor L", 'sensor', ['L']),
        #("Sensor H_e", 'sensor', ['H_e']),
        # ("Sensor rho_C_e", 'sensor', ['rho_C_e']),
    ]
    perturbation = 0.01  # +1% change

    results_amp = {}
    results_phase = {}

    # --- 4. Loop Through Parameters, Perturb, and Calculate Sensitivity ------------------------------------------------------
    print("Running sensitivity analysis...")
    for name, p_type, location in params_to_analyze:
        print(f"  Analyzing: {name}")

        perturbed_layers = copy.deepcopy(base_layers)
        perturbed_sensor_params = base_sensor_params.copy()

        # Apply perturbation
        if p_type == 'layer':
            layer_idx, prop = location
            original_val = perturbed_layers[layer_idx][prop]
            perturbed_layers[layer_idx][prop] *= (1 + perturbation)
            p_val = original_val
        else:  # sensor
            prop = location[0]
            original_val = perturbed_sensor_params[prop]
            perturbed_sensor_params[prop] *= (1 + perturbation)
            p_val = original_val

        # 确定本次计算应该使用的 b 值
        # 如果是传感器参数被扰动，就用新的 b，否则用基准 b
        if p_type == 'sensor':
            b_to_use = perturbed_sensor_params['b']
        else:
            b_to_use = base_sensor_params['b']

        # 使用正确的 b 值重新计算 V3
        Z_perturbed = calculate_analytical_Z(freqs, b_to_use, perturbed_layers)
        V3_perturbed = convert_Z_to_V3(Z_perturbed, freqs, b_to_use, perturbed_sensor_params)

        # Calculate sensitivity S = (dY/Y) / (dP/P) ~= ((Y_pert - Y_base)/Y_base) / perturbation
        # This is the "normalized sensitivity" or "sensitivity index"
        sensitivity_amp = ((np.abs(V3_perturbed) - V3_base_amp) / V3_base_amp) / perturbation
        sensitivity_phase = ((np.unwrap(
            np.angle(V3_perturbed)) - V3_base_phase)) / perturbation  # Phase is already normalized in a way

        results_amp[name] = sensitivity_amp
        results_phase[name] = sensitivity_phase

    print("Analysis complete.")

    # --- 5. 参数相关性分析 ------------------------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("参数相关性分析")
    print("=" * 60)

    # 构建敏感度矩阵（每行是一个参数，每列是一个频率点）
    D_matrix = []
    param_names = []
    for name, sens in results_amp.items():
        D_matrix.append(sens)
        param_names.append(name)
    D_matrix = np.array(D_matrix)

    # 计算相关系数矩阵
    correlation_matrix = np.corrcoef(D_matrix)

    print("\n参数间相关系数矩阵:")
    print("(绝对值 > 0.7 表示强相关)")
    print("-" * 40)

    # 打印相关系数矩阵（只显示上三角部分）
    for i in range(len(param_names)):
        for j in range(i + 1, len(param_names)):
            corr = correlation_matrix[i, j]
            if abs(corr) > 0.7:  # 只显示强相关
                print(f"{param_names[i]:20s} vs {param_names[j]:20s}: {corr:+.3f}")

    # 特别分析 k_y 和 dR/dT 的相关性
    if "Substrate k_y" in param_names and "Sensor dR/dT" in param_names:
        idx_k = param_names.index("Substrate k_y")
        idx_dR = param_names.index("Sensor dR/dT")
        corr_k_dR = correlation_matrix[idx_k, idx_dR]
        print(f"\n特别关注:")
        print(f"k_y 与 dR/dT 的相关系数: {corr_k_dR:.3f}")
        if abs(corr_k_dR) > 0.7:
            print("→ 强相关！建议同时拟合这两个参数以避免补偿效应")

    print("\n" + "=" * 60)

    # --- 6. Plot the Results ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Sensitivity Analysis (Parameter change: +{perturbation * 100}%)', fontsize=16)

    # Plot Amplitude Sensitivities
    for name, sens in results_amp.items():
        ax1.semilogx(freqs, sens, label=name)
    ax1.set_ylabel('Amplitude Sensitivity [ (dV/V) / (dP/P) ]')
    ax1.legend(loc='best', fontsize='small')
    ax1.grid(True, which="both", ls="--")
    ax1.set_title('V3 Amplitude Sensitivity')

    # Plot Phase Sensitivities
    for name, sens in results_phase.items():
        ax2.semilogx(freqs, sens, label=name)
    ax2.set_ylabel('Phase Sensitivity [ d(Phase) / (dP/P) ]')
    ax2.legend(loc='best', fontsize='small')
    ax2.grid(True, which="both", ls="--")
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_title('V3 Phase Sensitivity')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    run_sensitivity_analysis()
