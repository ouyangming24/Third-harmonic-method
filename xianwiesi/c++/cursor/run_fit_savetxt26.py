import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', etc.
import matplotlib.pyplot as plt
from olson_model26 import ThermalModelFitter, calculate_analytical_Z, convert_Z_to_V3
import pandas as pd


def run_example():
    """
    Runs a fitting example based on the Si/SiO2 structure
    from Table II in the Olson et al. paper, fitting directly to V3_rms data.
    """
    # --- 1. Define Material Library First (Move to top for accessibility) ---
    # 修改材料定义：使用 rho 和 cap 代替 alpha_y
    materials_lib = {
        'Al2O3': {'k_y': 40.0, 'rho': 3965, 'cap': 1000, 'H': np.inf, 'psi': 1.0, 'Rc': 5E-5},
        'sio2': {'k_y': 1.12, 'rho': 2200, 'cap': 31, 'H': 1.47e-3, 'psi': 1.0, 'Rc': 1.e-4},
        'si': {'k_y': 140.0, 'rho': 232, 'cap': 700, 'H': 100e-6, 'psi': 1.0, 'Rc':5.1e-5},
        'Ag_grease': {'k_y': 30, 'rho': 1500, 'cap': 1000, 'H': 40e-6, 'psi': 1.0, 'Rc': 1e-8},
        'diamond': {'k_y': 1000, 'rho': 3500, 'cap': 500, 'H': 50e-6, 'psi': 1.0, 'Rc': 6e-9},
        'diamond_sensor': {'k_y': 000, 'rho': 3500, 'cap': 500, 'H': 1.0e-3, 'psi': 1.0, 'Rc': 1e-8},
        'air': {'k_y': 0.01, 'rho': 1.2, 'cap': 1000, 'H': 0.2e-6, 'psi': 1.0, 'Rc': 1e-8},
        'silicone_grease': {'k_y': 18, 'rho': 3000, 'cap': 1250, 'H': 63e-6, 'psi': 1.0, 'Rc': 5e-5},
        'glass': {'k_y': 1.4, 'rho': 2200, 'cap': 800, 'H': 10e-6, 'psi': 1.0, 'Rc': 1e-4},
        'zeolite': {'k_y': 1.07, 'rho': 2000, 'cap': 1250, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8},
        'steel': {'k_y': 40, 'rho': 7850, 'cap': 500, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8},
        'MgO': {'k_y': 50, 'rho': 3580, 'cap': 1000, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8},
        'P_si': {'k_y': 120, 'rho': 2329, 'cap': 700, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8},
        'pi': {'k_y':0.155, 'rho': 1894, 'cap': 1100, 'H': 25e-6, 'psi': 1.0, 'Rc': 6.88e-5},
        'sensor': {'k_y': 0.1, 'rho': 1900, 'cap': 500, 'H': 25e-6, 'psi': 1.0, 'Rc': 1e-8},
        'Al': {'k_y': 237, 'rho': 2700, 'cap': 800, 'H': 4e-3, 'psi': 1.0, 'Rc': 3.85E-5},
        'Cu': {'k_y': 400, 'rho': 8960, 'cap': 385, 'H': 4e-3, 'psi': 1.0, 'Rc': 3.8e-4},
        'yunmu': {'k_y':43.5, 'rho': 2800, 'cap': 135.4, 'H': 110e-6, 'psi': 1.0, 'Rc': 6.5e-6},
        'CNO': {'k_y':1.0465, 'rho': 460, 'cap': 5654, 'H': 4.2e-6, 'psi': 1.0, 'Rc': 1.4405e-6}
    }

    # For backward compatibility, create individual variables
    Al2O3 = materials_lib['Al2O3']
    sio2 = materials_lib['sio2']
    si = materials_lib['si']
    Ag_grease = materials_lib['Ag_grease']
    diamond = materials_lib['diamond']
    diamond_sensor = materials_lib['diamond_sensor']
    air = materials_lib['air']
    silicone_grease = materials_lib['silicone_grease']
    glass = materials_lib['glass']
    zeolite = materials_lib['zeolite']
    steel = materials_lib['steel']
    MgO = materials_lib['MgO']
    P_si = materials_lib['P_si']
    pi = materials_lib['pi']
    sensor = materials_lib['sensor']
    Al = materials_lib['Al']
    Cu = materials_lib['Cu']
    yunmu = materials_lib['yunmu']
    CNO = materials_lib['CNO']

    # --- 2. Define Experimental Parameters & True Layer Properties ---

    # 大传感器
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

    # 小传感器
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
        'V_rms': 0.4, 'H_e': 130e-9, 'rho_C_e': 2.5116e6
    }

    # sensor_params =  {
    #     'b': 50e-6, 'L': 2e-3, 'R': 9.032, 'dR_dT': 0.0153,
    #     'V_rms': 0.704, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    # }#金刚x1

    # sensor_params =   {
    #     'b': 200e-6, 'L': 5e-3, 'R': 6.532, 'dR_dT': 0.012,
    #     'V_rms': 0.264*0.89, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    # }#d3

    # sensor_params =   {
    #     'b': 250e-6, 'L': 6e-3, 'R': 7.5, 'dR_dT': 0.0142,
    #     'V_rms': 0.264, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    # }#pi-d2

    # sensor_params = {
    #     'b': 200e-6, 'L': 5e-3, 'R': 18.4, 'dR_dT': 0.025,
    #     'V_rms': 0.264, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    # }#hw-金刚石-镀

    # sensor_params = {
    #     'b': 150e-6, 'L': 4e-3, 'R': 6.68, 'dR_dT': 0.013,
    #     'V_rms': 0.264, 'H_e': 22e-9, 'rho_C_e': 2.5116e6
    # }

    # sensor_params = {
    #     'b': 150e-6, 'L': 4e-3, 'R': 7.15, 'dR_dT': 1.0e-02,
    #     'V_rms': 0.264, 'H_e': 100e-9, 'rho_C_e': 2.5116e6
    # }

    sensor_params =  {
        'b': 200e-6, 'L': 5e-3, 'R': 2.376, 'dR_dT': 0.004,
        'V_rms': 0.264, 'H_e': 1e-6, 'rho_C_e': 2.5116e6
    }

    # Layer properties from Table II (Reference values)
    # Layer 0: Silicon Substrate (semi-infinite)
    si_props = {
        'k_y': 140.0,  # W/mK
        'rho': 2329,  # kg/m³
        'cap': 700,  # J/kg·K
        # alpha_y = 140.0 / (2329 * 700) = 8.8e-5 m²/s
        'H': np.inf,
        'psi': 1.0,
        'Rc': 1.0e-8
    }
    si2 = {'k_y': 24.0, 'rho': 2329, 'cap': 1000, 'H': 80e-6, 'psi': 1.0, 'Rc': 1e-7}
    true_layers = [glass,yunmu, CNO]
    #true_layers = [pi]

    # Frequency range similar to the paper's experiment
    freqs = np.logspace(np.log10(0.0001), np.log10(1000), 100)

    # --- 2. Generate Synthetic Experimental Data ---
    print("Generating synthetic experimental V3_rms data based on reference values...")
    Z_true = calculate_analytical_Z(freqs, sensor_params['b'], true_layers)
    V3_true = convert_Z_to_V3(Z_true, freqs, sensor_params['b'], sensor_params)

    # Add some realistic noise to the V3 data
    noise_level = 0.0 # 1% noise relative to the mean V3 magnitude
    noise = noise_level * (np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))) * np.mean(np.abs(V3_true))
    V3_exp_noisy = V3_true + noise

    # --- 3. Set Up the Fitter ---
    # Choose your fitting mode here: 'both', 'real', 'imag', 'amp', 'phase'
    fit_mode_selection = 'amp'

    # Initial guesses for the properties we want to fit.
    si_props_guess = {
        'k_y': 160.0,  # Guess for Si conductivity (20% off)
        'rho': 2329,  # Si density (assumed known)
        'cap': 700,  # Si specific heat (assumed known)
        'H': np.inf,
        'psi': 1.0,
        'Rc': 1.0e-8
    }

    # --- 新增：选择拟合模式 ---
    # fit_type可以是: 'material' (仅材料), 'sensor' (仅传感器), 'combined' (同时拟合)
    fit_type = 'material'  # 改为combined进行测试

    if fit_type == 'combined':
        # ========== 同时拟合材料和传感器参数 ==========
        print("\n========== 同时拟合模式 ==========")
        initial_layers_guess = [Al2O3,si2]
        params_to_fit = {
            'layer_0': ['k_y', 'Rc'],  # 材料参数
            'sensor': ['dR_dT']  # 传感器参数
        }

        # 参数边界约束
        param_bounds = {
            'layer_0': {
                'k_y': (0.005, 150),  # SiO2 热导率范围
                'Rc': (1e-9, 1e-4)  # 界面热阻范围
            },
            'sensor': {
                'dR_dT': (0.001, 0.1)  # 温度系数范围
            }
        }

        print("\n参数边界约束:")
        for layer_key, bounds in param_bounds.items():
            print(f"  {layer_key}:")
            for prop, (lower, upper) in bounds.items():
                if prop == 'k_y':
                    print(f"    {prop}: [{lower:.2f}, {upper:.2f}] W/mK")
                elif prop == 'dR_dT':
                    print(f"    {prop}: [{lower:.3f}, {upper:.3f}] Ω/K")
                else:
                    print(f"    {prop}: [{lower:.2e}, {upper:.2e}]")
    elif fit_type == 'sensor':
        # ========== 传感器参数拟合模式 ==========
        print("\n========== 传感器参数拟合模式 ==========")
        # 假设材料属性是已知的（使用标准材料）
        # 这里以SiO2为例，实际应用中应该根据实际标准材料设置
        known_layers = [Al2O3]

        print("使用已知材料属性:")
        for i, layer in enumerate(known_layers):
            k_y = layer['k_y']
            rho = layer.get('rho', 2000)
            cap = layer.get('cap', 1000)
            alpha_y = k_y / (rho * cap)
            print(f"  层 {i}: k_y={k_y:.2f} W/mK, alpha_y={alpha_y:.2e} m²/s")

        # 定义要拟合的传感器参数
        params_to_fit = {
            'sensor': ['dR_dT']  # 拟合这些传感器参数
        }

        # 传感器参数边界约束
        param_bounds = {
            'sensor': {
                # 'R': (4.0, 10.0),  # 电阻范围
                'dR_dT': (0.001, 0.06),  # 温度系数范围
                # 'V_rms': (0.3, 0.6)  # 电压范围
            }
        }

        print("\n传感器参数初始猜测:")
        for param in params_to_fit['sensor']:
            print(f"  {param}: {sensor_params[param]:.4e}")

        print("\n传感器参数边界约束:")
        for param, (lower, upper) in param_bounds['sensor'].items():
            print(f"  {param}: [{lower:.3e}, {upper:.3e}]")

        initial_layers_guess = known_layers
    else:
        # fit_type == 'matrial'
        # ========== 材料参数拟合模式 ==========
        print("\n========== 材料参数拟合模式 ==========")
        initial_layers_guess = [glass,yunmu, CNO]
        #initial_layers_guess = [pi]
        params_to_fit = {
            'layer_0': ['Rc'],  # Fit k_y and Rc of substrate
           # 'layer_1': ['k_y','Rc'] # 可以拟合 k_y, rho 和 cap
        }

        param_bounds = {
            'layer_1': {
                'Rc': (1E-10, 1E-3),  # 热导率限制在 0.1-10 W/mK
                'k_y': (0.1, 322),  # 界面热阻限制在 1e-8 到 1e-4 K·m²/W
                #'rho': (1000, 3000),  # 密度范围示例
                # 'cap': (500, 1500)  # 比热容范围示例
            }
            ,
            'layer_0': {
                'k_y': (0.1, 5),  # 热导率限制在 0.1-10 W/mK
                'Rc': (1e-11,1e-3),  # 界面热阻限制在 1e-8 到 1e-4 K·m²/W
                # 'rho': (1000, 3000),  # 密度范围示例
                # 'cap': (500, 1500)  # 比热容范围示例
            }


        }

        print("\n参数边界约束设置:")
        for layer_key, bounds in param_bounds.items():
            print(f"  {layer_key}:")
            for prop, (lower, upper) in bounds.items():
                print(f"    {prop}: [{lower:.2e}, {upper:.2e}]")

    fitter = ThermalModelFitter(
        freqs=freqs,
        V3_exp=V3_exp_noisy,
        b=sensor_params['b'],
        sensor_params=sensor_params,
        initial_layers=initial_layers_guess,
        params_to_fit=params_to_fit,
        fit_mode=fit_mode_selection,
        param_bounds=param_bounds,  # 传入边界约束
        fit_type=fit_type  # 新增：传入拟合类型
    )

    # --- 4. Run the Fit ---
    fitted_result = fitter.fit(max_iter=30, tol=1e-6, zeta=0.5)

    # 处理拟合结果
    if fit_type == 'combined':
        # 同时拟合返回字典
        fitted_layers = fitted_result['layers']
        fitted_sensor_params = fitted_result['sensor_params']
        sensor_params_for_plot = fitted_sensor_params
    elif fit_type == 'sensor':
        fitted_sensor_params = fitted_result
        fitted_layers = initial_layers_guess
        # 使用拟合后的传感器参数
        sensor_params_for_plot = fitted_sensor_params
    else:
        fitted_layers = fitted_result
        sensor_params_for_plot = sensor_params

    # --- 5. Visualize the Results ---
    # Calculate V3 for initial guess and final fitted model
    Z_initial_guess = calculate_analytical_Z(freqs, sensor_params['b'], initial_layers_guess)
    V3_initial_guess = convert_Z_to_V3(Z_initial_guess, freqs, sensor_params['b'], sensor_params)

    Z_fitted = calculate_analytical_Z(freqs, sensor_params_for_plot['b'], fitted_layers)
    V3_fitted = convert_Z_to_V3(Z_fitted, freqs, sensor_params_for_plot['b'], sensor_params_for_plot)

    # --- 5. Visualize the Results ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'3ω Fitting Results (Mode: {fit_mode_selection}) for Si/SiO2 Structure', fontsize=16)

    if fit_mode_selection in ['amp', 'phase']:
        # Plot Amplitude of V3 in ax1
        ax1.semilogx(freqs, np.abs(V3_exp_noisy) * 1e3, 'ro', label='Experimental Amp (Noisy)')
        ax1.semilogx(freqs, np.abs(V3_initial_guess) * 1e3, 'b--', label='Initial Guess')
        ax1.semilogx(freqs, np.abs(V3_fitted) * 1e3, 'g-', label='Fitted Model')
        ax1.set_ylabel('Amplitude(V3) [mV]')

        # Plot Phase of V3 in ax2
        # ax2.semilogx(freqs, np.unwrap(np.angle(V3_exp_noisy, deg=True)), 'ro', label='Experimental Phase (Noisy)')
        # ax2.semilogx(freqs, np.unwrap(np.angle(V3_initial_guess, deg=True)), 'b--', label='Initial Guess')
        # ax2.semilogx(freqs, np.unwrap(np.angle(V3_fitted, deg=True)), 'g-', label='Fitted Model')
        ax2.semilogx(freqs, np.unwrap(np.angle(V3_exp_noisy)) * 180/np.pi, 'ro', label='Experimental Phase (Noisy)')
        ax2.semilogx(freqs, np.unwrap(np.angle(V3_initial_guess)) * 180/np.pi, 'b--', label='Initial Guess')
        ax2.semilogx(freqs, np.unwrap(np.angle(V3_fitted)) * 180/np.pi, 'g-', label='Fitted Model')
        ax2.set_ylabel('Phase(V3) [degrees]')

    else:  # 'both', 'real', 'imag'
        # Plot Real Part of V3
        ax1.semilogx(freqs, V3_exp_noisy.real * 1e3, 'ro', label='Experimental Real (Noisy)')
        ax1.semilogx(freqs, V3_initial_guess.real * 1e3, 'b--', label='Initial Guess')
        ax1.semilogx(freqs, V3_fitted.real * 1e3, 'g-', label='Fitted Model')
        ax1.set_ylabel('Real(V3) [mV]')

        # Plot Imaginary Part of V3
        ax2.semilogx(freqs, V3_exp_noisy.imag * 1e3, 'ro', label='Experimental Imag (Noisy)')
        ax2.semilogx(freqs, V3_initial_guess.imag * 1e3, 'b--', label='Initial Guess')
        ax2.semilogx(freqs, V3_fitted.imag * 1e3, 'g-', label='Fitted Model')
        ax2.set_ylabel('Imag(V3) [mV]')

    ax1.legend()
    ax1.grid(True, which="both", ls="--")
    ax2.legend()
    ax2.grid(True, which="both", ls="--")
    ax2.set_xlabel('Frequency [Hz]')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Optional: Save the synthetic data ---
    save_v3_data(freqs, V3_exp_noisy)


def save_v3_data(freqs, V3_exp_noisy):
    """Saves the synthetic V3 omega data to an Excel file."""
    df_v3omega = pd.DataFrame({
        'Frequency (Hz)': freqs,
        'V3_Real (V)': V3_exp_noisy.real,
        'V3_Imag (V)': V3_exp_noisy.imag,
        'V3_Magnitude (V)': np.abs(V3_exp_noisy),
        'V3_Phase (deg)': np.rad2deg(np.angle(V3_exp_noisy))
    })
    df_v3omega.to_excel('测试pi.xlsx', index=False)
    print("\n模拟的V3ω数据已保存到 测试.xlsx'")


if __name__ == '__main__':
    run_example()
