"""
生成示例实验数据文件 - 用于测试GUI的导入功能
"""
import numpy as np
import pandas as pd
from olson_model26 import calculate_analytical_Z, convert_Z_to_V3

def generate_sample_experimental_data():
    """生成模拟的实验数据(带噪声)"""
    
    # 传感器参数
    sensor_params = {
        'b': 200e-6,
        'L': 5e-3,
        'R': 6.74,
        'dR_dT': 0.014391,
        'V_rms': 0.4,
        'H_e': 100e-9,
        'rho_C_e': 2.5116e6
    }
    
    # 材料层 (Si/SiO2双层结构)
    layers = [
        {'k_y': 140.0, 'rho': 2329, 'cap': 700, 'H': np.inf, 'psi': 1.0, 'Rc': 5.1e-5},
        {'k_y': 1.12, 'rho': 2200, 'cap': 31, 'H': 1.47e-6, 'psi': 1.0, 'Rc': 1.e-4}
    ]
    
    # 频率范围
    freqs = np.logspace(np.log10(0.0001), np.log10(1000), 50)
    
    # 计算理论值
    Z_true = calculate_analytical_Z(freqs, sensor_params['b'], layers)
    V3_true = convert_Z_to_V3(Z_true, freqs, sensor_params['b'], sensor_params)
    
    # 添加噪声 (2% 相对噪声)
    noise_level = 0.02
    noise = noise_level * (np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))) * np.mean(np.abs(V3_true))
    V3_exp = V3_true + noise
    
    # 创建DataFrame
    df = pd.DataFrame({
        '频率 (Hz)': freqs,
        'V3_实部 (V)': V3_exp.real,
        'V3_虚部 (V)': V3_exp.imag,
        'V3_幅值 (V)': np.abs(V3_exp),
        'V3_相位 (度)': np.rad2deg(np.angle(V3_exp))
    })
    
    # 保存为Excel
    filename = '示例实验数据_SiO2.xlsx'
    df.to_excel(filename, index=False)
    print(f"✓ 已生成示例实验数据: {filename}")
    print(f"  频率范围: {freqs.min():.4f} - {freqs.max():.1f} Hz")
    print(f"  数据点数: {len(freqs)}")
    print(f"  幅值范围: {np.abs(V3_exp).min()*1e3:.3f} - {np.abs(V3_exp).max()*1e3:.3f} mV")
    print(f"  噪声水平: {noise_level*100}%")
    
    return filename


if __name__ == '__main__':
    print("=" * 50)
    print("生成示例实验数据")
    print("=" * 50)
    print()
    
    filename = generate_sample_experimental_data()
    
    print()
    print("=" * 50)
    print("使用方法:")
    print("=" * 50)
    print("1. 启动GUI: python gui_3omega.py")
    print("2. 点击 [📥 导入实验数据] 按钮")
    print(f"3. 选择文件: {filename}")
    print("4. 观察理论曲线与实验数据的对比")
    print()




