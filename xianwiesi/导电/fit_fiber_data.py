import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
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

# ==================== 预配置参数 ====================
# 
# 【配置说明】
# 当前配置已与 run_fiber_simulation.py 完全对齐，确保拟合结果一致性：
# 1. 参数边界：kappa(0.1,5000), C_p(50,2000) - 与run_fiber_simulation.py相同
# 2. 优化参数：max_iter=30, zeta=0.5 - 与run_fiber_simulation.py相同
# 3. 频率筛选：关闭(use_range=False) - 使用全部数据点
# 4. 初值偏离：启用(enabled=True) - κ×1.2, Cp×0.8，模拟run_fiber_simulation.py的行为
# 5. 随机种子：run_fiber_simulation.py已固定为42，确保噪声可重复
#
# 使用方法：
# - 先运行: python run_fiber_simulation.py (生成数据)
# - 再运行: python fit_fiber_data.py (拟合数据)
# - 结果应完全一致（差异<0.1%）
#
# ====================================================

# 数据文件配置
DATA_FILE_CONFIG = {
    'file_path': 'fiber_3omega_simulation_碳纳米管束测试.xlsx',  # 修改为您的数据文件路径
    'backup_files': [  # 备选文件列表
        'fiber_3omega_simulation_铂丝测试.xlsx',
        'test_fiber_data.xlsx',
        '测试.xlsx'
    ]
}

# 纤维参数预配置
FIBER_CONFIG = {
    'config_name': 'carbon_nanotube_bundle',  # 可选: 'platinum_wire_20um', 'carbon_nanotube_bundle', 'thin_wire_1um', 'custom'
    'custom_params': {  # 当config_name为'custom'时使用
        'L': 8e-3,                    # 长度 [m]
        'diameter': 20e-6,           # 直径 [m]
        'S': np.pi * (20e-6)**2,      # 横截面积 [m²]
        'R': 5,                    # 电阻 [Ω]
        'R_prime': 1.0e-02,          # 温度系数 [Ω/K]
        'I_rms': 10e-3,               # RMS电流 [A]
        'description': '自定义纤维配置'
    }
}

# 材料初值配置
MATERIAL_CONFIG = {
    'material_name': 'carbon_nanotube',  # 可选: 'platinum', 'carbon_nanotube', 'copper', 'tungsten', 'nichrome', 'custom'
    'custom_material': {  # 当material_name为'custom'时使用
        'kappa': 3000,             # 热导率 [W/m·K] - 碳纳米管示例值
        'rho': 1300,               # 密度 [kg/m³]
        'C_p': 700,                # 比热容 [J/kg·K]
        'description': '自定义材料'
    },
    # 初值偏离设置（用于测试收敛性，模拟run_fiber_simulation.py的行为）
    'perturb_initial': {
        'enabled': True,           # 是否启用初值偏离 - 启用以匹配run_fiber_simulation.py
        'kappa_factor': 1.2,       # 热导率乘数（1.2表示偏离+20%）
        'C_p_factor': 0.8,         # 比热容乘数（0.8表示偏离-20%）
        'rho_factor': 1.0          # 密度乘数
    }
}

# 拟合参数配置
FITTING_CONFIG = {
    'params_to_fit': ['kappa', 'C_p'],  # 可选组合: ['kappa'], ['kappa', 'C_p'], ['kappa', 'rho'], ['kappa', 'C_p', 'rho']
    'fit_mode': 'amplitude',            # 可选: 'amplitude', 'phase', 'both'
    'param_bounds': {
        'kappa': (0.1, 5000),      # 热导率边界 [W/m·K] - 与run_fiber_simulation.py一致
        'C_p': (50, 2000),          # 比热容边界 [J/kg·K] - 与run_fiber_simulation.py一致
        'rho': (100, 25000),        # 密度边界 [kg/m³]
    },
    'optimization': {
        'max_iter': 30,        # 最大迭代次数 - 与run_fiber_simulation.py一致
        'tolerance': 1e-5,     # 收敛容差
        'zeta': 0.5            # 步长因子 - 与run_fiber_simulation.py一致
    },
    'frequency_range': {
        'use_range': False,     # 是否使用频率范围筛选 - 关闭以匹配run_fiber_simulation.py
        'min_freq': 1,        # 最小频率 [Hz]
        'max_freq': 1000.0,      # 最大频率 [Hz]
        'use_indices': False,   # 使用索引选择而非频率值
        'start_index': 0,       # 起始索引（当use_indices=True时）
        'end_index': -1,        # 结束索引（-1表示最后，当use_indices=True时）
        'skip_points': 1        # 跳过点数（1=使用全部，2=每隔一个点，等）
    }
}

# 批量处理配置
BATCH_CONFIG = {
    'enabled': False,  # 是否启用批量处理
    'data_directory': '.',  # 数据文件目录
    'file_pattern': '*.xlsx',  # 文件匹配模式
    'selected_files': 'all',  # 'all' 或文件名列表
}

# 输出配置
OUTPUT_CONFIG = {
    'save_results': True,
    'show_plots': True,
    'save_plots': False,
    'plot_format': 'png',
    'output_directory': './results'
}

def load_fiber_data(file_path):
    """
    从Excel文件加载纤维丝3ω实验数据
    
    Args:
        file_path (str): Excel文件路径
        
    Returns:
        tuple: (freqs, V3_exp, success)
    """
    try:
        print(f"正在加载数据文件: {file_path}")
        df = pd.read_excel(file_path)
        
        # 处理列名变化 (转换为小写，便于匹配)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # 寻找频率列
        freq_cols = [c for c in df.columns if 'freq' in c]
        if not freq_cols:
            raise ValueError("未找到频率列 (应包含'freq')")
        freq_col = freq_cols[0]
        
        # 寻找V3ω实部和虚部列
        real_cols = [c for c in df.columns if 'real' in c and 'v3' in c]
        imag_cols = [c for c in df.columns if 'imag' in c and 'v3' in c]
        
        if not real_cols:
            raise ValueError("未找到V3ω实部列 (应包含'real'和'v3')")
        if not imag_cols:
            raise ValueError("未找到V3ω虚部列 (应包含'imag'和'v3')")
            
        real_col = real_cols[0]
        imag_col = imag_cols[0]
        
        # 提取数据
        freqs = df[freq_col].values
        V3_real = df[real_col].values
        V3_imag = df[imag_col].values
        
        # 构建复数数据
        V3_exp = V3_real + 1j * V3_imag
        
        # 数据验证
        if len(freqs) < 5:
            raise ValueError(f"数据点太少: {len(freqs)} < 5")
        
        if np.any(freqs <= 0):
            raise ValueError("频率必须为正值")
            
        if np.all(np.abs(V3_exp) == 0):
            raise ValueError("V3ω信号全为零")
        
        print(f"数据加载成功:")
        print(f"  数据点数: {len(freqs)}")
        print(f"  频率范围: {np.min(freqs):.2f} - {np.max(freqs):.2f} Hz")
        print(f"  V3ω幅值范围: {np.min(np.abs(V3_exp)):.2e} - {np.max(np.abs(V3_exp)):.2e} V")
        
        return freqs, V3_exp, True
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("\n请确保Excel文件包含以下列:")
        print("  - 频率列 (列名包含'freq')")
        print("  - V3ω实部列 (列名包含'real'和'v3')")
        print("  - V3ω虚部列 (列名包含'imag'和'v3')")
        return None, None, False

def interactive_fiber_setup():
    """
    交互式设置纤维参数
    
    Returns:
        dict: 纤维参数配置
    """
    print("\n" + "="*50)
    print("纤维参数设置")
    print("="*50)
    
    # 显示预设配置
    configs = create_typical_fiber_params()
    print("\n可用的预设配置:")
    for i, (name, config) in enumerate(configs.items()):
        print(f"  {i+1}. {name}: {config['description']}")
    
    print(f"  {len(configs)+1}. 自定义配置")
    
    while True:
        try:
            choice = input(f"\n请选择配置 (1-{len(configs)+1}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(configs):
                # 使用预设配置
                config_name = list(configs.keys())[choice_idx]
                fiber_params = configs[config_name].copy()
                print(f"\n已选择: {config_name}")
                break
            elif choice_idx == len(configs):
                # 自定义配置
                fiber_params = create_custom_fiber_params()
                break
            else:
                print("无效选择，请重新输入")
        except ValueError:
            print("请输入数字")
    
    # 显示最终配置
    print(f"\n最终纤维参数:")
    print(f"  长度 L: {fiber_params['L']*1e3:.2f} mm")
    print(f"  横截面积 S: {fiber_params['S']*1e12:.2f} μm²")
    if 'diameter' in fiber_params:
        print(f"  直径: {fiber_params['diameter']*1e6:.2f} μm")
    print(f"  电阻 R: {fiber_params['R']:.2f} Ω")
    print(f"  温度系数 R': {fiber_params['R_prime']:.4f} Ω/K")
    print(f"  RMS电流: {fiber_params['I_rms']*1e3:.2f} mA")
    
    return fiber_params

def create_custom_fiber_params():
    """创建自定义纤维参数"""
    print("\n自定义纤维参数:")
    
    params = {}
    
    # 长度
    while True:
        try:
            L_mm = float(input("  长度 L [mm]: "))
            if L_mm > 0:
                params['L'] = L_mm * 1e-3
                break
            else:
                print("长度必须为正值")
        except ValueError:
            print("请输入数字")
    
    # 直径或横截面积
    area_input = input("  输入方式 - 直径[d] 或 横截面积[a]: ").lower().strip()
    
    if area_input == 'd':
        while True:
            try:
                diameter_um = float(input("  直径 [μm]: "))
                if diameter_um > 0:
                    params['diameter'] = diameter_um * 1e-6
                    params['S'] = np.pi * (diameter_um * 1e-6 / 2)**2
                    break
                else:
                    print("直径必须为正值")
            except ValueError:
                print("请输入数字")
    else:
        while True:
            try:
                area_um2 = float(input("  横截面积 [μm²]: "))
                if area_um2 > 0:
                    params['S'] = area_um2 * 1e-12
                    break
                else:
                    print("横截面积必须为正值")
            except ValueError:
                print("请输入数字")
    
    # 电阻
    while True:
        try:
            R = float(input("  电阻 R [Ω]: "))
            if R > 0:
                params['R'] = R
                break
            else:
                print("电阻必须为正值")
        except ValueError:
            print("请输入数字")
    
    # 温度系数
    while True:
        try:
            R_prime = float(input("  温度系数 R' [Ω/K]: "))
            params['R_prime'] = R_prime
            break
        except ValueError:
            print("请输入数字")
    
    # RMS电流
    while True:
        try:
            I_mA = float(input("  RMS电流 [mA]: "))
            if I_mA > 0:
                params['I_rms'] = I_mA * 1e-3
                break
            else:
                print("电流必须为正值")
        except ValueError:
            print("请输入数字")
    
    params['description'] = "用户自定义配置"
    
    return params

def interactive_material_setup():
    """
    交互式设置材料初值
    
    Returns:
        dict: 材料属性初值
    """
    print("\n" + "="*50)
    print("材料属性初值设置")
    print("="*50)
    
    materials = create_fiber_material_library()
    print("\n可用的材料预设:")
    for i, (name, props) in enumerate(materials.items()):
        print(f"  {i+1}. {name}: {props['description']}")
        print(f"      κ={props['kappa']} W/m·K, ρ={props['rho']} kg/m³, C_p={props['C_p']} J/kg·K")
    
    print(f"  {len(materials)+1}. 自定义材料")
    
    while True:
        try:
            choice = input(f"\n请选择材料 (1-{len(materials)+1}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(materials):
                # 使用预设材料
                material_name = list(materials.keys())[choice_idx]
                material_props = materials[material_name].copy()
                print(f"\n已选择: {material_name}")
                break
            elif choice_idx == len(materials):
                # 自定义材料
                material_props = create_custom_material()
                break
            else:
                print("无效选择，请重新输入")
        except ValueError:
            print("请输入数字")
    
    return material_props

def create_custom_material():
    """创建自定义材料属性"""
    print("\n自定义材料属性:")
    
    props = {}
    
    # 热导率
    while True:
        try:
            kappa = float(input("  热导率 κ [W/m·K]: "))
            if kappa > 0:
                props['kappa'] = kappa
                break
            else:
                print("热导率必须为正值")
        except ValueError:
            print("请输入数字")
    
    # 密度
    while True:
        try:
            rho = float(input("  密度 ρ [kg/m³]: "))
            if rho > 0:
                props['rho'] = rho
                break
            else:
                print("密度必须为正值")
        except ValueError:
            print("请输入数字")
    
    # 比热容
    while True:
        try:
            C_p = float(input("  比热容 C_p [J/kg·K]: "))
            if C_p > 0:
                props['C_p'] = C_p
                break
            else:
                print("比热容必须为正值")
        except ValueError:
            print("请输入数字")
    
    props['description'] = "用户自定义材料"
    
    return props

def get_fiber_params():
    """根据配置获取纤维参数"""
    config_name = FIBER_CONFIG['config_name']
    
    if config_name == 'custom':
        fiber_params = FIBER_CONFIG['custom_params'].copy()
    else:
        fiber_configs = create_typical_fiber_params()
        if config_name in fiber_configs:
            fiber_params = fiber_configs[config_name].copy()
        else:
            print(f"警告: 未找到配置 '{config_name}', 使用默认配置")
            fiber_params = fiber_configs['platinum_wire_20um'].copy()
    
    return fiber_params

def get_material_props():
    """根据配置获取材料属性"""
    material_name = MATERIAL_CONFIG['material_name']
    
    if material_name == 'custom':
        material_props = MATERIAL_CONFIG['custom_material'].copy()
    else:
        materials = create_fiber_material_library()
        if material_name in materials:
            material_props = materials[material_name].copy()
        else:
            print(f"警告: 未找到材料 '{material_name}', 使用默认材料")
            material_props = materials['platinum'].copy()
    
    # 应用初值偏离（如果启用）
    if MATERIAL_CONFIG.get('perturb_initial', {}).get('enabled', False):
        perturb = MATERIAL_CONFIG['perturb_initial']
        if 'kappa' in material_props:
            material_props['kappa'] *= perturb.get('kappa_factor', 1.0)
        if 'C_p' in material_props:
            material_props['C_p'] *= perturb.get('C_p_factor', 1.0)
        if 'rho' in material_props:
            material_props['rho'] *= perturb.get('rho_factor', 1.0)
        print(f"  已应用初值偏离: κ×{perturb.get('kappa_factor', 1.0)}, "
              f"Cp×{perturb.get('C_p_factor', 1.0)}, ρ×{perturb.get('rho_factor', 1.0)}")
    
    return material_props

def find_data_file():
    """查找数据文件"""
    # 首先尝试主文件
    main_file = DATA_FILE_CONFIG['file_path']
    if os.path.exists(main_file):
        return main_file
    
    print(f"主数据文件 '{main_file}' 未找到，尝试备选文件...")
    
    # 尝试备选文件
    for backup_file in DATA_FILE_CONFIG['backup_files']:
        if os.path.exists(backup_file):
            print(f"使用备选文件: {backup_file}")
            return backup_file
    
    print("错误: 未找到任何数据文件!")
    print("请检查以下文件是否存在:")
    print(f"  主文件: {main_file}")
    for backup_file in DATA_FILE_CONFIG['backup_files']:
        print(f"  备选: {backup_file}")
    
    return None

def filter_frequency_data(freqs, V3_exp):
    """
    根据配置筛选频率数据
    
    Args:
        freqs (np.ndarray): 原始频率数组
        V3_exp (np.ndarray): 原始V3数据
        
    Returns:
        tuple: (筛选后的频率, 筛选后的V3数据, 使用的索引)
    """
    freq_config = FITTING_CONFIG.get('frequency_range', {})
    
    if not freq_config.get('use_range', False):
        # 不使用频率筛选，返回原始数据
        return freqs, V3_exp, np.arange(len(freqs))
    
    print("\n应用频率范围筛选...")
    
    # 创建初始索引
    indices = np.arange(len(freqs))
    
    if freq_config.get('use_indices', False):
        # 使用索引选择
        start_idx = freq_config.get('start_index', 0)
        end_idx = freq_config.get('end_index', -1)
        
        if end_idx == -1:
            end_idx = len(freqs)
        
        indices = indices[start_idx:end_idx]
        print(f"  使用索引范围: [{start_idx}, {end_idx})")
        
    else:
        # 使用频率值选择
        min_freq = freq_config.get('min_freq', np.min(freqs))
        max_freq = freq_config.get('max_freq', np.max(freqs))
        
        # 找出在频率范围内的索引
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        indices = indices[freq_mask]
        
        print(f"  频率范围: [{min_freq:.2f}, {max_freq:.2f}] Hz")
    
    # 应用跳过点数
    skip = freq_config.get('skip_points', 1)
    if skip > 1:
        indices = indices[::skip]
        print(f"  每 {skip} 个点取一个")
    
    # 提取筛选后的数据
    filtered_freqs = freqs[indices]
    filtered_V3 = V3_exp[indices]
    
    print(f"  原始数据点: {len(freqs)}")
    print(f"  筛选后数据点: {len(filtered_freqs)}")
    
    if len(filtered_freqs) > 0:
        print(f"  实际频率范围: [{filtered_freqs[0]:.2f}, {filtered_freqs[-1]:.2f}] Hz")
    
    return filtered_freqs, filtered_V3, indices

def fit_fiber_experimental_data():
    """
    拟合纤维丝实验数据的主函数（使用预配置参数）
    """
    print("纤维丝3ω方法实验数据拟合")
    print("="*60)
    
    # --- 1. 数据加载 ---
    file_path = find_data_file()
    if file_path is None:
        return None
    
    freqs, V3_exp, success = load_fiber_data(file_path)
    if not success:
        return None
    
    # --- 1.5 应用频率筛选 ---
    freqs_filtered, V3_exp_filtered, freq_indices = filter_frequency_data(freqs, V3_exp)
    
    # 保存原始数据用于可视化
    freqs_original = freqs.copy()
    V3_exp_original = V3_exp.copy()
    
    # 使用筛选后的数据进行拟合
    freqs = freqs_filtered
    V3_exp = V3_exp_filtered
    
    # --- 2. 使用预配置参数 ---
    print("\n" + "="*50)
    print("使用预配置参数")
    print("="*50)
    
    fiber_params = get_fiber_params()
    initial_material = get_material_props()
    
    print(f"\n纤维配置: {FIBER_CONFIG['config_name']}")
    print(f"  长度 L: {fiber_params['L']*1e3:.2f} mm")
    print(f"  横截面积 S: {fiber_params['S']*1e12:.2f} μm²")
    if 'diameter' in fiber_params:
        print(f"  直径: {fiber_params['diameter']*1e6:.2f} μm")
    print(f"  电阻 R: {fiber_params['R']:.2f} Ω")
    print(f"  温度系数 R': {fiber_params['R_prime']:.4f} Ω/K")
    print(f"  RMS电流: {fiber_params['I_rms']*1e3:.2f} mA")
    
    print(f"\n材料配置: {MATERIAL_CONFIG['material_name']}")
    print(f"  热导率 κ: {initial_material['kappa']:.2f} W/m·K")
    print(f"  密度 ρ: {initial_material['rho']:.0f} kg/m³")
    print(f"  比热容 C_p: {initial_material['C_p']:.0f} J/kg·K")
    
    # --- 3. 拟合参数配置 ---
    params_to_fit = FITTING_CONFIG['params_to_fit']
    fit_mode = FITTING_CONFIG['fit_mode']
    param_bounds = {}
    
    print(f"\n拟合配置:")
    print(f"  拟合参数: {params_to_fit}")
    print(f"  拟合模式: {fit_mode}")
    print(f"  参数边界:")
    
    for param in params_to_fit:
        if param in FITTING_CONFIG['param_bounds']:
            bounds = FITTING_CONFIG['param_bounds'][param]
        else:
            # 默认边界
            current_val = initial_material[param]
            bounds = (current_val * 0.1, current_val * 10)
        
        param_bounds[param] = bounds
        print(f"    {param}: [{bounds[0]:.2e}, {bounds[1]:.2e}]")
    
    # --- 4. 执行拟合 ---
    print(f"\n{'='*20} 开始拟合 {'='*20}")
    
    fitter = FiberThermalFitter(
        freqs=freqs,
        V3_exp=V3_exp,
        fiber_params=fiber_params,
        initial_material_props=initial_material,
        params_to_fit=params_to_fit,
        fit_mode=fit_mode,
        param_bounds=param_bounds
    )
    
    result = fitter.fit(
        max_iter=FITTING_CONFIG['optimization']['max_iter'],
        tol=FITTING_CONFIG['optimization'].get('tolerance', 1e-5),
        zeta=FITTING_CONFIG['optimization'].get('zeta', 0.5)
    )
    
    # --- 5. 结果分析 ---
    print(f"\n{'='*20} 结果分析 {'='*20}")
    
    # 计算初始猜测的理论曲线
    V3_initial, _, _ = calculate_fiber_V3omega(freqs, fiber_params, initial_material)
    
    # 可视化结果
    if OUTPUT_CONFIG['show_plots']:
        visualize_fit_results(freqs, V3_exp, V3_initial, result, fiber_params)
    
    # --- 6. 保存结果 ---
    if OUTPUT_CONFIG['save_results']:
        save_fit_results(file_path, result, fiber_params, params_to_fit)
    
    return result

def visualize_fit_results(freqs, V3_exp, V3_initial, result, fiber_params):
    """可视化拟合结果"""
    
    V3_fitted = result['V3_fitted']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('纤维丝3ω方法拟合结果', fontsize=16)
    
    # 子图1: 幅值对比 (对数坐标)
    ax1 = axes[0, 0]
    ax1.loglog(freqs, np.abs(V3_exp) * 1e6, 'ro', label='实验数据', markersize=8)
    ax1.loglog(freqs, np.abs(V3_initial) * 1e6, 'b--', label='初始猜测', linewidth=2)
    ax1.loglog(freqs, np.abs(V3_fitted) * 1e6, 'g-', label='拟合结果', linewidth=2)
    ax1.set_xlabel('频率 [Hz]')
    ax1.set_ylabel('|V₃ω| [μV]')
    ax1.set_title('3ω电压幅值')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    
    # 子图2: 相位对比
    ax2 = axes[0, 1]
    phase_exp = np.unwrap(np.angle(V3_exp)) * 180/np.pi
    phase_initial = np.unwrap(np.angle(V3_initial)) * 180/np.pi
    phase_fitted = np.unwrap(np.angle(V3_fitted)) * 180/np.pi
    
    ax2.semilogx(freqs, phase_exp, 'ro', label='实验数据', markersize=8)
    ax2.semilogx(freqs, phase_initial, 'b--', label='初始猜测', linewidth=2)
    ax2.semilogx(freqs, phase_fitted, 'g-', label='拟合结果', linewidth=2)
    ax2.set_xlabel('频率 [Hz]')
    ax2.set_ylabel('相位 [度]')
    ax2.set_title('3ω电压相位')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    
    # 子图3: 残差分析
    ax3 = axes[1, 0]
    residuals_amp = (np.abs(V3_exp) - np.abs(V3_fitted)) / np.abs(V3_exp) * 100
    residuals_phase = (phase_exp - phase_fitted)
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.semilogx(freqs, residuals_amp, 'ro-', label='幅值残差')
    line2 = ax3_twin.semilogx(freqs, residuals_phase, 'bs-', label='相位残差')
    
    ax3.set_xlabel('频率 [Hz]')
    ax3.set_ylabel('幅值残差 [%]', color='r')
    ax3_twin.set_ylabel('相位残差 [度]', color='b')
    ax3.set_title('拟合残差')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3_twin.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    ax3.grid(True, which="both", ls="--", alpha=0.3)
    
    # 子图4: 拟合参数总结
    ax4 = axes[1, 1]
    ax4.axis('off')  # 关闭坐标轴
    
    # 创建参数总结文本
    summary_text = "拟合结果总结\n" + "="*25 + "\n\n"
    
    fitted_params = result['fitted_params']
    for param, value in fitted_params.items():
        if param == 'kappa':
            summary_text += f"热导率 κ: {value:.2f} W/m·K\n"
        elif param == 'C_p':
            summary_text += f"比热容 C_p: {value:.0f} J/kg·K\n"
        elif param == 'rho':
            summary_text += f"密度 ρ: {value:.0f} kg/m³\n"
    
    summary_text += f"\n热扩散率 α: {result['alpha']:.2e} m²/s\n"
    summary_text += f"热时间常数 γ: {result['gamma']:.2e} s\n"
    summary_text += f"RMSE: {result['rmse']:.2e}\n\n"
    
    # 纤维参数
    summary_text += "纤维参数\n" + "-"*15 + "\n"
    summary_text += f"长度 L: {fiber_params['L']*1e3:.2f} mm\n"
    summary_text += f"横截面积 S: {fiber_params['S']*1e12:.1f} μm²\n"
    summary_text += f"电阻 R: {fiber_params['R']:.2f} Ω\n"
    summary_text += f"温度系数: {fiber_params['R_prime']:.4f} Ω/K\n"
    summary_text += f"RMS电流: {fiber_params['I_rms']*1e3:.1f} mA\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def save_fit_results(original_file, result, fiber_params, params_fitted):
    """保存拟合结果"""
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(original_file))[0]
    output_file = f"{base_name}_fiber_fit_results.xlsx"
    
    # 准备结果数据
    results_data = []
    
    # 拟合参数
    for param, value in result['fitted_params'].items():
        results_data.append({
            'Parameter': param,
            'Value': value,
            'Unit': {'kappa': 'W/m·K', 'C_p': 'J/kg·K', 'rho': 'kg/m³'}.get(param, ''),
            'Type': 'Fitted'
        })
    
    # 导出参数
    results_data.extend([
        {'Parameter': 'alpha', 'Value': result['alpha'], 'Unit': 'm²/s', 'Type': 'Derived'},
        {'Parameter': 'gamma', 'Value': result['gamma'], 'Unit': 's', 'Type': 'Derived'},
        {'Parameter': 'RMSE', 'Value': result['rmse'], 'Unit': '', 'Type': 'Quality'}
    ])
    
    # 纤维参数
    fiber_data = []
    for key, value in fiber_params.items():
        if key != 'description':
            unit = {
                'L': 'm', 'S': 'm²', 'R': 'Ω', 'R_prime': 'Ω/K', 
                'I_rms': 'A', 'diameter': 'm'
            }.get(key, '')
            fiber_data.append({
                'Parameter': key,
                'Value': value,
                'Unit': unit
            })
    
    # 写入Excel文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 拟合结果
        pd.DataFrame(results_data).to_excel(writer, sheet_name='Fit_Results', index=False)
        
        # 纤维参数
        pd.DataFrame(fiber_data).to_excel(writer, sheet_name='Fiber_Params', index=False)
        
        # 有效性检查
        validity = result['validity_check']
        validity_data = [
            {'Check': 'Condition_10', 'Value': validity['condition_10_value'], 'Satisfied': validity['condition_10_satisfied']},
            {'Check': 'Condition_31', 'Value': validity['condition_31_value'], 'Satisfied': validity['condition_31_satisfied']},
            {'Check': 'Frequency_Range', 'Value': validity['max_2omega_gamma'], 'Satisfied': validity['frequency_range_ok']},
            {'Check': 'Gamma', 'Value': validity['gamma'], 'Satisfied': True}
        ]
        pd.DataFrame(validity_data).to_excel(writer, sheet_name='Validity_Check', index=False)
    
    print(f"\n拟合结果已保存到: {output_file}")

"""批量处理多个数据文件（使用预配置参数）"""
def batch_process_files():
    """批量处理多个数据文件（使用预配置参数）"""
    print("批量处理模式")
    print("="*40)
    
    # 使用预配置的目录和文件模式
    data_dir = BATCH_CONFIG['data_directory']
    file_pattern = BATCH_CONFIG['file_pattern']
    
    if not os.path.exists(data_dir):
        print(f"目录不存在: {data_dir}")
        return
    
    # 查找匹配的文件
    import glob
    pattern_path = os.path.join(data_dir, file_pattern)
    all_files = glob.glob(pattern_path)
    excel_files = [os.path.basename(f) for f in all_files if f.endswith(('.xlsx', '.xls'))]
    
    if not excel_files:
        print(f"在目录 {data_dir} 中未找到匹配 {file_pattern} 的Excel文件")
        return
    
    print(f"\n找到 {len(excel_files)} 个Excel文件:")
    for i, file in enumerate(excel_files[:10]):  # 只显示前10个
        print(f"  {i+1}. {file}")
    if len(excel_files) > 10:
        print(f"  ... 和其他 {len(excel_files)-10} 个文件")
    
    # 根据配置选择文件
    if BATCH_CONFIG['selected_files'] == 'all':
        selected_files = excel_files
    elif isinstance(BATCH_CONFIG['selected_files'], list):
        selected_files = [f for f in BATCH_CONFIG['selected_files'] if f in excel_files]
    else:
        selected_files = excel_files
    
    if not selected_files:
        print("未选择任何文件")
        return
    
    print(f"\n将处理 {len(selected_files)} 个文件")
    
    # 使用预配置参数
    fiber_params = get_fiber_params()
    initial_material = get_material_props()
    params_to_fit = FITTING_CONFIG['params_to_fit']
    fit_mode = FITTING_CONFIG['fit_mode']
    param_bounds = {param: FITTING_CONFIG['param_bounds'][param] 
                   for param in params_to_fit 
                   if param in FITTING_CONFIG['param_bounds']}
    
    print(f"\n使用预配置参数:")
    print(f"  纤维配置: {FIBER_CONFIG['config_name']}")
    print(f"  材料配置: {MATERIAL_CONFIG['material_name']}")
    print(f"  拟合参数: {params_to_fit}")
    print(f"  拟合模式: {fit_mode}")
    
    # 处理每个文件
    results_summary = []
    
    for i, file in enumerate(selected_files):
        print(f"\n{'='*20} 处理文件 {i+1}/{len(selected_files)}: {file} {'='*20}")
        
        file_path = os.path.join(data_dir, file)
        
        try:
            # 加载数据
            freqs, V3_exp, success = load_fiber_data(file_path)
            if not success:
                continue
            
            # 应用频率筛选
            freqs, V3_exp, _ = filter_frequency_data(freqs, V3_exp)
            
            # 执行拟合
            fitter = FiberThermalFitter(
                freqs=freqs,
                V3_exp=V3_exp,
                fiber_params=fiber_params,
                initial_material_props=initial_material,
                params_to_fit=params_to_fit,
                fit_mode=fit_mode,
                param_bounds=param_bounds
            )
            
            result = fitter.fit(
                max_iter=FITTING_CONFIG['optimization']['max_iter'],
                tol=FITTING_CONFIG['optimization'].get('tolerance', 1e-5),
                zeta=FITTING_CONFIG['optimization'].get('zeta', 0.5)
            )
            
            # 保存结果
            if OUTPUT_CONFIG['save_results']:
                save_fit_results(file_path, result, fiber_params, params_to_fit)
            
            # 记录总结
            summary_entry = {
                'File': file,
                'Alpha': result['alpha'],
                'Gamma': result['gamma'],
                'RMSE': result['rmse']
            }
            
            # 添加拟合参数到总结
            for param, value in result['fitted_params'].items():
                if param == 'kappa':
                    summary_entry['Kappa (W/m·K)'] = value
                elif param == 'C_p':
                    summary_entry['C_p (J/kg·K)'] = value
                elif param == 'rho':
                    summary_entry['Rho (kg/m³)'] = value
                else:
                    summary_entry[param] = value
                    
            results_summary.append(summary_entry)
            
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            continue
    
    # 保存批量处理总结
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_file = os.path.join(data_dir, "batch_processing_summary.xlsx")
        summary_df.to_excel(summary_file, index=False)
        print(f"\n批量处理总结已保存到: {summary_file}")
        
        # 显示统计信息
        print(f"\n批量处理完成! 成功处理 {len(results_summary)} 个文件")
        
        # 根据拟合参数显示统计
        for param in params_to_fit:
            if param == 'kappa' and 'Kappa (W/m·K)' in summary_df.columns:
                values = summary_df['Kappa (W/m·K)']
                print(f"热导率范围: {values.min():.2f} - {values.max():.2f} W/m·K")
            elif param == 'C_p' and 'C_p (J/kg·K)' in summary_df.columns:
                values = summary_df['C_p (J/kg·K)']
                print(f"比热容范围: {values.min():.0f} - {values.max():.0f} J/kg·K")
            elif param == 'rho' and 'Rho (kg/m³)' in summary_df.columns:
                values = summary_df['Rho (kg/m³)']
                print(f"密度范围: {values.min():.0f} - {values.max():.0f} kg/m³")

def print_current_config():
    """打印当前配置信息"""
    print("\n当前配置信息:")
    print("="*50)
    
    print(f"数据文件配置:")
    print(f"  主文件: {DATA_FILE_CONFIG['file_path']}")
    print(f"  备选文件: {len(DATA_FILE_CONFIG['backup_files'])} 个")
    
    print(f"\n纤维配置:")
    print(f"  配置名称: {FIBER_CONFIG['config_name']}")
    if FIBER_CONFIG['config_name'] == 'custom':
        params = FIBER_CONFIG['custom_params']
        print(f"  长度: {params['L']*1e3:.2f} mm")
        print(f"  直径: {params['diameter']*1e6:.2f} μm")
        print(f"  电阻: {params['R']:.2f} Ω")
    
    print(f"\n材料配置:")
    print(f"  材料名称: {MATERIAL_CONFIG['material_name']}")
    if MATERIAL_CONFIG['material_name'] == 'custom':
        props = MATERIAL_CONFIG['custom_material']
        print(f"  热导率: {props['kappa']:.2f} W/m·K")
        print(f"  密度: {props['rho']:.0f} kg/m³")
        print(f"  比热容: {props['C_p']:.0f} J/kg·K")
    
    print(f"\n拟合配置:")
    print(f"  拟合参数: {FITTING_CONFIG['params_to_fit']}")
    print(f"  拟合模式: {FITTING_CONFIG['fit_mode']}")
    print(f"  迭代方法: 灵敏度矩阵迭代法")
    print(f"  最大迭代: {FITTING_CONFIG['optimization']['max_iter']}")
    print(f"  收敛容差: {FITTING_CONFIG['optimization']['tolerance']}")
    print(f"  步长因子: {FITTING_CONFIG['optimization']['zeta']}")
    
    freq_config = FITTING_CONFIG.get('frequency_range', {})
    if freq_config.get('use_range', False):
        print(f"\n频率筛选配置:")
        if freq_config.get('use_indices', False):
            print(f"  使用索引: [{freq_config.get('start_index', 0)}, {freq_config.get('end_index', -1)})")
        else:
            print(f"  频率范围: [{freq_config.get('min_freq', 0):.1f}, {freq_config.get('max_freq', 1000):.1f}] Hz")
        if freq_config.get('skip_points', 1) > 1:
            print(f"  跳过点数: 每 {freq_config['skip_points']} 个点取一个")
    
    print(f"\n批量处理配置:")
    print(f"  启用批量处理: {BATCH_CONFIG['enabled']}")
    if BATCH_CONFIG['enabled']:
        print(f"  数据目录: {BATCH_CONFIG['data_directory']}")
        print(f"  文件模式: {BATCH_CONFIG['file_pattern']}")

if __name__ == "__main__":
    print("纤维丝3ω方法数据分析工具 (预配置模式)")
    print("="*60)
    
    # 打印当前配置
    print_current_config()
    
    # 根据配置决定运行模式
    if BATCH_CONFIG['enabled']:
        print(f"\n运行批量处理模式...")
        batch_process_files()
    else:
        print(f"\n运行单文件拟合模式...")
        result = fit_fiber_experimental_data()
        
        if result:
            print("\n拟合完成!")
            print("="*50)
            
            # 输出关键结果
            fitted_params = result['fitted_params']
            print("拟合结果总结:")
            for param, value in fitted_params.items():
                if param == 'kappa':
                    print(f"  热导率 κ: {value:.3f} W/m·K")
                elif param == 'C_p':
                    print(f"  比热容 C_p: {value:.1f} J/kg·K")
                elif param == 'rho':
                    print(f"  密度 ρ: {value:.1f} kg/m³")
            
            print(f"  热扩散率 α: {result['alpha']:.2e} m²/s")
            print(f"  热时间常数 γ: {result['gamma']:.2e} s")
            print(f"  拟合精度 RMSE: {result['rmse']:.2e}")
        else:
            print("\n拟合失败，请检查以下设置:")
            print("1. 数据文件路径是否正确")
            print("2. 纤维参数配置是否合适")
            print("3. 材料初值是否合理")
            print("4. 参数边界是否适当")
            
            print("\n提示：如果是用run_fiber_simulation.py生成的数据，")
            print("请确保两边配置一致（详见文件顶部的配置说明）")
