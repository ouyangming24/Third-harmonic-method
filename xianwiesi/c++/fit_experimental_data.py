import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from fiber_model import (
    FiberThermalFitter,
    create_fiber_material_library,
    create_typical_fiber_params,
    check_validity_conditions
)

# 中文字体支持（优先使用 DejaVu 提供完整符号）
rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'dejavusans'

DEFAULT_CONFIG_PATH = Path("experimental_fit_config.jsonc")


def strip_jsonc(text: str) -> str:
    """移除 JSONC 风格注释，保留合法 JSON 文本。"""
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)
    text = re.sub(r'//.*', '', text)
    return text


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"未找到配置文件: {config_path}\n"
            "请复制 experimental_fit_config.jsonc 模板并填写参数。"
        )
    raw = config_path.read_text(encoding='utf-8')
    sanitized = strip_jsonc(raw)
    try:
        config = json.loads(sanitized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"解析配置文件失败: {exc}") from exc
    return config


def load_experimental_data(data_cfg: dict):
    data_path = Path(data_cfg['file'])
    if not data_path.exists():
        raise FileNotFoundError(f"未找到实验数据文件: {data_path}")

    if data_path.suffix.lower() in {'.xls', '.xlsx'}:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("数据文件为空或缺少有效内容")

    columns_cfg = data_cfg.get('columns', {})
    phase_unit = columns_cfg.get('phase_unit', 'deg').lower()

    column_lookup = {c.lower().strip(): c for c in df.columns}

    def resolve_column(explicit_name, keywords):
        if explicit_name:
            norm = explicit_name.lower().strip()
            if norm in column_lookup:
                return column_lookup[norm]
            raise ValueError(f"指定列 {explicit_name} 不存在")
        for norm_name, original in column_lookup.items():
            if all(keyword in norm_name for keyword in keywords):
                return original
        return None

    freq_col = columns_cfg.get('frequency')
    freq_col = resolve_column(freq_col, ['freq']) if freq_col is None else freq_col
    if freq_col is None:
        raise ValueError("未能自动识别频率列，请在配置中指定 columns.frequency")

    real_col = columns_cfg.get('real')
    imag_col = columns_cfg.get('imag')
    mag_col = columns_cfg.get('magnitude')
    phase_col = columns_cfg.get('phase')

    if real_col is None:
        real_col = resolve_column(None, ['v3', 'real'])
    if imag_col is None:
        imag_col = resolve_column(None, ['v3', 'imag'])
    if mag_col is None:
        mag_col = resolve_column(None, ['mag'])
    if phase_col is None:
        phase_col = resolve_column(None, ['phase'])

    freqs = df[freq_col].to_numpy(dtype=float)

    if real_col and imag_col:
        V3_real = df[real_col].to_numpy(dtype=float)
        V3_imag = df[imag_col].to_numpy(dtype=float)
        V3_exp = V3_real + 1j * V3_imag
    elif mag_col and phase_col:
        magnitude = df[mag_col].to_numpy(dtype=float)
        phase = df[phase_col].to_numpy(dtype=float)
        if phase_unit == 'deg':
            phase = np.deg2rad(phase)
        V3_exp = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    else:
        raise ValueError("请在配置中提供实部+虚部列或幅值+相位列")

    valid_mask = np.isfinite(freqs) & np.isfinite(V3_exp.real) & np.isfinite(V3_exp.imag)
    freqs = freqs[valid_mask]
    V3_exp = V3_exp[valid_mask]

    if len(freqs) < 5:
        raise ValueError("有效数据点不足 5 个，无法拟合")

    order = np.argsort(freqs)
    freqs = freqs[order]
    V3_exp = V3_exp[order]

    print(f"数据加载完成: {data_path.name}")
    print(f"  数据点数: {len(freqs)}")
    print(f"  频率范围: {freqs.min():.3f} - {freqs.max():.3f} Hz")
    print(f"  |V3ω| 范围: {np.abs(V3_exp).min():.3e} - {np.abs(V3_exp).max():.3e} V")

    return freqs, V3_exp


def apply_frequency_selection(freqs, V3_exp, freq_cfg: dict):
    selected_freqs = freqs
    selected_V3 = V3_exp

    if freq_cfg:
        min_freq = freq_cfg.get('min')
        max_freq = freq_cfg.get('max')
        if min_freq is not None or max_freq is not None:
            lower = min_freq if min_freq is not None else -np.inf
            upper = max_freq if max_freq is not None else np.inf
            mask = (selected_freqs >= lower) & (selected_freqs <= upper)
            selected_freqs = selected_freqs[mask]
            selected_V3 = selected_V3[mask]

        start_index = freq_cfg.get('start_index')
        end_index = freq_cfg.get('end_index')
        if start_index is not None or end_index is not None:
            start = start_index or 0
            end = end_index if end_index is not None else len(selected_freqs)
            selected_freqs = selected_freqs[start:end]
            selected_V3 = selected_V3[start:end]

        skip_every = max(freq_cfg.get('skip_every', 1), 1)
        if skip_every > 1:
            selected_freqs = selected_freqs[::skip_every]
            selected_V3 = selected_V3[::skip_every]

    if len(selected_freqs) < 5:
        raise ValueError("频率筛选过度，剩余点数 < 5")

    print(f"频率筛选后数据点数: {len(selected_freqs)}")
    return selected_freqs, selected_V3


def build_fiber_params(fiber_cfg: dict):
    configs = create_typical_fiber_params()
    preset_name = fiber_cfg.get('preset', 'platinum_wire_20um')

    if preset_name in configs:
        fiber_params = configs[preset_name].copy()
    elif preset_name.lower() == 'custom':
        fiber_params = {}
    else:
        raise ValueError(f"未知的纤维配置: {preset_name}")

    overrides = fiber_cfg.get('overrides', {})

    if overrides.get('L_mm') is not None:
        fiber_params['L'] = overrides['L_mm'] * 1e-3
    if overrides.get('diameter_um') is not None:
        diameter_m = overrides['diameter_um'] * 1e-6
        fiber_params['diameter'] = diameter_m
        fiber_params['S'] = np.pi * (diameter_m / 2)**2
    elif overrides.get('area_um2') is not None:
        fiber_params['S'] = overrides['area_um2'] * 1e-12
        fiber_params.pop('diameter', None)

    if overrides.get('R') is not None:
        fiber_params['R'] = overrides['R']
    if overrides.get('R_prime') is not None:
        fiber_params['R_prime'] = overrides['R_prime']
    if overrides.get('I_mA') is not None:
        fiber_params['I_rms'] = overrides['I_mA'] * 1e-3

    required = ['L', 'S', 'R', 'R_prime', 'I_rms']
    missing = [key for key in required if key not in fiber_params]
    if missing:
        raise ValueError(f"纤维参数缺少字段: {missing}，请在配置中补齐")

    fiber_params.setdefault('description', fiber_cfg.get('description', '实验纤维样品'))
    return fiber_params


def build_material_props(material_cfg: dict):
    materials = create_fiber_material_library()
    preset_name = material_cfg.get('preset', 'platinum')

    if preset_name in materials:
        material_props = materials[preset_name].copy()
    elif preset_name.lower() == 'custom':
        material_props = {}
    else:
        raise ValueError(f"未知材料名称: {preset_name}")

    overrides = material_cfg.get('overrides', {})
    for key in ('kappa', 'rho', 'C_p'):
        if overrides.get(key) is not None:
            material_props[key] = overrides[key]

    required = ['kappa', 'rho', 'C_p']
    missing = [key for key in required if key not in material_props]
    if missing:
        raise ValueError(f"材料初值缺少字段: {missing}")
    return material_props


def normalize_params_to_fit(raw_params):
    """
    将配置文件中的参数列表转换为新接口的字典格式
    返回: {'material': ['kappa', 'C_p', ...]}
    """
    if raw_params is None:
        raise ValueError("配置中必须提供 fitting.params_to_fit")
    if isinstance(raw_params, str):
        tokens = raw_params.replace(';', ',').split(',')
    else:
        tokens = raw_params

    allowed = {
        'kappa': 'kappa', 'k': 'kappa',
        'c_p': 'C_p', 'cp': 'C_p', 'c': 'C_p',
        'rho': 'rho'
    }
    params = []
    for token in tokens:
        name = token.strip().lower()
        if not name:
            continue
        if name not in allowed:
            raise ValueError(f"不支持的拟合参数: {token}")
        norm = allowed[name]
        if norm not in params:
            params.append(norm)

    if not params:
        raise ValueError("拟合参数列表为空")
    
    # 返回新接口格式
    return {'material': params}


def build_param_bounds(bounds_cfg: dict, params_to_fit):
    """
    构建参数边界，返回新接口格式
    params_to_fit: {'material': ['kappa', 'C_p', ...]}
    返回: {'material': {'kappa': (min, max), ...}}
    """
    defaults = {
        'kappa': (0.1, 5000),
        'C_p': (50, 5000),
        'rho': (100, 25000)
    }
    
    # 新接口格式
    bounds = {'material': {}}
    param_list = params_to_fit.get('material', [])
    
    for param in param_list:
        custom = bounds_cfg.get(param) if bounds_cfg else None
        if custom:
            bounds['material'][param] = tuple(custom)
        else:
            bounds['material'][param] = defaults[param]
    return bounds


def print_validity_report(validity_dict):
    print("\n模型有效性检查:")
    print(f"  条件(10)功率均匀性: {validity_dict['condition_10_value']:.3e} "
          f"({'满足' if validity_dict['condition_10_satisfied'] else '不满足'})")
    print(f"  条件(31)辐射损失: {validity_dict['condition_31_value']:.3e} "
          f"({'满足' if validity_dict['condition_31_satisfied'] else '不满足'})")
    print(f"  频率范围 2ωγ_max: {validity_dict['max_2omega_gamma']:.3f} "
          f"({'合适' if validity_dict['frequency_range_ok'] else '偏高'})")
    print(f"  等效直径: {validity_dict['equivalent_diameter']*1e6:.2f} μm")
    print(f"  热时间常数 γ: {validity_dict['gamma']:.3e} s")


def plot_results(freqs, V3_exp, V3_fit, save_path=None, show_plot=False, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title = title or '实验数据拟合结果'
    fig.suptitle(title)

    axes[0].loglog(freqs, np.abs(V3_exp), 'o', label='实验数据', alpha=0.7)
    axes[0].loglog(freqs, np.abs(V3_fit), '-', label='拟合曲线')
    axes[0].set_xlabel('频率 [Hz]')
    axes[0].set_ylabel('|V₃ω| [V]')
    axes[0].set_title('幅值')
    axes[0].grid(True, which='both', ls='--', alpha=0.3)
    axes[0].legend()

    phase_exp = np.unwrap(np.angle(V3_exp)) * 180 / np.pi
    phase_fit = np.unwrap(np.angle(V3_fit)) * 180 / np.pi
    axes[1].semilogx(freqs, phase_exp, 'o', label='实验数据', alpha=0.7)
    axes[1].semilogx(freqs, phase_fit, '-', label='拟合曲线')
    axes[1].set_xlabel('频率 [Hz]')
    axes[1].set_ylabel('相位 [度]')
    axes[1].set_title('相位')
    axes[1].grid(True, which='both', ls='--', alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {out_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def export_summary_json(result, fiber_params, material_props, path, gamma, alpha):
    """
    导出拟合摘要JSON
    新接口: result 是 material_props (dict)
    """
    def to_serializable(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        'fitted_material_props': result,  # 新接口直接返回
        'initial_material_props': material_props,
        'fiber_params': fiber_params,
        'alpha': alpha,
        'gamma': gamma
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2, default=to_serializable)
    print(f"拟合摘要已保存: {out_path}")


def export_fitted_dataset(freqs, V3_exp, V3_fit, path):
    df = pd.DataFrame({
        'Frequency_Hz': freqs,
        'V3_Real_Exp': V3_exp.real,
        'V3_Imag_Exp': V3_exp.imag,
        'V3_Magnitude_Exp': np.abs(V3_exp),
        'V3_Phase_Exp_deg': np.unwrap(np.angle(V3_exp)) * 180 / np.pi,
        'V3_Real_Fitted': V3_fit.real,
        'V3_Imag_Fitted': V3_fit.imag,
        'V3_Magnitude_Fitted': np.abs(V3_fit),
        'V3_Phase_Fitted_deg': np.unwrap(np.angle(V3_fit)) * 180 / np.pi,
    })
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"拟合数据已导出: {out_path}")


def main(config_path: Path):
    config = load_config(config_path)

    freqs, V3_exp = load_experimental_data(config['data'])
    freqs, V3_exp = apply_frequency_selection(
        freqs, V3_exp, config['data'].get('frequency_filter', {})
    )

    fiber_params = build_fiber_params(config['fiber'])
    initial_material_props = build_material_props(config['material'])

    fitting_cfg = config['fitting']
    params_to_fit = normalize_params_to_fit(fitting_cfg.get('params_to_fit'))
    bounds = build_param_bounds(fitting_cfg.get('bounds'), params_to_fit)
    fit_mode = fitting_cfg.get('fit_mode', 'both')
    max_iter = fitting_cfg.get('max_iter', 50)
    tolerance = fitting_cfg.get('tolerance', 1e-5)
    zeta = fitting_cfg.get('zeta', 0.5)

    print("\n拟合配置概览:")
    print(f"  纤维配置: {fiber_params.get('description', fiber_params)}")
    print(f"  拟合参数: {params_to_fit['material']}")  # 访问material列表
    print(f"  拟合模式: {fit_mode}")

    validity = check_validity_conditions(fiber_params, initial_material_props, freqs)
    print_validity_report(validity)

    # 使用新接口创建拟合器
    fitter = FiberThermalFitter(
        freqs=freqs,
        V3_exp=V3_exp,
        fiber_params=fiber_params,
        initial_material_props=initial_material_props.copy(),
        params_to_fit=params_to_fit,  # 已经是字典格式
        fit_mode=fit_mode,
        param_bounds=bounds,
        fit_type='material'  # 明确指定类型
    )

    # 执行拟合，新接口直接返回 material_props
    fit_result = fitter.fit(max_iter=max_iter, tol=tolerance, zeta=zeta)
    
    # 重新计算V3_fitted（新接口不返回V3_fitted）
    from fiber_model import calculate_fiber_V3omega
    V3_fitted, gamma_fitted, alpha_fitted = calculate_fiber_V3omega(freqs, fiber_params, fit_result)

    print("\n拟合总结:")
    for name in params_to_fit['material']:
        value = fit_result[name]
        bound = bounds['material'].get(name)
        print(f"  {name}: {value:.5g}  (边界: {bound})")
    print(f"  热扩散率 alpha: {alpha_fitted:.4e} m^2/s")
    print(f"  热时间常数 gamma: {gamma_fitted:.4e} s")

    outputs = config.get('outputs', {})
    save_plot_path = outputs.get('save_plot')
    show_plot = outputs.get('show_plot', False)
    if save_plot_path or show_plot:
        title = Path(config['data']['file']).stem + " 拟合对比"
        plot_results(freqs, V3_exp, V3_fitted,
                     save_path=save_plot_path,
                     show_plot=show_plot,
                     title=title)

    if outputs.get('export_json'):
        export_summary_json(
            fit_result, fiber_params, initial_material_props, 
            outputs['export_json'], gamma_fitted, alpha_fitted
        )
    if outputs.get('export_data'):
        export_fitted_dataset(freqs, V3_exp, V3_fitted, outputs['export_data'])


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    main(config_arg)
