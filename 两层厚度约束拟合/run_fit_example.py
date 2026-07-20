"""
示例：使用 TwoLayerThicknessFitter 拟合两相邻层厚度
=====================================================

用法
----
1. 把下面 CONFIG 中的「数据文件」「传感器」「层结构」「H_total」改成你的实际值
2. 标记哪两层做厚度耦合 (layer_top_idx / layer_bot_idx，按 GUI 顺序，相邻)
3. 直接 python run_fit_example.py 即可

输出
----
- 终端打印迭代过程与最终结果
- 弹出 matplotlib 窗口：左 = V3 幅相对比，右 = 代价函数随 f 的扫描曲线
- 把结果保存到 fitting_result_two_layer.json
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from two_layer_thickness_fitter import TwoLayerThicknessFitter  # noqa: E402

# =========================================================== #
# CONFIG —— 把这里改成你的实际参数                              #
# =========================================================== #
CONFIG = {
    # ----- 1. 数据文件 ----------------------------------------
    # 兼容主程序导出的列：'频率 (Hz)','V3_Real','V3_Imag','V3_幅值','V3_相位'
    # 或 'frequency','v3_real','v3_imag'，会自动识别
    'data_file': r'',  # 例如 r'E:\项目\3Ω\...\measurement_xxx.xlsx'

    # ----- 2. 传感器参数 (与主程序一致) ------------------------
    'sensor': {
        'b':       250e-6,
        'L':       8e-3,
        'R':       6.225,
        'dR_dT':   0.0099,
        'V_rms':   0.264,
        'H_e':     100e-9,
        'rho_C_e': 2.5116e6,
    },

    # ----- 3. 层结构 (GUI 顺序: 顶层在前, 基底在后) ------------
    # 这里给一个示例：顶层 SiO2，中间 Cu（要拆分的两层之一），
    # 底层 Si 基底（H 用 np.inf）；实际请按自己的样品填
    'layers': [
        # 顶层
        {'k_y': 1.4,   'rho': 2200, 'cap': 730,  'H': 5e-6,
         'psi': 1.0, 'Rc': 1e-8},
        # 中间层 A（与下面 B 共享总厚度）
        {'k_y': 400.0, 'rho': 8960, 'cap': 385,  'H': 2e-6,
         'psi': 1.0, 'Rc': 1e-8},
        # 中间层 B（与上面 A 共享总厚度）
        {'k_y': 237.0, 'rho': 2900, 'cap': 1000, 'H': 2e-6,
         'psi': 1.0, 'Rc': 1e-8},
        # 基底 (Si)
        {'k_y': 140.0, 'rho': 2330, 'cap': 700,  'H': np.inf,
         'psi': 1.0, 'Rc': 0.0},
    ],

    # ----- 4. 两层厚度耦合 ------------------------------------
    'layer_top_idx': 1,           # 上层在 layers 中的索引 (GUI 顺序)
    'layer_bot_idx': 2,           # 下层在 layers 中的索引
    'H_total':       4e-6,        # 两层总厚度 (m)
    'f_init':        0.5,         # f = H_top / H_total 初值
    'f_bounds':      (0.05, 0.95),

    # ----- 5. (可选) 同时拟合的其它参数 ------------------------
    # 留空 {} 表示只拟合厚度；下面示例同时拟合两层各自的 k_y
    # 'extra_params_to_fit': {
    #     'layer_1': ['k_y'],
    #     'layer_2': ['k_y'],
    # },
    # 'extra_param_bounds': {
    #     'layer_1': {'k_y': (50, 800)},
    #     'layer_2': {'k_y': (50, 800)},
    # },
    'extra_params_to_fit': {},
    'extra_param_bounds':  {},

    # ----- 6. 拟合方式 ----------------------------------------
    'fit_mode': 'both',           # both/real/imag/amp/phase
    'max_iter': 400,

    # ----- 7. 频率筛选 (None 表示全部用) ----------------------
    'freq_min': None,
    'freq_max': None,

    # ----- 8. 输出 ---------------------------------------------
    'save_json': True,
    'show_plot': True,
    'scan_f_plot': True,          # 是否额外扫描 f 看代价函数
    'scan_n':     61,
}
# =========================================================== #


def _read_data_file(path: str):
    if path.lower().endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    cols = {c.lower(): c for c in df.columns}

    if '频率 (hz)' in cols:
        freq_col = cols['频率 (hz)']
    elif 'frequency' in cols:
        freq_col = cols['frequency']
    else:
        freq_col = df.columns[0]

    if 'v3_real' in cols and 'v3_imag' in cols:
        real = df[cols['v3_real']].values.astype(float)
        imag = df[cols['v3_imag']].values.astype(float)
    elif 'V3_Real' in df.columns:
        real = df['V3_Real'].values.astype(float)
        imag = df['V3_Imag'].values.astype(float)
    else:
        real = df.iloc[:, 1].values.astype(float)
        imag = df.iloc[:, 2].values.astype(float)

    freqs = df[freq_col].values.astype(float)
    V3 = real + 1j * imag
    return freqs, V3


def _plot_result(freqs, V3_exp, result, f_scan=None):
    if f_scan is not None:
        fig = plt.figure(figsize=(13, 5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])
        ax_amp = fig.add_subplot(gs[0, 0])
        ax_ph  = fig.add_subplot(gs[1, 0], sharex=ax_amp)
        ax_sc  = fig.add_subplot(gs[:, 1])
    else:
        fig, (ax_amp, ax_ph) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax_sc = None

    ax_amp.loglog(freqs, np.abs(V3_exp)*1e3, 'ro', ms=5, label='Exp')
    if result.V3_initial is not None:
        ax_amp.loglog(freqs, np.abs(result.V3_initial)*1e3, 'g--',
                      lw=1.0, label='Initial')
    if result.V3_fit is not None:
        ax_amp.loglog(freqs, np.abs(result.V3_fit)*1e3, 'b-',
                      lw=2.0, label='Fit')
    ax_amp.set_ylabel('|V3| (mV)')
    ax_amp.grid(True, which='both', ls='--', alpha=0.3)
    ax_amp.legend()
    ax_amp.set_title(
        f'H_total = {result.H_total*1e6:.2f} μm | '
        f'H_top = {result.H_top*1e6:.3f} μm, '
        f'H_bot = {result.H_bot*1e6:.3f} μm  '
        f'(f={result.f_fraction:.3f})'
    )

    ph_exp = np.unwrap(np.angle(V3_exp)) * 180 / np.pi
    ax_ph.semilogx(freqs, ph_exp, 'ro', ms=5, label='Exp')
    if result.V3_initial is not None:
        ax_ph.semilogx(freqs, np.unwrap(np.angle(result.V3_initial)) * 180 / np.pi,
                       'g--', lw=1.0, label='Initial')
    if result.V3_fit is not None:
        ax_ph.semilogx(freqs, np.unwrap(np.angle(result.V3_fit)) * 180 / np.pi,
                       'b-', lw=2.0, label='Fit')
    ax_ph.set_xlabel('Frequency (Hz)')
    ax_ph.set_ylabel('Phase (deg)')
    ax_ph.grid(True, which='both', ls='--', alpha=0.3)
    ax_ph.legend()

    if ax_sc is not None and f_scan is not None:
        f_arr, cost_arr = f_scan
        ax_sc.semilogy(f_arr, cost_arr, 'k-')
        ax_sc.axvline(result.f_fraction, color='r', ls='--',
                      label=f'fit: f={result.f_fraction:.3f}')
        ax_sc.set_xlabel('f = H_top / H_total')
        ax_sc.set_ylabel('cost')
        ax_sc.grid(True, which='both', ls='--', alpha=0.3)
        ax_sc.set_title('Cost vs f (scan)')
        ax_sc.legend()

    fig.tight_layout()
    return fig


def main(cfg=None):
    cfg = cfg or CONFIG

    if not cfg['data_file']:
        print("[错误] 请先在 CONFIG['data_file'] 填入实验数据文件路径")
        return
    if not os.path.exists(cfg['data_file']):
        print(f"[错误] 数据文件不存在: {cfg['data_file']}")
        return

    freqs, V3_exp = _read_data_file(cfg['data_file'])

    if cfg.get('freq_min') is not None or cfg.get('freq_max') is not None:
        lo = -np.inf if cfg.get('freq_min') is None else cfg['freq_min']
        hi =  np.inf if cfg.get('freq_max') is None else cfg['freq_max']
        m = (freqs >= lo) & (freqs <= hi)
        if not np.any(m):
            print("[错误] 选定频率区间没有数据点")
            return
        freqs = freqs[m]
        V3_exp = V3_exp[m]
    print(f"加载数据: {os.path.basename(cfg['data_file'])}, 共 {len(freqs)} 点")

    fitter = TwoLayerThicknessFitter(
        freqs=freqs,
        V3_exp=V3_exp,
        b=cfg['sensor']['b'],
        sensor_params=cfg['sensor'],
        initial_layers=cfg['layers'],
        layer_top_idx=cfg['layer_top_idx'],
        layer_bot_idx=cfg['layer_bot_idx'],
        H_total=cfg['H_total'],
        extra_params_to_fit=cfg.get('extra_params_to_fit') or {},
        extra_param_bounds=cfg.get('extra_param_bounds') or {},
        fit_mode=cfg['fit_mode'],
        f_init=cfg['f_init'],
        f_bounds=cfg['f_bounds'],
    )

    f_scan = None
    if cfg.get('scan_f_plot'):
        f_scan = fitter.scan_f(n=cfg.get('scan_n', 61))

    result = fitter.fit(max_iter=cfg['max_iter'], verbose=1)

    if cfg.get('save_json', True):
        out = {
            'timestamp':   datetime.now().isoformat(),
            'data_file':   os.path.basename(cfg['data_file']),
            'H_total_m':   result.H_total,
            'f_fraction':  result.f_fraction,
            'H_top_m':     result.H_top,
            'H_bot_m':     result.H_bot,
            'layer_top_idx': cfg['layer_top_idx'],
            'layer_bot_idx': cfg['layer_bot_idx'],
            'extra_params': result.extra_params,
            'success':     result.success,
            'message':     result.message,
            'cost':        result.cost,
            'residual_norm': result.residual_norm,
            'n_iter':      result.n_iter,
            'fit_mode':    cfg['fit_mode'],
        }
        out_path = os.path.join(
            HERE, f'fit_two_layer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {out_path}")

    if cfg.get('show_plot', True):
        _plot_result(freqs, V3_exp, result, f_scan=f_scan)
        plt.show()

    return result


if __name__ == '__main__':
    main()
