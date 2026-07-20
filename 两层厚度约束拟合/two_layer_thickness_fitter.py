"""
两层厚度约束拟合器
=====================

适用场景：相邻两层样品的总厚度已知（例如已用千分尺/SEM 量出 H_total = H1 + H2），
但每层各自的厚度未知，需要从 3ω 频谱反推。

实现思路
--------
引入归一化变量 f = H1 / H_total ∈ (0, 1)，则：
    H1 = f * H_total
    H2 = (1 - f) * H_total
自动满足总厚度约束，避免在两个变量上各自设边界导致的耦合问题。

可以单独拟合 f，也可以同时拟合其他材料参数（如 k_y、Rc 等）。
底层直接调用现有的 calculate_analytical_Z / convert_Z_to_V3，不修改现有模块。

层顺序约定
----------
本模块外部接口使用 GUI 顺序，即 layers = [顶层, ..., 基底]。
内部传给 Olson 模型时会自动反转为 [基底, ..., 顶层]。
layer_top_idx / layer_bot_idx 也按 GUI 顺序给出，且必须相邻
（典型情况：layer_top_idx = i, layer_bot_idx = i + 1）。
"""

from __future__ import annotations

import os
import sys
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from olson_model26_3 import calculate_analytical_Z, convert_Z_to_V3  # noqa: E402


@dataclass
class TwoLayerThicknessFitResult:
    """拟合结果数据结构"""
    success: bool
    message: str
    H_total: float
    f_fraction: float           # H_top / H_total
    H_top: float                # 上层厚度（GUI 视角）
    H_bot: float                # 下层厚度（GUI 视角）
    extra_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    layers_fitted: List[Dict[str, float]] = field(default_factory=list)  # GUI 顺序
    residual_norm: float = float('nan')
    cost: float = float('nan')
    n_iter: int = 0
    V3_fit: Optional[np.ndarray] = None
    V3_initial: Optional[np.ndarray] = None
    history: List[Dict[str, float]] = field(default_factory=list)


class TwoLayerThicknessFitter:
    """
    在「相邻两层总厚度已知」约束下，拟合各层厚度（以及可选的其它材料参数）。

    Parameters
    ----------
    freqs : np.ndarray
        实验频率列表 (Hz)
    V3_exp : np.ndarray
        实验 V3ω (复数)
    b : float
        加热器半线宽 (m)
    sensor_params : dict
        传感器参数 {'R','L','V_rms','dR_dT','H_e','rho_C_e'}
    initial_layers : list[dict]
        GUI 顺序的层结构 [顶层, ..., 基底]，每层含 {'k_y','rho','cap','H','psi','Rc'}
    layer_top_idx, layer_bot_idx : int
        被厚度约束绑定的两相邻层在 initial_layers 中的索引（GUI 顺序）
    H_total : float
        两层总厚度 (m)，作为硬约束保持不变
    extra_params_to_fit : dict, optional
        额外要拟合的参数，键名格式与 ThermalModelFitter 一致：
            {'layer_0': ['k_y','Rc'], 'layer_1': ['k_y'],
             'sensor':  ['R', 'dR_dT', 'V_rms', ...], ...}
        层索引按 GUI 顺序；'sensor' 是特殊键，里面列传感器参数名。
    extra_param_bounds : dict, optional
        对应的边界：
            {'layer_0': {'k_y': (lo, hi), ...},
             'sensor':  {'R': (lo, hi), 'dR_dT': (lo, hi), ...}}
    fit_mode : str
        'both' | 'real' | 'imag' | 'amp' | 'phase'
    f_init : float
        f 的初值
    f_bounds : tuple
        f 的边界，默认 (0.01, 0.99)；可以根据物理预判收窄
    """

    PROP_DEFAULT_BOUNDS = {
        'k_y': (0.001, 5000.0),
        'rho': (100.0, 20000.0),
        'cap': (100.0, 5000.0),
        'psi': (0.01, 100.0),
        'Rc':  (0.0, 1e-3),
    }

    # 传感器参数的默认相对边界（无显式 bounds 时使用）
    SENSOR_DEFAULT_REL = {
        'R':       (0.5, 2.0),
        'V_rms':   (0.5, 2.0),
        'L':       (0.9, 1.1),
        'b':       (0.9, 1.1),
        'dR_dT':   (0.1, 10.0),
        'H_e':     (0.1, 10.0),
        'rho_C_e': (0.1, 10.0),
    }

    def __init__(
        self,
        freqs: np.ndarray,
        V3_exp: np.ndarray,
        b: float,
        sensor_params: Dict[str, float],
        initial_layers: List[Dict[str, float]],
        layer_top_idx: int,
        layer_bot_idx: int,
        H_total: float,
        extra_params_to_fit: Optional[Dict[str, List[str]]] = None,
        extra_param_bounds: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
        fit_mode: str = 'both',
        f_init: float = 0.5,
        f_bounds: Tuple[float, float] = (0.01, 0.99),
    ):
        self.freqs = np.asarray(freqs, dtype=float)
        self.V3_exp = np.asarray(V3_exp, dtype=complex)
        self.b = float(b)
        self.sensor_params = dict(sensor_params)
        self.initial_layers = [dict(l) for l in initial_layers]
        self.layers = [dict(l) for l in initial_layers]

        if not (0 <= layer_top_idx < len(initial_layers)):
            raise ValueError("layer_top_idx 越界")
        if not (0 <= layer_bot_idx < len(initial_layers)):
            raise ValueError("layer_bot_idx 越界")
        if abs(layer_top_idx - layer_bot_idx) != 1:
            raise ValueError(
                "layer_top_idx 与 layer_bot_idx 必须相邻 "
                f"(给定 {layer_top_idx}, {layer_bot_idx})"
            )
        if H_total <= 0:
            raise ValueError("H_total 必须为正")
        if fit_mode not in ('both', 'real', 'imag', 'amp', 'phase'):
            raise ValueError("fit_mode 必须是 'both'/'real'/'imag'/'amp'/'phase'")
        if not (0.0 < f_bounds[0] < f_bounds[1] < 1.0):
            raise ValueError("f_bounds 必须满足 0 < lo < hi < 1")
        if not (f_bounds[0] <= f_init <= f_bounds[1]):
            f_init = 0.5 * (f_bounds[0] + f_bounds[1])

        self.layer_top_idx = int(layer_top_idx)
        self.layer_bot_idx = int(layer_bot_idx)
        self.H_total = float(H_total)
        self.fit_mode = fit_mode
        self.f_init = float(f_init)
        self.f_bounds = (float(f_bounds[0]), float(f_bounds[1]))

        self.extra_params_to_fit = extra_params_to_fit or {}
        self.extra_param_bounds = extra_param_bounds or {}

        self._param_map: List[Dict] = []
        self._build_param_map()

        self._history: List[Dict[str, float]] = []

    # --------------------------------------------------------------------- #
    # 参数向量管理
    # --------------------------------------------------------------------- #
    def _build_param_map(self):
        """参数向量布局：[f, extra_layer_params..., extra_sensor_params...]"""
        pm: List[Dict] = [{'kind': 'f'}]

        # 先添加材料层额外参数
        for layer_key, props in self.extra_params_to_fit.items():
            if layer_key == 'sensor':
                continue  # 留到后面处理
            if not layer_key.startswith('layer_'):
                raise ValueError(
                    f"extra_params_to_fit 键必须是 'layer_i' 或 'sensor'，得到 {layer_key}"
                )
            gui_idx = int(layer_key.split('_', 1)[1])
            if not (0 <= gui_idx < len(self.initial_layers)):
                raise ValueError(f"layer_key {layer_key} 越界")
            for prop in props:
                if prop == 'H':
                    raise ValueError("不要把 H 放进 extra_params_to_fit，厚度由 H_total + f 约束")
                pm.append({'kind': 'extra', 'gui_idx': gui_idx, 'prop': prop})

        # 再添加传感器参数
        sensor_props = self.extra_params_to_fit.get('sensor', [])
        for prop in sensor_props:
            if prop not in self.sensor_params:
                raise ValueError(f"传感器参数 '{prop}' 不在 sensor_params 中")
            pm.append({'kind': 'sensor', 'prop': prop})

        self._param_map = pm

    def _initial_vector(self) -> np.ndarray:
        x0 = []
        for pm in self._param_map:
            if pm['kind'] == 'f':
                x0.append(self.f_init)
            elif pm['kind'] == 'sensor':
                x0.append(self.sensor_params[pm['prop']])
            else:
                x0.append(self.initial_layers[pm['gui_idx']][pm['prop']])
        return np.asarray(x0, dtype=float)

    def _bounds_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = [], []
        for pm in self._param_map:
            if pm['kind'] == 'f':
                lo.append(self.f_bounds[0])
                hi.append(self.f_bounds[1])
            elif pm['kind'] == 'sensor':
                prop = pm['prop']
                bnds = self.extra_param_bounds.get('sensor', {}).get(prop)
                if bnds is None:
                    rel = self.SENSOR_DEFAULT_REL.get(prop, (0.1, 10.0))
                    cur = self.sensor_params[prop]
                    if cur > 0:
                        bnds = (cur * rel[0], cur * rel[1])
                    else:
                        bnds = (-abs(cur) * 10 - 1e-12, abs(cur) * 10 + 1e-12)
                lo.append(bnds[0])
                hi.append(bnds[1])
            else:
                bnds = self.extra_param_bounds.get(
                    f'layer_{pm["gui_idx"]}', {}
                ).get(pm['prop'])
                if bnds is None:
                    bnds = self.PROP_DEFAULT_BOUNDS.get(pm['prop'])
                if bnds is None:
                    cur = self.initial_layers[pm['gui_idx']][pm['prop']]
                    bnds = (cur * 0.01 if cur > 0 else 1e-10,
                            cur * 100.0 if cur > 0 else 1.0)
                lo.append(bnds[0])
                hi.append(bnds[1])
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    def _apply_vector_to_layers(self, x: np.ndarray):
        """把参数向量写回 self.layers / self.sensor_params，并按 H_total 约束分配厚度"""
        for val, pm in zip(x, self._param_map):
            if pm['kind'] == 'f':
                f = float(val)
                self.layers[self.layer_top_idx]['H'] = f * self.H_total
                self.layers[self.layer_bot_idx]['H'] = (1.0 - f) * self.H_total
            elif pm['kind'] == 'sensor':
                self.sensor_params[pm['prop']] = float(val)
            else:
                self.layers[pm['gui_idx']][pm['prop']] = float(val)

    # --------------------------------------------------------------------- #
    # V3 计算与残差
    # --------------------------------------------------------------------- #
    def _compute_V3(self) -> np.ndarray:
        model_layers = [dict(l) for l in self.layers[::-1]]
        Z = calculate_analytical_Z(self.freqs, self.b, model_layers)
        return convert_Z_to_V3(Z, self.freqs, self.b, self.sensor_params)

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        self._apply_vector_to_layers(x)
        try:
            V3_th = self._compute_V3()
        except Exception:
            return np.full(self._residual_length(), 1e6)
        if np.any(np.isnan(V3_th)) or np.any(np.isinf(V3_th)):
            return np.full(self._residual_length(), 1e6)

        if self.fit_mode == 'both':
            r = np.concatenate([
                (self.V3_exp.real - V3_th.real),
                (self.V3_exp.imag - V3_th.imag),
            ])
            scale = max(np.max(np.abs(self.V3_exp.real)),
                        np.max(np.abs(self.V3_exp.imag)), 1e-20)
            r = r / scale
        elif self.fit_mode == 'real':
            r = self.V3_exp.real - V3_th.real
            r = r / max(np.max(np.abs(self.V3_exp.real)), 1e-20)
        elif self.fit_mode == 'imag':
            r = self.V3_exp.imag - V3_th.imag
            r = r / max(np.max(np.abs(self.V3_exp.imag)), 1e-20)
        elif self.fit_mode == 'amp':
            A_exp = np.abs(self.V3_exp)
            r = (A_exp - np.abs(V3_th)) / max(np.max(A_exp), 1e-20)
        else:  # phase
            ph_exp = np.unwrap(np.angle(self.V3_exp))
            ph_th = np.unwrap(np.angle(V3_th))
            r = (ph_exp - ph_th) / (2 * np.pi)
        return r

    def _residual_length(self) -> int:
        if self.fit_mode == 'both':
            return 2 * len(self.freqs)
        return len(self.freqs)

    # --------------------------------------------------------------------- #
    # 主入口
    # --------------------------------------------------------------------- #
    def fit(
        self,
        max_iter: int = 200,
        ftol: float = 1e-10,
        xtol: float = 1e-10,
        gtol: float = 1e-10,
        verbose: int = 1,
    ) -> TwoLayerThicknessFitResult:
        """
        运行约束拟合。

        verbose: 0 静默 / 1 简略 / 2 每步详细
        """
        x0 = self._initial_vector()
        lo, hi = self._bounds_vectors()

        if verbose >= 1:
            print("=" * 60)
            print("两层厚度约束拟合 (H_total = H_top + H_bot 已知)")
            print(f"  GUI 第 {self.layer_top_idx} 层 (top) <-> 第 {self.layer_bot_idx} 层 (bot)")
            print(f"  H_total = {self.H_total*1e6:.3f} μm")
            print(f"  自由参数数: {len(x0)} | 残差长度: {self._residual_length()}")
            print(f"  fit_mode = {self.fit_mode}")
            print(f"  初值: f = {x0[0]:.4f} -> "
                  f"H_top = {x0[0]*self.H_total*1e6:.3f} um, "
                  f"H_bot = {(1-x0[0])*self.H_total*1e6:.3f} um")
            for i, pm in enumerate(self._param_map[1:], 1):
                if pm['kind'] == 'sensor':
                    tag = f"sensor.{pm['prop']}"
                else:
                    tag = f"{pm['prop']}@layer_{pm['gui_idx']}"
                print(f"        {tag} = {x0[i]:.4e} "
                      f"(范围 [{lo[i]:.2e}, {hi[i]:.2e}])")
            print("-" * 60)

        self._history.clear()

        def wrapped_residuals(x):
            r = self._residuals(x)
            self._history.append({
                'f': float(x[0]),
                'H_top': float(x[0] * self.H_total),
                'H_bot': float((1 - x[0]) * self.H_total),
                'cost': 0.5 * float(np.dot(r, r)),
            })
            if verbose >= 2 and len(self._history) % max(1, max_iter // 20) == 0:
                last = self._history[-1]
                print(f"  iter={len(self._history):3d}  cost={last['cost']:.4e}  "
                      f"f={last['f']:.4f}  "
                      f"H_top={last['H_top']*1e6:.3f}um  "
                      f"H_bot={last['H_bot']*1e6:.3f}um")
            return r

        V3_initial = self._compute_V3()

        try:
            res = least_squares(
                wrapped_residuals, x0,
                bounds=(lo, hi),
                method='trf',
                max_nfev=max_iter,
                ftol=ftol, xtol=xtol, gtol=gtol,
                verbose=0,
            )
            success = res.success
            message = res.message
            x_opt = res.x
            cost = float(res.cost)
            n_iter = int(res.nfev)
            residual_norm = float(np.linalg.norm(res.fun))
        except Exception as e:
            success = False
            message = f"least_squares 失败: {e}"
            x_opt = x0
            cost = float('nan')
            n_iter = 0
            residual_norm = float('nan')

        self._apply_vector_to_layers(x_opt)
        V3_fit = self._compute_V3()

        f_opt = float(x_opt[0])
        result = TwoLayerThicknessFitResult(
            success=success,
            message=message,
            H_total=self.H_total,
            f_fraction=f_opt,
            H_top=f_opt * self.H_total,
            H_bot=(1.0 - f_opt) * self.H_total,
            extra_params=self._collect_extra_params(x_opt),
            layers_fitted=[dict(l) for l in self.layers],
            residual_norm=residual_norm,
            cost=cost,
            n_iter=n_iter,
            V3_fit=V3_fit,
            V3_initial=V3_initial,
            history=list(self._history),
        )

        if verbose >= 1:
            print("-" * 60)
            print(f"拟合结束: success={success}, msg={message}")
            print(f"  迭代次数 = {n_iter}, cost = {cost:.4e}, |r| = {residual_norm:.4e}")
            print(f"  f = {f_opt:.6f}")
            print(f"  H_top (layer {self.layer_top_idx}) = "
                  f"{result.H_top*1e6:.4f} μm")
            print(f"  H_bot (layer {self.layer_bot_idx}) = "
                  f"{result.H_bot*1e6:.4f} μm")
            for key, props in result.extra_params.items():
                for prop, val in props.items():
                    print(f"  {key}.{prop} = {val:.6e}")
            print(f"  (最终传感器参数: " +
                  ", ".join(f"{k}={self.sensor_params[k]:.4e}"
                            for k in ('R','dR_dT','V_rms','b','L'))
                  + ")")
            print("=" * 60)

        return result

    def _collect_extra_params(self, x: np.ndarray) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for val, pm in zip(x, self._param_map):
            if pm['kind'] == 'extra':
                out.setdefault(f'layer_{pm["gui_idx"]}', {})[pm['prop']] = float(val)
            elif pm['kind'] == 'sensor':
                out.setdefault('sensor', {})[pm['prop']] = float(val)
        return out

    # --------------------------------------------------------------------- #
    # 工具：扫描 f 看代价函数（用于诊断是否多解 / 哪段下降）
    # --------------------------------------------------------------------- #
    def scan_f(self, n: int = 81) -> Tuple[np.ndarray, np.ndarray]:
        """
        固定其余参数为初值，仅扫描 f 看代价函数曲线，便于诊断初值与边界。
        返回 (f_array, cost_array)。
        """
        f_arr = np.linspace(self.f_bounds[0], self.f_bounds[1], int(n))
        cost_arr = np.empty_like(f_arr)
        layers_backup = copy.deepcopy(self.layers)
        sensor_backup = dict(self.sensor_params)
        x0 = self._initial_vector()
        for i, f in enumerate(f_arr):
            x = x0.copy()
            x[0] = f
            r = self._residuals(x)
            cost_arr[i] = 0.5 * np.dot(r, r)
        self.layers = layers_backup
        self.sensor_params = sensor_backup
        return f_arr, cost_arr
