"""
绝缘纤维丝（镀层）3ω 模型 — Xing / Munro / Ban (IMECE2015 及 ATE 2014)

绝缘丝本身不导电，表面镀金属膜（Au 等）作加热器与温度计。
轴向一维导热 + 两端恒温热沉 + 径向辐射。

公式（文中式 8–10；式8 按低频极限与式9 一致补全为 I0³）:
  m² = i 2ω/α + 4 h_r / (D k)
  h_r = 4 ε σ T0³
  V3_amp = I0³ R0 R' [2-2cosh(mL)+mL sinh(mL)]
           / (L² m³ k π D² sinh(mL))
  低频: V3_amp = I0³ R0 R' L / (12 k π D²)
  高频: V3_amp = I0³ R0 R' / (i 2ω ρ C_p π D² L)

约定与导电丝 GUI 一致:
  用户输入 I_rms、实验 V3 为锁相 rms；
  I0 = √2 I_rms（幅值），算出的 V3_amp 再 /√2 得到 V3_rms。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from lu2001_fiber_model import (
    FiberGeometry, MaterialProps, RadiationParams, SIGMA_SB,
    diameter_from_area, thermal_diffusivity,
)


def hr_radiation(emissivity: float, T0: float) -> float:
    """h_r = 4 ε σ T0³  [W/m²K]"""
    if emissivity <= 0:
        return 0.0
    return 4.0 * emissivity * SIGMA_SB * (T0 ** 3)


def complex_m(omega: np.ndarray, alpha: float, kappa: float, D: float,
              hr: float) -> np.ndarray:
    """m = sqrt(i 2ω/α + 4 hr/(D k))"""
    return np.sqrt(1j * 2.0 * omega / alpha + 4.0 * hr / (D * kappa))


def _transfer_factor(m: np.ndarray, L: float) -> np.ndarray:
    """
    F(m) = [2-2cosh(mL)+mL sinh(mL)] / (L² m³ sinh(mL))
    等价稳定式: F = [x - 2 coth(x) + 2 csch(x)] / (L² m³), x=mL
    m→0 时 F → L/12；|x| 大时 → (x-2)/(L² m³) ≈ 1/(L² m²)
    """
    m = np.asarray(m, dtype=complex)
    x = m * L
    F = np.empty_like(x, dtype=complex)

    for i in range(x.size):
        xi = x.flat[i]
        mi = m.flat[i]
        ax = abs(xi)
        if ax < 1e-6:
            F.flat[i] = L / 12.0
            continue
        # 大 |x|：coth→sign(Re), csch→0；取 Re(x)≥0 分支的渐近
        if ax > 40.0:
            F.flat[i] = (xi - 2.0) / ((L ** 2) * (mi ** 3))
            continue
        # 中等：用 e^{-x} 缩放避免 sinh/cosh 溢出（要求 Re(x) 不太负）
        # num/sinh = x - 2 cosh/sinh + 2/sinh
        sh = np.sinh(xi)
        ch = np.cosh(xi)
        if abs(sh) < 1e-300:
            F.flat[i] = (xi - 2.0) / ((L ** 2) * (mi ** 3))
        else:
            F.flat[i] = (xi - 2.0 * ch / sh + 2.0 / sh) / ((L ** 2) * (mi ** 3))
    return F


def calculate_V3omega_insulating(
    freqs: np.ndarray,
    fiber: FiberGeometry,
    material: MaterialProps,
    *,
    diameter: Optional[float] = None,
    radiation: Optional[RadiationParams] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    正演绝缘丝镀层 3ω 电压（返回 rms 复数，与锁相一致）。

    R, R' 为镀层电阻参数；κ,ρ,C_p 为纤维芯热物性；D 为纤维外径。
    """
    freqs = np.asarray(freqs, dtype=float)
    D = diameter if diameter is not None else diameter_from_area(fiber.S)
    if D <= 0:
        raise ValueError("纤维直径 D 必须 > 0")

    kappa = material.kappa
    alpha = thermal_diffusivity(kappa, material.rho, material.C_p)
    omega = 2.0 * np.pi * freqs

    eps = 0.0
    T0 = 300.0
    if radiation is not None and radiation.emissivity > 0:
        eps = radiation.emissivity
        T0 = radiation.T0
        if radiation.diameter:
            D = radiation.diameter
    hr = hr_radiation(eps, T0)
    m = complex_m(omega, alpha, kappa, D, hr)
    F = _transfer_factor(m, fiber.L)

    # I0 幅值；V3 公式给出幅值，再转 rms
    I0 = fiber.I_rms * np.sqrt(2.0)
    # 注意: R' 取绝对值进幅值，相位由复数 F 与 R' 符号共同决定
    # V3_amp ∝ I0³ R0 R' F / (k π D²)  —— F 已含几何因子
    # 完整: I0³ R R' F / (k π D²)
    pref = (I0 ** 3 * fiber.R * fiber.R_prime) / (kappa * np.pi * D ** 2)
    V3_amp = pref * F
    V3_rms = V3_amp / np.sqrt(2.0)

    # 低频/高频参考
    V3_lf_amp = (I0 ** 3 * fiber.R * abs(fiber.R_prime) * fiber.L
                 / (12.0 * kappa * np.pi * D ** 2))
    info = {
        'alpha': alpha,
        'kappa': kappa,
        'C_p': material.C_p,
        'D': D,
        'hr': hr,
        'emissivity': eps,
        'T0': T0,
        'V3_lf_amp': V3_lf_amp,
        'V3_lf_rms': V3_lf_amp / np.sqrt(2.0),
        'model': 1.0,  # tag insulating
    }
    return V3_rms, info


class InsulatingFiberFitter:
    """
    绝缘丝镀层 3ω：拟合材料 κ/C_p/ρ 和/或几何 L、D 与电学 R、α_R、V_rms。

    驱动按电压源：V_rms 独立，I_rms = V_rms / R。
    α_R 与 R 相互独立；拟合 D 时同步 S=πD²/4。
    兼容旧键 'R_prime'、'I_rms'。
    """

    MATERIAL_KEYS = frozenset({'kappa', 'C_p', 'rho'})
    FIBER_KEYS = frozenset({'L', 'D', 'R', 'alpha_R', 'R_prime', 'V_rms', 'I_rms'})

    def __init__(
        self,
        freqs: np.ndarray,
        V3_exp: np.ndarray,
        fiber: FiberGeometry,
        material0: MaterialProps,
        params_to_fit: Sequence[str] = ('kappa', 'C_p'),
        fit_mode: str = 'amp',
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        diameter: Optional[float] = None,
        radiation: Optional[RadiationParams] = None,
        alpha_R: Optional[float] = None,
        V_rms: Optional[float] = None,
    ):
        self.freqs = np.asarray(freqs, dtype=float)
        self.V3_exp = np.asarray(V3_exp, dtype=complex)
        self.fiber = FiberGeometry(
            fiber.L, fiber.S, fiber.R, fiber.R_prime, fiber.I_rms)
        self.material = MaterialProps(material0.kappa, material0.rho, material0.C_p)
        if alpha_R is not None:
            self.alpha_R = float(alpha_R)
        elif abs(fiber.R) > 1e-30:
            self.alpha_R = float(fiber.R_prime) / float(fiber.R)
        else:
            self.alpha_R = 0.0
        self.fiber.R_prime = self.alpha_R * self.fiber.R
        if V_rms is not None:
            self.V_rms = float(V_rms)
        else:
            self.V_rms = float(fiber.I_rms) * float(fiber.R)
        self._sync_I_from_V()
        if diameter is not None and diameter > 0:
            self.diameter = float(diameter)
        else:
            self.diameter = diameter_from_area(fiber.S) if fiber.S > 0 else 0.0
        self._sync_S_from_D()
        self.params_to_fit = list(params_to_fit)
        self.fit_mode = fit_mode
        self.param_bounds = param_bounds or {}
        self.radiation = radiation
        self._sync_radiation_D()
        self.last_info: Dict[str, float] = {}
        self.success = False
        self.message = ''
        self.cost = np.nan

        allowed = self.MATERIAL_KEYS | self.FIBER_KEYS
        for p in self.params_to_fit:
            if p not in allowed:
                raise ValueError(f"未知拟合参数 '{p}', 允许: {sorted(allowed)}")
        if 'alpha_R' in self.params_to_fit and 'R_prime' in self.params_to_fit:
            raise ValueError("不能同时拟合 alpha_R 与 R_prime")
        if 'V_rms' in self.params_to_fit and 'I_rms' in self.params_to_fit:
            raise ValueError("不能同时拟合 V_rms 与 I_rms")

    def _sync_R_prime(self) -> None:
        self.fiber.R_prime = self.alpha_R * self.fiber.R

    def _sync_I_from_V(self) -> None:
        R = float(self.fiber.R)
        if abs(R) < 1e-30:
            raise ValueError("R 过小，无法由 V_rms 换算 I_rms")
        self.fiber.I_rms = self.V_rms / R

    def _sync_S_from_D(self) -> None:
        if self.diameter > 0:
            self.fiber.S = 0.25 * np.pi * self.diameter ** 2

    def _sync_radiation_D(self) -> None:
        if self.radiation is not None and self.diameter > 0:
            self.radiation.diameter = self.diameter

    def _get_param(self, p: str) -> float:
        if p in self.MATERIAL_KEYS:
            return float(getattr(self.material, p))
        if p == 'alpha_R':
            return float(self.alpha_R)
        if p == 'D':
            return float(self.diameter)
        if p == 'V_rms':
            return float(self.V_rms)
        return float(getattr(self.fiber, p))

    def _set_param(self, p: str, v: float) -> None:
        if p in self.MATERIAL_KEYS:
            setattr(self.material, p, float(v))
            return
        if p == 'alpha_R':
            self.alpha_R = float(v)
            self._sync_R_prime()
            return
        if p == 'R':
            self.fiber.R = float(v)
            self._sync_R_prime()
            self._sync_I_from_V()
            return
        if p == 'R_prime':
            self.fiber.R_prime = float(v)
            if abs(self.fiber.R) > 1e-30:
                self.alpha_R = self.fiber.R_prime / self.fiber.R
            return
        if p == 'V_rms':
            self.V_rms = float(v)
            self._sync_I_from_V()
            return
        if p == 'I_rms':
            self.fiber.I_rms = float(v)
            self.V_rms = self.fiber.I_rms * self.fiber.R
            return
        if p == 'D':
            self.diameter = float(v)
            self._sync_S_from_D()
            self._sync_radiation_D()
            return
        if p == 'L':
            self.fiber.L = float(v)
            return
        setattr(self.fiber, p, float(v))

    def _pack(self) -> np.ndarray:
        return np.array([self._get_param(p) for p in self.params_to_fit], dtype=float)

    def _unpack(self, x: np.ndarray) -> None:
        for p, v in zip(self.params_to_fit, x):
            self._set_param(p, float(v))
        if 'R_prime' not in self.params_to_fit:
            self._sync_R_prime()
        if 'I_rms' not in self.params_to_fit:
            self._sync_I_from_V()

    def _bounds_vectors(self):
        defaults = {
            'kappa': (1e-3, 5e3),
            'C_p': (10.0, 5e3),
            'rho': (10.0, 3e4),
            'L': (1e-5, 0.1),
            'D': (1e-8, 1e-3),
            'R': (1e-3, 1e5),
            'alpha_R': (-0.1, 0.1),
            'R_prime': (-10.0, 10.0),
            'V_rms': (1e-6, 10.0),
            'I_rms': (1e-6, 1.0),
        }
        lo, hi = [], []
        for p in self.params_to_fit:
            if p in self.param_bounds:
                a, b = self.param_bounds[p]
            else:
                a, b = defaults[p]
                if p in ('alpha_R', 'R_prime'):
                    v0 = self._get_param(p)
                    if abs(v0) > 0:
                        a, b = (min(v0 * 0.1, v0 * 10.0), max(v0 * 0.1, v0 * 10.0))
                        if v0 < 0:
                            a, b = b, a
                elif p in ('R', 'I_rms', 'V_rms', 'L', 'D'):
                    v0 = abs(self._get_param(p)) or 1.0
                    a, b = v0 * 0.1, v0 * 10.0
            lo.append(a)
            hi.append(b)
        return np.array(lo, dtype=float), np.array(hi, dtype=float)

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        self._unpack(x)
        V3_model, info = calculate_V3omega_insulating(
            self.freqs, self.fiber, self.material,
            diameter=self.diameter, radiation=self.radiation,
        )
        self.last_info = info
        # 相对残差，避免 |V3|≪1 时 gtol 过早终止
        scale = np.maximum(np.abs(self.V3_exp), 1e-30)
        if self.fit_mode == 'amp':
            return (np.abs(self.V3_exp) - np.abs(V3_model)) / scale
        if self.fit_mode == 'phase':
            return np.unwrap(np.angle(self.V3_exp)) - np.unwrap(np.angle(V3_model))
        if self.fit_mode == 'real':
            return (self.V3_exp.real - V3_model.real) / scale
        if self.fit_mode == 'imag':
            return (self.V3_exp.imag - V3_model.imag) / scale
        return np.concatenate([
            (self.V3_exp.real - V3_model.real) / scale,
            (self.V3_exp.imag - V3_model.imag) / scale,
        ])

    def fit(self, max_nfev: int = 200, verbose: int = 0) -> MaterialProps:
        res = least_squares(
            self._residuals, self._pack(),
            bounds=self._bounds_vectors(),
            max_nfev=max_nfev, verbose=verbose,
            ftol=1e-12, xtol=1e-12, gtol=1e-12,
        )
        self._unpack(res.x)
        self.success = bool(res.success)
        self.message = res.message
        self.cost = float(res.cost)
        _, self.last_info = calculate_V3omega_insulating(
            self.freqs, self.fiber, self.material,
            diameter=self.diameter, radiation=self.radiation,
        )
        return MaterialProps(self.material.kappa, self.material.rho, self.material.C_p)

    def predict(self, freqs=None) -> np.ndarray:
        f = self.freqs if freqs is None else np.asarray(freqs, dtype=float)
        V3, info = calculate_V3omega_insulating(
            f, self.fiber, self.material,
            diameter=self.diameter, radiation=self.radiation,
        )
        self.last_info = info
        return V3

    def get_param_labels(self, params=None):
        from lu2001_fiber_model import FIBER_PARAM_LABELS
        ps = list(params) if params is not None else list(self.params_to_fit)
        return [FIBER_PARAM_LABELS.get(p, p) for p in ps]

    def calculate_sensitivity_matrix(self, params=None, rel_step: float = 1e-6):
        from lu2001_fiber_model import calculate_fiber_sensitivity_matrix
        return calculate_fiber_sensitivity_matrix(self, params=params, rel_step=rel_step)

    def compute_parameter_correlation(self, params=None):
        from lu2001_fiber_model import compute_fiber_parameter_correlation
        return compute_fiber_parameter_correlation(self, params=params)


def validity_report_insulating(
    fiber: FiberGeometry,
    material: MaterialProps,
    freqs: np.ndarray,
    diameter: Optional[float] = None,
    radiation: Optional[RadiationParams] = None,
) -> str:
    D = diameter or diameter_from_area(fiber.S)
    alpha = thermal_diffusivity(material.kappa, material.rho, material.C_p)
    eps = radiation.emissivity if radiation else 0.0
    T0 = radiation.T0 if radiation else 300.0
    hr = hr_radiation(eps, T0)
    # 特征: 辐射项 vs 扩散项量级 @ 中频
    f_mid = float(np.median(np.asarray(freqs))) if len(freqs) else 10.0
    omega = 2 * np.pi * f_mid
    rad_term = 4.0 * hr / (D * material.kappa) if material.kappa > 0 else np.nan
    diff_term = 2.0 * omega / alpha if alpha > 0 else np.nan

    lines = [
        "=== 绝缘丝镀层 3ω 适用性 / 说明 ===",
        "  模型: Xing/Munro (IMECE2015) — 镀层加热，纤维芯导热",
        f"  D = {D*1e6:.4g} μm, α = {alpha:.4e} m²/s",
        f"  h_r = {hr:.4g} W/m²K  (ε={eps:g}, T0={T0:g} K)",
        f"  @ f≈{f_mid:.4g} Hz:  2ω/α = {diff_term:.3e},  4hr/(Dk) = {rad_term:.3e}",
        "  注意: R、R' 为镀层参数；需退火稳定后再测",
        "  细长丝务必计入辐射，否则 κ 易偏高",
    ]
    return "\n".join(lines)


# 绝缘丝常用材料初值（纤维芯）
INSULATING_MATERIAL_PRESETS = {
    'glass': MaterialProps(kappa=1.38, rho=2200.0, C_p=740.0),
    'spider_silk': MaterialProps(kappa=1.2, rho=1300.0, C_p=1500.0),
    'polymer': MaterialProps(kappa=0.25, rho=1200.0, C_p=1500.0),
    'carbon_fiber_AS4': MaterialProps(kappa=6.0, rho=1790.0, C_p=700.0),
}
