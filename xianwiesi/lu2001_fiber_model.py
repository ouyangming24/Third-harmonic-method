"""
Lu et al. (2001) 纤维丝 3ω 模型 — 基于论文原文实现

L. Lu, W. Yi, D. L. Zhang,
"3ω method for specific heat and thermal conductivity measurements"
Rev. Sci. Instrum. 72, 2996 (2001).

主要公式:
  α = κ / (ρ C_p)
  γ = L² / (π² α)                         热时间常数
  V_{3ω} ≈ 4 I³ L R R' / (π⁴ κ S √(1+(2ωγ)²))   式(19), I/V 为 rms
  tan φ ≈ 2 ω γ                           式(18)
  C_p = π² γ κ / (ρ L²)                   式(20)

可选修正式(24)（2ωγ ≤ 4）降低截断 n>1 误差。
径向辐射: gγ≪1 时忽略; 否则用表观 κ_ap、γ_ap（式 29–33）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares


SIGMA_SB = 5.67e-8  # Stefan–Boltzmann W/m²K⁴


@dataclass
class FiberGeometry:
    """几何与电学参数（Lu 四线悬空段）。"""
    L: float          # 电压触点间距 [m]
    S: float          # 横截面积 [m²]
    R: float          # 电阻 [Ω]（衬底温度 T0）
    R_prime: float    # dR/dT [Ω/K]
    I_rms: float      # 电流有效值 [A]


@dataclass
class MaterialProps:
    """材料热物性。"""
    kappa: float      # 热导率 [W/m·K]
    rho: float        # 密度 [kg/m³]
    C_p: float        # 比热容 [J/kg·K]


@dataclass
class RadiationParams:
    """圆柱径向辐射（论文 Sec. V）。ε=0 表示忽略。"""
    emissivity: float = 0.0
    T0: float = 300.0       # [K]
    diameter: Optional[float] = None  # [m]; None 则由 S 推等直径


def thermal_diffusivity(kappa: float, rho: float, C_p: float) -> float:
    return kappa / (rho * C_p)


def thermal_time_constant(L: float, alpha: float) -> float:
    """γ = L² / (π² α)"""
    return L ** 2 / (np.pi ** 2 * alpha)


def specific_heat_from_gamma(gamma: float, kappa: float, rho: float, L: float) -> float:
    """式(20): C_p = π² γ κ / (ρ L²)"""
    return np.pi ** 2 * gamma * kappa / (rho * L ** 2)


def radiation_g(rho: float, C_p: float, D: float, emissivity: float, T0: float) -> float:
    """g = 16 ε σ T0³ / (ρ C_p D)  （圆柱）"""
    if emissivity <= 0 or D <= 0:
        return 0.0
    return 16.0 * emissivity * SIGMA_SB * (T0 ** 3) / (rho * C_p * D)


def diameter_from_area(S: float) -> float:
    return float(np.sqrt(4.0 * S / np.pi))


def condition_10(fiber: FiberGeometry, kappa: float, n: int = 1) -> float:
    """式(10) 左侧: I0² R' L / (n² π² κ S)。I0 = √2 I_rms。应 ≪ 1。"""
    I0 = fiber.I_rms * np.sqrt(2.0)
    return (I0 ** 2 * abs(fiber.R_prime) * fiber.L) / (n ** 2 * np.pi ** 2 * kappa * fiber.S)


def condition_31_g_gamma(g: float, gamma: float) -> float:
    """式(31): gγ，应 ≪ 1。"""
    return g * gamma


def calculate_V3omega(
    freqs: np.ndarray,
    fiber: FiberGeometry,
    material: MaterialProps,
    *,
    use_eq24: bool = False,
    radiation: Optional[RadiationParams] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    正演复数 V3ω（rms）。

    相位约定（论文 Sec. VI，相对 1ω 置零后）:
      R' < 0 → 相位 = -φ
      R' > 0 → 相位 = π - φ
    其中 tan φ = 2 ω γ_ap，φ = arctan(2 ω γ_ap) ∈ (0, π/2)。

    Returns
    -------
    V3 : complex ndarray
    info : dict 含 alpha, gamma, kappa_used, gamma_used, g, g_gamma, Delta0 等
    """
    freqs = np.asarray(freqs, dtype=float)
    kappa = material.kappa
    rho = material.rho
    C_p = material.C_p
    alpha = thermal_diffusivity(kappa, rho, C_p)
    gamma = thermal_time_constant(fiber.L, alpha)

    g = 0.0
    if radiation is not None and radiation.emissivity > 0:
        D = radiation.diameter if radiation.diameter else diameter_from_area(fiber.S)
        g = radiation_g(rho, C_p, D, radiation.emissivity, radiation.T0)
        # 式(29)(30): 表观参数
        kappa_ap = (1.0 + g * gamma) * kappa
        gamma_ap = gamma / (1.0 + g * gamma)
    else:
        kappa_ap = kappa
        gamma_ap = gamma

    omega = 2.0 * np.pi * freqs
    two_omega_gamma = 2.0 * omega * gamma_ap
    denom = np.sqrt(1.0 + two_omega_gamma ** 2)

    # 式(19) 幅值前因子（用表观 κ）
    pref = (4.0 * fiber.I_rms ** 3 * fiber.L * fiber.R * abs(fiber.R_prime)
            / (np.pi ** 4 * kappa_ap * fiber.S))

    if use_eq24:
        # 式(24): 仅建议 2ωγ ≤ 4
        amp = (pref / 1.01) * (1.0 / denom + 0.01)
    else:
        amp = pref / denom

    phi = np.arctan(two_omega_gamma)  # ∈ (0, π/2)
    if fiber.R_prime < 0:
        phase = -phi
    else:
        phase = np.pi - phi

    V3 = amp * np.exp(1j * phase)

    # Δ0 = 2 I0² R / (π κ S / L) = 4 I_rms² R L / (π κ S)  （论文定义，用真实 κ）
    I0 = fiber.I_rms * np.sqrt(2.0)
    Delta0 = 2.0 * I0 ** 2 * fiber.R / (np.pi * kappa * fiber.S / fiber.L)

    info = {
        'alpha': alpha,
        'gamma': gamma,
        'kappa': kappa,
        'kappa_ap': kappa_ap,
        'gamma_ap': gamma_ap,
        'C_p': C_p,
        'g': g,
        'g_gamma': g * gamma,
        'Delta0': Delta0,
        'cond10': condition_10(fiber, kappa),
        'use_eq24': float(use_eq24),
    }
    return V3, info


def high_freq_limit_V3_amp(freqs, fiber: FiberGeometry, material: MaterialProps) -> np.ndarray:
    """式(23) 高频极限幅值（Holland）: I³ R R' / (4 ω ρ C_p L S)"""
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    return (fiber.I_rms ** 3 * fiber.R * abs(fiber.R_prime)
            / (4.0 * omega * material.rho * material.C_p * fiber.L * fiber.S))


def low_freq_limit_V3_amp(fiber: FiberGeometry, kappa: float) -> float:
    """式(22) 低频极限: 4 I³ R R' L / (π⁴ κ S)"""
    return (4.0 * fiber.I_rms ** 3 * fiber.R * abs(fiber.R_prime) * fiber.L
            / (np.pi ** 4 * kappa * fiber.S))


class FiberModelFitter:
    """
    非线性最小二乘拟合 κ 和/或 C_p（或等价地 κ 与 γ）。

    拟合参数可选: 'kappa', 'C_p', 'rho'（少见）。
    内部用 κ、C_p 正演；γ 由 α 推出。
    """

    def __init__(
        self,
        freqs: np.ndarray,
        V3_exp: np.ndarray,
        fiber: FiberGeometry,
        material0: MaterialProps,
        params_to_fit: Sequence[str] = ('kappa', 'C_p'),
        fit_mode: str = 'amp',
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        use_eq24: bool = False,
        radiation: Optional[RadiationParams] = None,
        freq_window_2wgamma_max: float = 4.0,
    ):
        self.freqs = np.asarray(freqs, dtype=float)
        self.V3_exp = np.asarray(V3_exp, dtype=complex)
        self.fiber = fiber
        self.material = MaterialProps(material0.kappa, material0.rho, material0.C_p)
        self.params_to_fit = list(params_to_fit)
        self.fit_mode = fit_mode
        self.param_bounds = param_bounds or {}
        self.use_eq24 = use_eq24
        self.radiation = radiation
        self.freq_window_2wgamma_max = freq_window_2wgamma_max
        self.last_info: Dict[str, float] = {}
        self.success = False
        self.message = ''
        self.cost = np.nan

        allowed = {'kappa', 'C_p', 'rho'}
        for p in self.params_to_fit:
            if p not in allowed:
                raise ValueError(f"未知拟合参数 '{p}', 允许: {allowed}")
        if fit_mode not in ('amp', 'phase', 'both', 'real', 'imag'):
            raise ValueError("fit_mode 须为 amp|phase|both|real|imag")

    def _pack(self) -> np.ndarray:
        return np.array([getattr(self.material, p) for p in self.params_to_fit], dtype=float)

    def _unpack(self, x: np.ndarray) -> None:
        for p, v in zip(self.params_to_fit, x):
            setattr(self.material, p, float(v))

    def _bounds_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = [], []
        defaults = {
            'kappa': (1e-3, 5e3),
            'C_p': (10.0, 5e3),
            'rho': (10.0, 3e4),
        }
        for p in self.params_to_fit:
            if p in self.param_bounds:
                a, b = self.param_bounds[p]
            else:
                a, b = defaults[p]
            lo.append(a)
            hi.append(b)
        return np.array(lo), np.array(hi)

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        self._unpack(x)
        V3_model, info = calculate_V3omega(
            self.freqs, self.fiber, self.material,
            use_eq24=self.use_eq24, radiation=self.radiation,
        )
        self.last_info = info
        scale = np.maximum(np.abs(self.V3_exp), 1e-30)

        if self.fit_mode == 'amp':
            r = (np.abs(self.V3_exp) - np.abs(V3_model)) / scale
        elif self.fit_mode == 'phase':
            r = np.unwrap(np.angle(self.V3_exp)) - np.unwrap(np.angle(V3_model))
        elif self.fit_mode == 'real':
            r = (self.V3_exp.real - V3_model.real) / scale
        elif self.fit_mode == 'imag':
            r = (self.V3_exp.imag - V3_model.imag) / scale
        else:  # both
            r = np.concatenate([
                (self.V3_exp.real - V3_model.real) / scale,
                (self.V3_exp.imag - V3_model.imag) / scale,
            ])
        return r

    def fit(self, max_nfev: int = 200, verbose: int = 0) -> MaterialProps:
        x0 = self._pack()
        lo, hi = self._bounds_vectors()
        res = least_squares(
            self._residuals, x0, bounds=(lo, hi),
            max_nfev=max_nfev, verbose=verbose,
            ftol=1e-12, xtol=1e-12, gtol=1e-12,
        )
        self._unpack(res.x)
        self.success = bool(res.success)
        self.message = res.message
        self.cost = float(res.cost)
        # 刷新 info
        _, self.last_info = calculate_V3omega(
            self.freqs, self.fiber, self.material,
            use_eq24=self.use_eq24, radiation=self.radiation,
        )
        return MaterialProps(self.material.kappa, self.material.rho, self.material.C_p)

    def predict(self, freqs: Optional[np.ndarray] = None) -> np.ndarray:
        f = self.freqs if freqs is None else np.asarray(freqs, dtype=float)
        V3, info = calculate_V3omega(
            f, self.fiber, self.material,
            use_eq24=self.use_eq24, radiation=self.radiation,
        )
        self.last_info = info
        return V3


def validity_report(
    fiber: FiberGeometry,
    material: MaterialProps,
    freqs: np.ndarray,
    radiation: Optional[RadiationParams] = None,
) -> str:
    """生成适用性检查文本（式 10、31、2ωγ<4）。"""
    alpha = thermal_diffusivity(material.kappa, material.rho, material.C_p)
    gamma = thermal_time_constant(fiber.L, alpha)
    c10 = condition_10(fiber, material.kappa)
    g = 0.0
    if radiation and radiation.emissivity > 0:
        D = radiation.diameter or diameter_from_area(fiber.S)
        g = radiation_g(material.rho, material.C_p, D, radiation.emissivity, radiation.T0)
    g_gamma = g * gamma
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    two_wg = 2.0 * omega * gamma
    f_max_rec = None
    if gamma > 0:
        # 2ωγ < 4 → f < 4/(4π γ) = 1/(π γ)
        f_max_rec = 1.0 / (np.pi * gamma)

    lines = [
        "=== Lu 2001 模型适用性检查 ===",
        f"  α = {alpha:.4e} m²/s",
        f"  γ = {gamma:.4e} s",
        f"  条件(10) I0²|R'|L/(π²κS) = {c10:.3e}  （应 ≪ 1）"
        f"  {'✅' if c10 < 0.1 else '⚠️'}",
        f"  条件(31) gγ = {g_gamma:.3e}  （应 ≪ 1；ε=0 时为 0）"
        f"  {'✅' if g_gamma < 0.1 else '⚠️'}",
    ]
    if f_max_rec:
        lines.append(f"  建议 f 上限 (2ωγ<4): f ≲ {f_max_rec:.4g} Hz")
        in_win = float(np.mean(two_wg <= 4.0))
        lines.append(f"  当前频点落在 2ωγ≤4 的比例: {in_win*100:.0f}%")
        lines.append(f"  当前 2ωγ 范围: [{two_wg.min():.3g}, {two_wg.max():.3g}]")
    Delta0 = 2.0 * (fiber.I_rms * np.sqrt(2)) ** 2 * fiber.R / (
        np.pi * material.kappa * fiber.S / fiber.L)
    lines.append(f"  中心直流温升 Δ0 ≈ {Delta0:.3g} K  （论文建议约 ~1 K）")
    return "\n".join(lines)


# 常用预设
FIBER_PRESETS: Dict[str, dict] = {
    'platinum_20um_Lu': {
        # 论文 Sec.VI: D=20μm, L=8mm
        'L': 8e-3,
        'D': 20e-6,
        'R': 5.0,          # 需实测覆盖
        'R_prime': 0.01,
        'I_rms': 5e-3,
    },
    'custom': {
        'L': 1e-3,
        'D': 10e-6,
        'R': 100.0,
        'R_prime': 0.1,
        'I_rms': 1e-3,
    },
}

MATERIAL_PRESETS: Dict[str, MaterialProps] = {
    'platinum': MaterialProps(kappa=71.6, rho=21450.0, C_p=133.0),
    'carbon_nanotube': MaterialProps(kappa=300.0, rho=1300.0, C_p=700.0),
    'copper': MaterialProps(kappa=401.0, rho=8960.0, C_p=385.0),
    'tungsten': MaterialProps(kappa=173.0, rho=19300.0, C_p=132.0),
}
