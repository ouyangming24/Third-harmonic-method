"""
Xing / Munro / Ban 纤维丝 3ω 闭式模型 (IMECE2015 及 ATE 2014)

轴向一维导热 + 两端恒温热沉 + 径向辐射（写入 m²）。
导电丝与镀层绝缘丝共用同一套热学公式；镀层绝缘丝时 R、R' 取镀层，
拟合得到的 κ 为芯+膜等效轴向热导率，纤维芯 κ 需再用 Moon 2012 扣除镀层。

公式（文中式 8–10；式8 按低频极限与式9 一致补全为 I0³）:
  m² = i 2ω/α + 4 h_r / (D k)
  h_r = 4 ε σ T0³
  V3_amp = I0³ R0 R' [2-2cosh(mL)+mL sinh(mL)]
           / (L² m³ k π D² sinh(mL))
  低频: V3_amp = I0³ R0 R' L / (12 k π D²)
  高频: V3_amp = I0³ R0 R' / (i 2ω ρ C_p π D² L)

约定与 Lu GUI 一致:
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


def calculate_V3omega_xing(
    freqs: np.ndarray,
    fiber: FiberGeometry,
    material: MaterialProps,
    *,
    diameter: Optional[float] = None,
    radiation: Optional[RadiationParams] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    正演 Xing 闭式 3ω 电压（返回 rms 复数，与锁相一致）。

    κ,ρ,C_p 为等效轴向热物性（镀层绝缘丝时为芯+膜等效）；D 为外径。
    相位约定：V3 ∝ R'·F(m)；R'>0 时为热滞后（负相位），与 Lu 正演对齐。
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
    # R' 带符号；相位由复数 F 决定（R'>0 → 热滞后为负相位，与 Lu 正演对齐）
    # V3_amp ∝ I0³ R R' F / (k π D²)
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
        'model': 1.0,  # tag xing
    }
    return V3_rms, info


# 旧名兼容
calculate_V3omega_insulating = calculate_V3omega_xing


class XingFiberFitter:
    """
    Xing 闭式 3ω：拟合材料 κ/C_p/ρ 和/或几何 L、D 与电学 R、α_R、V_rms。

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
        V3_model, info = calculate_V3omega_xing(
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
        _, self.last_info = calculate_V3omega_xing(
            self.freqs, self.fiber, self.material,
            diameter=self.diameter, radiation=self.radiation,
        )
        return MaterialProps(self.material.kappa, self.material.rho, self.material.C_p)

    def predict(self, freqs=None) -> np.ndarray:
        f = self.freqs if freqs is None else np.asarray(freqs, dtype=float)
        V3, info = calculate_V3omega_xing(
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


# 旧名兼容
InsulatingFiberFitter = XingFiberFitter


def _pass_tag(ok: bool) -> str:
    return "【通过】" if ok else "【注意】"


def validity_report_xing(
    fiber: FiberGeometry,
    material: MaterialProps,
    freqs: np.ndarray,
    diameter: Optional[float] = None,
    radiation: Optional[RadiationParams] = None,
    insulating_fiber: bool = False,
) -> str:
    """Xing 模型适用性检查（编号清单；不套用 Lu 的 (10)(31)(2ωγ) 原式）。"""
    D = float(diameter or diameter_from_area(fiber.S))
    r = 0.5 * D if D > 0 else np.nan
    S = 0.25 * np.pi * D * D if D > 0 else np.nan
    L = float(fiber.L)
    kappa = float(material.kappa)
    alpha = thermal_diffusivity(kappa, material.rho, material.C_p)
    eps = float(radiation.emissivity) if radiation else 0.0
    T0 = float(radiation.T0) if radiation else 300.0
    hr = hr_radiation(eps, T0)

    freqs = np.asarray(freqs, dtype=float)
    if freqs.size == 0:
        freqs = np.array([10.0])
    f_min = float(np.min(freqs))
    f_mid = float(np.median(freqs))
    f_max = float(np.max(freqs))

    def _terms(f):
        om = 2.0 * np.pi * f
        diff = 2.0 * om / alpha if alpha > 0 else np.nan
        rad = 4.0 * hr / (D * kappa) if (D > 0 and kappa > 0) else np.nan
        m = complex_m(np.array([om]), alpha, kappa, D, hr)[0]
        ml = m * L
        return diff, rad, m, ml

    diff_mid, rad_mid, _, ml_mid = _terms(f_mid)
    _, _, _, ml_min = _terms(f_min)
    _, _, _, ml_max = _terms(f_max)
    ratio_mid = (rad_mid / diff_mid) if (np.isfinite(rad_mid) and np.isfinite(diff_mid)
                                         and abs(diff_mid) > 0) else np.nan
    slender = (L / D) if D > 0 else np.nan
    rad_on = eps > 0 and hr > 0

    # 2. 小温升
    I0 = float(fiber.I_rms) * np.sqrt(2.0)
    if kappa > 0 and S > 0 and L > 0:
        dT_avg = (I0 ** 2 * fiber.R * L) / (24.0 * kappa * S)
        lin_analog = (I0 ** 2 * abs(fiber.R_prime) * L) / (np.pi ** 2 * kappa * S)
    else:
        dT_avg = np.nan
        lin_analog = np.nan
    ok_dT = np.isfinite(dT_avg) and dT_avg < 5.0
    ok_lin = np.isfinite(lin_analog) and lin_analog < 0.1

    ok_rad = True
    if np.isfinite(slender) and slender >= 50.0 and not rad_on:
        ok_rad = False
    elif np.isfinite(ratio_mid) and ratio_mid > 0.05 and not rad_on:
        ok_rad = False

    f_rad_lim = (alpha / (2.0 * np.pi * r * r)) if (alpha > 0 and r > 0) else np.nan
    frac_radial = float(np.mean(freqs < 0.1 * f_rad_lim)) if np.isfinite(f_rad_lim) else np.nan
    ok_radial = np.isfinite(frac_radial) and frac_radial >= 0.8

    abs_ml_min = abs(ml_min)
    abs_ml_mid = abs(ml_mid)
    abs_ml_max = abs(ml_max)

    sample_note = (
        "样品: 绝缘纤维（镀层加热/测温）；本检查中的 κ 为拟合等效 κ_tot。"
        if insulating_fiber else
        "样品: 本体传感（或未勾选绝缘）；本检查中的 κ 为模型拟合轴向热导率。"
    )

    lines = [
        "=== Xing 闭式 3ω 适用性检查 ===",
        "模型: Xing/Munro (IMECE2015) — 全模态 + 辐射写入 m²。",
        "说明: 与 Lu 模型同类一维物理，但不用 (10)/(31)/2ωγ 截断判据。",
        f"      {sample_note}",
        "",
        "1. 几何与热物性摘要",
        f"   D = {D * 1e6:.4g} μm，L = {L * 1e3:.4g} mm，L/D = {slender:.4g}",
        f"   κ = {kappa:.4g} W/(m·K)，ρ = {material.rho:.4g} kg/m³，"
        f"C_p = {material.C_p:.4g} J/(kg·K)",
        f"   α = κ/(ρ C_p) = {alpha:.4e} m²/s",
        f"   频率: {f_min:.4g} – {f_max:.4g} Hz（中位 {f_mid:.4g} Hz，共 {freqs.size} 点）",
        "",
        "2. 小温升 / 线性化（类比 Lu 条件(10)，公式已改）",
        "   要求: 直流/平均温升不宜过大，电阻反馈可忽略。",
        f"   估计平均温升 ΔT ≈ I0² R L / (24 κ S) = {dT_avg:.3g} K"
        f"   （建议 ≲ 1–5 K）  {_pass_tag(ok_dT)}",
        f"   反馈量级 I0²|R'|L/(π²κS) = {lin_analog:.3e}  （建议 ≪ 0.1）  "
        f"{_pass_tag(ok_lin)}",
        "   不通过时: 减小驱动电流/电压。",
        "",
        "3. 侧向辐射（写入 m²，非 Lu 的 gγ 表观修正）",
        "   m² = i 2ω/α + 4 h_r/(D κ)，h_r = 4 ε σ T0³。",
        f"   ε = {eps:g}，T0 = {T0:g} K，h_r = {hr:.4g} W/(m²·K)",
        f"   中位频率: 2ω/α = {diff_mid:.3e}，4 h_r/(Dκ) = {rad_mid:.3e}，"
        f"辐射/扩散 ≈ {ratio_mid:.3e}",
        f"   细长比 L/D = {slender:.4g}；辐射已计入: {'是' if rad_on else '否'}",
        f"   {_pass_tag(ok_rad)}  细长丝（L/D ≳ 50）或辐射项不可忽略时必须开 ε。",
        "   关闭辐射时 κ 容易偏高。",
        "",
    ]

    if insulating_fiber:
        lines.extend([
            "4. 镀层几何（详细数值见下方 Moon 专节）",
            "   要求: t ≪ D（建议 t/D<0.01）；并联权重 βλ_film/λ_tot≥30%【注意】、≥50%【不适用】。",
            "   λ_film 按厚度档选用（<50 nm 薄膜 / ≥200 nm 近块体或 WF）。",
            "   【说明】  勾选绝缘后，请查看「镀层 / Moon 2012 检查」。",
            "",
        ])
    else:
        lines.extend([
            "4. 镀层",
            "   未勾选绝缘纤维：不按镀层样品处理，跳过 t/D 与 Moon 扣除。",
            "",
        ])

    lines.extend([
        "5. 径向近似均匀",
        "   要求: 径向扩散跟得上 2ω 加热，f ≪ α/(2π r²)"
        + ("（绝缘丝宜用径向 α 估上限）" if insulating_fiber else "") + "。",
        f"   r = D/2 = {r * 1e6:.4g} μm",
        f"   径向时间 τ_r ~ r²/α = {(r * r / alpha) if alpha > 0 else np.nan:.3e} s",
        f"   建议 f ≪ {f_rad_lim:.4g} Hz；当前频点落在 0.1×该上限以内的比例: "
        f"{(frac_radial * 100) if np.isfinite(frac_radial) else float('nan'):.0f}%",
        f"   {_pass_tag(ok_radial)}  高频点过多时径向均匀假设变差。",
        "",
        "6. 频率窗口 |mL|（Xing 不用 2ωγ≲4）",
        "   |mL|≪1 为低频平台（偏 κ），≫1 为高频 1/ω（偏 C_p）。",
        f"   |mL|(fmin) = {abs_ml_min:.3g}，|mL|(中位) = {abs_ml_mid:.3g}，"
        f"|mL|(fmax) = {abs_ml_max:.3g}",
        "   【说明】  同时拟合 κ 与 C_p 时，频段最好覆盖 |mL| 从 <1 到 >1。",
        "",
        "7. 两端为热沉  ΔT(0)=ΔT(L)=0",
        "   要求: 焊盘/热沉导热 ≫ 纤维轴向热导 κS/L。",
        "   【需人工确认】  程序无法从 V3 曲线自动验证接点质量。",
        "",
        "8. 电学参数 R、R'（= α_R·R）",
    ])
    if insulating_fiber:
        lines.extend([
            "   要求: R、α_R 属于镀层，须退火稳定后再测。",
            f"   当前 R = {fiber.R:.4g} Ω，R' = {fiber.R_prime:.4g} Ω/K",
            "   【需人工确认】  未稳定的镀层电阻会使 κ_tot 系统偏差。",
        ])
    else:
        lines.extend([
            "   要求: R、α_R 属于丝本体传感器。",
            f"   当前 R = {fiber.R:.4g} Ω，R' = {fiber.R_prime:.4g} Ω/K",
            "   【需人工确认】  标定须线性、可重复。",
        ])
    lines.extend([
        "",
        "9. 总评（Xing 热学）",
        f"   自动项: 温升 {_pass_tag(ok_dT)}，线性化 {_pass_tag(ok_lin)}，"
        f"辐射 {_pass_tag(ok_rad)}，径向 {_pass_tag(ok_radial)}。",
        "   标记【注意】不表示拟合一定失败，只表示该条假设可能被动摇。",
    ])
    return "\n".join(lines)


# 旧名兼容
validity_report_insulating = validity_report_xing

# Xing / 绝缘芯常用材料初值
XING_MATERIAL_PRESETS = {
    'glass': MaterialProps(kappa=1.38, rho=2200.0, C_p=740.0),
    'spider_silk': MaterialProps(kappa=1.2, rho=1300.0, C_p=1500.0),
    'polymer': MaterialProps(kappa=0.25, rho=1200.0, C_p=1500.0),
    'carbon_fiber_AS4': MaterialProps(kappa=6.0, rho=1790.0, C_p=700.0),
}
INSULATING_MATERIAL_PRESETS = XING_MATERIAL_PRESETS
