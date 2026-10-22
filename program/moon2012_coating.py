"""
Moon et al. (2012) 镀层绝缘纤维：等效热导率 → 纤维芯热导率校正

J. Moon et al., Rev. Sci. Instrum. 83, 016103 (2012).

稳态/3ω 一维模型拟合得到的是芯+膜并联的等效轴向热导率 λ_tot。
蒸发镀膜通常只盖住上半周，截面积比：

  β = (π D δ / 2) / (π D² / 4) = 2 δ / D     （半周蒸发，Moon 原文）
  β = (π D δ)     / (π D² / 4) = 4 δ / D     （整周共形镀）

  λ_tot = β λ_film + (1 − β) λ_CF
  λ_CF  = (λ_tot − β λ_film) / (1 − β)

薄膜金属热导率远低于块体；Moon 对 10 nm e-beam Pt 取 15 W/(m·K)。
厚膜（≳电子平均自由程）应接近块体或用 WF 自测，勿套用 10 nm 薄膜值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# 覆盖模式
COVERAGE_HALF = "half"          # 半周蒸发（Moon 原文）
COVERAGE_CONFORMAL = "conformal"  # 整周共形

# 适用性软/硬门槛（经验；非论文硬性公式）
T_OVER_D_SOFT = 0.01            # t/D < 1%：薄镀层假设较稳
BETA_WARN = 0.05                # β 已不小
# 镀层并联导热占比 f = β λ_film / λ_tot
FILM_SHARE_WARN = 0.30          # ≥30%：【注意】κ_CF 不确定度大
FILM_SHARE_UNFIT = 0.50         # ≥50%：【不适用】大数相减，芯导热不可靠
# 厚度分档（相对贵金属电子 MFP ~40–50 nm）
DELTA_THIN_M = 50e-9            # <50 nm：薄膜散射主导
DELTA_THICK_M = 200e-9          # ≥200 nm：宜近块体 / WF


@dataclass
class CoatingParams:
    """镀层几何与热导率。"""
    thickness: float          # δ [m]
    kappa_film: float         # λ_film [W/m·K]
    coverage: str = COVERAGE_HALF


@dataclass
class MoonCorrectionResult:
    """Moon 校正结果。"""
    kappa_tot: float          # 拟合等效热导率
    kappa_fiber: float        # 纤维芯热导率 λ_CF
    beta: float               # 截面积比
    kappa_film: float
    thickness: float
    diameter: float
    coverage: str
    relative_deviation: float  # (λ_tot − λ_CF) / λ_tot
    film_thermal_share: float = 0.0  # β λ_film / λ_tot
    note: str = ""


def coating_area_fraction(
    diameter: float,
    thickness: float,
    coverage: str = COVERAGE_HALF,
) -> float:
    """β：镀层截面积 / 纤维外接圆截面积。"""
    if diameter <= 0:
        raise ValueError("纤维直径 D 必须 > 0")
    if thickness < 0:
        raise ValueError("镀层厚度 δ 不能为负")
    if coverage == COVERAGE_HALF:
        # Moon: β = (π D δ / 2) / (π D² / 4) = 2δ/D
        return 2.0 * thickness / diameter
    if coverage == COVERAGE_CONFORMAL:
        # 整周薄壳: β = (π D δ) / (π D² / 4) = 4δ/D（δ≪D）
        return 4.0 * thickness / diameter
    raise ValueError(f"未知覆盖模式 '{coverage}'，应为 '{COVERAGE_HALF}' 或 '{COVERAGE_CONFORMAL}'")


def correct_kappa_moon(
    kappa_tot: float,
    diameter: float,
    coating: CoatingParams,
) -> MoonCorrectionResult:
    """
    由拟合等效热导率 λ_tot 扣除镀层并联贡献，得到纤维芯 λ_CF。

    若 β≥1 或校正后 λ_CF≤0，抛出 ValueError。
    """
    beta = coating_area_fraction(diameter, coating.thickness, coating.coverage)
    if beta >= 1.0:
        raise ValueError(
            f"镀层截面积比 β={beta:.3g} ≥ 1（δ={coating.thickness*1e9:.4g} nm, "
            f"D={diameter*1e6:.4g} μm），无法扣除。请检查厚度/直径。")
    if beta <= 0.0:
        return MoonCorrectionResult(
            kappa_tot=float(kappa_tot),
            kappa_fiber=float(kappa_tot),
            beta=0.0,
            kappa_film=float(coating.kappa_film),
            thickness=float(coating.thickness),
            diameter=float(diameter),
            coverage=coating.coverage,
            relative_deviation=0.0,
            film_thermal_share=0.0,
            note="δ=0，未做镀层扣除。",
        )

    kappa_cf = (float(kappa_tot) - beta * float(coating.kappa_film)) / (1.0 - beta)
    if kappa_cf <= 0:
        raise ValueError(
            f"扣除镀层后 λ_CF={kappa_cf:.4g} ≤ 0。"
            f"λ_tot={kappa_tot:.4g}, β={beta:.4g}, λ_film={coating.kappa_film:.4g}。"
            "请检查镀层热导率或厚度是否偏大。")

    kt = float(kappa_tot)
    share = (beta * float(coating.kappa_film) / kt) if abs(kt) > 1e-30 else 0.0
    # 相对 (λ_tot−λ_CF)/λ_tot：与「并联权重」share 不同，一并保留便于对照
    dev = (kt - kappa_cf) / kt if abs(kt) > 1e-30 else 0.0
    cov_cn = "半周蒸发" if coating.coverage == COVERAGE_HALF else "整周共形"
    note = (
        f"Moon 2012：β={beta:.4g}（{cov_cn}），"
        f"λ_tot={kappa_tot:.6g} → λ_CF={kappa_cf:.6g} W/(m·K)，"
        f"镀层并联权重 βλ_film/λ_tot={(share*100):.2f}%。"
    )
    return MoonCorrectionResult(
        kappa_tot=float(kappa_tot),
        kappa_fiber=float(kappa_cf),
        beta=float(beta),
        kappa_film=float(coating.kappa_film),
        thickness=float(coating.thickness),
        diameter=float(diameter),
        coverage=coating.coverage,
        relative_deviation=float(dev),
        film_thermal_share=float(share),
        note=note,
    )


def moon_guidance(kappa_fiber: float) -> str:
    """Moon 文中关于何时必须扣除镀层的经验判据。"""
    if kappa_fiber > 2.0:
        return "芯 λ≳2 W/(m·K)：Moon 认为镀层影响通常可忽略。"
    if kappa_fiber >= 0.2:
        return "芯 λ~0.2 W/(m·K) 量级：镀层引入误差通常 ≲10%。"
    if kappa_fiber >= 0.1:
        return "芯 λ~0.1–0.2：建议报告扣除前后两个值。"
    return "芯 λ≲0.1：Moon 建议必须扣除镀层并联导热后再报告 λ_CF。"


def film_thickness_regime(delta_m: float) -> str:
    """按厚度分档：thin / intermediate / thick。"""
    if delta_m < DELTA_THIN_M:
        return "thin"
    if delta_m < DELTA_THICK_M:
        return "intermediate"
    return "thick"


def lambda_film_advice(delta_m: float, kappa_film: float) -> Tuple[str, bool]:
    """
    按 δ 给出 λ_film 取值建议；返回 (文案, 是否与厚度档明显不符)。

    不符：薄δ却用近块体值，或厚δ却用 ~10 nm 薄膜值。
    """
    regime = film_thickness_regime(delta_m)
    d_nm = delta_m * 1e9
    mismatch = False
    if regime == "thin":
        tip = (
            f"δ={d_nm:.4g} nm < 50 nm：属薄膜区，λ_film 远低于块体；"
            "Moon 对 10 nm e-beam Pt 取 15；Au 膜常见约 30–130（初猜约 50）。"
            "宜用同工艺文献或 WF（注意纳米膜 Lorenz 数可能偏离块体）。"
        )
        if kappa_film >= 200.0:
            mismatch = True
            tip += f" 当前 λ_film={kappa_film:.4g} 偏块体，对薄膜可能偏高。"
    elif regime == "intermediate":
        tip = (
            f"δ={d_nm:.4g} nm（50–200 nm）：过渡区，λ_film 介于薄膜与块体之间；"
            "优先用电阻 + WF（L≈2.45e-8）自测，勿直接套 15 或块体表。"
        )
        if kappa_film <= 25.0 or kappa_film >= 400.0:
            mismatch = True
            tip += f" 当前 λ_film={kappa_film:.4g} 与过渡区常见范围不太相符，请核对。"
    else:
        tip = (
            f"δ={d_nm:.4g} nm ≥ 200 nm：近块体区，λ_film 宜接近块体或 WF 自测"
            "（Ag 约 300–429，Au 约 200–317）；勿再用 10 nm 薄膜预设 15/50。"
        )
        if kappa_film <= 80.0:
            mismatch = True
            tip += f" 当前 λ_film={kappa_film:.4g} 仍像纳米薄膜，对厚镀层可能严重偏低。"
    return tip, mismatch


def film_share_verdict(share: float) -> Tuple[str, str]:
    """
    镀层并联权重 f=βλ_film/λ_tot 的判定。

    返回 (标签, 说明)：【通过】/【注意】/【不适用】
    """
    if share >= FILM_SHARE_UNFIT:
        return (
            "【不适用】",
            f"镀层并联权重 βλ_film/λ_tot = {share*100:.1f}% ≥ {FILM_SHARE_UNFIT*100:.0f}%："
            "热量几乎走金属壳，κ_CF 为大数相减，不建议报告芯导热。"
            "可减薄镀层、改低导热膜，或只报告 κ_tot。",
        )
    if share >= FILM_SHARE_WARN:
        return (
            "【注意】",
            f"镀层并联权重 βλ_film/λ_tot = {share*100:.1f}% ≥ {FILM_SHARE_WARN*100:.0f}%："
            "κ_CF 对 δ、覆盖方式与 λ_film 极敏感，请做参数扫描并谨慎引用。",
        )
    return (
        "【通过】",
        f"镀层并联权重 βλ_film/λ_tot = {share*100:.1f}% "
        f"< {FILM_SHARE_WARN*100:.0f}%：扣膜后芯导热尚可分辨。",
    )


def validity_report_coating(
    diameter: float,
    coating: Optional[CoatingParams] = None,
    kappa_tot: Optional[float] = None,
    insulating: bool = True,
) -> str:
    """
    绝缘镀层 / Moon 校正适用性检查。

    insulating=False 时仅提示「未勾选，跳过」。
    kappa_tot 若给出，会试算 λ_CF、镀层并联权重，并给出硬门槛判定。
    """
    if not insulating:
        return (
            "=== 镀层 / Moon 2012 检查 ===\n"
            "未勾选「绝缘纤维（镀层）」：不扣除膜并联导热，"
            "报告 κ 即模型拟合的等效轴向热导率。\n"
        )

    lines = [
        "=== 镀层 / Moon 2012 检查 ===",
        "绝缘纤维：模型拟合的是 κ_tot（芯+膜）；最终报告 κ_CF 需 Moon 扣除。",
        f"   D = {diameter * 1e6:.4g} μm",
        "   门槛: t/D<0.01；β≥0.05 提醒；"
        f"并联权重 βλ_film/λ_tot ≥{FILM_SHARE_WARN*100:.0f}%【注意】、"
        f"≥{FILM_SHARE_UNFIT*100:.0f}%【不适用】。",
    ]

    if coating is None:
        lines.extend([
            "   【注意】 未提供镀层参数，无法自动算 β / κ_CF。",
            "   请填写 δ、λ_film 与覆盖方式后再检查或拟合。",
        ])
        return "\n".join(lines)

    try:
        beta = coating_area_fraction(diameter, coating.thickness, coating.coverage)
    except ValueError as err:
        return "\n".join(lines + [f"   【注意】 {err}"])

    cov_cn = "半周蒸发 β=2δ/D" if coating.coverage == COVERAGE_HALF else "整周共形 β=4δ/D"
    t_over_D = coating.thickness / diameter if diameter > 0 else float("nan")
    ok_thin = bool(t_over_D < T_OVER_D_SOFT)
    tag = lambda ok: "【通过】" if ok else "【注意】"
    regime = film_thickness_regime(coating.thickness)
    regime_cn = {"thin": "薄膜", "intermediate": "过渡", "thick": "近块体"}[regime]
    tip_film, mismatch_film = lambda_film_advice(coating.thickness, coating.kappa_film)

    lines.extend([
        f"   δ = {coating.thickness * 1e9:.4g} nm，λ_film = {coating.kappa_film:.4g} W/(m·K)",
        f"   覆盖: {cov_cn}，β = {beta:.4g}，t/D = {t_over_D:.3e}",
        f"   厚度档: {regime_cn}（<50 nm 薄膜 / 50–200 nm 过渡 / ≥200 nm 近块体）",
        f"   薄镀层 t/D ≪ 1（建议 < {T_OVER_D_SOFT}）  {tag(ok_thin)}",
    ])
    if not ok_thin:
        lines.append(
            f"   【注意】 t/D={t_over_D:.3e} ≥ {T_OVER_D_SOFT}："
            "一维「薄壳」假设偏弱，β 公式与芯径向均匀性更可疑。"
        )
    if beta >= BETA_WARN:
        lines.append(
            f"   【注意】 β={beta:.3g} ≥ {BETA_WARN}，镀层截面积已不小，必须做 Moon 扣除。"
        )
    elif beta <= 0:
        lines.append("   【注意】 δ=0 → β=0，Moon 扣除无实际效果。")

    lines.append(f"   λ_film 建议: {tip_film}")
    if mismatch_film:
        lines.append("   【注意】 当前 λ_film 与厚度档明显不符，请按建议改值或 WF 自测。")

    # 无 κ_tot 时：用 β·λ_film 的绝对量级预警（对低导热芯）
    parallel_abs = beta * float(coating.kappa_film)
    lines.append(
        f"   镀层等效并联贡献 β·λ_film = {parallel_abs:.4g} W/(m·K)"
        "（与 κ_tot 同量纲的权重项）"
    )
    if kappa_tot is None and parallel_abs >= 2.0:
        lines.append(
            "   【注意】 β·λ_film ≳ 2：对聚合物/丝等低导热芯，拟合后并联权重很容易 ≥30–50%；"
            "请准备减薄镀层或接受可能【不适用】扣芯。"
        )

    ok_share = True
    share_label = "【说明】"
    if kappa_tot is not None:
        try:
            res = correct_kappa_moon(float(kappa_tot), diameter, coating)
            share_label, share_msg = film_share_verdict(res.film_thermal_share)
            ok_share = share_label == "【通过】"
            lines.extend([
                f"   κ_tot = {res.kappa_tot:.6g} → κ_CF = {res.kappa_fiber:.6g} W/(m·K)",
                f"   (λ_tot−λ_CF)/λ_tot = {res.relative_deviation * 100:.2f}%",
                f"   {share_label}  {share_msg}",
            ])
            if res.film_thermal_share < FILM_SHARE_WARN:
                lines.append(f"   {moon_guidance(res.kappa_fiber)}")
            else:
                lines.append(
                    "   【说明】 并联权重已偏高，Moon「芯 λ≳2 时镀层可忽略」的经验不适用本样品。"
                )
            if res.kappa_fiber <= 0:
                lines.append("   【注意】 κ_CF≤0，参数不合理。")
                ok_share = False
        except ValueError as err:
            lines.append(f"   【不适用】 Moon 校正失败: {err}")
            ok_share = False
    else:
        lines.append(
            "   【说明】 尚未拟合 κ_tot；拟合后将判定并联权重并给出 κ_CF。"
        )

    lines.extend([
        "   【需人工确认】 镀层须退火稳定；Moon 不修正镀层热容对 C_p 的污染。",
        "   【需人工确认】 两端热沉导热须 ≫ 纤维轴向热导 κS/L（程序无法从曲线自动验证）。",
        "",
        "   总评（镀层 / Moon）",
        f"   t/D {tag(ok_thin)}，λ_film 与厚度档 "
        f"{'【通过】' if not mismatch_film else '【注意】'}，"
        f"扣芯可分辨性 {share_label if kappa_tot is not None else '【待拟合】'}。",
    ])
    if kappa_tot is not None and not ok_share:
        lines.append(
            "   结论: 当前参数下不建议把 κ_CF 当作可靠芯导热；可报告 κ_tot 并说明镀层主导。"
        )
    return "\n".join(lines)


# 常用 λ_film 初值（按厚度选用；厚银请用 WF 覆盖）
FILM_KAPPA_PRESETS = {
    'Pt_10nm_ebeam': 15.0,       # Moon 原文安全取值
    'Au_thin_film_~10nm': 50.0,  # 纳米金膜量级
    'Ag_near_bulk_≳200nm': 350.0,  # 厚银近块体初猜，宜 WF 覆盖
    'custom': 15.0,
}
