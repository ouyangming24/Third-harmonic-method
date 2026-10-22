"""纤维 3ω 实验报告（HTML）。版式参考实验室 Word 测试报告。"""

from __future__ import annotations

import base64
import html as html_lib
import io
import os
import pathlib
import shutil
import subprocess

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.pyplot as plt

MODEL_LU = "lu"
MODEL_XING = "xing"
MODEL_CONDUCTIVE = MODEL_LU      # 旧键兼容
MODEL_INSULATING = MODEL_XING    # 旧键兼容
MODEL_LABELS = {
    MODEL_LU: "Lu 模型 (2001)",
    MODEL_XING: "Xing 模型 (IMECE2015)",
    # 旧报告 JSON
    "conductive": "Lu 模型 (2001)",
    "insulating": "Xing 模型 (IMECE2015)",
}

_REFS_HTML = """
<h2 id="refs">参考文献</h2>
<ol class="refs">
<li id="ref1">D. G. Cahill, Thermal conductivity measurement from 30 to 750&nbsp;K: the 3ω method,
<i>Rev. Sci. Instrum.</i> <b>61</b>, 802–808 (1990).</li>
<li id="ref2">L. Lu, W. Yi, D. L. Zhang, 3ω method for specific heat and thermal conductivity measurements,
<i>Rev. Sci. Instrum.</i> <b>72</b>, 2996–3003 (2001).</li>
<li id="ref3">T. Munro, C. Xing, H. Ban, et al., Thermal property measurement of thin fibers by complementary methods,
IMECE2015-52056 / related fiber $3\\omega$–TET work (Utah State University &amp; KU Leuven).</li>
<li id="ref4">J. Moon, K. Weaver, B. Feng, et al., Note: Thermal conductivity measurement of individual
poly(ether ketone)/carbon nanotube fibers using a steady-state dc thermal bridge method,
<i>Rev. Sci. Instrum.</i> <b>83</b>, 016103 (2012).</li>
</ol>
"""


def _cite(*nums: int) -> str:
    """文中标准引用标识，如 [1] 或 [2,4]，链到文末参考文献。"""
    parts = []
    for n in nums:
        parts.append(f'<a class="cite" href="#ref{n}">[{n}]</a>')
    return '<sup>' + ','.join(parts) + '</sup>'


def _configure_mpl_font():
    plt.rcParams['axes.unicode_minus'] = False
    try:
        from matplotlib import font_manager
        for name in ('Microsoft YaHei', 'SimHei', 'SimSun'):
            if any(name == f.name for f in font_manager.fontManager.ttflist):
                plt.rcParams['font.sans-serif'] = [name] + list(
                    plt.rcParams.get('font.sans-serif', []))
                break
    except Exception:
        pass


def _fig_to_png_b64(fig, dpi=160) -> str:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _asset_image_b64(*filenames: str):
    """按优先级读取 assets 图片，返回 (mime, base64) 或 None。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for filename in filenames:
        path = os.path.join(here, 'assets', filename)
        if not os.path.isfile(path):
            continue
        with open(path, 'rb') as f:
            raw = f.read()
        if raw[:3] == b'\xff\xd8\xff':
            mime = 'image/jpeg'
        elif raw[:8] == b'\x89PNG\r\n\x1a\n':
            mime = 'image/png'
        elif filename.lower().endswith(('.jpg', '.jpeg')):
            mime = 'image/jpeg'
        else:
            mime = 'image/png'
        return mime, base64.b64encode(raw).decode('ascii')
    return None


def _asset_png_b64(filename: str) -> str | None:
    """兼容：只返回 base64。"""
    got = _asset_image_b64(filename)
    return None if got is None else got[1]


def _program_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _flowchart_html() -> str:
    """
    测量原理用流程图：优先嵌入 program/3omega-measurement-flowchart.html 内 SVG；
    若缺失则回退 assets PNG / 简易示意。
    """
    html_path = os.path.join(_program_dir(), '3omega-measurement-flowchart.html')
    if os.path.isfile(html_path):
        try:
            with open(html_path, encoding='utf-8') as f:
                text = f.read()
            i0 = text.find('<svg')
            i1 = text.find('</svg>')
            if i0 >= 0 and i1 > i0:
                svg = text[i0:i1 + len('</svg>')]
                # 报告页宽约 900px：保留 viewBox，用 CSS 缩放
                svg = svg.replace('width="1560"', 'width="100%"', 1)
                svg = svg.replace('height="960"', 'height="auto"', 1)
                return (
                    f'<div class="flowchart">{svg}</div>'
                    f'<p class="figcap">图  $3\\omega$ 法热导率测量流程图'
                    f'（激励 / 参考抵消 / 锁相检测）</p>'
                )
        except OSError:
            pass

    got = _asset_image_b64(
        '3omega_measurement_flowchart_hq.png',
        '3omega_measurement_flowchart.png',
    )
    if got:
        mime, setup = got
        return (
            f'<img class="fig" alt="3ω法热导率测量流程图" '
            f'src="data:{mime};base64,{setup}"/>'
            f'<p class="figcap">图  $3\\omega$ 法热导率测量流程图</p>'
        )

    _configure_mpl_font()
    fig = Figure(figsize=(8.6, 2.8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis('off')
    boxes = [
        (0.3, 1.0, 2.0, 1.2, '信号源\n(ω)'),
        (3.6, 1.0, 2.6, 1.2, '纤维样品\n加热 / 测温'),
        (7.4, 1.0, 2.2, 1.2, '锁相放大器\n(检测 3ω)'),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
            facecolor='#F5F5F5', edgecolor='#333', lw=1.4))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=11)
    fig.tight_layout()
    return (
        f'<img class="fig" alt="3ω法热导率测量流程图" '
        f'src="data:image/png;base64,{_fig_to_png_b64(fig, dpi=140)}"/>'
        f'<p class="figcap">图  $3\\omega$ 法热导率测量流程图</p>'
    )


def _moon_principle_html(moon: dict | None) -> str:
    """镀层 / Moon 热导率转换公式与说明（绝缘纤维时插入原理节）。"""
    e = html_lib.escape
    nums = ""
    if moon:
        share = moon.get("film_thermal_share")
        share_txt = (
            f'{100 * float(share):.2f}%' if share is not None else "—"
        )
        delta_nm = float(moon.get("thickness_m", 0.0) or 0.0) * 1e9
        nums = f"""
<p class="indent">本次样品镀层参数与校正结果：
厚度 $\\delta$ = {delta_nm:.4g} nm，
$\\lambda_{{\\mathrm{{film}}}}$ = {fmt_sci(moon.get("kappa_film"))} W/m·K，
$\\beta$ = {fmt_sci(moon.get("beta"))}；
拟合 $\\kappa_{{\\mathrm{{tot}}}}$ = {fmt_sci(moon.get("kappa_tot"))} W/m·K →
纤维芯 $\\kappa_{{\\mathrm{{CF}}}}$ = {fmt_sci(moon.get("kappa_fiber"))} W/m·K；
并联权重 $\\beta\\lambda_{{\\mathrm{{film}}}}/\\lambda_{{\\mathrm{{tot}}}}$ = {share_txt}。
{e(moon.get("note", "") or "")}</p>
"""
    return rf"""
<p class="indent"><strong>绝缘纤维镀层校正</strong>{_cite(4)}：
金属镀层同时作加热器与温度计时，$3\omega$ 拟合得到的是芯+膜并联的等效轴向热导率
$\kappa_{{\mathrm{{tot}}}}$（文中亦记 $\lambda_{{\mathrm{{tot}}}}$）。
截面积比为</p>
$$\beta=\frac{{\pi D\delta/2}}{{\pi D^{{2}}/4}}=\frac{{2\delta}}{{D}}$$
<p class="indent">其中 $D$ 为纤维外径，$\delta$ 为 Pt 镀层厚度。并联导热：</p>
$$\lambda_{{\mathrm{{tot}}}}=\beta\,\lambda_{{\mathrm{{film}}}}+(1-\beta)\,\lambda_{{\mathrm{{CF}}}}$$
$$\lambda_{{\mathrm{{CF}}}}=\frac{{\lambda_{{\mathrm{{tot}}}}-\beta\,\lambda_{{\mathrm{{film}}}}}}{{1-\beta}}$$
""" + nums


def _principle_block(
    model: str,
    *,
    insulating_fiber: bool = False,
    moon: dict | None = None,
) -> str:
    """稍详细的原理 + 核心 V3ω 公式 + 流程图 +（绝缘时）Moon 镀层公式。"""
    setup_html = _flowchart_html()

    common = f"""
<p class="indent">$3\\omega$ 方法最初用于平面加热器测固体热导率{_cite(1)}。
对纤维丝样品，激励电流角频率为 $\\omega$，焦耳热功率含直流与 $2\\omega$ 分量，
引起温度振荡 $\\Delta T(2\\omega)$；若样品（或镀层）电阻随温度变化，
则电压中出现与热物性相关的 $3\\omega$ 分量，由锁相放大器提取。</p>
<p class="indent">电阻温度系数 $\\alpha_R$ 标定与 $3\\omega$ 热导率测量均在
<strong>精密真空温控控制台</strong>中进行：样品置于可控温真空腔，
工作真空度控制在 <strong>2&nbsp;Pa 以下</strong>，以抑制气体对流换热，
使热损失以轴向导热（及必要时计入的辐射）为主，与一维模型假设一致。</p>
""" + setup_html + r"""
<p class="indent">激励与传感的基本关系为：</p>
$$I(t)=I_0\cos(\omega t)$$
$$P(t)=I^2(t)R \;\propto\; \tfrac12 I_0^2 R\bigl[1+\cos(2\omega t)\bigr]$$
$$R(T)\approx R_0\bigl[1+\alpha_R\,\Delta T\bigr],\qquad
  R'=\frac{\mathrm{d}R}{\mathrm{d}T}=\alpha_R R_0$$
<p class="indent">差分输入按「样品 $-$ 参考」接线，相对 $1\omega$ 置零后读取复数 $V_{3\omega}$。
扫频后用最小二乘法拟合热物性。</p>
"""

    moon_html = _moon_principle_html(moon) if insulating_fiber else ""

    if model in (MODEL_XING, "insulating", MODEL_INSULATING):
        geom = _schematic_insulating_b64() if insulating_fiber else _schematic_conductive_b64()
        geom_cap = (
            "镀层绝缘纤维结构示意（截面与侧视）"
            if insulating_fiber else "纤维测量段结构示意"
        )
        intro = (
            f"采用 <strong>Xing / Munro</strong> 闭式一维 $3\\omega$ 模型{_cite(3)}。"
            + (
                f"绝缘芯+镀层时，$R$、$\\alpha_R$ 取镀层；拟合得等效 $\\kappa_{{\\mathrm{{tot}}}}$，"
                f"再按下方公式{_cite(4)}得纤维芯 $\\kappa_{{\\mathrm{{CF}}}}$。"
                if insulating_fiber else
                "导电丝时 $R$、$\\alpha_R$ 取丝本体；拟合直接得到轴向 $\\kappa$。"
            )
        )
        return common + f"""
<p class="indent">{intro}</p>
<img class="fig" alt="{geom_cap}" src="data:image/png;base64,{geom}"/>
<p class="figcap">图  {geom_cap}</p>
<p class="indent">热扩散率与复波数（含侧向辐射）为</p>
$$\\alpha=\\frac{{\\kappa}}{{\\rho C_p}}$$
$$m^2=i\\frac{{2\\omega}}{{\\alpha}}+\\frac{{4 h_r}}{{D\\kappa}},\\qquad
  h_r=4\\varepsilon\\sigma T_0^3$$
<p class="indent">两端恒温边界下的几何传递函数</p>
$$F(m)=\\frac{{mL\\sinh(mL)-2\\cosh(mL)+2}}{{L^2 m^3\\sinh(mL)}}
  =\\frac{{2-2\\cosh(mL)+mL\\sinh(mL)}}{{L^2 m^3\\sinh(mL)}}$$
<p class="indent"><strong>核心正演（rms 复数，与锁相一致）：</strong></p>
$$V_{{3\\omega}}=
\\frac{{I_0^3 R R'}}{{\\kappa\\,\\pi D^2\\sqrt{{2}}}}\\,F(m)
=\\frac{{2 I_{{\\mathrm{{rms}}}}^3 R R'}}{{\\kappa\\,\\pi D^2}}\\,F(m)$$
<p class="indent">其中 $I_0=\\sqrt{{2}}\\,I_{{\\mathrm{{rms}}}}$，$R'$ 带符号；$V_{{3\\omega}}$ 为复数，
同时含幅值与相位。$R'>0$ 时相位呈热滞后（负相位）。
低频极限（$|mL|\\to 0$，$F\\to L/12$）为</p>
$$V_{{3\\omega}}\\to
\\frac{{I_0^3 R |R'| L}}{{12\\,\\kappa\\,\\pi D^2\\sqrt{{2}}}}
=\\frac{{I_{{\\mathrm{{rms}}}}^3 R |R'| L}}{{6\\,\\kappa\\,\\pi D^2}}$$
<p class="indent">拟合中的 $\\kappa$ 对绝缘镀层样品即为 $\\kappa_{{\\mathrm{{tot}}}}$。</p>
""" + moon_html

    # Lu
    geom = _schematic_insulating_b64() if insulating_fiber else _schematic_conductive_b64()
    geom_cap = (
        "镀层绝缘纤维结构示意（截面与侧视）"
        if insulating_fiber else "纤维测量段结构示意"
    )
    intro = (
        f"采用 <strong>Lu, Yi, Zhang</strong> 一维 $3\\omega$ 模型（一阶轴向模态）{_cite(2)}。"
        + (
            f"绝缘镀层样品拟合等效 $\\kappa_{{\\mathrm{{tot}}}}$，再经校正{_cite(4)}得 $\\kappa_{{\\mathrm{{CF}}}}$。"
            if insulating_fiber else
            "导电丝直接拟合轴向 $\\kappa$。"
        )
    )
    return common + f"""
<p class="indent">{intro}</p>
<img class="fig" alt="{geom_cap}" src="data:image/png;base64,{geom}"/>
<p class="figcap">图  {geom_cap}</p>
<p class="indent">热扩散率与热时间常数为</p>
$$\\alpha=\\frac{{\\kappa}}{{\\rho C_p}},\\qquad
  \\gamma=\\frac{{L^2}}{{\\pi^2\\alpha}}$$
<p class="indent">其中 $L$ 为测量长度，$S$ 为截面积。
<strong>核心正演（rms 复数，与 Xing 相位约定对齐）：</strong></p>
$$V_{{3\\omega}}=
\\frac{{4 I_{{\\mathrm{{rms}}}}^3 R R' L}}{{\\pi^4\\kappa S}}\\,
\\frac{{1}}{{1+i\\,2\\omega\\gamma}}$$
<p class="indent">因而幅值与相位满足</p>
$$\\lvert V_{{3\\omega}}\\rvert=
\\frac{{4 I_{{\\mathrm{{rms}}}}^3 R\\lvert R'\\rvert L}}{{\\pi^4\\kappa S}}\\,
\\frac{{1}}{{\\sqrt{{1+(2\\omega\\gamma)^2}}}}$$
$$\\tan\\phi=2\\omega\\gamma$$
<p class="indent">$R'>0$ 时相位为 $-\\phi$（热滞后）；$R'<0$ 时约为 $\\pi-\\phi$
（差分「样品$-$参考」、相对 $1\\omega$ 置零）。
文献{_cite(2)} Sec.VI 对 $R'>0$ 报 $\\pi-\\phi$，与本程序（与文献{_cite(3)}一致的负滞后约定）差一整体符号。
低频、高频极限为</p>
$$\\lvert V_{{3\\omega}}\\rvert\\xrightarrow{{\\omega\\to 0}}
\\frac{{4 I_{{\\mathrm{{rms}}}}^3 R\\lvert R'\\rvert L}}{{\\pi^4\\kappa S}}$$
$$\\lvert V_{{3\\omega}}\\rvert\\xrightarrow{{\\omega\\to\\infty}}
\\frac{{I_{{\\mathrm{{rms}}}}^3 R\\lvert R'\\rvert}}{{4\\omega\\,\\rho C_p L S}}$$
<p class="indent">若计入径向辐射，可用表观参数 $\\kappa_{{\\mathrm{{ap}}}}=(1+g\\gamma)\\kappa$、
$\\gamma_{{\\mathrm{{ap}}}}=\\gamma/(1+g\\gamma)$ 代入上式。
由 $V_{{3\\omega}}(\\omega)$ 拟合得到 $\\kappa$（绝缘时为 $\\kappa_{{\\mathrm{{tot}}}}$）。</p>
""" + moon_html


def _schematic_setup_b64():
    """兼容旧调用：返回流程图 PNG（优先 assets，否则由 SVG 页说明改走 _flowchart_html）。"""
    got = _asset_image_b64(
        '3omega_measurement_flowchart_hq.png',
        '3omega_measurement_flowchart.png',
    )
    if got:
        return got
    _configure_mpl_font()
    fig = Figure(figsize=(8.6, 2.8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis('off')
    boxes = [
        (0.3, 1.0, 2.0, 1.2, '信号源\n(ω)'),
        (3.6, 1.0, 2.6, 1.2, '纤维样品\n加热 / 测温'),
        (7.4, 1.0, 2.2, 1.2, '锁相放大器\n(检测 3ω)'),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
            facecolor='#F5F5F5', edgecolor='#333', lw=1.4))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=11)
    fig.tight_layout()
    return 'image/png', _fig_to_png_b64(fig, dpi=140)


def _schematic_conductive_b64() -> str:
    """导电丝：四探针 + 轴向导热示意。"""
    _configure_mpl_font()
    fig = Figure(figsize=(8.6, 3.0))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis('off')

    # pads
    for x in (1.2, 8.0):
        ax.add_patch(Rectangle((x, 1.35), 0.85, 0.9,
                               facecolor='#90CAF9', edgecolor='#1565C0', lw=1.2))
    ax.text(1.62, 1.05, '电极', ha='center', fontsize=9)
    ax.text(8.42, 1.05, '电极', ha='center', fontsize=9)

    # fiber
    ax.plot([2.05, 8.0], [1.8, 1.8], color='#333', lw=3.5)
    ax.annotate('', xy=(6.5, 1.8), xytext=(3.5, 1.8),
                arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=1.6))
    ax.text(5.0, 2.05, 'L（测量段）', ha='center', fontsize=10, color='#2E7D32')

    # current / voltage labels
    ax.annotate('', xy=(1.6, 2.55), xytext=(1.6, 3.15),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.3))
    ax.text(1.95, 2.95, 'I(ω)', fontsize=10, color='#1565C0')
    ax.annotate('', xy=(8.4, 2.55), xytext=(8.4, 3.15),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.3))
    ax.text(8.75, 2.95, 'V3ω', fontsize=10, color='#C62828')

    ax.text(5.0, 0.55,
            '导电丝同时作为加热器与温度传感器；热流主要沿轴向一维传导',
            ha='center', fontsize=10)
    fig.tight_layout()
    return _fig_to_png_b64(fig, dpi=140)


def _schematic_insulating_b64() -> str:
    """绝缘丝：芯 + 金属镀层截面示意。"""
    _configure_mpl_font()
    fig = Figure(figsize=(8.6, 3.2))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.set_aspect('equal')
    ax.axis('off')

    # cross section
    cx, cy = 2.6, 1.9
    ax.add_patch(Circle((cx, cy), 1.05, facecolor='#FFECB3',
                        edgecolor='#F9A825', lw=1.5, label='metal'))
    ax.add_patch(Circle((cx, cy), 0.72, facecolor='#E3F2FD',
                        edgecolor='#1565C0', lw=1.5))
    ax.text(cx, cy, '芯', ha='center', va='center', fontsize=11, color='#0D47A1')
    ax.text(cx + 0.95, cy + 0.85, '金属镀层', fontsize=10, color='#F57F17')
    ax.annotate('', xy=(cx + 0.72, cy), xytext=(cx + 1.05, cy),
                arrowprops=dict(arrowstyle='<->', color='#E65100', lw=1.2))
    ax.text(cx + 1.25, cy - 0.05, 't', fontsize=10, color='#E65100')

    # side view
    ax.add_patch(FancyBboxPatch(
        (5.2, 1.45), 3.8, 0.9, boxstyle='round,pad=0.02,rounding_size=0.08',
        facecolor='#FFECB3', edgecolor='#F9A825', lw=1.3))
    ax.add_patch(FancyBboxPatch(
        (5.35, 1.62), 3.5, 0.56, boxstyle='round,pad=0.02,rounding_size=0.06',
        facecolor='#E3F2FD', edgecolor='#1565C0', lw=1.2))
    ax.text(7.1, 2.55, '侧视：镀层加热 / 芯材导热', ha='center', fontsize=10)
    ax.text(7.1, 1.05, '电流在镀层中产生焦耳热，温升经芯材轴向扩散',
            ha='center', fontsize=10)

    # radiation arrow
    ax.annotate('', xy=(cx, cy + 1.55), xytext=(cx, cy + 1.15),
                arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.3))
    ax.text(cx + 0.15, cy + 1.65, '辐射 hr', fontsize=9, color='#6A1B9A')

    fig.tight_layout()
    return _fig_to_png_b64(fig, dpi=140)


def report_plot_kind(fit_mode_key: str) -> str:
    """按拟合目标决定报告只出哪一类图。"""
    return {
        'amp': 'amp',
        'phase': 'phase',
        'both': 're_im',
        'real': 'real',
        'imag': 'imag',
    }.get(fit_mode_key, 'amp_phase')


def _load_image_b64(path: str):
    """读取本地图片为 (mime, b64)，失败返回 None。"""
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        raw = f.read()
    low = path.lower()
    if raw[:3] == b'\xff\xd8\xff' or low.endswith(('.jpg', '.jpeg')):
        mime = 'image/jpeg'
    elif raw[:8] == b'\x89PNG\r\n\x1a\n' or low.endswith('.png'):
        mime = 'image/png'
    else:
        mime = 'image/jpeg'
    return mime, base64.b64encode(raw).decode('ascii')


def _instrument_photo_b64():
    """
    报告用 Harmonic ONE 仪器照片。
    优先 program/image.png，其次 assets 下 harmonic_one_*.png。
    返回 (mime, b64) 或 None。
    """
    got = _load_image_b64(os.path.join(_program_dir(), 'image.png'))
    if got is not None:
        return got
    return _asset_image_b64('harmonic_one_hq.png', 'harmonic_one.png')


def _instrument_showcase_html() -> str:
    """仪器实物图 + 说明（配合真空温控台）。"""
    instrument = _instrument_photo_b64()
    if not instrument:
        return ""
    mime, b64 = instrument
    return (
        f'<p class="indent"><strong>测量主机</strong>为 '
        f'<strong>Harmonic ONE</strong>（AUTINST）：'
        f'提供 $3\\omega$ 激励、参考与锁相检测；'
        f'与精密真空温控控制台联用，完成'
        f'电阻温度系数标定与热导率扫频。</p>'
        f'<img class="fig instrument" alt="Harmonic ONE" '
        f'src="data:{mime};base64,{b64}"/>'
        f'<p class="figcap">图  测量仪器 Harmonic ONE（AUTINST）；'
        f'配合精密真空温控控制台</p>'
    )


def _software_ui_gallery_html() -> str:
    """
    Three Omega Analyzer 软件界面展示（报告封面/过程说明用）。
    图片位于 program/ 根目录的 img_v3_*.jpg。
    """
    here = _program_dir()
    # 按测量→样品→系统→拟合 逻辑排序；文件名来自用户提供截图
    items = [
        (
            'img_v3_0214p_240015f1-a57f-4226-a812-cbbd01b007dg.jpg',
            '测量工程 — Project Setting（频率列表与台架扫描；左侧为温控与锁相状态）',
        ),
        (
            'img_v3_0214p_8e5c40bf-b429-4901-b482-3c42854ced7g.jpg',
            '测量工程 — Sample Parameter（样品/分层热物性与拟合上下限）',
        ),
        (
            'img_v3_0214p_59d3289c-d337-445c-8369-22bce42e218g.jpg',
            '测量工程 — System Parameter（拟合模式与系统参数）',
        ),
        (
            'img_v3_0214p_3dd16857-1f46-47cb-b8ae-4c5a7d9e467g.jpg',
            '拟合工程 — Curve（幅值–频率曲线；配合精密真空温控与锁相采集）',
        ),
    ]
    blocks = []
    for i, (fname, cap) in enumerate(items, start=1):
        got = _load_image_b64(os.path.join(here, fname))
        if got is None:
            # 兼容 assets 下可能的副本
            got = _asset_image_b64(fname)
        if got is None:
            continue
        mime, b64 = got
        blocks.append(
            f'<figure class="ui-shot">'
            f'<img class="fig software" alt="{html_lib.escape(cap)}" '
            f'src="data:{mime};base64,{b64}"/>'
            f'<figcaption class="figcap">图  软件界面 {i}　'
            f'{html_lib.escape(cap)}</figcaption>'
            f'</figure>'
        )
    if not blocks:
        return ""
    intro = (
        '<p class="indent"><strong>测量与分析软件</strong>采用 '
        '<strong>Three Omega Analyzer</strong>：左侧 '
        '<em>Temperature Control Panel</em> 对接精密真空温控控制台'
        '（温控精度 0.1&nbsp;℃），'
        '<em>Lockin Status</em> 显示锁相放大器状态；'
        '主区完成 $3\\omega$ 扫频采集、样品/系统参数设置与曲线拟合。</p>'
    )
    return intro + '<div class="ui-gallery">' + ''.join(blocks) + '</div>'


def _y_of(V, kind: str):
    if V is None:
        return None
    V = np.asarray(V, dtype=complex)
    if kind == 'amp':
        return np.abs(V) * 1e6
    if kind == 'pha':
        return np.angle(V, deg=True)
    if kind == 're':
        return V.real * 1e6
    if kind == 'im':
        return V.imag * 1e6
    return None


def _add_kappa_pm20(ax, freqs, V_plus, V_minus, kind: str, *, log_y=False):
    """在坐标轴上叠加 κ ±20% 正演虚线。"""
    y_p = _y_of(V_plus, kind)
    y_m = _y_of(V_minus, kind)
    plot_fn = ax.loglog if log_y else ax.semilogx
    if y_p is not None and np.any(np.isfinite(y_p)):
        plot_fn(freqs, y_p, '--', color='#ff7f0e', label='κ +20%', lw=1.5, alpha=0.95)
    if y_m is not None and np.any(np.isfinite(y_m)):
        plot_fn(freqs, y_m, '--', color='#2ca02c', label='κ -20%', lw=1.5, alpha=0.95)


def make_fit_report_plot(freqs, V3_exp, V3_fit, title: str, plot_kind: str = 'amp_phase',
                         V3_kappa_plus=None, V3_kappa_minus=None):
    """按拟合目标生成报告图与数据表；可选叠加 κ ±20% 虚线。"""
    _configure_mpl_font()
    freqs = np.asarray(freqs, dtype=float)
    Ve = np.asarray(V3_exp, dtype=complex)
    Vm = np.asarray(V3_fit, dtype=complex)
    V_plus = None if V3_kappa_plus is None else np.asarray(V3_kappa_plus, dtype=complex)
    V_minus = None if V3_kappa_minus is None else np.asarray(V3_kappa_minus, dtype=complex)
    pha_e = np.angle(Ve, deg=True)
    pha_m = np.angle(Vm, deg=True)
    amp_e = np.abs(Ve) * 1e6
    amp_m = np.abs(Vm) * 1e6
    figures = []

    def _ax_fig(h=4.2):
        fig = Figure(figsize=(8.4, h))
        return fig, fig.add_subplot(111)

    if plot_kind == 'phase':
        fig, ax = _ax_fig()
        ax.semilogx(freqs, pha_e, 'ro', label='实验', markersize=5)
        ax.semilogx(freqs, pha_m, 'b-', label='拟合', lw=2)
        _add_kappa_pm20(ax, freqs, V_plus, V_minus, 'pha')
        ax.set_ylabel('相位 (°)')
        ax.set_xlabel('频率 (Hz)')
        ax.set_title(title + ' — 相位')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', ls='--', alpha=0.35)
        fig.tight_layout()
        figures.append(('图  相位角与频率的关系（实验、拟合及 κ±20%）', _fig_to_png_b64(fig)))
        headers = ['频率 (Hz)', '实验相位 (°)', '拟合相位 (°)']
        rows = [(freqs[i], pha_e[i], pha_m[i]) for i in range(freqs.size)]
        cap = '附表  相位拟合所用数据'
    elif plot_kind == 'amp':
        fig, ax = _ax_fig()
        ax.loglog(freqs, amp_e, 'ro', label='实验', markersize=5)
        ax.loglog(freqs, amp_m, 'b-', label='拟合', lw=2)
        _add_kappa_pm20(ax, freqs, V_plus, V_minus, 'amp', log_y=True)
        ax.set_ylabel('|V3| (μV)')
        ax.set_xlabel('频率 (Hz)')
        ax.set_title(title + ' — 幅值')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', ls='--', alpha=0.35)
        fig.tight_layout()
        figures.append(('图  三次谐波幅值与频率的关系（实验、拟合及 κ±20%）', _fig_to_png_b64(fig)))
        headers = ['频率 (Hz)', '实验 |V3| (μV)', '拟合 |V3| (μV)']
        rows = [(freqs[i], amp_e[i], amp_m[i]) for i in range(freqs.size)]
        cap = '附表  幅值拟合所用数据'
    elif plot_kind == 'real':
        fig, ax = _ax_fig()
        ax.semilogx(freqs, Ve.real * 1e6, 'ro', label='实验', markersize=5)
        ax.semilogx(freqs, Vm.real * 1e6, 'b-', label='拟合', lw=2)
        _add_kappa_pm20(ax, freqs, V_plus, V_minus, 're')
        ax.set_ylabel('Re(V3) (μV)')
        ax.set_xlabel('频率 (Hz)')
        ax.set_title(title + ' — 实部')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', ls='--', alpha=0.35)
        fig.tight_layout()
        figures.append(('图  三次谐波实部与频率的关系（含 κ±20%）', _fig_to_png_b64(fig)))
        headers = ['频率 (Hz)', '实验 Re(V3) (μV)', '拟合 Re(V3) (μV)']
        rows = [(freqs[i], Ve.real[i] * 1e6, Vm.real[i] * 1e6) for i in range(freqs.size)]
        cap = '附表  实部拟合所用数据'
    elif plot_kind == 'imag':
        fig, ax = _ax_fig()
        ax.semilogx(freqs, Ve.imag * 1e6, 'ro', label='实验', markersize=5)
        ax.semilogx(freqs, Vm.imag * 1e6, 'b-', label='拟合', lw=2)
        _add_kappa_pm20(ax, freqs, V_plus, V_minus, 'im')
        ax.set_ylabel('Im(V3) (μV)')
        ax.set_xlabel('频率 (Hz)')
        ax.set_title(title + ' — 虚部')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', ls='--', alpha=0.35)
        fig.tight_layout()
        figures.append(('图  三次谐波虚部与频率的关系（含 κ±20%）', _fig_to_png_b64(fig)))
        headers = ['频率 (Hz)', '实验 Im(V3) (μV)', '拟合 Im(V3) (μV)']
        rows = [(freqs[i], Ve.imag[i] * 1e6, Vm.imag[i] * 1e6) for i in range(freqs.size)]
        cap = '附表  虚部拟合所用数据'
    elif plot_kind == 're_im':
        fig = Figure(figsize=(8.4, 6.2))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax1.semilogx(freqs, Ve.real * 1e6, 'ro', label='实验', markersize=5)
        ax1.semilogx(freqs, Vm.real * 1e6, 'b-', label='拟合', lw=2)
        _add_kappa_pm20(ax1, freqs, V_plus, V_minus, 're')
        ax1.set_ylabel('Re(V3) (μV)')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, which='both', ls='--', alpha=0.35)
        ax1.set_title(title + ' — 实部 / 虚部')
        ax2.semilogx(freqs, Ve.imag * 1e6, 'ro', label='实验', markersize=5)
        ax2.semilogx(freqs, Vm.imag * 1e6, 'b-', label='拟合', lw=2)
        _add_kappa_pm20(ax2, freqs, V_plus, V_minus, 'im')
        ax2.set_ylabel('Im(V3) (μV)')
        ax2.set_xlabel('频率 (Hz)')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, which='both', ls='--', alpha=0.35)
        fig.tight_layout()
        figures.append(('图  三次谐波实部、虚部与频率的关系（含 κ±20%）', _fig_to_png_b64(fig)))
        headers = ['频率 (Hz)', '实验 Re (μV)', '拟合 Re (μV)', '实验 Im (μV)', '拟合 Im (μV)']
        rows = [
            (freqs[i], Ve.real[i] * 1e6, Vm.real[i] * 1e6,
             Ve.imag[i] * 1e6, Vm.imag[i] * 1e6)
            for i in range(freqs.size)
        ]
        cap = '附表  实部+虚部拟合所用数据'
    else:
        fig = Figure(figsize=(8.4, 6.2))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax1.loglog(freqs, amp_e, 'ro', label='实验', markersize=5)
        ax1.loglog(freqs, amp_m, 'b-', label='拟合', lw=2)
        _add_kappa_pm20(ax1, freqs, V_plus, V_minus, 'amp', log_y=True)
        ax1.set_ylabel('|V3| (μV)')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, which='both', ls='--', alpha=0.35)
        ax1.set_title(title)
        ax2.semilogx(freqs, pha_e, 'ro', label='实验', markersize=5)
        ax2.semilogx(freqs, pha_m, 'b-', label='拟合', lw=2)
        _add_kappa_pm20(ax2, freqs, V_plus, V_minus, 'pha')
        ax2.set_ylabel('相位 (°)')
        ax2.set_xlabel('频率 (Hz)')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, which='both', ls='--', alpha=0.35)
        fig.tight_layout()
        figures.append(('图  三次谐波幅值、相位与频率的关系（含 κ±20%）', _fig_to_png_b64(fig)))
        headers = ['频率 (Hz)', '实验 |V3| (μV)', '拟合 |V3| (μV)', '实验相位 (°)', '拟合相位 (°)']
        rows = [
            (freqs[i], amp_e[i], amp_m[i], pha_e[i], pha_m[i])
            for i in range(freqs.size)
        ]
        cap = '附表  拟合所用数据'

    return {
        'figures': figures,
        'data_caption': cap,
        'data_headers': headers,
        'data_rows': rows,
    }


def fmt_sci(v, digits=4):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "—"
    ax = abs(x)
    if ax == 0:
        return "0"
    if 1e-3 <= ax < 1e4:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}e}"


def fit_tcr_calibration(T_C, R_ohm, T_ref_C=None):
    """
    由 R–T 标定做线性拟合，求电阻温度系数。

    R = a + b·T_C，α_R = (1/R_ref)·dR/dT = b / R(T_ref)。
    T_ref 默认取标定温度均值。
    """
    T = np.asarray(T_C, dtype=float)
    R = np.asarray(R_ohm, dtype=float)
    if T.size < 2:
        raise ValueError("TCR 标定至少需要 2 个温度点")
    if T.size != R.size:
        raise ValueError("T 与 R 长度不一致")
    b, a = np.polyfit(T, R, 1)
    T_ref = float(np.mean(T)) if T_ref_C is None else float(T_ref_C)
    R_ref = float(a + b * T_ref)
    if abs(R_ref) < 1e-30:
        raise ValueError("参考电阻过小，无法计算 α_R")
    alpha_R = float(b / R_ref)
    R_fit = a + b * T
    ss_res = float(np.sum((R - R_fit) ** 2))
    ss_tot = float(np.sum((R - np.mean(R)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        'T_C': T,
        'R_ohm': R,
        'slope': float(b),
        'intercept': float(a),
        'T_ref_C': T_ref,
        'R_ref': R_ref,
        'alpha_R': alpha_R,
        'r_squared': float(r2),
        'R_fit': R_fit,
    }


def make_tcr_calib_figure(calib: dict) -> str:
    """R–T 线性拟合图，返回 PNG base64。"""
    _configure_mpl_font()
    T = np.asarray(calib['T_C'], dtype=float)
    R = np.asarray(calib['R_ohm'], dtype=float)
    a = calib['intercept']
    b = calib['slope']
    T_line = np.linspace(float(T.min()), float(T.max()), 80)
    R_line = a + b * T_line

    fig = Figure(figsize=(7.2, 4.4))
    ax = fig.add_subplot(111)
    ax.plot(T, R, 'ro', markersize=7, label='实验标定')
    ax.plot(T_line, R_line, 'b-', lw=2,
            label=f'线性拟合  R = {b:.4g}·T + {a:.4g}')
    ax.set_xlabel('温度 T (℃)')
    ax.set_ylabel('电阻 R (Ω)')
    ax.set_title('电阻–温度标定与线性拟合')
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(loc='best', fontsize=9)
    txt = (f"α_R = {calib['alpha_R']:.4e} /K\n"
           f"R_ref({calib['T_ref_C']:.1f}℃) = {calib['R_ref']:.4g} Ω\n"
           f"R² = {calib['r_squared']:.5f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    fig.tight_layout()
    return _fig_to_png_b64(fig)


def build_tcr_section_html(calib: dict, times=None) -> str:
    """报告中「α_R 标定」段落：说明 + 表 + 拟合图。"""
    e = html_lib.escape
    T = np.asarray(calib['T_C'], dtype=float)
    R = np.asarray(calib['R_ohm'], dtype=float)
    fig_b64 = make_tcr_calib_figure(calib)
    times = list(times) if times is not None else [""] * T.size

    rows = []
    for i in range(T.size):
        tstr = e(str(times[i])) if i < len(times) and times[i] not in (None, "") else "—"
        rows.append(
            f"<tr><td class='num'>{i + 1}</td>"
            f"<td class='num'>{fmt_sci(T[i], 5)}</td>"
            f"<td class='num'>{fmt_sci(R[i], 5)}</td>"
            f"<td class='num'>{tstr}</td></tr>"
        )

    return f"""
<p><strong>电阻温度系数 $\\alpha_R$ 标定：</strong></p>
<p class="indent">在<strong>精密真空温控控制台</strong>中，于真空度
<strong>＜&nbsp;2&nbsp;Pa</strong> 条件下，对样品在若干温度点测量电阻，
作线性拟合 $R = a + b\\cdot T$。
取参考温度 $T_{{\\mathrm{{ref}}}}$ = {fmt_sci(calib['T_ref_C'])} ℃ 处电阻
$R_{{\\mathrm{{ref}}}}$ = {fmt_sci(calib['R_ref'])} Ω，则</p>
$$\\alpha_R = \\frac{{1}}{{R_{{\\mathrm{{ref}}}}}}\\frac{{\\mathrm{{d}}R}}{{\\mathrm{{d}}T}} = \\frac{{b}}{{R_{{\\mathrm{{ref}}}}}} = {fmt_sci(calib['alpha_R'], 5)}\\,/\\mathrm{{K}}
\\quad (R^2 = {fmt_sci(calib['r_squared'], 5)})$$
<table class="data">
<thead><tr><th>序号</th><th>温度 $T$ (℃)</th><th>电阻 $R$ (Ω)</th><th>时间</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
<img class="fig" alt="电阻温度标定拟合" src="data:image/png;base64,{fig_b64}"/>
<p class="figcap">图  电阻–温度标定数据与线性拟合曲线（真空温控台，＜2&nbsp;Pa）</p>
"""


def build_fiber_experiment_html(ctx: dict) -> str:
    e = html_lib.escape
    model = ctx['model']
    principle = _principle_block(
        model,
        insulating_fiber=bool(ctx.get('insulating_fiber')),
        moon=ctx.get('moon'),
    )
    mat = ctx['material']
    fiber = ctx['fiber']
    alpha = ctx['alpha']
    plot = ctx.get('plot') or {}
    figures = plot.get('figures') or []
    data_headers = plot.get('data_headers') or []
    data_rows = plot.get('data_rows') or []

    fig_html = ""
    for cap, b64 in figures:
        fig_html += (
            f'<img class="fig" alt="{e(cap)}" src="data:image/png;base64,{b64}"/>'
            f'<p class="figcap">{e(cap)}</p>'
        )

    appendix_html = ""
    if data_headers and data_rows:
        th = "".join(f"<th>{e(h)}</th>" for h in data_headers)
        trs = []
        for row in data_rows:
            tds = "".join(f"<td class='num'>{fmt_sci(v, 5)}</td>" for v in row)
            trs.append(f"<tr>{tds}</tr>")
        appendix_html = (
            f"<h2>附录</h2>"
            f"<p class='indent'>本附录列出本次拟合所用频率点及对应实验/拟合数据。"
            f"图中橙色、绿色虚线分别为热导率 $\\kappa$ 增加 20% 与减小 20% 时的正演曲线。</p>"
            f"<p class='figcap'>{e(plot.get('data_caption', '拟合数据'))}</p>"
            f"<table class='data'><thead><tr>{th}</tr></thead>"
            f"<tbody>{''.join(trs)}</tbody></table>"
        )

    software_html = _software_ui_gallery_html()
    instrument_html = _instrument_showcase_html()
    # 先仪器实物，再软件界面
    showcase_html = instrument_html + software_html

    T_show = ctx.get('temperature_C')
    T_txt = fmt_sci(T_show) if T_show is not None else "—"

    sample_type = "绝缘纤维（镀层）" if ctx.get("insulating_fiber") else "纤维样品"
    kappa_header = r"$\\kappa_{CF}$" if ctx.get("moon") else r"$\\kappa$"
    moon = ctx.get("moon") or {}
    if moon:
        moon_note = (
            f'<p class="indent">镀层并联导热校正{_cite(4)}：拟合等效 $\\kappa_{{\\mathrm{{tot}}}}$ = '
            f'{fmt_sci(moon["kappa_tot"])} W/m·K，'
            f'纤维芯 $\\kappa_{{\\mathrm{{CF}}}}$ = {fmt_sci(moon["kappa_fiber"])} W/m·K，'
            f'$\\beta$ = {fmt_sci(moon["beta"])}。'
            f'{e(moon.get("note", ""))}</p>'
        )
        process_extra = (
            f"（绝缘丝镀金属层作加热/测温，结果已按文献{_cite(4)}扣除镀层）"
        )
    else:
        moon_note = ""
        process_extra = ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>材料热物性测试报告</title>
<script>
window.__MATHJAX_DONE__ = false;
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }},
  svg: {{ fontCache: 'global' }},
  options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }},
  startup: {{
    pageReady: function () {{
      return MathJax.startup.defaultPageReady().then(function () {{
        window.__MATHJAX_DONE__ = true;
        document.documentElement.classList.add('mathjax-ready');
      }});
    }}
  }}
}};
</script>
<!-- SVG 输出便于打印/导出 PDF（CHTML 网页字体常导致公式变黑条） -->
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
  body {{ font-family: "SimSun", "Songti SC", "Times New Roman", serif;
         color: #111; max-width: 900px; margin: 28px auto; padding: 0 22px;
         line-height: 1.7; font-size: 14px; }}
  h1 {{ text-align: center; font-size: 22px; font-weight: bold;
        letter-spacing: 4px; margin: 8px 0 18px;
        font-family: "SimSun", "Songti SC", serif; }}
  h2 {{ font-size: 16px; margin: 1.4em 0 0.6em;
        font-family: "SimSun", "Songti SC", serif; }}
  .cover {{ width: 100%; border-collapse: collapse; margin: 0 0 18px; }}
  .cover td {{ border: none; padding: 3px 8px; }}
  .lab {{ text-align: center; font-size: 15px; margin: 6px 0 20px; }}
  table.result, table.data {{ border-collapse: collapse; width: 100%;
        margin: 10px 0 16px; font-size: 13px; }}
  table.result th, table.result td, table.data th, table.data td {{
        border: 1px solid #333; padding: 5px 7px; }}
  table.result th, table.data th {{ background: #f3f3f3; text-align: center; }}
  td.num {{ font-family: "Times New Roman", Consolas, serif; text-align: center; }}
  mjx-container {{ font-size: 105% !important; }}
  mjx-container[display="true"] {{ margin: 0.6em 0 !important; }}
  /* 打印/PDF：保证公式 SVG 与图片颜色被嵌入 */
  @media print {{
    mjx-container, mjx-container svg {{
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
      color-adjust: exact !important;
    }}
    img.fig, img.software, img.instrument {{
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
      max-width: 100% !important;
      page-break-inside: avoid;
    }}
    .ui-shot, figure {{ page-break-inside: avoid; }}
  }}
  html:not(.mathjax-ready) mjx-container {{ visibility: hidden; }}
  img.fig {{ max-width: 100%; height: auto; display: block; margin: 10px auto;
             image-rendering: -webkit-optimize-contrast; }}
  img.instrument {{ max-width: 640px; }}
  img.software {{ max-width: 100%; width: 100%; border: 1px solid #ccc; }}
  .ui-gallery {{ display: grid; grid-template-columns: 1fr;
               gap: 8px; margin: 8px 0 18px; }}
  .ui-shot {{ margin: 0; }}
  @media (min-width: 900px) {{
    .ui-gallery {{ grid-template-columns: 1fr 1fr; }}
  }}
  img.fig[alt*="流程图"] {{ max-width: 100%; width: 100%; }}
  .flowchart {{ width: 100%; max-width: 100%; margin: 10px auto 4px;
               overflow-x: auto; background: #f7f4ee; border: 1px solid #e6e0d4;
               border-radius: 8px; }}
  .flowchart svg {{ display: block; width: 100%; height: auto; }}
  .figcap {{ text-align: center; font-size: 13px; margin: 0 0 14px; }}
  .indent {{ text-indent: 2em; }}
  .footer {{ margin-top: 28px; font-size: 13px; }}
  ol.refs {{ font-size: 13px; line-height: 1.55; padding-left: 1.6em; margin: 8px 0 16px; }}
  ol.refs li {{ margin: 0.35em 0; }}
  a.cite {{ color: #1565C0; text-decoration: none; font-weight: 600; }}
  a.cite:hover {{ text-decoration: underline; }}
  .box {{ font-size: 12px; color: #444; border: 1px dashed #999;
         padding: 8px 12px; margin: 10px 0 16px; }}
  @media print {{
    body {{ margin: 14mm 12mm 18mm; max-width: none; }}
    .noprint {{ display: none; }}
    img.fig, table {{ page-break-inside: avoid; }}
  }}
  /* Firefox 等支持 @page 页边框时：仅页码、无 file:// 地址 */
  @page {{
    margin: 14mm 12mm 18mm;
    @bottom-center {{
      content: counter(page) " / " counter(pages);
      font-size: 10pt;
      color: #333;
    }}
  }}
</style>
</head>
<body>
<h1>材料热物性测试报告</h1>
<table class="cover">
<tr><td>样品类型：{e(sample_type)}</td>
    <td>报告时间：{e(ctx.get('report_date') or ctx['timestamp'])}</td></tr>
<tr><td>测试时间：{e(ctx['timestamp'])}</td>
    <td>频率范围：{fmt_sci(ctx['freq_min'])} – {fmt_sci(ctx['freq_max'])} Hz</td></tr>
<tr><td>数据点数：{ctx['n_points']}</td>
    <td></td></tr>
</table>
<p class="lab">纤维丝 $3\\omega$ 热物性测量系统</p>
{showcase_html}
<p class="noprint box">
<strong>另存为 PDF（去掉 file:// 地址、保留页码）：</strong><br/>
① 推荐在本软件中直接「导出实验报告」并选择 PDF。<br/>
② 若用浏览器 Ctrl+P：在「更多设置」中<strong>取消勾选「页眉和页脚」</strong>
（Edge/Chrome 会同时去掉左侧网址；该开关无法只留页码）。
取消后若需页码，请改用 Firefox 打印（本报告已设置打印页码），
或改用本软件导出 PDF。
</p>

<h2>一、测量原理</h2>
{principle}

<h2>二、测量过程及测量结果</h2>
<p class="indent">将单根纤维固定于焊盘之间{process_extra}，
置于<strong>精密真空温控控制台</strong>内，腔体真空度控制在
<strong>2&nbsp;Pa 以下</strong>，以消除对流对导热测量的干扰。
在该条件下完成电阻温度系数 $\\alpha_R$ 标定及 $3\\omega$ 热导率扫频测量。
测量长度 $L$ = {fmt_sci(fiber['L'] * 1e3)} mm，直径 $D$ = {fmt_sci(fiber['D'] * 1e6)} μm。
电阻温度系数 $\\alpha_R$ = {fmt_sci(fiber['alpha_R'])} /K
（$\\mathrm{{d}}R/\\mathrm{{d}}T = \\alpha_R\\cdot R$）。
采用最小二乘法拟合目标参数，
成功标志：{e(str(ctx.get('success')))}。</p>

{ctx.get('tcr_html', '')}

<p><strong>待测纤维材料热导率测量结果</strong>
（精密真空温控控制台，真空度 ＜&nbsp;2&nbsp;Pa）：</p>
<table class="result">
<thead>
<tr>
<th>温度<br/>(℃)</th>
<th>纤维长度<br/>(mm)</th>
<th>电阻<br/>(Ω)</th>
<th>电阻温度系数 $\\alpha_R$<br/>(/K)</th>
<th>纤维直径<br/>(μm)</th>
<th>热导率 $\\kappa$<br/>(W/m·K)</th>
<th>热扩散率 $\\alpha$<br/>(m²/s)</th>
<th>密度 $\\rho$<br/>(kg/m³)</th>
<th>比热容 $C_p$<br/>(J/kg·K)</th>
</tr>
</thead>
<tbody>
<tr>
<td class="num">{T_txt}</td>
<td class="num">{fmt_sci(fiber['L'] * 1e3)}</td>
<td class="num">{fmt_sci(fiber['R'])}</td>
<td class="num">{fmt_sci(fiber['alpha_R'])}</td>
<td class="num">{fmt_sci(fiber['D'] * 1e6)}</td>
<td class="num"><b>{fmt_sci(mat['kappa'])}</b></td>
<td class="num"><b>{fmt_sci(alpha)}</b></td>
<td class="num">{fmt_sci(mat['rho'])}</td>
<td class="num">{fmt_sci(mat['C_p'])}</td>
</tr>
</tbody>
</table>
<p class="indent">表中热导率为本次拟合结果；热扩散率由 $\\alpha = \\kappa/(\\rho\\cdot C_p)$ 计算。
测量结果仅对本次样品与本次拟合设置负责。
拟合曲线中虚线为 $\\kappa$ 单独变动 ±20% 时的模型正演结果。</p>

{fig_html}

{appendix_html}

{_REFS_HTML}
</body>
</html>
"""


def html_file_to_pdf(html_path: str, pdf_path: str) -> str:
    """
    将 HTML 报告转为 PDF：页脚仅页码，不含 file:// 地址。

    优先 Playwright（等待 MathJax SVG 渲染完成后再打印）；
    否则用本机 Edge/Chrome headless（加长虚拟时间预算以等待 CDN）。
    返回实际使用的后端名：'playwright' | 'edge' | 'chrome'。
    """
    html_path = os.path.abspath(html_path)
    pdf_path = os.path.abspath(pdf_path)
    if os.path.isfile(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError:
            pass
    uri = pathlib.Path(html_path).as_uri()

    def _pdf_kwargs():
        return dict(
            path=pdf_path,
            format='A4',
            print_background=True,
            display_header_footer=True,
            header_template='<span></span>',
            footer_template=(
                '<div style="width:100%;text-align:center;'
                'font-size:10px;color:#333;padding-top:4px;">'
                '<span class="pageNumber"></span> / '
                '<span class="totalPages"></span></div>'
            ),
            margin={
                'top': '14mm', 'bottom': '18mm',
                'left': '12mm', 'right': '12mm',
            },
        )

    # 1) Playwright：等 MathJax 完成
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sync_playwright = None  # type: ignore

    if sync_playwright is not None:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--font-render-hinting=none'])
            page = browser.new_page()
            page.goto(uri, wait_until='load', timeout=180_000)
            # 等网络与 MathJax（CDN）；失败则再等固定时间
            try:
                page.wait_for_function(
                    'window.__MATHJAX_DONE__ === true',
                    timeout=120_000,
                )
            except Exception:
                page.wait_for_timeout(8000)
            # 再等图片/布局稳定
            page.wait_for_timeout(800)
            n_svg = page.evaluate(
                'document.querySelectorAll("mjx-container svg").length')
            if int(n_svg or 0) < 1:
                # 可能 CDN 失败：再等一会
                page.wait_for_timeout(5000)
            page.pdf(**_pdf_kwargs())
            browser.close()
        return 'playwright'

    # 2) Edge / Chrome headless
    candidates = []
    for env_key in ('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH', 'CHROME_PATH', 'EDGE_PATH'):
        v = os.environ.get(env_key)
        if v:
            candidates.append(v)
    candidates.extend([
        shutil.which('msedge'),
        shutil.which('chrome'),
        shutil.which('google-chrome'),
        r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    ])
    exe = next((c for c in candidates if c and os.path.isfile(c)), None)
    if exe is None:
        raise RuntimeError(
            "未找到 Playwright 或 Edge/Chrome。\n"
            "建议安装以获得正确公式与页码:\n"
            "  pip install playwright\n"
            "  playwright install chromium\n"
            "或用浏览器打开 HTML，等公式全部显示后再 Ctrl+P；"
            "并取消勾选「页眉和页脚」。"
        )

    # 虚拟时间预算：给 MathJax CDN 与排版留时间（毫秒）
    cmd = [
        exe,
        '--headless=new',
        '--disable-gpu',
        '--no-pdf-header-footer',
        '--run-all-compositor-stages-before-draw',
        '--virtual-time-budget=90000',
        f'--print-to-pdf={pdf_path}',
        uri,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if r.returncode != 0 or not os.path.isfile(pdf_path):
        cmd2 = [
            exe, '--headless', '--disable-gpu',
            '--run-all-compositor-stages-before-draw',
            '--virtual-time-budget=90000',
            f'--print-to-pdf={pdf_path}', uri,
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=240)
        if r2.returncode != 0 or not os.path.isfile(pdf_path):
            err = (r.stderr or r.stdout or '') + (r2.stderr or r2.stdout or '')
            raise RuntimeError(
                f"浏览器导出 PDF 失败：{err.strip() or 'unknown error'}\n"
                "可改用: pip install playwright && playwright install chromium"
            )
    backend = 'edge' if 'edge' in exe.lower() else 'chrome'
    return backend
