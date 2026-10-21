"""多层薄膜 / 块体 3ω 实验报告（HTML / PDF），模型为 Olson 多层解析模型。"""

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
from matplotlib.patches import FancyBboxPatch, Rectangle, Arc
import matplotlib.pyplot as plt


_REFS_HTML = """
<h2 id="refs">参考文献</h2>
<ol class="refs">
<li id="ref1">D. G. Cahill, Thermal conductivity measurement from 30 to 750&nbsp;K: the 3ω method,
<i>Rev. Sci. Instrum.</i> <b>61</b>, 802–808 (1990).</li>
<li id="ref2">B. W. Olson, S. Graham, K. Chen, A practical extension of the 3ω method to multilayer structures,
<i>Rev. Sci. Instrum.</i> <b>76</b>, 053901 (2005).</li>
<li id="ref3">T. Borca-Tasciuc, A. R. Kumar, G. Chen, Data reduction in 3ω method for thin-film thermal
conductivity determination, <i>Rev. Sci. Instrum.</i> <b>72</b>, 2139–2147 (2001).</li>
</ol>
"""

PM20_PLUS_COLORS = ['#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22']
PM20_MINUS_COLORS = ['#2ca02c', '#17becf', '#1f77b4', '#7f7f7f', '#98df8a', '#aec7e8']
LAYER_COLORS = ['#BBDEFB', '#C8E6C9', '#FFE0B2', '#E1BEE7', '#B2EBF2', '#F8BBD0', '#DCEDC8']
SUBSTRATE_COLOR = '#CFD8DC'

FIT_MODE_LABELS = {
    'amp': '幅值 |V3ω|',
    'phase': '相位',
    'both': '实部 + 虚部（复数 V3ω）',
    'real': '仅实部',
    'imag': '仅虚部',
}


def _cite(*nums: int) -> str:
    parts = [f'<a class="cite" href="#ref{n}">[{n}]</a>' for n in nums]
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


def _program_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_image_b64(path: str):
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        raw = f.read()
    low = path.lower()
    if raw[:3] == b'\xff\xd8\xff' or low.endswith(('.jpg', '.jpeg')):
        mime = 'image/jpeg'
    else:
        mime = 'image/png'
    return mime, base64.b64encode(raw).decode('ascii')


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


# ---------------------------------------------------------------- 示意图

def _flowchart_html() -> str:
    """嵌入 3omega-measurement-flowchart.html 中的 SVG；文件缺失时不输出。"""
    html_path = os.path.join(_program_dir(), '3omega-measurement-flowchart.html')
    if not os.path.isfile(html_path):
        return ""
    try:
        with open(html_path, encoding='utf-8') as f:
            text = f.read()
    except OSError:
        return ""
    i0 = text.find('<svg')
    i1 = text.find('</svg>')
    if i0 < 0 or i1 <= i0:
        return ""
    svg = text[i0:i1 + len('</svg>')]
    svg = svg.replace('width="1560"', 'width="100%"', 1)
    svg = svg.replace('height="960"', 'height="auto"', 1)
    return (
        f'<div class="flowchart">{svg}</div>'
        f'<p class="figcap">图  $3\\omega$ 法测量系统流程图（激励 / 参考抵消 / 锁相检测）</p>'
    )


def _layer_display_name(layer: dict, idx: int, n: int) -> str:
    name = (layer.get('name') or '').strip()
    role = '基底' if idx == n - 1 else f'第 {idx + 1} 层'
    return f'{role}（{name}）' if name else role


def _schematic_stack_b64(layers: list, sensor: dict) -> str:
    """样品截面示意：顶部加热线 + 各层（GUI 顺序：顶层→基底）+ 界面热阻位置。"""
    _configure_mpl_font()
    n = len(layers)
    thin_h = 0.62
    sub_h = 1.5
    total_h = thin_h * (n - 1) + sub_h
    fig_h = 2.4 + total_h * 0.95
    fig = Figure(figsize=(8.8, fig_h))
    ax = fig.add_subplot(111)
    x0, x1 = 1.4, 7.0
    y_top = total_h + 0.2
    ax.set_xlim(0, 11.4)
    ax.set_ylim(-0.35, y_top + 1.25)
    ax.axis('off')

    y = y_top
    for i, layer in enumerate(layers):
        is_sub = i == n - 1
        h = sub_h if is_sub else thin_h
        color = SUBSTRATE_COLOR if is_sub else LAYER_COLORS[i % len(LAYER_COLORS)]
        ax.add_patch(Rectangle((x0, y - h), x1 - x0, h, facecolor=color,
                               edgecolor='#455A64', lw=1.1))
        if is_sub:
            for k in range(6):
                xx = x0 + 0.25 + k * (x1 - x0 - 0.5) / 5
                ax.plot([xx, xx + 0.18], [y - h + 0.03, y - h - 0.15], color='#607D8B', lw=0.9)
        H_txt = '半无限' if is_sub else f"H = {fmt_sci(layer.get('H', np.nan) * 1e6)} μm"
        k_txt = f"k_y = {fmt_sci(layer.get('k_y'))}"
        if not layer.get('isotropic', True):
            k_txt += f", k_x = {fmt_sci(layer.get('k_x'))}"
        k_txt += " W/(m·K)"
        ax.text(x1 + 0.25, y - h / 2 + 0.12, _layer_display_name(layer, i, n),
                fontsize=10, va='center', fontweight='bold')
        ax.text(x1 + 0.25, y - h / 2 - 0.16, f"{H_txt}；{k_txt}", fontsize=8.6, va='center')

        Rc = float(layer.get('Rc', 0.0) or 0.0)
        if Rc > 0:
            ax.plot([x0, x1], [y, y], color='#C62828', lw=2.0, ls='--')
            ax.text(x1 - 0.06, y - 0.04, f"界面热阻 Rc = {Rc:.2e} m²·K/W", fontsize=7.5,
                    color='#C62828', ha='right', va='top', zorder=4,
                    bbox=dict(facecolor=color, edgecolor='none', pad=0.6, alpha=0.9))
        y -= h

    xc = 0.5 * (x0 + x1)
    w_h = 0.9
    ax.add_patch(Rectangle((xc - w_h / 2, y_top), w_h, 0.16, facecolor='#FFB300',
                           edgecolor='#E65100', lw=1.2))
    ax.annotate('', xy=(xc - w_h / 2, y_top + 0.36), xytext=(xc + w_h / 2, y_top + 0.36),
                arrowprops=dict(arrowstyle='<->', color='#E65100', lw=1.2))
    b = float(sensor.get('b', np.nan))
    ax.text(xc, y_top + 0.52, f"金属加热线/温度计  宽 2b = {fmt_sci(2 * b * 1e6)} μm，"
            f"长 L = {fmt_sci(sensor.get('L', np.nan) * 1e3)} mm（垂直纸面）",
            ha='center', fontsize=9, color='#BF360C')

    for r in (0.45, 0.85, 1.3):
        ax.add_patch(Arc((xc, y_top), 2 * r * 1.25, 2 * r, theta1=180, theta2=360,
                         ls=':', ec='#D84315', lw=1.1, alpha=0.8))

    ax.annotate('', xy=(0.15, y_top - 1.0), xytext=(0.15, y_top + 0.9),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.3))
    ax.text(0.27, y_top - 0.85, 'y（深度）', color='#1565C0', fontsize=9)
    ax.annotate('', xy=(1.05, y_top + 0.9), xytext=(0.15, y_top + 0.9),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.3))
    ax.text(0.6, y_top + 1.0, 'x（面内）', color='#1565C0', fontsize=9, ha='center')
    fig.tight_layout()
    return _fig_to_png_b64(fig, dpi=150)


def _schematic_heater_top_b64(sensor: dict) -> str:
    """加热线俯视示意：四端引线（I+、I−、V+、V−）与测量长度 L。"""
    _configure_mpl_font()
    fig = Figure(figsize=(8.6, 2.7))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.0)
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.3, 0.25), 9.4, 2.5, boxstyle='round,pad=0.02,rounding_size=0.1',
                                facecolor='#ECEFF1', edgecolor='#90A4AE', lw=1.0))
    ax.text(0.5, 2.5, '样品表面（俯视）', fontsize=9, color='#546E7A')
    ax.plot([1.6, 8.4], [1.5, 1.5], color='#E65100', lw=4)
    for x, lab in ((1.1, 'I+'), (8.9, 'I−')):
        ax.add_patch(Rectangle((x - 0.45, 1.15), 0.9, 0.7, facecolor='#FFCC80',
                               edgecolor='#E65100', lw=1.1))
        ax.text(x, 1.5, lab, ha='center', va='center', fontsize=10)
    for x, lab in ((3.0, 'V+'), (7.0, 'V−')):
        ax.plot([x, x], [1.5, 2.15], color='#E65100', lw=2)
        ax.add_patch(Rectangle((x - 0.35, 2.15), 0.7, 0.45, facecolor='#FFCC80',
                               edgecolor='#E65100', lw=1.1))
        ax.text(x, 2.37, lab, ha='center', va='center', fontsize=9)
    ax.annotate('', xy=(3.0, 1.05), xytext=(7.0, 1.05),
                arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=1.4))
    ax.text(5.0, 0.72, f"测量长度 L = {fmt_sci(sensor.get('L', np.nan) * 1e3)} mm"
            f"（电压引线之间）；线宽 2b = {fmt_sci(2 * sensor.get('b', np.nan) * 1e6)} μm",
            ha='center', fontsize=9.5, color='#2E7D32')
    fig.tight_layout()
    return _fig_to_png_b64(fig, dpi=140)


# ---------------------------------------------------------------- 原理

def _principle_block(ctx: dict) -> str:
    layers = ctx['layers']
    sensor = ctx['sensors'][0]['params']
    n = len(layers)
    stack_b64 = _schematic_stack_b64(layers, sensor)
    top_b64 = _schematic_heater_top_b64(sensor)
    flow = _flowchart_html()

    part1 = (
        f'<h3>1.1 $3\\omega$ 法基本原理</h3>'
        f'<p class="indent">$3\\omega$ 法由 Cahill 提出{_cite(1)}。在样品表面沉积一条细长金属线，'
        f'线宽记为 $2b$、电压引线间长度记为 $L$，它同时充当<strong>加热器</strong>与'
        f'<strong>电阻温度计</strong>。通以角频率为 $\\omega$ 的交流电流时，焦耳热含直流和 $2\\omega$ 分量，'
        f'使样品表面温度以 $2\\omega$ 振荡；金属线电阻随温度变化，电阻也以 $2\\omega$ 振荡，'
        f'与 $\\omega$ 的电流相乘后在电压中出现 $3\\omega$ 分量。该分量的幅值与相位携带样品热物性信息，'
        f'由锁相放大器提取。</p>'
        f'<img class="fig" alt="加热线俯视示意" src="data:image/png;base64,{top_b64}"/>'
        f'<p class="figcap">图  加热线俯视示意（四端法：外侧通电流，内侧测电压）</p>'
    )
    part1 += r"""
<p class="indent">激励电流为</p>
$$I(t)=I_0\cos(\omega t)$$
<p class="indent">其中 $I_0$ 为电流峰值，$\omega=2\pi f$ 为激励角频率，$f$ 为激励频率（扫频变量）。加热功率为</p>
$$P(t)=I^2(t)R_0=\tfrac12 I_0^2R_0\bigl[1+\cos(2\omega t)\bigr]$$
<p class="indent">$R_0$ 为加热线在环境温度下的电阻。功率中的 $\cos(2\omega t)$ 项使加热线平均温升 $\Delta T$ 以 $2\omega$ 振荡，
电阻随之变化：</p>
$$R(t)=R_0\bigl[1+\alpha_R\,\Delta T_{2\omega}\cos(2\omega t+\varphi)\bigr],\qquad
\alpha_R=\frac{1}{R_0}\frac{\mathrm dR}{\mathrm dT}$$
<p class="indent">$\alpha_R$ 为电阻温度系数，$\varphi$ 为温升相对加热功率的相位滞后。电压 $V=IR$ 中由此出现 $3\omega$ 项，
取有效值后得到温升与 $3\omega$ 电压的关系：</p>
$$\Delta T_{2\omega}=\frac{2\,V_{3\omega}}{\alpha_R\,V_{1\omega}}$$
<p class="indent">$V_{1\omega}$ 为加热线两端基波电压有效值，$V_{3\omega}$ 为三次谐波电压（复数：实部为与激励同相的分量，
虚部为正交分量）。由于 $V_{3\omega}$ 通常比 $V_{1\omega}$ 小 3 个数量级以上，测量时先用参考电阻或数控电阻
抵消 $1\omega$ 大信号，再由锁相放大器在 $3$ 倍参考频率处检测。</p>
"""
    part1 += flow

    part2 = (
        f'<h3>1.2 多层结构导热模型（Olson 模型）</h3>'
        f'<p class="indent">本样品为 {n} 层结构（含基底），采用 Olson、Graham、Chen 提出的多层解析模型{_cite(2)}，'
        f'各向异性处理参考 Borca-Tasciuc 等{_cite(3)}。样品截面示意如下（层厚未按比例绘制，红色虚线表示设置了'
        f'界面热阻的位置，虚线圆弧示意热波由加热线向下扩散）：</p>'
        f'<img class="fig" alt="多层样品截面示意" src="data:image/png;base64,{stack_b64}"/>'
        f'<p class="figcap">图  多层样品截面与加热线示意（第 1 层为顶层，最后一层为基底）</p>'
    )
    part2 += r"""
<p class="indent">模型假设：(1) 加热线足够长（$L\gg b$），可按二维截面处理；(2) 各层在面内无限延伸，基底在深度方向
半无限厚；(3) 加热线宽度方向热流均匀；(4) 每层可为各向异性，面内与法向热导率分别为 $k_x$、$k_y$；
(5) 每层上表面可有一个界面热阻 $R_c$（单位 m²·K/W），最顶层的 $R_c$ 即加热线与样品之间的接触热阻。</p>
<p class="indent">第 $n$ 层内温度满足热扩散方程：</p>
$$\rho_n c_n\frac{\partial T}{\partial t}=k_{x,n}\frac{\partial^2T}{\partial x^2}+k_{y,n}\frac{\partial^2T}{\partial y^2}$$
<p class="indent">其中 $x$ 为面内、垂直于加热线的方向，$y$ 为深度方向；$\rho_n$、$c_n$ 为该层密度与比热容。
对 $x$ 做傅里叶变换（把加热线的空间分布分解为一系列正弦热波），并引入无量纲波数 $\chi$（波数乘以 $b$）；
温度随时间以 $2\omega$ 振荡，时间导数变为 $i\cdot 2\omega=i\,4\pi f$。于是每层得到复数“衰减系数”：</p>
$$\phi_n=\sqrt{\psi_n\chi^2+\frac{i\,4\pi f\,b^2}{\alpha_{y,n}}},\qquad
\psi_n=\frac{k_{x,n}}{k_{y,n}},\qquad \alpha_{y,n}=\frac{k_{y,n}}{\rho_n c_n}$$
<p class="indent">$\psi_n$ 为各向异性比（各向同性时为 1），$\alpha_{y,n}$ 为法向热扩散率。
定义某一空间频率分量下表面温度与热流之比为“传递阻抗” $z(\chi,f)$，模型按 <strong>基底 → 顶层</strong> 编号
$n=1,\dots,N$ 逐层递推。半无限基底为</p>
$$z_1=-\frac{b}{k_{y,1}\,\phi_1}$$
<p class="indent">每层上表面的界面热阻串联到该层的阻抗上（负号与上式的符号约定一致）：</p>
$$\tilde z_n=z_n-R_{c,n}$$
<p class="indent">第 $n$ 层（厚度 $H_n$）把下方的 $\tilde z_{n-1}$ 变换为本层上表面的 $z_n$：</p>
$$z_n=\frac{b}{k_{y,n}\phi_n}\cdot
\frac{\dfrac{k_{y,n}\phi_n}{b}\,\tilde z_{n-1}-\tanh\!\Bigl(\dfrac{\phi_nH_n}{b}\Bigr)}
{1-\dfrac{k_{y,n}\phi_n}{b}\,\tilde z_{n-1}\tanh\!\Bigl(\dfrac{\phi_nH_n}{b}\Bigr)}$$
<p class="indent">$\tanh$ 项描述热波穿过该层时的衰减：层很厚（$\phi_nH_n/b\gg1$）时 $z_n$ 退化为该层自身的半无限解，
下层不再可见；层很薄时 $z_n\approx\tilde z_{n-1}$ 加上一个薄层热阻。最后把顶层阻抗按加热线宽度做加权平均：</p>
$$Z(f)=\frac{2}{\pi}\int_0^{4\pi}\left(\frac{\sin\chi}{\chi}\right)^{2}\tilde z_N(\chi,f)\,\mathrm d\chi$$
<p class="indent">$(\sin\chi/\chi)^2$ 来自两次“宽度平均”：加热线在 $-b\sim b$ 内均匀发热（矩形热源的傅里叶变换为
$\sin\chi/\chi$），测得的是线宽范围内的平均温度（再乘一次 $\sin\chi/\chi$）。积分上限截断到 $4\pi$ 引入的误差可忽略。
程序中该积分用对数分段 Gauss–Legendre 求积计算。</p>
<p class="indent">模型 $Z$ 与实测 $V_{3\omega}$ 的换算：先由电压得到实验阻抗</p>
$$\hat Z=-\frac{4R^2\,b\,L}{V_{1\omega}^3\,(\mathrm dR/\mathrm dT)}\,V_{3\omega}$$
<p class="indent">$R$ 为加热线电阻，$\mathrm dR/\mathrm dT=\alpha_R R$。再扣除加热线自身热容（厚度 $H_e$、体积热容
$(\rho c)_e$）的影响：</p>
$$\frac{1}{Z}=\frac{1}{\hat Z}+i\,4\pi f\,H_e\,(\rho c)_e$$
<p class="indent">等价地，加热线平均温升为 $\Delta T=-\dfrac{P}{2bL}\,Z$，其中 $P=V_{1\omega}^2/R$ 为加热功率。</p>
<p class="indent"><strong>参数拟合：</strong>以选定的待拟合参数 $\mathbf p$（如某层 $k_y$、$k_x$、厚度、界面热阻等）为自变量，
用阻尼最小二乘法（Levenberg–Marquardt / 信赖域）最小化</p>
$$S(\mathbf p)=\sum_{j}\Bigl[y_j^{\mathrm{exp}}-y_j^{\mathrm{model}}(\mathbf p)\Bigr]^2$$
<p class="indent">$y_j$ 为第 $j$ 个频点上的拟合目标（幅值、相位、实部或虚部，由拟合目标决定），上标 exp 为实验值、model 为模型值。
收敛后由灵敏度矩阵 $\mathbf J$ 与残差估计参数协方差 $\mathrm{Cov}=s^2(\mathbf J^{\mathsf T}\mathbf J)^{-1}$，
其中 $s^2=\mathrm{RSS}/(N-M)$，RSS 为残差平方和，$N$ 为残差个数，$M$ 为拟合参数个数；
协方差对角元开方即参数标准误差 $\sigma$。</p>
"""

    part3 = ""
    if ctx.get('slope'):
        part3 = (
            f'<h3>1.3 斜率法（交叉验证）</h3>'
            f'<p class="indent">当热穿透深度 $\\lambda=\\sqrt{{\\alpha/(4\\pi f)}}$（$\\alpha$ 为热扩散率）满足 '
            f'$b\\ll\\lambda\\ll$ 被测层厚度，且上方各层已被“穿透”时，同相温升随 $\\ln f$ 线性下降{_cite(1)}，斜率只取决于'
            f'被测层的热导率：</p>'
            r'$$k=-\frac{P}{2\pi L}\left(\frac{\mathrm d\,\Delta T_{\mathrm{in}}}{\mathrm d\ln f}\right)^{-1}$$'
            f'<p class="indent">$\\Delta T_{{\\mathrm{{in}}}}$ 为同相（实部）温升。对各向异性层，该斜率给出的是 '
            f'$\\sqrt{{k_xk_y}}$；界面热阻只平移截距、不改变斜率。此方法不依赖全模型，可用于检验拟合结果。</p>'
        )
    return part1 + part2 + part3


# ---------------------------------------------------------------- 图与数据

def report_plot_kind(fit_mode_key: str) -> str:
    return {
        'amp': 'amp',
        'phase': 'phase',
        'both': 're_im',
        'real': 'real',
        'imag': 'imag',
    }.get(fit_mode_key, 'amp_phase')


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


def _add_pm20(ax, freqs, pm20, kind: str, *, log_y=False):
    plot_fn = ax.loglog if log_y else ax.semilogx
    for i, s in enumerate(pm20 or []):
        cp = PM20_PLUS_COLORS[i % len(PM20_PLUS_COLORS)]
        cm = PM20_MINUS_COLORS[i % len(PM20_MINUS_COLORS)]
        y_p = _y_of(s.get('V3_plus'), kind)
        y_m = _y_of(s.get('V3_minus'), kind)
        if y_p is not None and np.any(np.isfinite(y_p)):
            plot_fn(freqs, y_p, '--', color=cp, label=f"{s['label']} +20%", lw=1.3, alpha=0.9)
        if y_m is not None and np.any(np.isfinite(y_m)):
            plot_fn(freqs, y_m, ':', color=cm, label=f"{s['label']} −20%", lw=1.5, alpha=0.9)


def _style_ax(ax, ylabel, title=None, xlabel=True):
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel('激励频率 f (Hz)')
    if title:
        ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, which='both', ls='--', alpha=0.35)


def make_fit_report_plot(freqs, V3_exp, V3_fit, title: str, plot_kind: str = 'amp_phase',
                         pm20=None):
    """按拟合目标生成报告图与数据表；pm20 为 [{label, V3_plus, V3_minus}]。"""
    _configure_mpl_font()
    freqs = np.asarray(freqs, dtype=float)
    Ve = np.asarray(V3_exp, dtype=complex)
    Vm = np.asarray(V3_fit, dtype=complex)
    pha_e, pha_m = np.angle(Ve, deg=True), np.angle(Vm, deg=True)
    amp_e, amp_m = np.abs(Ve) * 1e6, np.abs(Vm) * 1e6
    pm_note = '及各拟合参数 ±20%' if pm20 else ''

    def single(kind, ylabel, ye, ym, log_y, name):
        fig = Figure(figsize=(8.4, 4.4))
        ax = fig.add_subplot(111)
        plot_fn = ax.loglog if log_y else ax.semilogx
        plot_fn(freqs, ye, 'ro', label='实验', markersize=4.5)
        plot_fn(freqs, ym, 'b-', label='拟合', lw=2)
        _add_pm20(ax, freqs, pm20, kind, log_y=log_y)
        _style_ax(ax, ylabel, f'{title} — {name}')
        fig.tight_layout()
        return _fig_to_png_b64(fig)

    def double(k1, l1, e1, m1, log1, k2, l2, e2, m2, name):
        fig = Figure(figsize=(8.4, 6.6))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        (ax1.loglog if log1 else ax1.semilogx)(freqs, e1, 'ro', label='实验', markersize=4.5)
        (ax1.loglog if log1 else ax1.semilogx)(freqs, m1, 'b-', label='拟合', lw=2)
        _add_pm20(ax1, freqs, pm20, k1, log_y=log1)
        _style_ax(ax1, l1, f'{title} — {name}', xlabel=False)
        ax2.semilogx(freqs, e2, 'ro', label='实验', markersize=4.5)
        ax2.semilogx(freqs, m2, 'b-', label='拟合', lw=2)
        _add_pm20(ax2, freqs, pm20, k2)
        _style_ax(ax2, l2)
        fig.tight_layout()
        return _fig_to_png_b64(fig)

    if plot_kind == 'phase':
        b64 = single('pha', '相位 (°)', pha_e, pha_m, False, '相位')
        cap = f'图  $V_{{3\\omega}}$ 相位与频率的关系（实验、拟合{pm_note}）'
        headers = ['频率 (Hz)', '实验相位 (°)', '拟合相位 (°)']
        rows = [(freqs[i], pha_e[i], pha_m[i]) for i in range(freqs.size)]
    elif plot_kind == 'amp':
        b64 = single('amp', '|V3ω| (μV)', amp_e, amp_m, True, '幅值')
        cap = f'图  $V_{{3\\omega}}$ 幅值与频率的关系（实验、拟合{pm_note}）'
        headers = ['频率 (Hz)', '实验 |V3| (μV)', '拟合 |V3| (μV)']
        rows = [(freqs[i], amp_e[i], amp_m[i]) for i in range(freqs.size)]
    elif plot_kind == 'real':
        b64 = single('re', 'Re(V3ω) (μV)', Ve.real * 1e6, Vm.real * 1e6, False, '实部')
        cap = f'图  $V_{{3\\omega}}$ 实部与频率的关系（实验、拟合{pm_note}）'
        headers = ['频率 (Hz)', '实验 Re(V3) (μV)', '拟合 Re(V3) (μV)']
        rows = [(freqs[i], Ve.real[i] * 1e6, Vm.real[i] * 1e6) for i in range(freqs.size)]
    elif plot_kind == 'imag':
        b64 = single('im', 'Im(V3ω) (μV)', Ve.imag * 1e6, Vm.imag * 1e6, False, '虚部')
        cap = f'图  $V_{{3\\omega}}$ 虚部与频率的关系（实验、拟合{pm_note}）'
        headers = ['频率 (Hz)', '实验 Im(V3) (μV)', '拟合 Im(V3) (μV)']
        rows = [(freqs[i], Ve.imag[i] * 1e6, Vm.imag[i] * 1e6) for i in range(freqs.size)]
    elif plot_kind == 're_im':
        b64 = double('re', 'Re(V3ω) (μV)', Ve.real * 1e6, Vm.real * 1e6, False,
                     'im', 'Im(V3ω) (μV)', Ve.imag * 1e6, Vm.imag * 1e6, '实部 / 虚部')
        cap = f'图  $V_{{3\\omega}}$ 实部、虚部与频率的关系（实验、拟合{pm_note}）'
        headers = ['频率 (Hz)', '实验 Re (μV)', '拟合 Re (μV)', '实验 Im (μV)', '拟合 Im (μV)']
        rows = [(freqs[i], Ve.real[i] * 1e6, Vm.real[i] * 1e6, Ve.imag[i] * 1e6, Vm.imag[i] * 1e6)
                for i in range(freqs.size)]
    else:
        b64 = double('amp', '|V3ω| (μV)', amp_e, amp_m, True,
                     'pha', '相位 (°)', pha_e, pha_m, '幅值 / 相位')
        cap = f'图  $V_{{3\\omega}}$ 幅值、相位与频率的关系（实验、拟合{pm_note}）'
        headers = ['频率 (Hz)', '实验 |V3| (μV)', '拟合 |V3| (μV)', '实验相位 (°)', '拟合相位 (°)']
        rows = [(freqs[i], amp_e[i], amp_m[i], pha_e[i], pha_m[i]) for i in range(freqs.size)]

    return {'figures': [(cap, b64)], 'data_headers': headers, 'data_rows': rows}


def _make_slope_figure(slope: dict) -> str:
    _configure_mpl_font()
    f = np.asarray(slope['freqs'], dtype=float)
    dT = np.asarray(slope['dT_in'], dtype=float)
    sel = np.asarray(slope['sel'], dtype=bool)
    fig = Figure(figsize=(8.0, 4.0))
    ax = fig.add_subplot(111)
    ax.semilogx(f[~sel], dT[~sel], 'o', color='#B0BEC5', ms=4, label='窗口外数据')
    ax.semilogx(f[sel], dT[sel], 'o', color='#1565C0', ms=5, label='线性回归数据')
    f_line = np.logspace(np.log10(slope['f_lo']), np.log10(slope['f_hi']), 60)
    ax.semilogx(f_line, slope['intercept'] + slope['slope'] * np.log(f_line), '-',
                color='#d62728', lw=1.6, label=f"线性拟合  k = {fmt_sci(slope['k'])} W/(m·K)")
    ax.set_xlabel('激励频率 f (Hz)')
    ax.set_ylabel('同相温升 ΔT_in (K)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', ls='--', alpha=0.35)
    fig.tight_layout()
    return _fig_to_png_b64(fig)


# ---------------------------------------------------------------- 表格

def _val_cell(value, sigma=None, fitted=False, scale=1.0, digits=4):
    if value is None:
        return "<td class='num'>—</td>"
    txt = fmt_sci(float(value) * scale, digits)
    if sigma is not None and np.isfinite(sigma[0]):
        txt += f"<br/><span class='sig'>±{fmt_sci(sigma[0] * scale, 3)} ({sigma[1] * 100:.1f}%)</span>"
    if fitted:
        return f"<td class='num fit'><b>{txt}</b></td>"
    return f"<td class='num'>{txt}</td>"


def _layer_table_html(ctx: dict) -> str:
    layers = ctx['layers']
    n = len(layers)
    rows = []
    for i, L in enumerate(layers):
        fitted = set(L.get('fitted') or [])
        sig = L.get('sigma') or {}
        iso = L.get('isotropic', True)
        is_sub = i == n - 1
        try:
            alpha_y = float(L['k_y']) / (float(L['rho']) * float(L['cap']))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            alpha_y = np.nan
        kx_fitted = 'k_x' in fitted or (iso and 'k_y' in fitted)
        kx_sig = sig.get('k_x') or (sig.get('k_y') if iso else None)
        H_cell = ("<td class='num'>半无限</td>" if is_sub
                  else _val_cell(L.get('H'), sig.get('H'), 'H' in fitted, 1e6))
        rows.append(
            f"<tr><td>{html_lib.escape(_layer_display_name(L, i, n))}</td>"
            f"{H_cell}"
            f"{_val_cell(L.get('k_y'), sig.get('k_y'), 'k_y' in fitted)}"
            f"{_val_cell(L.get('k_x'), kx_sig, kx_fitted)}"
            f"<td class='num'>{'是' if iso else '否'}</td>"
            f"{_val_cell(L.get('rho'), sig.get('rho'), 'rho' in fitted)}"
            f"{_val_cell(L.get('cap'), sig.get('cap'), 'cap' in fitted)}"
            f"<td class='num'>{fmt_sci(alpha_y)}</td>"
            f"{_val_cell(L.get('Rc'), sig.get('Rc'), 'Rc' in fitted)}</tr>"
        )
    return (
        "<table class='result'><thead><tr>"
        "<th>层</th><th>厚度 $H$<br/>(μm)</th><th>法向热导率 $k_y$<br/>(W/m·K)</th>"
        "<th>面内热导率 $k_x$<br/>(W/m·K)</th><th>各向<br/>同性</th><th>密度 $\\rho$<br/>(kg/m³)</th>"
        "<th>比热容 $c$<br/>(J/kg·K)</th><th>法向热扩散率 $\\alpha_y$<br/>(m²/s)</th>"
        "<th>上表面界面热阻 $R_c$<br/>(m²·K/W)</th>"
        "</tr></thead><tbody>" + ''.join(rows) + "</tbody></table>"
    )


_SENSOR_ROWS = [
    ('加热线半宽 $b$', 'μm', 'b', 1e6),
    ('测量长度 $L$', 'mm', 'L', 1e3),
    ('加热线电阻 $R$', 'Ω', 'R', 1.0),
    ('电阻温度系数 $\\alpha_R$', '1/K', 'alpha_R', 1.0),
    ('$\\mathrm dR/\\mathrm dT=\\alpha_R R$', 'Ω/K', 'dR_dT', 1.0),
    ('基波电压有效值 $V_{1\\omega}$', 'V', 'V_rms', 1.0),
    ('加热功率 $P=V_{1\\omega}^2/R$', 'mW', 'P', 1e3),
    ('线功率 $P/L$', 'W/m', 'P_per_L', 1.0),
    ('加热线厚度 $H_e$', 'nm', 'H_e', 1e9),
    ('加热线体积热容 $(\\rho c)_e$', 'J/(m³·K)', 'rho_C_e', 1.0),
]


def _sensor_table_html(ctx: dict) -> str:
    """ctx['sensors']: [{label, params, fitted, sigma}]；多线宽联合拟合时每个数据集一列。"""
    sensors = ctx['sensors']
    cols = []
    for s in sensors:
        p = dict(s['params'])
        try:
            p['P'] = float(p['V_rms']) ** 2 / float(p['R'])
            p['P_per_L'] = p['P'] / float(p['L'])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            p['P'] = p['P_per_L'] = np.nan
        cols.append((p, set(s.get('fitted') or []), s.get('sigma') or {}))
    head = ''.join(f"<th>{html_lib.escape(s.get('label') or '数值')}</th>" for s in sensors)
    rows = []
    for name, unit, key, scale in _SENSOR_ROWS:
        cells = ''.join(_val_cell(p.get(key), sig.get(key), key in fitted, scale)
                        for p, fitted, sig in cols)
        rows.append(f"<tr><td>{name}</td><td class='num'>{unit}</td>{cells}</tr>")
    cls = 'result narrow' if len(sensors) == 1 else 'result'
    return (f"<table class='{cls}'><thead><tr><th>参数</th><th>单位</th>{head}</tr></thead>"
            "<tbody>" + ''.join(rows) + "</tbody></table>")


DEFAULT_INSTRUMENT_PHOTO = os.path.join(_program_dir(), 'assets', 'harmonic_one.png')


def _instrument_showcase_html(extra_images) -> str:
    """封面仪器图：用户指定照片优先，否则用 assets/harmonic_one.png。"""
    e = html_lib.escape
    images = list(extra_images or [])
    if images:
        out = ""
        for path, cap in images:
            got = _load_image_b64(path)
            if got:
                mime, b64 = got
                out += (f'<img class="fig photo" alt="{e(cap)}" src="data:{mime};base64,{b64}"/>'
                        f'<p class="figcap">图  {e(cap)}</p>')
        if out:
            return out
    got = _load_image_b64(DEFAULT_INSTRUMENT_PHOTO)
    if not got:
        return ""
    mime, b64 = got
    return (
        '<p class="indent"><strong>测量主机</strong>为 <strong>Harmonic ONE</strong>（AUTINST），'
        '集成 $3\\omega$ 激励、参考抵消与锁相检测，完成扫频采集；数据由本软件按多层 Olson 模型拟合。</p>'
        f'<img class="fig photo" alt="Harmonic ONE" src="data:{mime};base64,{b64}"/>'
        '<p class="figcap">图  测量仪器 Harmonic ONE（AUTINST）</p>'
    )


def _data_table_html(headers, rows) -> str:
    e = html_lib.escape
    th = "".join(f"<th>{e(h)}</th>" for h in headers)
    trs = ["<tr>" + "".join(f"<td class='num'>{fmt_sci(v, 5)}</td>" for v in row) + "</tr>"
           for row in rows]
    return f"<table class='data'><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _fit_quality(Ve, Vm):
    Ve = np.asarray(Ve, dtype=complex)
    Vm = np.asarray(Vm, dtype=complex)
    amp_e = np.abs(Ve)
    rel_amp = np.sqrt(np.mean(((np.abs(Vm) - amp_e) / np.where(amp_e > 0, amp_e, np.nan)) ** 2))
    d_pha = np.angle(Vm / Ve, deg=True)
    return float(rel_amp), float(np.sqrt(np.mean(d_pha ** 2)))


# ---------------------------------------------------------------- 主体

def build_film_experiment_html(ctx: dict) -> str:
    """
    ctx 字段：
      info: {sample_name, sample_id, temperature, environment, operator, test_date, report_date, note, data_file}
      sensors: [{label, params: {b, L, R, alpha_R, dR_dT, V_rms, H_e, rho_C_e}, fitted, sigma}]
      layers: GUI 顺序（顶层→基底）[{name, H, k_y, k_x, rho, cap, Rc, isotropic, fitted, sigma}]
      fit: {mode_text, fit_mode_code, cancelled, converged, n_params, uncertainty}
      datasets: [{label, freqs, V3_exp, V3_fit, pm20}]
      slope: slope_method_estimate 结果（可选）
      include_pm20 / include_appendix
    """
    e = html_lib.escape
    info = ctx.get('info') or {}
    fit = ctx.get('fit') or {}
    datasets = ctx.get('datasets') or []
    layers = ctx['layers']
    n = len(layers)
    principle = _principle_block(ctx)

    all_f = np.concatenate([np.asarray(d['freqs'], dtype=float) for d in datasets]) if datasets else np.array([np.nan])
    n_points = int(sum(len(d['freqs']) for d in datasets))
    mode_code = fit.get('fit_mode_code', 'both')
    kind = report_plot_kind(mode_code)
    method_label = FIT_MODE_LABELS.get(mode_code, mode_code)

    fig_html = ""
    appendix_html = ""
    quality_rows = []
    for di, ds in enumerate(datasets):
        title = ds.get('label') or '拟合结果'
        plot = make_fit_report_plot(ds['freqs'], ds['V3_exp'], ds['V3_fit'], title, kind,
                                    pm20=ds.get('pm20') if ctx.get('include_pm20', True) else None)
        for cap, b64 in plot['figures']:
            fig_html += (f'<img class="fig" alt="拟合曲线" src="data:image/png;base64,{b64}"/>'
                         f'<p class="figcap">{cap}{"（" + e(title) + "）" if len(datasets) > 1 else ""}</p>')
        rel_amp, rms_pha = _fit_quality(ds['V3_exp'], ds['V3_fit'])
        quality_rows.append(
            f"<tr><td>{e(title)}</td><td class='num'>{len(ds['freqs'])}</td>"
            f"<td class='num'>{fmt_sci(np.min(ds['freqs']))} – {fmt_sci(np.max(ds['freqs']))}</td>"
            f"<td class='num'>{rel_amp * 100:.3g}%</td><td class='num'>{rms_pha:.3g}°</td></tr>")
        if ctx.get('include_appendix', True):
            appendix_html += (f"<p class='figcap'>附表 {di + 1}  {e(title)} 拟合所用数据</p>"
                              + _data_table_html(plot['data_headers'], plot['data_rows']))
    if appendix_html:
        appendix_html = ("<h2>附录  拟合数据</h2><p class='indent'>下表列出本次拟合实际使用的频点"
                         "（已应用频率范围筛选、剔点与信号校正）及对应的实验值与模型值。</p>" + appendix_html)

    quality_html = (
        "<table class='data'><thead><tr><th>数据集</th><th>点数</th><th>频率范围 (Hz)</th>"
        "<th>幅值相对均方根偏差</th><th>相位均方根偏差</th></tr></thead><tbody>"
        + ''.join(quality_rows) + "</tbody></table>"
    ) if quality_rows else ""

    unc = fit.get('uncertainty') or {}
    if unc.get('error'):
        unc_html = f"<p class='indent'>参数标准误差未能估计：{e(str(unc['error']))}。</p>"
    elif unc:
        unc_html = (f"<p class='indent'>由拟合残差估计的测量噪声水平 $s$ = {fmt_sci(unc.get('noise_std'))}"
                    f"（残差自由度 {unc.get('dof')}）；表中加粗数值为拟合参数，其下方为标准误差 $\\pm\\sigma$"
                    f"及相对误差。该误差只反映随机噪声，不含传感器参数（$b$、$L$、$\\alpha_R$ 等）与固定参数"
                    f"取值偏差带来的系统误差。</p>")
    else:
        unc_html = ""

    slope_html = ""
    slope = ctx.get('slope')
    if slope:
        slope_b64 = _make_slope_figure(slope)
        slope_html = (
            "<h3>3.4 斜率法交叉验证</h3>"
            f"<p class='indent'>在 {fmt_sci(slope['f_lo'])} – {fmt_sci(slope['f_hi'])} Hz"
            f"（{slope['decades']:.2f} 个数量级，{slope['n']} 个点）内，对同相温升 $\\Delta T_{{\\mathrm{{in}}}}$ "
            f"与 $\\ln f$ 做线性回归，斜率 = {fmt_sci(slope['slope'])} K，$R^2$ = {slope['r2']:.5f}，"
            f"得到 $k$ = <b>{fmt_sci(slope['k'])}</b> ± {fmt_sci(slope['k_se'], 3)} W/(m·K)"
            f"（各向异性层对应 $\\sqrt{{k_xk_y}}$）。"
            f"{e(slope.get('compare_text', ''))}</p>"
            f'<img class="fig" alt="斜率法" src="data:image/png;base64,{slope_b64}"/>'
            f'<p class="figcap">图  同相温升与频率对数的线性回归（斜率法）</p>'
        )

    pm_note = ("图中虚线 / 点线分别为对应拟合参数单独增大 / 减小 20% 时的模型曲线，"
               "曲线分离越明显说明数据对该参数越敏感。"
               if ctx.get('include_pm20', True) else "")
    if fit.get('cancelled'):
        status = "拟合被中止，以下为中止时的中间结果。"
    elif fit.get('converged') is False:
        status = "注意：拟合达到最大迭代次数仍未满足收敛判据，结果仅供参考。"
    elif fit.get('converged'):
        status = "拟合已收敛。"
    else:
        status = "拟合已完成。"

    temp = info.get('temperature')
    temp_txt = f"{e(str(temp))} ℃" if temp not in (None, "") else "—"
    data_file = info.get('data_file') or "—"
    note = info.get('note') or ""

    showcase = _instrument_showcase_html(ctx.get('extra_images'))

    structure = " / ".join(
        (L.get('name') or f'第{i + 1}层') for i, L in enumerate(layers))

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
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
  body {{ font-family: "SimSun", "Songti SC", "Times New Roman", serif;
         color: #111; max-width: 900px; margin: 28px auto; padding: 0 22px;
         line-height: 1.7; font-size: 14px; }}
  h1 {{ text-align: center; font-size: 22px; letter-spacing: 4px; margin: 8px 0 18px; }}
  h2 {{ font-size: 16px; margin: 1.4em 0 0.6em; }}
  h3 {{ font-size: 14.5px; margin: 1.1em 0 0.4em; }}
  .cover {{ width: 100%; border-collapse: collapse; margin: 0 0 18px; }}
  .cover td {{ border: none; padding: 3px 8px; }}
  .lab {{ text-align: center; font-size: 15px; margin: 6px 0 20px; }}
  table.result, table.data {{ border-collapse: collapse; width: 100%; margin: 10px 0 16px; font-size: 12.5px; }}
  table.narrow {{ width: 75%; margin-left: auto; margin-right: auto; }}
  table.result th, table.result td, table.data th, table.data td {{ border: 1px solid #333; padding: 4px 6px; }}
  table.result th, table.data th {{ background: #f3f3f3; text-align: center; }}
  td.num {{ font-family: "Times New Roman", Consolas, serif; text-align: center; }}
  td.fit {{ background: #FFF8E1; }}
  span.sig {{ font-size: 11px; color: #444; font-weight: normal; }}
  mjx-container {{ font-size: 105% !important; }}
  mjx-container[display="true"] {{ margin: 0.6em 0 !important; overflow-x: auto; overflow-y: hidden; }}
  html:not(.mathjax-ready) mjx-container {{ visibility: hidden; }}
  img.fig {{ max-width: 100%; height: auto; display: block; margin: 10px auto; }}
  img.photo {{ max-width: 640px; }}
  .flowchart {{ width: 100%; margin: 10px auto 4px; background: #f7f4ee;
               border: 1px solid #e6e0d4; border-radius: 8px; }}
  .flowchart svg {{ display: block; width: 100%; height: auto; }}
  .figcap {{ text-align: center; font-size: 13px; margin: 0 0 14px; }}
  .indent {{ text-indent: 2em; }}
  ol.refs {{ font-size: 13px; line-height: 1.55; padding-left: 1.6em; }}
  a.cite {{ color: #1565C0; text-decoration: none; font-weight: 600; }}
  .box {{ font-size: 12px; color: #444; border: 1px dashed #999; padding: 8px 12px; margin: 10px 0 16px; }}
  @media print {{
    body {{ margin: 0; max-width: none; }}
    .noprint {{ display: none; }}
    img.fig, table.result, .flowchart {{ page-break-inside: avoid; }}
    table.data thead {{ display: table-header-group; }}
    table.data tr {{ page-break-inside: avoid; }}
    * {{ -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
  }}
  @page {{ margin: 14mm 12mm 18mm; }}
</style>
</head>
<body>
<h1>材料热物性测试报告</h1>
<table class="cover">
<tr><td>样品名称：{e(info.get('sample_name') or '—')}</td>
    <td>样品编号：{e(info.get('sample_id') or '—')}</td></tr>
<tr><td>样品结构（顶层→基底）：{e(structure)}</td>
    <td>测试温度：{temp_txt}</td></tr>
<tr><td>测试环境：{e(info.get('environment') or '—')}</td>
    <td>测试人员：{e(info.get('operator') or '—')}</td></tr>
<tr><td>测试日期：{e(info.get('test_date') or '—')}</td>
    <td>报告日期：{e(info.get('report_date') or '—')}</td></tr>
<tr><td>频率范围：{fmt_sci(np.nanmin(all_f))} – {fmt_sci(np.nanmax(all_f))} Hz</td>
    <td>数据点数：{n_points}</td></tr>
<tr><td colspan="2">数据文件：{e(str(data_file))}</td></tr>
</table>
<p class="lab">多层材料 $3\\omega$ 热物性测量系统</p>
{showcase}
<p class="noprint box">提示：建议在软件中直接导出 PDF；若用浏览器打印，请等待公式全部显示后再打印，
并在打印设置中取消“页眉和页脚”。</p>

<h2>一、测量原理</h2>
{principle}

<h2>二、样品与测量条件</h2>
<p class="indent">样品为 {n} 层结构，自上而下依次为 {e(structure)}。加热线经光刻 / 蒸镀制作在样品顶面，
采用四端法连接。测量在“{e(info.get('environment') or '—')}”环境、{temp_txt} 下进行，
对每个频点记录复数 $V_{{3\\omega}}$。{e(note)}</p>
<p><strong>加热线（传感器）参数：</strong>（加粗项为参与拟合的参数）</p>
{_sensor_table_html(ctx)}

<h2>三、测量结果</h2>
<h3>3.1 热物性参数</h3>
<p class="indent">拟合模式：{e(fit.get('mode_text') or '—')}；拟合目标：{e(method_label)}；
拟合参数个数：{fit.get('n_params', '—')}。{status}
下表中<b>加粗且底色高亮</b>的为本次拟合得到的参数，其余为按已知值固定的参数；
各向同性层的 $k_x$ 与 $k_y$ 相同；热扩散率由 $\\alpha_y=k_y/(\\rho c)$ 计算。</p>
{_layer_table_html(ctx)}
{unc_html}

<h3>3.2 拟合曲线</h3>
<p class="indent">红色圆点为实验数据，蓝色实线为最终参数下的模型曲线。{pm_note}</p>
{fig_html}

<h3>3.3 拟合质量</h3>
{quality_html}
{slope_html}
<p class="indent">测量结果仅对本次样品及本次拟合设置负责。</p>

{appendix_html}

{_REFS_HTML}
</body>
</html>
"""


def html_file_to_pdf(html_path: str, pdf_path: str) -> str:
    """HTML → PDF（页脚只显示页码）。优先 Playwright 等待 MathJax 渲染，否则用 Edge/Chrome headless。"""
    html_path = os.path.abspath(html_path)
    pdf_path = os.path.abspath(pdf_path)
    if os.path.isfile(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError:
            pass
    uri = pathlib.Path(html_path).as_uri()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sync_playwright = None

    if sync_playwright is not None:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(args=['--font-render-hinting=none'])
                page = browser.new_page()
                page.goto(uri, wait_until='load', timeout=180_000)
                try:
                    page.wait_for_function('window.__MATHJAX_DONE__ === true', timeout=120_000)
                except Exception:
                    page.wait_for_timeout(8000)
                page.wait_for_timeout(800)
                page.pdf(
                    path=pdf_path, format='A4', print_background=True,
                    display_header_footer=True,
                    header_template='<span></span>',
                    footer_template=(
                        '<div style="width:100%;text-align:center;font-size:10px;color:#333;">'
                        '<span class="pageNumber"></span> / <span class="totalPages"></span></div>'
                    ),
                    margin={'top': '14mm', 'bottom': '18mm', 'left': '12mm', 'right': '12mm'},
                )
                browser.close()
            return 'playwright'
        except Exception:
            pass  # 未安装 chromium 内核等情况，回退到本机浏览器

    candidates = [os.environ.get(k) for k in ('CHROME_PATH', 'EDGE_PATH')]
    candidates += [
        shutil.which('msedge'), shutil.which('chrome'), shutil.which('google-chrome'),
        r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    ]
    exe = next((c for c in candidates if c and os.path.isfile(c)), None)
    if exe is None:
        raise RuntimeError(
            "未找到 Playwright 或 Edge/Chrome。\n"
            "可执行: pip install playwright && playwright install chromium\n"
            "或导出 HTML 后用浏览器打开，等公式显示后再打印为 PDF。"
        )
    base = [exe, '--disable-gpu', '--run-all-compositor-stages-before-draw',
            '--virtual-time-budget=90000', f'--print-to-pdf={pdf_path}']
    r = subprocess.run(base[:1] + ['--headless=new', '--no-pdf-header-footer'] + base[1:] + [uri],
                       capture_output=True, text=True, timeout=240)
    if r.returncode != 0 or not os.path.isfile(pdf_path):
        r2 = subprocess.run(base[:1] + ['--headless'] + base[1:] + [uri],
                            capture_output=True, text=True, timeout=240)
        if r2.returncode != 0 or not os.path.isfile(pdf_path):
            err = (r.stderr or r.stdout or '') + (r2.stderr or r2.stdout or '')
            raise RuntimeError(f"浏览器导出 PDF 失败：{err.strip() or 'unknown error'}")
    return 'edge' if 'edge' in exe.lower() else 'chrome'
