"""
纤维丝 3ω 热物性测量与拟合 GUI

支持两种模型自由切换:
  - 导电丝: Lu et al., Rev. Sci. Instrum. 72, 2996 (2001)
  - 绝缘丝镀层: Xing/Munro 等 (IMECE2015) — 镀金属加热/测温

界面风格对齐主程序 thermal_measurement_gui2.py（拟合 / 可视化 / 正向模拟）。

启动:
  cd xianwiesi
  python fiber_thermal_gui.py
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFormLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QCheckBox, QTabWidget, QTextEdit, QFileDialog, QMessageBox, QStatusBar,
    QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QAction, QScrollArea, QListWidget, QListWidgetItem,
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from lu2001_fiber_model import (
    FiberGeometry, MaterialProps, RadiationParams,
    calculate_V3omega, FiberModelFitter, validity_report,
    FIBER_PRESETS, MATERIAL_PRESETS, diameter_from_area,
    thermal_time_constant, thermal_diffusivity,
    FIBER_PARAM_LABELS, compute_fiber_parameter_correlation,
)
from insulating_fiber_model import (
    calculate_V3omega_insulating, InsulatingFiberFitter,
    validity_report_insulating, INSULATING_MATERIAL_PRESETS,
)

# 模型标识
MODEL_CONDUCTIVE = "conductive"   # Lu 2001
MODEL_INSULATING = "insulating"   # 镀层绝缘丝
MODEL_LABELS = {
    MODEL_CONDUCTIVE: "导电丝 (Lu 2001)",
    MODEL_INSULATING: "绝缘丝镀层 (3ω+辐射)",
}


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


_configure_mpl_font()


def _parse_float(text: str, name: str = 'value') -> float:
    t = text.strip().lower()
    if t in ('inf', 'infinity'):
        return np.inf
    return float(t)


def _alpha_R_from_Rp(R: float, R_prime: float) -> float:
    """α_R = (dR/dT)/R  [1/K]。"""
    if abs(R) < 1e-30:
        return 0.0
    return float(R_prime) / float(R)


def _Rp_from_alpha_R(R: float, alpha_R: float) -> float:
    """dR/dT = α_R · R  [Ω/K]。"""
    return float(alpha_R) * float(R)


def _I_from_V(V_rms: float, R: float) -> float:
    """电压源：I_rms = V_rms / R。"""
    if abs(R) < 1e-30:
        raise ValueError("R 过小，无法由 V_rms 换算 I_rms")
    return float(V_rms) / float(R)


def _V_from_I(I_rms: float, R: float) -> float:
    """V_rms = I_rms · R。"""
    return float(I_rms) * float(R)


def _preset_alpha_R(p: dict) -> float:
    """纤维预设：优先 alpha_R；旧键 R_prime 则换算。"""
    if 'alpha_R' in p:
        return float(p['alpha_R'])
    R = float(p.get('R', 0.0))
    return _alpha_R_from_Rp(R, float(p.get('R_prime', 0.0)))


def _preset_V_rms(p: dict) -> float:
    """纤维预设：优先 V_rms；旧键 I_rms 则 V=I·R。"""
    if 'V_rms' in p:
        return float(p['V_rms'])
    R = float(p.get('R', 0.0))
    return _V_from_I(float(p.get('I_rms', 0.0)), R)

def parse_fiber_v3_dataframe(df: pd.DataFrame):
    """从 DataFrame 解析频率与复数 V3，返回 (freqs, V3) 升序。"""
    freq_col = None
    for key in ('频率 (hz)', '频率(hz)', 'freq', 'frequency', 'f', '频率'):
        for c in df.columns:
            if c.lower().strip() == key or key in c.lower():
                freq_col = c
                break
        if freq_col:
            break
    if freq_col is None:
        freq_col = df.columns[0]
    freqs = df[freq_col].values.astype(float)

    V3 = None
    real_c = imag_c = None
    for c in df.columns:
        cl = c.lower()
        if 'real' in cl or '实部' in c or cl in ('x', 'v3_x'):
            real_c = c
        if 'imag' in cl or '虚部' in c or cl in ('y', 'v3_y'):
            imag_c = c
    if real_c and imag_c:
        V3 = df[real_c].values.astype(float) + 1j * df[imag_c].values.astype(float)
    else:
        amp_c = pha_c = None
        for c in df.columns:
            cl = c.lower()
            if '幅' in c or 'amp' in cl or 'mag' in cl:
                amp_c = c
            if '相' in c or 'phase' in cl or 'pha' in cl:
                pha_c = c
        if amp_c and pha_c:
            amp = df[amp_c].values.astype(float)
            pha = df[pha_c].values.astype(float)
            if np.nanmax(np.abs(pha)) > 2 * np.pi + 0.5:
                pha = np.deg2rad(pha)
            V3 = amp * np.exp(1j * pha)
        elif df.shape[1] >= 3:
            amp = df.iloc[:, 1].values.astype(float)
            pha = np.deg2rad(df.iloc[:, 2].values.astype(float))
            V3 = amp * np.exp(1j * pha)
        else:
            raise ValueError(
                "无法识别 V3 列。请提供 V3_Real/V3_Imag 或 幅值/相位，"
                "或 频率 + 幅值 + 相位(度)。")

    order = np.argsort(freqs)
    return freqs[order], V3[order]


# ---------------------------------------------------------------------------
# 数据拟合页
# ---------------------------------------------------------------------------
class FiberFittingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.fitting_data = None
        self.freqs = None
        self.V3_exp = None
        self.fitter = None
        self.fitted_material = None
        self._model_type = MODEL_CONDUCTIVE
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 模型选择（全局）
        model_bar = QHBoxLayout()
        model_g = QGroupBox("物理模型")
        mg = QHBoxLayout()
        self.model_combo = QComboBox()
        for key, label in MODEL_LABELS.items():
            self.model_combo.addItem(label, key)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        mg.addWidget(QLabel("当前模型:"))
        mg.addWidget(self.model_combo, 1)
        self.model_hint = QLabel("")
        self.model_hint.setWordWrap(True)
        self.model_hint.setStyleSheet("color:#1565C0; font-size:12px;")
        mg.addWidget(self.model_hint, 2)
        model_g.setLayout(mg)
        model_bar.addWidget(model_g)
        layout.addLayout(model_bar)

        # 顶栏
        top = QHBoxLayout()
        load_g = QGroupBox("数据加载")
        load_l = QVBoxLayout()
        self.load_btn = QPushButton("加载数据文件 (xlsx/csv)")
        self.load_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        self.load_btn.clicked.connect(self.load_data)
        self.data_label = QLabel("未加载数据")
        load_l.addWidget(self.load_btn)
        load_l.addWidget(self.data_label)
        load_g.setLayout(load_l)
        top.addWidget(load_g)

        mode_g = QGroupBox("拟合设置")
        mode_l = QFormLayout()
        self.fit_mode = QComboBox()
        self.fit_mode.addItems(["Amplitude", "Phase", "Real+Imag", "Real Only", "Imag Only"])
        self.fit_mode.setCurrentText("Amplitude")
        mode_l.addRow("拟合目标:", self.fit_mode)
        self.use_eq24 = QCheckBox("使用修正式(24)（建议 2ωγ≤4）")
        mode_l.addRow(self.use_eq24)
        self.fit_kappa = QCheckBox("拟合 κ")
        self.fit_kappa.setChecked(True)
        self.fit_Cp = QCheckBox("拟合 C_p")
        self.fit_Cp.setChecked(True)
        self.fit_L = QCheckBox("拟合 L")
        self.fit_D = QCheckBox("拟合 D")
        self.fit_R = QCheckBox("拟合 R")
        self.fit_Rp = QCheckBox("拟合 α_R")
        self.fit_I = QCheckBox("拟合 V_rms")
        row = QHBoxLayout()
        row.addWidget(self.fit_kappa)
        row.addWidget(self.fit_Cp)
        mode_l.addRow("材料:", row)
        row_geo = QHBoxLayout()
        row_geo.addWidget(self.fit_L)
        row_geo.addWidget(self.fit_D)
        mode_l.addRow("几何:", row_geo)
        row2 = QHBoxLayout()
        row2.addWidget(self.fit_R)
        row2.addWidget(self.fit_Rp)
        row2.addWidget(self.fit_I)
        mode_l.addRow("电学/传感器:", row2)
        self.sensor_fit_hint = QLabel(
            "提示: 驱动按电压源 V_rms，I=V/R；|V3|∝(V/R)³ R|α_R|L/(κS)。"
            "κ 与 R/α_R/V/L/D 勿同时乱拟合；几何/电学宜先标定。")
        self.sensor_fit_hint.setWordWrap(True)
        self.sensor_fit_hint.setStyleSheet("color:#555; font-size:11px;")
        mode_l.addRow(self.sensor_fit_hint)
        mode_g.setLayout(mode_l)
        top.addWidget(mode_g)

        freq_g = QGroupBox("频率筛选")
        freq_l = QFormLayout()
        self.fmin = QDoubleSpinBox()
        self.fmin.setDecimals(4)
        self.fmin.setRange(0, 1e8)
        self.fmin.setEnabled(False)
        self.fmax = QDoubleSpinBox()
        self.fmax.setDecimals(4)
        self.fmax.setRange(0, 1e8)
        self.fmax.setEnabled(False)
        freq_l.addRow("f_min (Hz):", self.fmin)
        freq_l.addRow("f_max (Hz):", self.fmax)
        freq_g.setLayout(freq_l)
        top.addWidget(freq_g)
        layout.addLayout(top)

        # 参数区
        params = QHBoxLayout()

        # 纤维
        self.fiber_g = QGroupBox("纤维参数 (Lu 四线悬空段)")
        fl = QFormLayout()
        self.fiber_preset = QComboBox()
        self.fiber_preset.addItems(list(FIBER_PRESETS.keys()))
        self.fiber_preset.currentTextChanged.connect(self._load_fiber_preset)
        fl.addRow("预设:", self.fiber_preset)
        self.inp_L = QLineEdit("8e-3")
        self.inp_D = QLineEdit("20e-6")
        self.inp_S = QLineEdit("")
        self.inp_S.setPlaceholderText("留空则由 D 计算 πD²/4")
        self.inp_R = QLineEdit("5.0")
        self.inp_Rp = QLineEdit("0.002")
        self.inp_V = QLineEdit("0.025")
        self.lbl_R = QLabel("R (Ω):")
        self.lbl_Rp = QLabel("α_R (1/K, 与 R 独立):")
        self.lbl_V = QLabel("V_rms (V):")
        self.lbl_I_hint = QLabel("I = V/R ≈ —")
        self.lbl_I_hint.setStyleSheet("color:#555; font-size:11px;")
        fl.addRow("L (m):", self.inp_L)
        fl.addRow("D (m):", self.inp_D)
        fl.addRow("S (m²):", self.inp_S)
        fl.addRow(self.lbl_R, self.inp_R)
        fl.addRow(self.lbl_Rp, self.inp_Rp)
        fl.addRow(self.lbl_V, self.inp_V)
        fl.addRow("", self.lbl_I_hint)
        self.inp_R.textChanged.connect(self._update_I_hint)
        self.inp_V.textChanged.connect(self._update_I_hint)
        self.fiber_g.setLayout(fl)
        params.addWidget(self.fiber_g)

        # 材料初值
        self.mat_g = QGroupBox("材料初值")
        ml = QFormLayout()
        self.mat_preset = QComboBox()
        self.mat_preset.addItems(list(MATERIAL_PRESETS.keys()) + ['custom'])
        self.mat_preset.currentTextChanged.connect(self._load_mat_preset)
        ml.addRow("预设:", self.mat_preset)
        self.inp_k = QLineEdit("71.6")
        self.inp_rho = QLineEdit("21450")
        self.inp_Cp = QLineEdit("133")
        ml.addRow("κ (W/m·K):", self.inp_k)
        ml.addRow("ρ (kg/m³):", self.inp_rho)
        ml.addRow("C_p (J/kg·K):", self.inp_Cp)
        self.mat_g.setLayout(ml)
        params.addWidget(self.mat_g)

        # 辐射（可选）
        self.rad_g = QGroupBox("径向辐射（可选）")
        rl = QFormLayout()
        self.use_rad = QCheckBox("计入辐射修正 (ε>0)")
        self.inp_eps = QLineEdit("0")
        self.inp_T0 = QLineEdit("300")
        rl.addRow(self.use_rad)
        rl.addRow("发射率 ε:", self.inp_eps)
        rl.addRow("T0 (K):", self.inp_T0)
        self.rad_hint = QLabel("辐射强时 κ 为表观值；C_p 仍可靠 (Lu 式33)")
        self.rad_hint.setWordWrap(True)
        self.rad_hint.setStyleSheet("color:#555; font-size:11px;")
        rl.addRow(self.rad_hint)
        self.rad_g.setLayout(rl)
        params.addWidget(self.rad_g)
        layout.addLayout(params)

        # 按钮
        btn_row = QHBoxLayout()
        self.fit_btn = QPushButton("开始拟合")
        self.fit_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-size: 15px; padding: 8px; }")
        self.fit_btn.clicked.connect(self.run_fitting)
        self.prefit_corr_btn = QPushButton("拟合前相关性")
        self.prefit_corr_btn.setStyleSheet(
            "QPushButton { background-color: #607D8B; color: white; font-size: 14px; padding: 8px; }")
        self.prefit_corr_btn.setToolTip(
            "基于当前初值与已勾选拟合参数，预判参数耦合（无需先运行拟合）")
        self.prefit_corr_btn.clicked.connect(self.run_prefit_correlation)
        self.valid_btn = QPushButton("适用性检查")
        self.valid_btn.clicked.connect(self.run_validity)
        self.export_btn = QPushButton("导出结果 JSON")
        self.export_btn.clicked.connect(self.export_results)
        btn_row.addWidget(self.fit_btn)
        btn_row.addWidget(self.prefit_corr_btn)
        btn_row.addWidget(self.valid_btn)
        btn_row.addWidget(self.export_btn)
        layout.addLayout(btn_row)

        # 结果 + 相关性
        bottom = QHBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Consolas", 10))
        bottom.addWidget(self.results_text, 2)

        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["参数", "数值", "单位"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bottom.addWidget(self.results_table, 1)
        layout.addLayout(bottom)

        corr_row = QHBoxLayout()
        corr_hint_style = (
            "QLabel { color: #2c3e50; font-size: 12px; font-weight: 600; padding: 2px 0; }"
        )
        corr_summary_style = (
            "QLabel { color: #1a1a1a; font-size: 12px; line-height: 140%; padding: 2px 0; }"
        )
        prefit_group = QGroupBox("相关性 (拟合前)")
        prefit_l = QVBoxLayout()
        ph = QLabel("|ρ|≥0.7 为强耦合")
        ph.setStyleSheet(corr_hint_style)
        prefit_l.addWidget(ph)
        self.prefit_correlation_summary = QLabel("点击「拟合前相关性」")
        self.prefit_correlation_summary.setWordWrap(True)
        self.prefit_correlation_summary.setStyleSheet(corr_summary_style)
        prefit_l.addWidget(self.prefit_correlation_summary)
        self.prefit_correlation_table = QTableWidget()
        prefit_l.addWidget(self.prefit_correlation_table)
        prefit_group.setLayout(prefit_l)
        corr_row.addWidget(prefit_group, 1)

        postfit_group = QGroupBox("相关性 (拟合后)")
        postfit_l = QVBoxLayout()
        poh = QLabel("|ρ|≥0.7 为强耦合")
        poh.setStyleSheet(corr_hint_style)
        postfit_l.addWidget(poh)
        self.correlation_summary = QLabel("拟合完成后自动显示")
        self.correlation_summary.setWordWrap(True)
        self.correlation_summary.setStyleSheet(corr_summary_style)
        postfit_l.addWidget(self.correlation_summary)
        self.correlation_table = QTableWidget()
        postfit_l.addWidget(self.correlation_table)
        postfit_group.setLayout(postfit_l)
        corr_row.addWidget(postfit_group, 1)
        layout.addLayout(corr_row)

        self._load_fiber_preset(self.fiber_preset.currentText())
        self._load_mat_preset(self.mat_preset.currentText())
        self._on_model_changed()
        self._update_I_hint()

    def _update_I_hint(self):
        try:
            R = _parse_float(self.inp_R.text(), 'R')
            V = _parse_float(self.inp_V.text(), 'V_rms')
            I = _I_from_V(V, R)
            self.lbl_I_hint.setText(f"I_rms = V/R ≈ {I:.4g} A")
        except Exception:
            self.lbl_I_hint.setText("I_rms = V/R ≈ —")

    def current_model(self) -> str:
        return self.model_combo.currentData() or MODEL_CONDUCTIVE

    def _on_model_changed(self):
        self._model_type = self.current_model()
        is_ins = self._model_type == MODEL_INSULATING

        self.use_eq24.setEnabled(not is_ins)
        if is_ins:
            self.use_eq24.setChecked(False)

        if is_ins:
            self.fiber_g.setTitle("纤维参数 (镀层加热 — L 为热沉间距)")
            self.lbl_R.setText("R_镀层 (Ω):")
            self.lbl_Rp.setText("α_R,镀层 (1/K):")
            self.mat_g.setTitle("纤维芯材料初值")
            self.rad_g.setTitle("径向辐射 h_r=4εσT0³（细丝建议开启）")
            self.use_rad.setText("计入辐射 (ε>0，写入 m²)")
            self.rad_hint.setText(
                "绝缘细丝辐射往往不可忽略；R/α_R 为镀层标定值，κ/C_p 为纤维芯。")
            self.model_hint.setText(
                "绝缘丝：镀 Au 等作加热/温度计；m²=i2ω/α+4h_r/(Dk)；拟合纤维芯 κ、C_p。")
            # 刷新材料预设列表
            self.mat_preset.blockSignals(True)
            self.mat_preset.clear()
            self.mat_preset.addItems(
                list(INSULATING_MATERIAL_PRESETS.keys()) + ['custom'])
            self.mat_preset.blockSignals(False)
            if 'glass' in INSULATING_MATERIAL_PRESETS:
                self.mat_preset.setCurrentText('glass')
                self._load_mat_preset('glass')
        else:
            self.fiber_g.setTitle("纤维参数 (Lu 四线悬空段)")
            self.lbl_R.setText("R (Ω):")
            self.lbl_Rp.setText("α_R = (1/R)dR/dT (1/K):")
            self.mat_g.setTitle("材料初值")
            self.rad_g.setTitle("径向辐射（可选）")
            self.use_rad.setText("计入辐射修正 (ε>0)")
            self.rad_hint.setText("辐射强时 κ 为表观值；C_p 仍可靠 (Lu 式33)")
            self.model_hint.setText(
                "导电丝：丝本体既是加热器也是温度计；V3∝1/√(1+(2ωγ)²)，γ=L²/(π²α)。")
            self.mat_preset.blockSignals(True)
            self.mat_preset.clear()
            self.mat_preset.addItems(list(MATERIAL_PRESETS.keys()) + ['custom'])
            self.mat_preset.blockSignals(False)
            if 'platinum' in MATERIAL_PRESETS:
                self.mat_preset.setCurrentText('platinum')
                self._load_mat_preset('platinum')

        # 通知主窗口同步模拟页模型（若已挂接）
        mw = self.window()
        if hasattr(mw, 'on_fiber_model_changed'):
            mw.on_fiber_model_changed(self._model_type)

    def _load_fiber_preset(self, name: str):
        p = FIBER_PRESETS.get(name)
        if not p:
            return
        self.inp_L.setText(f"{p['L']:.6g}")
        self.inp_D.setText(f"{p['D']:.6g}")
        self.inp_S.setText("")
        self.inp_R.setText(f"{p['R']:.6g}")
        self.inp_Rp.setText(f"{_preset_alpha_R(p):.6g}")
        self.inp_V.setText(f"{_preset_V_rms(p):.6g}")
        self._update_I_hint()
        # 若预设绑定了材料名，同步材料初值（导电模型）
        mat_name = p.get('material')
        if mat_name and self.current_model() == MODEL_CONDUCTIVE:
            if mat_name in MATERIAL_PRESETS:
                self.mat_preset.blockSignals(True)
                self.mat_preset.setCurrentText(mat_name)
                self.mat_preset.blockSignals(False)
                self._load_mat_preset(mat_name)

    def _load_mat_preset(self, name: str):
        if self.current_model() == MODEL_INSULATING:
            m = INSULATING_MATERIAL_PRESETS.get(name)
        else:
            m = MATERIAL_PRESETS.get(name)
        if not m:
            return
        self.inp_k.setText(f"{m.kappa:.6g}")
        self.inp_rho.setText(f"{m.rho:.6g}")
        self.inp_Cp.setText(f"{m.C_p:.6g}")

    def get_fiber(self) -> FiberGeometry:
        L = _parse_float(self.inp_L.text(), 'L')
        D = _parse_float(self.inp_D.text(), 'D')
        S_txt = self.inp_S.text().strip()
        S = _parse_float(S_txt, 'S') if S_txt else np.pi * D ** 2 / 4.0
        R = _parse_float(self.inp_R.text(), 'R')
        alpha_R = _parse_float(self.inp_Rp.text(), 'α_R')
        V_rms = _parse_float(self.inp_V.text(), 'V_rms')
        return FiberGeometry(
            L=L, S=S, R=R,
            R_prime=_Rp_from_alpha_R(R, alpha_R),
            I_rms=_I_from_V(V_rms, R),
        )

    def get_V_rms(self) -> float:
        return _parse_float(self.inp_V.text(), 'V_rms')

    def get_material(self) -> MaterialProps:
        return MaterialProps(
            kappa=_parse_float(self.inp_k.text(), 'κ'),
            rho=_parse_float(self.inp_rho.text(), 'ρ'),
            C_p=_parse_float(self.inp_Cp.text(), 'C_p'),
        )

    def get_radiation(self):
        if not self.use_rad.isChecked():
            return None
        eps = _parse_float(self.inp_eps.text(), 'ε')
        if eps <= 0:
            return None
        D = _parse_float(self.inp_D.text(), 'D')
        return RadiationParams(
            emissivity=eps,
            T0=_parse_float(self.inp_T0.text(), 'T0'),
            diameter=D,
        )

    def get_model_config(self) -> dict:
        fiber = self.get_fiber()
        mat = self.get_material()
        return {
            'model_type': self.current_model(),
            'fiber': {
                'L': fiber.L, 'S': fiber.S, 'R': fiber.R,
                'R_prime': fiber.R_prime,
                'alpha_R': _alpha_R_from_Rp(fiber.R, fiber.R_prime),
                'V_rms': self.get_V_rms(),
                'I_rms': fiber.I_rms,
                'D': _parse_float(self.inp_D.text(), 'D'),
            },
            'material': {'kappa': mat.kappa, 'rho': mat.rho, 'C_p': mat.C_p},
            'use_eq24': self.use_eq24.isChecked(),
            'radiation': {
                'enabled': self.use_rad.isChecked(),
                'emissivity': _parse_float(self.inp_eps.text()) if self.inp_eps.text().strip() else 0,
                'T0': _parse_float(self.inp_T0.text()) if self.inp_T0.text().strip() else 300,
            },
        }

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载数据", "",
            "Data (*.xlsx *.xls *.csv *.txt);;All (*.*)")
        if not path:
            return
        try:
            if path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            self.fitting_data = df
            self._parse_v3_columns(df)
            self.data_label.setText(f"{path.split('/')[-1].split(chr(92))[-1]}  ({len(self.freqs)} 点)")
            self.results_text.append(f"已加载: {path}\n点数: {len(self.freqs)}\n")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            traceback.print_exc()

    def _parse_v3_columns(self, df: pd.DataFrame):
        self.freqs, self.V3_exp = parse_fiber_v3_dataframe(df)
        self.fmin.setEnabled(True)
        self.fmax.setEnabled(True)
        self.fmin.setRange(float(self.freqs.min()), float(self.freqs.max()))
        self.fmax.setRange(float(self.freqs.min()), float(self.freqs.max()))
        self.fmin.setValue(float(self.freqs.min()))
        self.fmax.setValue(float(self.freqs.max()))

    def _masked_data(self):
        if self.freqs is None:
            return None, None
        mask = (self.freqs >= self.fmin.value()) & (self.freqs <= self.fmax.value())
        if not np.any(mask):
            raise ValueError("频率筛选后无数据点")
        return self.freqs[mask], self.V3_exp[mask]

    def _fit_mode_key(self) -> str:
        m = self.fit_mode.currentText()
        return {
            'Amplitude': 'amp',
            'Phase': 'phase',
            'Real+Imag': 'both',
            'Real Only': 'real',
            'Imag Only': 'imag',
        }[m]

    def _collect_params_to_fit(self) -> list:
        params = []
        if self.fit_kappa.isChecked():
            params.append('kappa')
        if self.fit_Cp.isChecked():
            params.append('C_p')
        if self.fit_L.isChecked():
            params.append('L')
        if self.fit_D.isChecked():
            params.append('D')
        if self.fit_R.isChecked():
            params.append('R')
        if self.fit_Rp.isChecked():
            params.append('alpha_R')
        if self.fit_I.isChecked():
            params.append('V_rms')
        return params

    def _build_probe_fitter(self, params=None):
        """基于当前初值构建探针 fitter（不执行拟合），用于相关性分析。"""
        if self.freqs is None:
            return None, "请先加载实验数据。"
        params = params if params is not None else self._collect_params_to_fit()
        if not params:
            return None, "请至少勾选一个拟合参数。"
        freqs, V3 = self._masked_data()
        if len(freqs) == 0:
            return None, "选定频率范围内无数据点。"
        fiber = self.get_fiber()
        mat0 = self.get_material()
        alpha_R0 = _parse_float(self.inp_Rp.text(), 'α_R')
        D0 = _parse_float(self.inp_D.text(), 'D')
        V0 = _parse_float(self.inp_V.text(), 'V_rms')
        model = self.current_model()
        if model == MODEL_INSULATING:
            fitter = InsulatingFiberFitter(
                freqs, V3, fiber, mat0,
                params_to_fit=params,
                fit_mode=self._fit_mode_key(),
                diameter=D0,
                radiation=self.get_radiation(),
                alpha_R=alpha_R0,
                V_rms=V0,
            )
        else:
            fitter = FiberModelFitter(
                freqs, V3, fiber, mat0,
                params_to_fit=params,
                fit_mode=self._fit_mode_key(),
                use_eq24=self.use_eq24.isChecked(),
                radiation=self.get_radiation(),
                alpha_R=alpha_R0,
                diameter=D0,
                V_rms=V0,
            )
        return fitter, None

    def _display_correlation_result(self, result, target='post'):
        if target == 'prefit':
            table = self.prefit_correlation_table
            summary = self.prefit_correlation_summary
            prefix = "[拟合前] "
        else:
            table = self.correlation_table
            summary = self.correlation_summary
            prefix = "[拟合后] "
        labels = result.get('labels', [])
        matrix = result.get('matrix')
        error = result.get('error')
        if error:
            table.setRowCount(0)
            table.setColumnCount(0)
            summary.setText(prefix + error)
            return
        n = len(labels)
        if matrix is None or n == 0:
            table.setRowCount(0)
            table.setColumnCount(0)
            summary.setText(prefix + "无参与拟合的参数")
            return
        cond = result.get('condition_number', float('nan'))
        strong = result.get('strong_pairs', [])
        parts = [f"条件数 κ = {cond:.2e}"]
        if cond > 1e6:
            parts.append("整体病态，参数辨识性较差")
        if strong:
            pairs = [f"{a} ↔ {b} (ρ={rho:+.3f})" for a, b, rho in strong[:6]]
            parts.append("强耦合: " + "; ".join(pairs))
        else:
            parts.append("未发现强耦合参数对 (|ρ|≥0.7)")
        summary.setText(prefix + " | ".join(parts))
        table.setRowCount(n)
        table.setColumnCount(n)
        table.setHorizontalHeaderLabels(labels)
        table.setVerticalHeaderLabels(labels)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                text = "1.000" if i == j else f"{val:+.3f}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if i == j:
                    item.setBackground(QColor(240, 240, 240))
                else:
                    arho = abs(val)
                    if arho >= 0.9:
                        item.setBackground(QColor(255, 180, 180))
                    elif arho >= 0.7:
                        item.setBackground(QColor(255, 220, 180))
                    elif arho >= 0.5:
                        item.setBackground(QColor(255, 255, 200))
                table.setItem(i, j, item)

    def run_prefit_correlation(self):
        self.prefit_correlation_summary.setText("正在计算拟合前相关性…")
        try:
            fitter, err = self._build_probe_fitter()
            if err:
                self._display_correlation_result({'error': err, 'labels': []}, target='prefit')
                QMessageBox.warning(self, "提示", err)
                return
            result = fitter.compute_parameter_correlation()
            self._display_correlation_result(result, target='prefit')
        except Exception as e:
            self.prefit_correlation_table.setRowCount(0)
            self.prefit_correlation_table.setColumnCount(0)
            self.prefit_correlation_summary.setText(f"拟合前相关性分析失败: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"拟合前相关性分析失败:\n{e}")

    def update_correlation_analysis(self, fitter=None):
        fitter = fitter or self.fitter
        if fitter is None:
            self.correlation_table.setRowCount(0)
            self.correlation_table.setColumnCount(0)
            self.correlation_summary.setText("拟合完成后自动显示")
            return
        try:
            self.correlation_summary.setText("[拟合后] 正在计算参数相关性…")
            result = fitter.compute_parameter_correlation()
            self._display_correlation_result(result, target='post')
        except Exception as e:
            self.correlation_table.setRowCount(0)
            self.correlation_table.setColumnCount(0)
            self.correlation_summary.setText(f"相关性分析失败: {e}")
            traceback.print_exc()

    def run_validity(self):
        try:
            fiber = self.get_fiber()
            mat = self.get_material()
            freqs, _ = self._masked_data() if self.freqs is not None else (
                np.logspace(0, 2, 20), None)
            if freqs is None:
                freqs = np.logspace(0, 2, 20)
            if self.current_model() == MODEL_INSULATING:
                report = validity_report_insulating(
                    fiber, mat, freqs,
                    diameter=_parse_float(self.inp_D.text(), 'D'),
                    radiation=self.get_radiation(),
                )
            else:
                report = validity_report(fiber, mat, freqs, self.get_radiation())
            self.results_text.append("\n" + report + "\n")
        except Exception as e:
            QMessageBox.warning(self, "检查失败", str(e))

    def run_fitting(self):
        try:
            if self.freqs is None:
                QMessageBox.warning(self, "提示", "请先加载实验数据。")
                return
            freqs, V3 = self._masked_data()
            fiber = self.get_fiber()
            mat0 = self.get_material()
            params = []
            if self.fit_kappa.isChecked():
                params.append('kappa')
            if self.fit_Cp.isChecked():
                params.append('C_p')
            if self.fit_L.isChecked():
                params.append('L')
            if self.fit_D.isChecked():
                params.append('D')
            if self.fit_R.isChecked():
                params.append('R')
            if self.fit_Rp.isChecked():
                params.append('alpha_R')  # 独立于 R；模型内 R'=α_R·R
            if self.fit_I.isChecked():
                params.append('V_rms')  # 电压源；模型内 I=V/R
            if not params:
                QMessageBox.warning(self, "提示", "请至少勾选一个拟合参数。")
                return

            mat_keys = {'kappa', 'C_p', 'rho'}
            geo_keys = {'L', 'D'}
            sens_keys = {'R', 'alpha_R', 'R_prime', 'V_rms', 'I_rms'}
            has_mat = any(p in mat_keys for p in params)
            has_geo = any(p in geo_keys for p in params)
            if 'kappa' in params and (sens_keys | geo_keys).intersection(params):
                QMessageBox.warning(
                    self, "参数退化警告",
                    "同时拟合 κ 与 R / α_R / V_rms / L / D 时，幅值尺度高度相关，结果可能不可靠。\n"
                    "建议：日常测热物性只拟合 κ、C_p；几何/电学先独立标定后再固定。")

            alpha_R0 = _parse_float(self.inp_Rp.text(), 'α_R')
            D0 = _parse_float(self.inp_D.text(), 'D')
            V0 = _parse_float(self.inp_V.text(), 'V_rms')
            model = self.current_model()
            if model == MODEL_INSULATING:
                self.fitter = InsulatingFiberFitter(
                    freqs, V3, fiber, mat0,
                    params_to_fit=params,
                    fit_mode=self._fit_mode_key(),
                    diameter=D0,
                    radiation=self.get_radiation(),
                    alpha_R=alpha_R0,
                    V_rms=V0,
                )
            else:
                self.fitter = FiberModelFitter(
                    freqs, V3, fiber, mat0,
                    params_to_fit=params,
                    fit_mode=self._fit_mode_key(),
                    use_eq24=self.use_eq24.isChecked(),
                    radiation=self.get_radiation(),
                    alpha_R=alpha_R0,
                    diameter=D0,
                    V_rms=V0,
                )
            fitted = self.fitter.fit(verbose=0)
            self.fitted_material = fitted
            fitted_fiber = self.fitter.fiber
            # α_R 独立于 R：只拟合 R 时 α_R 不变；R'=α_R·R 已在 fitter 内同步
            alpha_R = float(getattr(self.fitter, 'alpha_R',
                                    _alpha_R_from_Rp(fitted_fiber.R, fitted_fiber.R_prime)))
            D_fit = float(getattr(self.fitter, 'diameter',
                                  diameter_from_area(fitted_fiber.S)))
            V_fit = float(getattr(self.fitter, 'V_rms',
                                  _V_from_I(fitted_fiber.I_rms, fitted_fiber.R)))

            # 写回初值框：仅写回实际拟合过的量（改 R 不改 α_R；电压源改 R 时 V 不变）
            if 'kappa' in params or has_mat:
                self.inp_k.setText(f"{fitted.kappa:.6g}")
                self.inp_Cp.setText(f"{fitted.C_p:.6g}")
            if 'L' in params:
                self.inp_L.setText(f"{fitted_fiber.L:.6g}")
            if 'D' in params:
                self.inp_D.setText(f"{D_fit:.6g}")
                self.inp_S.setText(f"{fitted_fiber.S:.6g}")
            if 'R' in params:
                self.inp_R.setText(f"{fitted_fiber.R:.6g}")
            if 'alpha_R' in params or 'R_prime' in params:
                self.inp_Rp.setText(f"{alpha_R:.6g}")
            if 'V_rms' in params or 'I_rms' in params:
                self.inp_V.setText(f"{V_fit:.6g}")
            self._update_I_hint()

            info = self.fitter.last_info
            alpha = info['alpha']

            param_labels = [
                'α_R' if p in ('alpha_R', 'R_prime') else p for p in params
            ]
            self.results_text.append("\n========== 拟合完成 ==========")
            self.results_text.append(
                f"模型: {MODEL_LABELS.get(model, model)}")
            self.results_text.append(f"成功: {self.fitter.success}  |  {self.fitter.message}")
            self.results_text.append(f"cost = {self.fitter.cost:.4e}")
            self.results_text.append(f"拟合参数: {', '.join(param_labels)}")
            self.results_text.append(f"κ = {fitted.kappa:.6g} W/m·K")
            self.results_text.append(f"C_p = {fitted.C_p:.6g} J/kg·K")
            self.results_text.append(f"ρ = {fitted.rho:.6g} kg/m³ (固定)" if 'rho' not in params
                                     else f"ρ = {fitted.rho:.6g} kg/m³")
            self.results_text.append(f"L = {fitted_fiber.L:.6g} m")
            self.results_text.append(f"D = {D_fit:.6g} m  (S = {fitted_fiber.S:.6g} m²)")
            self.results_text.append(f"R = {fitted_fiber.R:.6g} Ω")
            self.results_text.append(f"α_R = {alpha_R:.6g} 1/K  (与 R 独立)")
            self.results_text.append(
                f"R' = dR/dT = {fitted_fiber.R_prime:.6g} Ω/K  (= α_R·R)")
            self.results_text.append(f"V_rms = {V_fit:.6g} V")
            self.results_text.append(
                f"I_rms = V/R = {fitted_fiber.I_rms:.6g} A")
            self.results_text.append(f"α = {alpha:.4e} m²/s")
            if has_geo:
                self.results_text.append(
                    "注: 已拟合几何参数；κ 若同时自由，请核查是否退化。")
            if model == MODEL_INSULATING:
                self.results_text.append(
                    f"h_r = {info.get('hr', 0):.4g} W/m²K, "
                    f"ε = {info.get('emissivity', 0):g}")
                if info.get('V3_lf_rms'):
                    self.results_text.append(
                        f"低频极限 |V3|_rms ≈ {info['V3_lf_rms']:.4e} V")
                self.results_text.append(
                    validity_report_insulating(
                        fitted_fiber, fitted, freqs,
                        diameter=_parse_float(self.inp_D.text(), 'D'),
                        radiation=self.get_radiation(),
                    ))
            else:
                gamma = info['gamma_ap'] if self.get_radiation() else info['gamma']
                self.results_text.append(f"γ = {gamma:.4e} s")
                if self.get_radiation():
                    self.results_text.append(
                        f"κ_ap = {info['kappa_ap']:.6g}, γ_ap = {info['gamma_ap']:.4e}, "
                        f"gγ = {info['g_gamma']:.3e}")
                self.results_text.append(
                    validity_report(fitted_fiber, fitted, freqs, self.get_radiation()))
            self.results_text.append("")

            self._fill_table(fitted, info, model, fiber=fitted_fiber)
            self.update_correlation_analysis(self.fitter)
            # 通知主窗口画图
            mw = self.window()
            if hasattr(mw, 'on_fiber_fit_done'):
                mw.on_fiber_fit_done()

        except Exception as e:
            self.results_text.append(f"\n拟合失败: {e}\n")
            traceback.print_exc()
            QMessageBox.critical(self, "拟合失败", str(e))

    def _fill_table(self, mat: MaterialProps, info: dict, model: str = MODEL_CONDUCTIVE,
                    fiber: FiberGeometry = None):
        fiber = fiber or (self.fitter.fiber if self.fitter else self.get_fiber())
        if self.fitter is not None and hasattr(self.fitter, 'alpha_R'):
            alpha_R_show = float(self.fitter.alpha_R)
        else:
            alpha_R_show = _alpha_R_from_Rp(fiber.R, fiber.R_prime)
        if self.fitter is not None and getattr(self.fitter, 'diameter', None):
            D_show = float(self.fitter.diameter)
        else:
            D_show = diameter_from_area(fiber.S) if fiber.S > 0 else float('nan')
        if self.fitter is not None and hasattr(self.fitter, 'V_rms'):
            V_show = float(self.fitter.V_rms)
        else:
            V_show = _V_from_I(fiber.I_rms, fiber.R)
        sens_rows = [
            ('L', fiber.L, 'm'),
            ('D', D_show, 'm'),
            ('S', fiber.S, 'm²'),
            ('R', fiber.R, 'Ω'),
            ('α_R', alpha_R_show, '1/K'),
            ("R'=α_R·R", fiber.R_prime, 'Ω/K'),
            ('V_rms', V_show, 'V'),
            ('I_rms=V/R', fiber.I_rms, 'A'),
        ]
        if model == MODEL_INSULATING:
            rows = [
                ('κ', mat.kappa, 'W/m·K'),
                ('C_p', mat.C_p, 'J/kg·K'),
                ('ρ', mat.rho, 'kg/m³'),
                ('α', info.get('alpha', np.nan), 'm²/s'),
                ('D', info.get('D', np.nan), 'm'),
                ('h_r', info.get('hr', np.nan), 'W/m²K'),
                ('ε', info.get('emissivity', np.nan), '-'),
                ('|V3|_lf,rms', info.get('V3_lf_rms', np.nan), 'V'),
            ] + sens_rows
        else:
            rows = [
                ('κ', mat.kappa, 'W/m·K'),
                ('C_p', mat.C_p, 'J/kg·K'),
                ('ρ', mat.rho, 'kg/m³'),
                ('α', info.get('alpha', np.nan), 'm²/s'),
                ('γ', info.get('gamma', np.nan), 's'),
                ('γ_ap', info.get('gamma_ap', np.nan), 's'),
                ('κ_ap', info.get('kappa_ap', np.nan), 'W/m·K'),
                ('Δ0', info.get('Delta0', np.nan), 'K'),
                ('条件(10)', info.get('cond10', np.nan), '-'),
                ('gγ', info.get('g_gamma', np.nan), '-'),
            ] + sens_rows
        self.results_table.setRowCount(len(rows))
        for i, (n, v, u) in enumerate(rows):
            self.results_table.setItem(i, 0, QTableWidgetItem(n))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{v:.6g}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(u))

    def export_results(self):
        if self.fitted_material is None and self.fitter is None:
            QMessageBox.warning(self, "提示", "尚无拟合结果。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出", f"fiber_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON (*.json)")
        if not path:
            return
        mat = self.fitted_material or self.get_material()
        fiber = self.get_fiber()
        model = self.current_model()
        data = {
            'model': 'InsulatingCoated3omega' if model == MODEL_INSULATING else 'Lu2001',
            'model_type': model,
            'fiber': {
                'L_m': fiber.L, 'S_m2': fiber.S, 'R_ohm': fiber.R,
                'alpha_R_1_per_K': _alpha_R_from_Rp(fiber.R, fiber.R_prime),
                'R_prime_ohm_per_K': fiber.R_prime,
                'V_rms_V': self.get_V_rms(),
                'I_rms_A': fiber.I_rms,
                'D_m': _parse_float(self.inp_D.text(), 'D'),
            },
            'material': {
                'kappa': mat.kappa, 'rho': mat.rho, 'C_p': mat.C_p,
            },
            'info': {k: (float(v) if np.isscalar(v) else v)
                     for k, v in getattr(self.fitter, 'last_info', {}).items()},
            'fit_mode': self._fit_mode_key(),
            'use_eq24': self.use_eq24.isChecked() if model != MODEL_INSULATING else False,
            'success': getattr(self.fitter, 'success', None),
            'cost': getattr(self.fitter, 'cost', None),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        QMessageBox.information(self, "成功", f"已保存:\n{path}")


# ---------------------------------------------------------------------------
# 可视化（对齐主程序：拟合曲线 + 各拟合参数 ±20% 正演）
# ---------------------------------------------------------------------------
class FiberVizWidget(QWidget):
    PM20_PLUS_COLORS = ['#ff7f0e', '#d62728', '#9467bd', '#e377c2', '#8c564b']
    PM20_MINUS_COLORS = ['#2ca02c', '#17becf', '#bcbd22', '#1f77b4', '#7f7f7f']
    PARAM_LABELS = {
        'kappa': 'κ', 'C_p': 'C_p', 'rho': 'ρ',
        'L': 'L', 'D': 'D',
        'R': 'R', 'alpha_R': 'α_R', 'R_prime': "R'",
        'V_rms': 'V_rms', 'I_rms': 'I_rms',
    }

    def __init__(self):
        super().__init__()
        self.bundle = None
        self.pm20_checkboxes = []
        self.init_ui()

    def init_ui(self):
        main = QHBoxLayout(self)

        # 左侧：±20% 参数开关
        self.curve_panel = QWidget()
        panel_l = QVBoxLayout(self.curve_panel)
        panel_l.addWidget(QLabel("曲线显示开关:"))
        btn_row = QHBoxLayout()
        sel_all = QPushButton("全选")
        desel_all = QPushButton("全不选")
        sel_all.clicked.connect(lambda: self._set_all_pm20(True))
        desel_all.clicked.connect(lambda: self._set_all_pm20(False))
        btn_row.addWidget(sel_all)
        btn_row.addWidget(desel_all)
        panel_l.addLayout(btn_row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_w = QWidget()
        self.checkbox_layout = QVBoxLayout(scroll_w)
        self.checkbox_layout.addStretch()
        scroll.setWidget(scroll_w)
        panel_l.addWidget(scroll)
        self.curve_panel.setMaximumWidth(260)
        self.curve_panel.setVisible(False)
        main.addWidget(self.curve_panel)

        # 右侧：图
        right = QVBoxLayout()
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("显示:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Amplitude", "Phase", "Amplitude & Phase", "Real+Imag"])
        self.plot_type.currentTextChanged.connect(self.redraw)
        ctrl.addWidget(self.plot_type)
        self.show_pm20_cb = QCheckBox("显示参数 ±20%")
        self.show_pm20_cb.setChecked(True)
        self.show_pm20_cb.setToolTip(
            "对每个已拟合参数单独 ×1.2 / ×0.8 后重新正演理论曲线；"
            "左侧可勾选要显示的参数（默认全部）。")
        self.show_pm20_cb.toggled.connect(self.redraw)
        ctrl.addWidget(self.show_pm20_cb)
        ctrl.addStretch()
        self.save_btn = QPushButton("保存图")
        self.save_btn.clicked.connect(self.save_fig)
        ctrl.addWidget(self.save_btn)
        right.addLayout(ctrl)

        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        right.addWidget(self.canvas)
        right.addWidget(NavigationToolbar(self.canvas, self))
        main.addLayout(right, stretch=1)

    def _set_all_pm20(self, checked: bool):
        for cb in self.pm20_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self.redraw()

    def _clear_pm20_checkboxes(self):
        while self.checkbox_layout.count() > 1:
            item = self.checkbox_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.pm20_checkboxes = []

    def _setup_pm20_checkboxes(self, pm20_series):
        self._clear_pm20_checkboxes()
        if not pm20_series:
            self.curve_panel.setVisible(False)
            return
        hdr = QLabel("参数 ±20% 正演:")
        hdr.setStyleSheet("font-weight: bold; margin-top: 4px;")
        self.checkbox_layout.insertWidget(self.checkbox_layout.count() - 1, hdr)
        for item in pm20_series:
            cb = QCheckBox(f"{item['label']} ±20%")
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {item['color_plus']};")
            cb.toggled.connect(self.redraw)
            self.checkbox_layout.insertWidget(self.checkbox_layout.count() - 1, cb)
            self.pm20_checkboxes.append(cb)
        self.curve_panel.setVisible(True)

    def _selected_pm20_indices(self):
        if not self.show_pm20_cb.isChecked():
            return []
        return [i for i, cb in enumerate(self.pm20_checkboxes) if cb.isChecked()]

    @staticmethod
    def _copy_radiation(radiation):
        if radiation is None:
            return None
        return RadiationParams(
            emissivity=radiation.emissivity,
            T0=radiation.T0,
            diameter=radiation.diameter,
        )

    @classmethod
    def _forward_v3(cls, freqs, fiber, material, *, model, diameter, radiation,
                    use_eq24, alpha_R, V_rms=None):
        fib = FiberGeometry(fiber.L, fiber.S, fiber.R, fiber.R_prime, fiber.I_rms)
        # 保证 R' = α_R · R；电压源 I = V/R
        aR = float(alpha_R)
        fib.R_prime = aR * fib.R
        if V_rms is not None:
            fib.I_rms = _I_from_V(float(V_rms), fib.R)
        D = float(diameter) if diameter is not None else diameter_from_area(fib.S)
        rad = cls._copy_radiation(radiation)
        if rad is not None and D > 0:
            rad.diameter = D
        if model == MODEL_INSULATING:
            V3, _ = calculate_V3omega_insulating(
                freqs, fib, material, diameter=D, radiation=rad)
        else:
            V3, _ = calculate_V3omega(
                freqs, fib, material, use_eq24=use_eq24, radiation=rad)
        return V3

    @classmethod
    def _v3_with_param_scale(cls, freqs, fiber, material, param, factor, *,
                             model, diameter, radiation, use_eq24, alpha_R,
                             V_rms=None):
        """将指定参数 ×factor 后正演；失败返回 None。"""
        try:
            mat = MaterialProps(material.kappa, material.rho, material.C_p)
            fib = FiberGeometry(fiber.L, fiber.S, fiber.R, fiber.R_prime, fiber.I_rms)
            aR = float(alpha_R)
            D = float(diameter) if diameter is not None else diameter_from_area(fib.S)
            Vdrv = float(V_rms) if V_rms is not None else _V_from_I(fib.I_rms, fib.R)

            if param == 'kappa':
                mat.kappa = float(mat.kappa) * factor
            elif param == 'C_p':
                mat.C_p = float(mat.C_p) * factor
            elif param == 'rho':
                mat.rho = float(mat.rho) * factor
            elif param == 'L':
                fib.L = float(fib.L) * factor
            elif param == 'D':
                D = D * factor
                fib.S = 0.25 * np.pi * D ** 2
            elif param == 'R':
                fib.R = float(fib.R) * factor
                # α_R、V 不变 → R' 与 I 同步
                fib.R_prime = aR * fib.R
                fib.I_rms = _I_from_V(Vdrv, fib.R)
            elif param == 'alpha_R':
                aR = aR * factor
                fib.R_prime = aR * fib.R
            elif param == 'R_prime':
                fib.R_prime = float(fib.R_prime) * factor
                if abs(fib.R) > 1e-30:
                    aR = fib.R_prime / fib.R
            elif param == 'V_rms':
                Vdrv = Vdrv * factor
                fib.I_rms = _I_from_V(Vdrv, fib.R)
            elif param == 'I_rms':
                fib.I_rms = float(fib.I_rms) * factor
                Vdrv = _V_from_I(fib.I_rms, fib.R)
            else:
                return None

            return cls._forward_v3(
                freqs, fib, mat, model=model, diameter=D,
                radiation=radiation, use_eq24=use_eq24, alpha_R=aR, V_rms=Vdrv)
        except Exception:
            return None

    def _build_pm20_series(self, freqs, model_ctx):
        series = []
        if not model_ctx:
            return series
        params = model_ctx.get('params_fitted') or []
        fiber = model_ctx['fiber']
        material = model_ctx['material']
        for i, p in enumerate(params):
            label = self.PARAM_LABELS.get(p, p)
            V_plus = self._v3_with_param_scale(
                freqs, fiber, material, p, 1.2, **model_ctx['fwd_kw'])
            V_minus = self._v3_with_param_scale(
                freqs, fiber, material, p, 0.8, **model_ctx['fwd_kw'])
            if V_plus is None and V_minus is None:
                continue
            series.append({
                'param': p,
                'label': label,
                'V3_plus': V_plus,
                'V3_minus': V_minus,
                'color_plus': self.PM20_PLUS_COLORS[i % len(self.PM20_PLUS_COLORS)],
                'color_minus': self.PM20_MINUS_COLORS[i % len(self.PM20_MINUS_COLORS)],
            })
        return series

    def set_results(self, freqs, V3_exp, V3_fit, title="Fiber 3ω Fit", model_ctx=None):
        pm20_series = self._build_pm20_series(freqs, model_ctx)
        self.bundle = {
            'freqs': freqs,
            'V3_exp': V3_exp,
            'V3_fit': V3_fit,
            'title': title,
            'model_ctx': model_ctx,
            'pm20_series': pm20_series,
        }
        self._setup_pm20_checkboxes(pm20_series)
        self.redraw()

    def _plot_pm20(self, axes_map, pt, freqs, V_plus, V_minus, c_plus, c_minus, tag):
        def draw_pair(ax, y_p, y_m, log_x=True, log_y=False):
            plot_fn = ax.loglog if (log_x and log_y) else (ax.semilogx if log_x else ax.plot)
            if y_p is not None and np.any(np.isfinite(y_p)):
                plot_fn(freqs, y_p, '--', color=c_plus,
                        label=f'{tag} +20%', lw=1.4, alpha=0.9)
            if y_m is not None and np.any(np.isfinite(y_m)):
                plot_fn(freqs, y_m, '--', color=c_minus,
                        label=f'{tag} -20%', lw=1.4, alpha=0.9)

        def y_of(V, kind):
            if V is None:
                return None
            if kind == 'amp':
                return np.abs(V) * 1e6
            if kind == 'pha':
                return np.angle(V, deg=True)
            if kind == 're':
                return V.real * 1e6
            if kind == 'im':
                return V.imag * 1e6
            return None

        if 'single' in axes_map:
            ax = axes_map['single']
            if pt == "Amplitude":
                draw_pair(ax, y_of(V_plus, 'amp'), y_of(V_minus, 'amp'), True, True)
            elif pt == "Phase":
                draw_pair(ax, y_of(V_plus, 'pha'), y_of(V_minus, 'pha'))
            elif pt == "Real+Imag":
                pass  # 单轴不画 Real+Imag
        else:
            if 'amp' in axes_map:
                draw_pair(axes_map['amp'], y_of(V_plus, 'amp'), y_of(V_minus, 'amp'), True, True)
            if 'pha' in axes_map:
                draw_pair(axes_map['pha'], y_of(V_plus, 'pha'), y_of(V_minus, 'pha'))
            if 're' in axes_map:
                draw_pair(axes_map['re'], y_of(V_plus, 're'), y_of(V_minus, 're'))
            if 'im' in axes_map:
                draw_pair(axes_map['im'], y_of(V_plus, 'im'), y_of(V_minus, 'im'))

    def redraw(self):
        if not self.bundle:
            return
        freqs = self.bundle['freqs']
        Ve = self.bundle['V3_exp']
        Vm = self.bundle['V3_fit']
        pm20_series = self.bundle.get('pm20_series') or []
        sel = self._selected_pm20_indices()
        self.figure.clear()
        pt = self.plot_type.currentText()

        if pt == "Amplitude & Phase":
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)
            axes_map = {'amp': ax1, 'pha': ax2}
            ax1.loglog(freqs, np.abs(Ve) * 1e6, 'ro', label='Exp', markersize=5)
            ax1.loglog(freqs, np.abs(Vm) * 1e6, 'b-', label='Fit', lw=2)
            ax1.set_ylabel('|V3| (μV)')
            ax2.semilogx(freqs, np.angle(Ve, deg=True), 'ro', label='Exp', markersize=5)
            ax2.semilogx(freqs, np.angle(Vm, deg=True), 'b-', label='Fit', lw=2)
            ax2.set_ylabel('Phase (deg)')
            ax2.set_xlabel('Frequency (Hz)')
        elif pt == "Phase":
            ax = self.figure.add_subplot(111)
            axes_map = {'single': ax}
            ax.semilogx(freqs, np.angle(Ve, deg=True), 'ro', label='Exp')
            ax.semilogx(freqs, np.angle(Vm, deg=True), 'b-', label='Fit', lw=2)
            ax.set_ylabel('Phase (deg)')
            ax.set_xlabel('Frequency (Hz)')
        elif pt == "Real+Imag":
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)
            axes_map = {'re': ax1, 'im': ax2}
            ax1.semilogx(freqs, Ve.real * 1e6, 'ro', label='Exp Re')
            ax1.semilogx(freqs, Vm.real * 1e6, 'b-', label='Fit Re', lw=2)
            ax1.set_ylabel('Re(V3) (μV)')
            ax2.semilogx(freqs, Ve.imag * 1e6, 'ro', label='Exp Im')
            ax2.semilogx(freqs, Vm.imag * 1e6, 'b-', label='Fit Im', lw=2)
            ax2.set_ylabel('Im(V3) (μV)')
            ax2.set_xlabel('Frequency (Hz)')
        else:
            ax = self.figure.add_subplot(111)
            axes_map = {'single': ax}
            ax.loglog(freqs, np.abs(Ve) * 1e6, 'ro', label='Exp', markersize=6)
            ax.loglog(freqs, np.abs(Vm) * 1e6, 'b-', label='Fit', lw=2)
            ax.set_ylabel('|V3| (μV)')
            ax.set_xlabel('Frequency (Hz)')

        for j in sel:
            if j >= len(pm20_series):
                continue
            item = pm20_series[j]
            self._plot_pm20(
                axes_map, pt, freqs,
                item.get('V3_plus'), item.get('V3_minus'),
                item['color_plus'], item['color_minus'], item['label'],
            )

        for ax in self.figure.get_axes():
            ax.legend(fontsize='small', loc='best')
            ax.grid(True, which='both', ls='--', alpha=0.3)

        self.figure.suptitle(self.bundle['title'])
        self.figure.tight_layout()
        self.canvas.draw()

    def save_fig(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图", f"fiber_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "成功", f"已保存:\n{path}")


# ---------------------------------------------------------------------------
# 正向模拟（完整参数栏 + 导出，对齐主程序 SimulationWidget）
# ---------------------------------------------------------------------------
class FiberSimWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._fit_src = None
        self.last_sim_data = None
        self.exp_data = None
        self._syncing_model = False
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.run_sim_silent)
        self.init_ui()

    def set_fitting_source(self, w: FiberFittingWidget):
        self._fit_src = w

    def schedule_update(self):
        self.update_timer.start(300)

    def current_model(self) -> str:
        return self.model_combo.currentData() or MODEL_CONDUCTIVE

    def set_model(self, model_type: str, from_fitting: bool = False):
        """外部（拟合页）同步模型选择。"""
        idx = self.model_combo.findData(model_type)
        if idx < 0 or idx == self.model_combo.currentIndex():
            if idx == self.model_combo.currentIndex():
                self._apply_model_ui()
            return
        self._syncing_model = True
        self.model_combo.setCurrentIndex(idx)
        self._syncing_model = False
        self._apply_model_ui()
        if not from_fitting:
            self.schedule_update()

    def init_ui(self):
        layout = QHBoxLayout(self)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(340)
        left_scroll.setMaximumWidth(420)
        left_w = QWidget()
        left = QVBoxLayout(left_w)

        sync = QPushButton("从数据拟合同步参数")
        sync.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background-color: #7B1FA2; }"
        )
        sync.clicked.connect(self.sync_from_fitting)
        left.addWidget(sync)

        # ---- 模型选择 ----
        model_g = QGroupBox("物理模型")
        mg = QFormLayout()
        self.model_combo = QComboBox()
        for key, label in MODEL_LABELS.items():
            self.model_combo.addItem(label, key)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        mg.addRow("模型:", self.model_combo)
        self.model_hint = QLabel("")
        self.model_hint.setWordWrap(True)
        self.model_hint.setStyleSheet("color:#1565C0; font-size:11px;")
        mg.addRow(self.model_hint)
        model_g.setLayout(mg)
        left.addWidget(model_g)

        # ---- 纤维参数 ----
        self.fiber_g = QGroupBox("纤维参数 (Lu 四线悬空段)")
        fl = QFormLayout()
        self.fiber_preset = QComboBox()
        self.fiber_preset.addItems(list(FIBER_PRESETS.keys()))
        self.fiber_preset.currentTextChanged.connect(self._load_fiber_preset)
        fl.addRow("预设:", self.fiber_preset)
        self.inp_L = QLineEdit("8e-3")
        self.inp_D = QLineEdit("20e-6")
        self.inp_S = QLineEdit("")
        self.inp_S.setPlaceholderText("留空则由 D 计算 πD²/4")
        self.inp_R = QLineEdit("5.0")
        self.inp_Rp = QLineEdit("0.002")
        self.inp_V = QLineEdit("0.025")
        for w in (self.inp_L, self.inp_D, self.inp_S, self.inp_R, self.inp_Rp, self.inp_V):
            w.textChanged.connect(self.schedule_update)
        self.lbl_R = QLabel("R (Ω):")
        self.lbl_Rp = QLabel("α_R (1/K, 与 R 独立):")
        self.lbl_I_hint = QLabel("I = V/R ≈ —")
        self.lbl_I_hint.setStyleSheet("color:#555; font-size:11px;")
        fl.addRow("L (m):", self.inp_L)
        fl.addRow("D (m):", self.inp_D)
        fl.addRow("S (m²):", self.inp_S)
        fl.addRow(self.lbl_R, self.inp_R)
        fl.addRow(self.lbl_Rp, self.inp_Rp)
        fl.addRow("V_rms (V):", self.inp_V)
        fl.addRow("", self.lbl_I_hint)
        self.inp_R.textChanged.connect(self._update_I_hint)
        self.inp_V.textChanged.connect(self._update_I_hint)
        self.fiber_g.setLayout(fl)
        left.addWidget(self.fiber_g)

        # ---- 材料 ----
        self.mat_g = QGroupBox("材料参数")
        ml = QFormLayout()
        self.mat_preset = QComboBox()
        self.mat_preset.addItems(list(MATERIAL_PRESETS.keys()) + ['custom'])
        self.mat_preset.currentTextChanged.connect(self._load_mat_preset)
        ml.addRow("预设:", self.mat_preset)
        self.inp_k = QLineEdit("71.6")
        self.inp_rho = QLineEdit("21450")
        self.inp_Cp = QLineEdit("133")
        for w in (self.inp_k, self.inp_rho, self.inp_Cp):
            w.textChanged.connect(self.schedule_update)
        ml.addRow("κ (W/m·K):", self.inp_k)
        ml.addRow("ρ (kg/m³):", self.inp_rho)
        ml.addRow("C_p (J/kg·K):", self.inp_Cp)
        self.mat_g.setLayout(ml)
        left.addWidget(self.mat_g)

        # ---- 频率 ----
        freq_g = QGroupBox("频率范围")
        ff = QFormLayout()
        self.fmin = QLineEdit("1")
        self.fmax = QLineEdit("1000")
        self.npts = QLineEdit("60")
        self.freq_scale = QComboBox()
        self.freq_scale.addItems(["对数分布", "线性分布"])
        for w in (self.fmin, self.fmax, self.npts):
            w.textChanged.connect(self.schedule_update)
        self.freq_scale.currentIndexChanged.connect(self.schedule_update)
        ff.addRow("f_min (Hz):", self.fmin)
        ff.addRow("f_max (Hz):", self.fmax)
        ff.addRow("点数:", self.npts)
        ff.addRow("分布:", self.freq_scale)
        freq_g.setLayout(ff)
        left.addWidget(freq_g)

        # ---- 模型选项 ----
        opt_g = QGroupBox("模型选项")
        ol = QFormLayout()
        self.use_eq24 = QCheckBox("使用修正式(24)")
        self.use_eq24.toggled.connect(self.schedule_update)
        ol.addRow(self.use_eq24)
        self.use_rad = QCheckBox("计入辐射修正 (ε>0)")
        self.use_rad.toggled.connect(self.schedule_update)
        self.inp_eps = QLineEdit("0")
        self.inp_T0 = QLineEdit("300")
        self.inp_eps.textChanged.connect(self.schedule_update)
        self.inp_T0.textChanged.connect(self.schedule_update)
        ol.addRow(self.use_rad)
        ol.addRow("发射率 ε:", self.inp_eps)
        ol.addRow("T0 (K):", self.inp_T0)
        opt_g.setLayout(ol)
        left.addWidget(opt_g)

        # ---- 噪声（可选，便于生成伪实验数据）----
        noise_g = QGroupBox("噪声（可选）")
        nl = QFormLayout()
        self.add_noise = QCheckBox("叠加相对噪声后导出/显示")
        self.noise_amp = QDoubleSpinBox()
        self.noise_amp.setRange(0, 1)
        self.noise_amp.setDecimals(4)
        self.noise_amp.setSingleStep(0.001)
        self.noise_amp.setValue(0.01)
        self.noise_amp.setSuffix(" (相对幅值)")
        self.noise_amp.setEnabled(False)
        self.add_noise.toggled.connect(lambda c: self.noise_amp.setEnabled(c))
        self.add_noise.toggled.connect(self.schedule_update)
        self.noise_amp.valueChanged.connect(self.schedule_update)
        nl.addRow(self.add_noise)
        nl.addRow("噪声水平:", self.noise_amp)
        noise_g.setLayout(nl)
        left.addWidget(noise_g)

        # ---- 实验数据叠加 ----
        exp_g = QGroupBox("实验数据对比（可选）")
        el = QVBoxLayout()
        exp_btns = QHBoxLayout()
        load_exp = QPushButton("加载实验数据")
        load_exp.clicked.connect(self.load_experimental_data)
        clr_exp = QPushButton("清除")
        clr_exp.clicked.connect(self.clear_experimental_data)
        exp_btns.addWidget(load_exp)
        exp_btns.addWidget(clr_exp)
        el.addLayout(exp_btns)
        self.exp_file_label = QLabel("未加载")
        self.exp_file_label.setWordWrap(True)
        el.addWidget(self.exp_file_label)
        exp_g.setLayout(el)
        left.addWidget(exp_g)

        # ---- 操作按钮 ----
        self.export_btn = QPushButton("导出模拟数据")
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 8px; }")
        self.export_btn.setToolTip(
            "导出 CSV/Excel，列名与拟合页兼容：频率 (Hz), V3_Real, V3_Imag, V3_幅值, V3_相位"
        )
        self.export_btn.clicked.connect(self.export_simulation_data)
        left.addWidget(self.export_btn)

        self.run_btn = QPushButton("开始模拟计算")
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; padding: 8px; font-size: 14px; }")
        self.run_btn.clicked.connect(lambda: self.run_sim(silent=False))
        left.addWidget(self.run_btn)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(160)
        self.info.setFont(QFont("Consolas", 9))
        left.addWidget(self.info)
        left.addStretch()

        left_scroll.setWidget(left_w)
        layout.addWidget(left_scroll)

        # 右侧图
        right = QWidget()
        right_l = QVBoxLayout(right)
        plot_ctrl = QHBoxLayout()
        plot_ctrl.addWidget(QLabel("显示:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Amplitude & Phase", "Amplitude", "Phase", "Real+Imag"])
        self.plot_type.currentTextChanged.connect(self._redraw_from_cache)
        plot_ctrl.addWidget(self.plot_type)
        plot_ctrl.addStretch()
        right_l.addLayout(plot_ctrl)
        self.figure = Figure(figsize=(8, 5.5))
        self.canvas = FigureCanvas(self.figure)
        right_l.addWidget(NavigationToolbar(self.canvas, self))
        right_l.addWidget(self.canvas)
        layout.addWidget(right, 1)

        self._load_fiber_preset(self.fiber_preset.currentText())
        self._load_mat_preset(self.mat_preset.currentText())
        self._apply_model_ui()

    def _on_model_changed(self):
        self._apply_model_ui()
        if self._syncing_model:
            return
        # 反向同步到拟合页
        mw = self.window()
        if hasattr(mw, 'on_sim_model_changed'):
            mw.on_sim_model_changed(self.current_model())
        self.schedule_update()

    def _apply_model_ui(self):
        is_ins = self.current_model() == MODEL_INSULATING
        self.use_eq24.setEnabled(not is_ins)
        if is_ins:
            self.use_eq24.setChecked(False)
            self.fiber_g.setTitle("纤维参数 (镀层加热 — L 为热沉间距)")
            self.lbl_R.setText("R_镀层 (Ω):")
            self.lbl_Rp.setText("α_R,镀层 (1/K):")
            self.mat_g.setTitle("纤维芯材料参数")
            self.use_rad.setText("计入辐射 (ε>0，写入 m²)")
            self.model_hint.setText(
                "绝缘丝镀层模型：V3∝[2−2cosh(mL)+mL sinh(mL)]/(m³ sinh(mL))")
            cur = self.mat_preset.currentText()
            self.mat_preset.blockSignals(True)
            self.mat_preset.clear()
            self.mat_preset.addItems(
                list(INSULATING_MATERIAL_PRESETS.keys()) + ['custom'])
            self.mat_preset.blockSignals(False)
            if cur in INSULATING_MATERIAL_PRESETS:
                self.mat_preset.setCurrentText(cur)
            elif 'glass' in INSULATING_MATERIAL_PRESETS:
                self.mat_preset.setCurrentText('glass')
                self._load_mat_preset('glass')
        else:
            self.fiber_g.setTitle("纤维参数 (Lu 四线悬空段)")
            self.lbl_R.setText("R (Ω):")
            self.lbl_Rp.setText("α_R = (1/R)dR/dT (1/K):")
            self.mat_g.setTitle("材料参数")
            self.use_rad.setText("计入辐射修正 (ε>0)")
            self.model_hint.setText("导电丝 Lu 模型：V3∝1/√(1+(2ωγ)²)")
            cur = self.mat_preset.currentText()
            self.mat_preset.blockSignals(True)
            self.mat_preset.clear()
            self.mat_preset.addItems(list(MATERIAL_PRESETS.keys()) + ['custom'])
            self.mat_preset.blockSignals(False)
            if cur in MATERIAL_PRESETS:
                self.mat_preset.setCurrentText(cur)
            elif 'platinum' in MATERIAL_PRESETS:
                self.mat_preset.setCurrentText('platinum')
                self._load_mat_preset('platinum')

    def _update_I_hint(self):
        try:
            R = _parse_float(self.inp_R.text(), 'R')
            V = _parse_float(self.inp_V.text(), 'V_rms')
            I = _I_from_V(V, R)
            self.lbl_I_hint.setText(f"I_rms = V/R ≈ {I:.4g} A")
        except Exception:
            self.lbl_I_hint.setText("I_rms = V/R ≈ —")

    def _load_fiber_preset(self, name: str):
        p = FIBER_PRESETS.get(name)
        if not p:
            return
        self.inp_L.blockSignals(True)
        self.inp_D.blockSignals(True)
        self.inp_S.blockSignals(True)
        self.inp_R.blockSignals(True)
        self.inp_Rp.blockSignals(True)
        self.inp_V.blockSignals(True)
        self.inp_L.setText(f"{p['L']:.6g}")
        self.inp_D.setText(f"{p['D']:.6g}")
        self.inp_S.setText("")
        self.inp_R.setText(f"{p['R']:.6g}")
        self.inp_Rp.setText(f"{_preset_alpha_R(p):.6g}")
        self.inp_V.setText(f"{_preset_V_rms(p):.6g}")
        for w in (self.inp_L, self.inp_D, self.inp_S, self.inp_R, self.inp_Rp, self.inp_V):
            w.blockSignals(False)
        self._update_I_hint()
        mat_name = p.get('material')
        if mat_name and self.current_model() == MODEL_CONDUCTIVE:
            if mat_name in MATERIAL_PRESETS:
                self.mat_preset.blockSignals(True)
                self.mat_preset.setCurrentText(mat_name)
                self.mat_preset.blockSignals(False)
                self._load_mat_preset(mat_name)
        self.schedule_update()

    def _load_mat_preset(self, name: str):
        if self.current_model() == MODEL_INSULATING:
            m = INSULATING_MATERIAL_PRESETS.get(name)
        else:
            m = MATERIAL_PRESETS.get(name)
        if not m:
            return
        self.inp_k.blockSignals(True)
        self.inp_rho.blockSignals(True)
        self.inp_Cp.blockSignals(True)
        self.inp_k.setText(f"{m.kappa:.6g}")
        self.inp_rho.setText(f"{m.rho:.6g}")
        self.inp_Cp.setText(f"{m.C_p:.6g}")
        for w in (self.inp_k, self.inp_rho, self.inp_Cp):
            w.blockSignals(False)
        self.schedule_update()

    def get_fiber(self) -> FiberGeometry:
        L = _parse_float(self.inp_L.text(), 'L')
        D = _parse_float(self.inp_D.text(), 'D')
        S_txt = self.inp_S.text().strip()
        S = _parse_float(S_txt, 'S') if S_txt else np.pi * D ** 2 / 4.0
        R = _parse_float(self.inp_R.text(), 'R')
        alpha_R = _parse_float(self.inp_Rp.text(), 'α_R')
        V_rms = _parse_float(self.inp_V.text(), 'V_rms')
        return FiberGeometry(
            L=L, S=S, R=R,
            R_prime=_Rp_from_alpha_R(R, alpha_R),
            I_rms=_I_from_V(V_rms, R),
        )

    def get_material(self) -> MaterialProps:
        return MaterialProps(
            kappa=_parse_float(self.inp_k.text(), 'κ'),
            rho=_parse_float(self.inp_rho.text(), 'ρ'),
            C_p=_parse_float(self.inp_Cp.text(), 'C_p'),
        )

    def get_radiation(self):
        if not self.use_rad.isChecked():
            return None
        eps = _parse_float(self.inp_eps.text(), 'ε')
        if eps <= 0:
            return None
        return RadiationParams(
            emissivity=eps,
            T0=_parse_float(self.inp_T0.text(), 'T0'),
            diameter=_parse_float(self.inp_D.text(), 'D'),
        )

    def sync_from_fitting(self):
        if self._fit_src is None:
            QMessageBox.warning(self, "提示", "未关联拟合页")
            return
        try:
            cfg = self._fit_src.get_model_config()
            fib = cfg['fiber']
            mat = cfg['material']
            rad = cfg.get('radiation', {})

            # 先同步模型类型（会刷新预设列表）
            self.set_model(cfg.get('model_type', MODEL_CONDUCTIVE), from_fitting=True)

            widgets = [
                self.inp_L, self.inp_D, self.inp_S, self.inp_R, self.inp_Rp, self.inp_V,
                self.inp_k, self.inp_rho, self.inp_Cp, self.inp_eps, self.inp_T0,
            ]
            for w in widgets:
                w.blockSignals(True)

            self.inp_L.setText(f"{fib['L']:.6g}")
            self.inp_D.setText(f"{fib.get('D', diameter_from_area(fib['S'])):.6g}")
            self.inp_S.setText(f"{fib['S']:.6g}")
            self.inp_R.setText(f"{fib['R']:.6g}")
            if 'alpha_R' in fib:
                self.inp_Rp.setText(f"{fib['alpha_R']:.6g}")
            else:
                self.inp_Rp.setText(
                    f"{_alpha_R_from_Rp(fib['R'], fib.get('R_prime', 0.0)):.6g}")
            if 'V_rms' in fib:
                self.inp_V.setText(f"{fib['V_rms']:.6g}")
            else:
                self.inp_V.setText(
                    f"{_V_from_I(fib.get('I_rms', 0.0), fib['R']):.6g}")
            self.inp_k.setText(f"{mat['kappa']:.6g}")
            self.inp_rho.setText(f"{mat['rho']:.6g}")
            self.inp_Cp.setText(f"{mat['C_p']:.6g}")
            self.use_eq24.setChecked(bool(cfg.get('use_eq24', False)))
            self.use_rad.setChecked(bool(rad.get('enabled', False)))
            self.inp_eps.setText(f"{rad.get('emissivity', 0):.6g}")
            self.inp_T0.setText(f"{rad.get('T0', 300):.6g}")

            # 若拟合页已有拟合结果，优先用拟合材料
            if self._fit_src.fitted_material is not None:
                fm = self._fit_src.fitted_material
                self.inp_k.setText(f"{fm.kappa:.6g}")
                self.inp_rho.setText(f"{fm.rho:.6g}")
                self.inp_Cp.setText(f"{fm.C_p:.6g}")

            for w in widgets:
                w.blockSignals(False)
            self._update_I_hint()

            self.info.append(
                f"已同步拟合页（模型: {MODEL_LABELS.get(self.current_model())}）。\n")
            self.schedule_update()
            QMessageBox.information(self, "同步成功", "已同步拟合页参数到正向模拟。")
        except Exception as e:
            QMessageBox.critical(self, "同步失败", str(e))
            traceback.print_exc()

    def load_experimental_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载实验数据", "", "Excel/CSV (*.xlsx *.xls *.csv);;All (*.*)")
        if not path:
            return
        try:
            if path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            freqs, V3 = parse_fiber_v3_dataframe(df)
            self.exp_data = {'freqs': freqs, 'V3': V3}
            self.exp_file_label.setText(path.replace('\\', '/').split('/')[-1])
            self.schedule_update()
            QMessageBox.information(self, "成功", "实验数据已加载，可与模拟曲线对比。")
        except Exception as e:
            self.exp_data = None
            self.exp_file_label.setText("加载失败")
            QMessageBox.critical(self, "错误", str(e))

    def clear_experimental_data(self):
        self.exp_data = None
        self.exp_file_label.setText("未加载")
        self.schedule_update()

    def run_sim_silent(self):
        try:
            self.run_sim(silent=True)
        except Exception:
            pass

    def run_sim(self, silent: bool = False):
        try:
            fiber = self.get_fiber()
            mat = self.get_material()
            rad = self.get_radiation()
            use24 = self.use_eq24.isChecked()
            model = self.current_model()

            f0 = float(self.fmin.text())
            f1 = float(self.fmax.text())
            n = max(2, int(float(self.npts.text())))
            if f0 <= 0 or f1 <= 0 or f1 < f0:
                raise ValueError("频率范围无效")

            if self.freq_scale.currentText() == "线性分布":
                freqs = np.linspace(f0, f1, n)
            else:
                freqs = np.logspace(np.log10(f0), np.log10(f1), n)

            if model == MODEL_INSULATING:
                V3, info = calculate_V3omega_insulating(
                    freqs, fiber, mat,
                    diameter=_parse_float(self.inp_D.text(), 'D'),
                    radiation=rad,
                )
            else:
                V3, info = calculate_V3omega(
                    freqs, fiber, mat, use_eq24=use24, radiation=rad)

            if self.add_noise.isChecked():
                lvl = float(self.noise_amp.value())
                noise = (np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))) * lvl
                V3 = V3 * (1.0 + noise)

            self.last_sim_data = {
                'freqs': freqs,
                'V3': V3,
                'info': info,
                'fiber': fiber,
                'material': mat,
                'model': model,
            }
            self._redraw_from_cache()

            if model == MODEL_INSULATING:
                msg = (
                    f"[{MODEL_LABELS[model]}]\n"
                    f"α={info['alpha']:.4e} m²/s, h_r={info.get('hr', 0):.4g} W/m²K\n"
                    f"低频 |V3|_rms≈{info.get('V3_lf_rms', np.nan):.4e} V\n"
                    f"点数={len(freqs)}, |V3|范围 "
                    f"{np.abs(V3).min()*1e6:.3g}~{np.abs(V3).max()*1e6:.3g} μV\n"
                )
                report_fn = lambda: validity_report_insulating(
                    fiber, mat, freqs,
                    diameter=_parse_float(self.inp_D.text(), 'D'),
                    radiation=rad,
                )
            else:
                msg = (
                    f"[{MODEL_LABELS[model]}]\n"
                    f"α={info['alpha']:.4e} m²/s, γ={info['gamma']:.4e} s\n"
                    f"Δ0={info['Delta0']:.3g} K, 条件(10)={info['cond10']:.3e}, "
                    f"gγ={info['g_gamma']:.3e}\n"
                    f"点数={len(freqs)}, |V3|范围 "
                    f"{np.abs(V3).min()*1e6:.3g}~{np.abs(V3).max()*1e6:.3g} μV\n"
                )
                report_fn = lambda: validity_report(fiber, mat, freqs, rad)

            if not silent:
                self.info.append(msg)
                self.info.append(report_fn() + "\n")

        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "模拟失败", str(e))
                traceback.print_exc()
            raise

    def _redraw_from_cache(self):
        if not self.last_sim_data:
            return
        freqs = self.last_sim_data['freqs']
        V3 = self.last_sim_data['V3']
        self.figure.clear()
        pt = self.plot_type.currentText()

        def plot_exp_amp(ax):
            if self.exp_data is not None:
                ax.loglog(self.exp_data['freqs'], np.abs(self.exp_data['V3']) * 1e6,
                          'ro', label='Exp', markersize=4, alpha=0.7)

        def plot_exp_pha(ax):
            if self.exp_data is not None:
                ax.semilogx(self.exp_data['freqs'],
                            np.angle(self.exp_data['V3'], deg=True),
                            'ro', label='Exp', markersize=4, alpha=0.7)

        if pt == "Amplitude & Phase":
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)
            plot_exp_amp(ax1)
            ax1.loglog(freqs, np.abs(V3) * 1e6, 'b-', lw=2, label='Sim')
            ax1.set_ylabel('|V3| (μV)')
            ax1.legend(fontsize='small')
            ax1.grid(True, which='both', ls='--', alpha=0.3)
            plot_exp_pha(ax2)
            ax2.semilogx(freqs, np.angle(V3, deg=True), 'b-', lw=2, label='Sim')
            ax2.set_ylabel('Phase (deg)')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.legend(fontsize='small')
            ax2.grid(True, which='both', ls='--', alpha=0.3)
        elif pt == "Phase":
            ax = self.figure.add_subplot(111)
            plot_exp_pha(ax)
            ax.semilogx(freqs, np.angle(V3, deg=True), 'b-', lw=2, label='Sim')
            ax.set_ylabel('Phase (deg)')
            ax.set_xlabel('Frequency (Hz)')
            ax.legend()
            ax.grid(True, which='both', ls='--', alpha=0.3)
        elif pt == "Real+Imag":
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)
            if self.exp_data is not None:
                ax1.semilogx(self.exp_data['freqs'], self.exp_data['V3'].real * 1e6,
                             'ro', markersize=4, label='Exp Re')
                ax2.semilogx(self.exp_data['freqs'], self.exp_data['V3'].imag * 1e6,
                             'ro', markersize=4, label='Exp Im')
            ax1.semilogx(freqs, V3.real * 1e6, 'b-', lw=2, label='Sim Re')
            ax2.semilogx(freqs, V3.imag * 1e6, 'b-', lw=2, label='Sim Im')
            ax1.set_ylabel('Re(V3) (μV)')
            ax2.set_ylabel('Im(V3) (μV)')
            ax2.set_xlabel('Frequency (Hz)')
            ax1.legend(fontsize='small')
            ax2.legend(fontsize='small')
            ax1.grid(True, which='both', ls='--', alpha=0.3)
            ax2.grid(True, which='both', ls='--', alpha=0.3)
        else:
            ax = self.figure.add_subplot(111)
            plot_exp_amp(ax)
            ax.loglog(freqs, np.abs(V3) * 1e6, 'b-', lw=2, label='Sim')
            ax.set_ylabel('|V3| (μV)')
            ax.set_xlabel('Frequency (Hz)')
            ax.legend()
            ax.grid(True, which='both', ls='--', alpha=0.3)

        self.figure.suptitle(
            f"Forward simulation ({MODEL_LABELS.get(self.last_sim_data.get('model', MODEL_CONDUCTIVE), 'Fiber')})")
        self.figure.tight_layout()
        self.canvas.draw()

    def export_simulation_data(self):
        """导出模拟数据为 CSV/Excel，格式兼容拟合页输入。"""
        if self.last_sim_data is None:
            QMessageBox.warning(self, "提示", "请先运行一次模拟以生成数据。")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "保存模拟数据",
            f"fiber_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)",
        )
        if not filename:
            return

        try:
            freqs = self.last_sim_data['freqs']
            V3 = self.last_sim_data['V3']
            df = pd.DataFrame()
            df['频率 (Hz)'] = freqs
            df['V3_Real'] = V3.real
            df['V3_Imag'] = V3.imag
            df['V3_幅值'] = np.abs(V3)
            df['V3_相位'] = np.degrees(np.unwrap(np.angle(V3)))

            if filename.lower().endswith(('.xlsx', '.xls')):
                df.to_excel(filename, index=False)
            else:
                if not filename.lower().endswith('.csv'):
                    filename += '.csv'
                df.to_csv(filename, index=False, encoding='utf-8-sig')

            QMessageBox.information(
                self, "成功",
                f"数据已保存至:\n{filename}\n共 {len(df)} 个频点"
                f"（含 频率/Real/Imag/幅值/相位，可直接导入「数据拟合」）。"
            )
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存文件时出错:\n{str(e)}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# 灵敏度分析
# ---------------------------------------------------------------------------
class FiberSensitivityWidget(QWidget):
    """对选定参数做 1% 相对扰动，绘制 (dY/Y)/(dp/p) 灵敏度曲线。"""

    SENS_PARAMS = [
        ('kappa', 'κ'),
        ('C_p', 'C_p'),
        ('rho', 'ρ'),
        ('L', 'L'),
        ('D', 'D'),
        ('R', 'R'),
        ('alpha_R', 'α_R'),
        ('V_rms', 'V_rms'),
    ]

    def __init__(self):
        super().__init__()
        self._fit_src = None
        self.sensitivity_data = None
        self.init_ui()

    def set_fitting_source(self, w: FiberFittingWidget):
        self._fit_src = w

    def init_ui(self):
        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        sync = QPushButton("从数据拟合同步参数")
        sync.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 6px; }")
        sync.clicked.connect(self.sync_from_fitting)
        left.addWidget(sync)

        pg = QGroupBox("分析参数选择")
        pl = QVBoxLayout()
        self.param_list = QListWidget()
        for key, label in self.SENS_PARAMS:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if key in ('kappa', 'C_p') else Qt.Unchecked)
            item.setData(Qt.UserRole, key)
            self.param_list.addItem(item)
        pl.addWidget(self.param_list)
        row = QHBoxLayout()
        b_all = QPushButton("全选")
        b_none = QPushButton("全不选")
        b_all.clicked.connect(lambda: self._set_all(True))
        b_none.clicked.connect(lambda: self._set_all(False))
        row.addWidget(b_all)
        row.addWidget(b_none)
        pl.addLayout(row)
        pg.setLayout(pl)
        left.addWidget(pg)

        freq_g = QGroupBox("频率范围")
        ff = QFormLayout()
        self.fmin = QLineEdit("1")
        self.fmax = QLineEdit("1000")
        self.npts = QLineEdit("80")
        self.freq_scale = QComboBox()
        self.freq_scale.addItems(["对数分布", "线性分布"])
        for w in (self.fmin, self.fmax, self.npts):
            w.setToolTip("可自由修改；点「运行灵敏度分析」后按此范围重算")
        ff.addRow("f_min (Hz):", self.fmin)
        ff.addRow("f_max (Hz):", self.fmax)
        ff.addRow("点数:", self.npts)
        ff.addRow("分布:", self.freq_scale)
        fill_btn = QPushButton("从拟合页数据填充")
        fill_btn.setToolTip("用拟合页当前数据（或筛选窗）的最小/最大频率填入上方")
        fill_btn.clicked.connect(lambda: self.fill_freq_from_fitting(silent=False))
        ff.addRow(fill_btn)
        freq_g.setLayout(ff)
        left.addWidget(freq_g)

        dg = QGroupBox("显示类型")
        dl = QVBoxLayout()
        self.display_combo = QComboBox()
        self.display_combo.addItems([
            "Amplitude Sensitivity (Relative)",
            "Phase Sensitivity (Deg)",
            "Real Part Sensitivity (Norm. to Mag)",
            "Imag Part Sensitivity (Norm. to Mag)",
            "Amplitude & Phase (Dual Plot)",
        ])
        self.display_combo.currentIndexChanged.connect(self.update_plot)
        dl.addWidget(self.display_combo)
        dg.setLayout(dl)
        left.addWidget(dg)

        hint = QLabel(
            "相对灵敏度 ≈ (dY/Y)/(dp/p)，扰动 1%。\n"
            "幅值 |灵敏度| 大 → 该频段对该参数敏感。\n"
            "频率范围可自由设定，不依赖实验数据点。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555; font-size:11px;")
        left.addWidget(hint)

        self.calc_btn = QPushButton("运行灵敏度分析")
        self.calc_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-size: 14px; padding: 8px; }")
        self.calc_btn.clicked.connect(self.run_analysis)
        left.addWidget(self.calc_btn)
        left.addStretch()

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMaximumWidth(320)
        layout.addWidget(left_w)

        right = QVBoxLayout()
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        right.addWidget(self.canvas)
        right.addWidget(NavigationToolbar(self.canvas, self))
        layout.addLayout(right, stretch=1)

        self.status = QLabel("就绪：请同步参数后运行分析")
        self.status.setStyleSheet("color:#333;")
        # put status under left
        left.addWidget(self.status)

    def _set_all(self, checked: bool):
        st = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.param_list.count()):
            self.param_list.item(i).setCheckState(st)

    def sync_from_fitting(self):
        if self._fit_src is None:
            QMessageBox.warning(self, "提示", "未关联拟合页")
            return
        self.fill_freq_from_fitting(silent=True)
        self.status.setText("已从拟合页同步当前参数（分析时读取）")
        QMessageBox.information(
            self, "同步",
            "将使用拟合页当前纤维/材料/辐射参数进行分析。\n"
            "若拟合页已有数据，频率范围已按其 f_min/f_max 填充（仍可自行修改）。")

    def fill_freq_from_fitting(self, silent: bool = False):
        if self._fit_src is None:
            if not silent:
                QMessageBox.warning(self, "提示", "未关联拟合页")
            return
        src = self._fit_src
        f0 = f1 = None
        if src.freqs is not None and len(src.freqs) > 0:
            try:
                freqs_m, _ = src._masked_data()
                if len(freqs_m) > 0:
                    f0, f1 = float(np.min(freqs_m)), float(np.max(freqs_m))
            except Exception:
                f0 = float(np.min(src.freqs))
                f1 = float(np.max(src.freqs))
        if f0 is None:
            try:
                f0 = float(src.fmin.value())
                f1 = float(src.fmax.value())
                if f0 <= 0 or f1 <= 0 or f1 <= f0:
                    raise ValueError
            except Exception:
                if not silent:
                    QMessageBox.warning(self, "提示", "拟合页尚无有效频率数据可填充")
                return
        self.fmin.setText(f"{f0:g}")
        self.fmax.setText(f"{f1:g}")
        if not silent:
            self.status.setText(f"已填充频率范围: {f0:g} – {f1:g} Hz")

    def _build_freq_grid(self) -> np.ndarray:
        f0 = float(self.fmin.text().strip())
        f1 = float(self.fmax.text().strip())
        n = int(float(self.npts.text().strip()))
        if f0 <= 0 or f1 <= 0:
            raise ValueError("f_min / f_max 必须为正数")
        if f1 <= f0:
            raise ValueError("需要 f_max > f_min")
        if n < 2:
            raise ValueError("点数至少为 2")
        if self.freq_scale.currentText() == "线性分布":
            return np.linspace(f0, f1, n)
        return np.logspace(np.log10(f0), np.log10(f1), n)

    def _selected_params(self):
        out = []
        for i in range(self.param_list.count()):
            item = self.param_list.item(i)
            if item.checkState() == Qt.Checked:
                out.append((item.text(), item.data(Qt.UserRole)))
        return out

    def run_analysis(self):
        if self._fit_src is None:
            QMessageBox.warning(self, "提示", "未关联拟合页")
            return
        selected = self._selected_params()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个参数")
            return
        try:
            fiber = self._fit_src.get_fiber()
            mat = self._fit_src.get_material()
            rad = self._fit_src.get_radiation()
            model = self._fit_src.current_model()
            use24 = (model == MODEL_CONDUCTIVE and self._fit_src.use_eq24.isChecked())
            alpha_R = _parse_float(self._fit_src.inp_Rp.text(), 'α_R')
            D = _parse_float(self._fit_src.inp_D.text(), 'D')
            V_rms = self._fit_src.get_V_rms()

            freqs = self._build_freq_grid()

            fwd_kw = dict(
                model=model, diameter=D, radiation=rad,
                use_eq24=use24, alpha_R=alpha_R, V_rms=V_rms,
            )
            V3_base = FiberVizWidget._forward_v3(freqs, fiber, mat, **fwd_kw)
            amp_base = np.maximum(np.abs(V3_base), 1e-30)
            phase_base = np.unwrap(np.angle(V3_base))

            perturbation = 0.01
            results = {}
            for name, key in selected:
                V3_new = FiberVizWidget._v3_with_param_scale(
                    freqs, fiber, mat, key, 1.0 + perturbation, **fwd_kw)
                if V3_new is None:
                    continue
                sens_amp = (np.abs(V3_new) - amp_base) / amp_base / perturbation
                sens_phase = (np.unwrap(np.angle(V3_new)) - phase_base) * 180 / np.pi / perturbation
                sens_real = (V3_new.real - V3_base.real) / amp_base / perturbation
                sens_imag = (V3_new.imag - V3_base.imag) / amp_base / perturbation
                results[name] = {
                    'amp': sens_amp, 'phase': sens_phase,
                    'real': sens_real, 'imag': sens_imag,
                }

            self.sensitivity_data = {'freqs': freqs, 'results': results}
            self.status.setText(
                f"完成：{len(results)} 个参数，{len(freqs)} 频点 "
                f"({freqs[0]:g}–{freqs[-1]:g} Hz)，模型={MODEL_LABELS.get(model)}")
            self.update_plot()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "分析错误", str(e))

    def update_plot(self):
        if not self.sensitivity_data:
            return
        self.figure.clear()
        display = self.display_combo.currentText()
        freqs = self.sensitivity_data['freqs']
        results = self.sensitivity_data['results']

        if display == "Amplitude & Phase (Dual Plot)":
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)
            for name, data in results.items():
                ax1.semilogx(freqs, data['amp'], label=name)
                ax2.semilogx(freqs, data['phase'], label=name)
            ax1.set_ylabel('Amp Sens. (dA/A)/(dp/p)')
            ax1.set_title('Amplitude Sensitivity')
            ax1.legend(fontsize='small')
            ax1.grid(True, which='both', ls='--', alpha=0.3)
            ax2.set_ylabel('Phase Sens. (deg)/(dp/p)')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_title('Phase Sensitivity')
            ax2.legend(fontsize='small')
            ax2.grid(True, which='both', ls='--', alpha=0.3)
        else:
            ax = self.figure.add_subplot(111)
            key_map = {
                "Amplitude Sensitivity (Relative)": 'amp',
                "Phase Sensitivity (Deg)": 'phase',
                "Real Part Sensitivity (Norm. to Mag)": 'real',
                "Imag Part Sensitivity (Norm. to Mag)": 'imag',
            }
            data_key = key_map.get(display, 'amp')
            for name, data in results.items():
                ax.semilogx(freqs, data[data_key], label=name)
            ax.set_xlabel('Frequency (Hz)')
            if data_key == 'amp':
                ax.set_ylabel('Sensitivity (dA/A) / (dp/p)')
                ax.set_title('Amplitude Sensitivity')
            elif data_key == 'phase':
                ax.set_ylabel('Sensitivity (deg) / (dp/p)')
                ax.set_title('Phase Sensitivity')
            else:
                ax.set_ylabel('Sensitivity (dV/|V|) / (dp/p)')
                ax.set_title(display)
            ax.legend()
            ax.grid(True, which='both', ls='--', alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class FiberThermalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("纤维丝 3ω 热物性测量系统 — 导电/绝缘可切换 v1.1")
        self.resize(1280, 860)
        self._syncing = False
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab { padding: 10px 18px; font-size: 12px; }
            QTabBar::tab:selected { background: #2196F3; color: white; }
        """)

        self.fitting = FiberFittingWidget()
        self.tabs.addTab(self.fitting, "📊 数据拟合")

        self.viz = FiberVizWidget()
        self.tabs.addTab(self.viz, "📈 结果可视化")

        self.sim = FiberSimWidget()
        self.sim.set_fitting_source(self.fitting)
        self.tabs.addTab(self.sim, "🔁 正向模拟")

        self.sensitivity = FiberSensitivityWidget()
        self.sensitivity.set_fitting_source(self.fitting)
        self.tabs.addTab(self.sensitivity, "🎯 灵敏度分析")

        layout.addWidget(self.tabs)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._update_status_model(MODEL_CONDUCTIVE)

        menubar = self.menuBar()
        help_m = menubar.addMenu("帮助(&H)")
        about = QAction("关于", self)
        about.triggered.connect(self.show_about)
        help_m.addAction(about)

    def _update_status_model(self, model_type: str):
        self.status.showMessage(f"就绪 — 当前模型: {MODEL_LABELS.get(model_type, model_type)}")

    def on_fiber_model_changed(self, model_type: str):
        """拟合页切换模型 → 同步模拟页。"""
        if self._syncing:
            return
        self._syncing = True
        try:
            self.sim.set_model(model_type, from_fitting=True)
            self._update_status_model(model_type)
        finally:
            self._syncing = False

    def on_sim_model_changed(self, model_type: str):
        """模拟页切换模型 → 同步拟合页。"""
        if self._syncing:
            return
        self._syncing = True
        try:
            idx = self.fitting.model_combo.findData(model_type)
            if idx >= 0 and idx != self.fitting.model_combo.currentIndex():
                self.fitting.model_combo.setCurrentIndex(idx)
            self._update_status_model(model_type)
        finally:
            self._syncing = False

    def on_fiber_fit_done(self):
        fitter = self.fitting.fitter
        if fitter is None:
            return
        freqs = fitter.freqs
        V3_exp = fitter.V3_exp
        V3_fit = fitter.predict(freqs)
        model = self.fitting.current_model()
        fiber = fitter.fiber
        material = fitter.material
        alpha_R = float(getattr(fitter, 'alpha_R',
                                _alpha_R_from_Rp(fiber.R, fiber.R_prime)))
        diameter = float(getattr(fitter, 'diameter',
                                 diameter_from_area(fiber.S) if fiber.S > 0 else 0.0))
        radiation = self.fitting.get_radiation()
        use_eq24 = (model == MODEL_CONDUCTIVE and self.fitting.use_eq24.isChecked())
        params_fitted = list(getattr(fitter, 'params_to_fit', []) or [])
        model_ctx = {
            'fiber': FiberGeometry(
                fiber.L, fiber.S, fiber.R, fiber.R_prime, fiber.I_rms),
            'material': MaterialProps(material.kappa, material.rho, material.C_p),
            'params_fitted': params_fitted,
            'fwd_kw': {
                'model': model,
                'diameter': diameter,
                'radiation': radiation,
                'use_eq24': use_eq24,
                'alpha_R': alpha_R,
                'V_rms': float(getattr(fitter, 'V_rms',
                                       _V_from_I(fiber.I_rms, fiber.R))),
            },
        }
        self.viz.set_results(
            freqs, V3_exp, V3_fit,
            title=f"{MODEL_LABELS.get(model, model)} Fit ({self.fitting.fit_mode.currentText()})",
            model_ctx=model_ctx,
        )
        self.tabs.setCurrentWidget(self.viz)
        n_pm = len(params_fitted)
        self.status.showMessage(
            f"拟合完成 [{MODEL_LABELS.get(model)}] "
            f"κ={fitter.material.kappa:.4g}, C_p={fitter.material.C_p:.4g}"
            + (f"；已对 {n_pm} 个拟合参数分别 ±20% 正演" if n_pm else ""))

    def show_about(self):
        QMessageBox.about(
            self, "关于",
            "<h3>纤维丝 3ω 热物性测量系统</h3>"
            "<p>可在 <b>导电丝</b> 与 <b>绝缘丝镀层</b> 模型间自由切换。</p>"
            "<p><b>导电丝</b>: Lu, Yi, Zhang, <i>Rev. Sci. Instrum.</i> "
            "<b>72</b>, 2996 (2001).<br>"
            "丝本体加热/测温；式(18)(19)(20)，可选式(24)与辐射表观修正。</p>"
            "<p><b>绝缘丝镀层</b>: Munro / Xing / Ban, IMECE2015-52056。<br>"
            "镀金属作加热器与温度计；"
            "m²=i2ω/α+4h_r/(Dk)；V3 用 sinh 公式（文中式8 按式9 取 I₀³）。</p>"
            "<p>界面风格对齐主程序 thermal_measurement_gui2.py"
            "（含灵敏度分析、拟合前后相关性）。</p>"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(240, 240, 240))
    app.setPalette(pal)
    win = FiberThermalGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
