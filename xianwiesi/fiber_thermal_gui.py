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
    QSplitter, QAction, QScrollArea,
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
        row = QHBoxLayout()
        row.addWidget(self.fit_kappa)
        row.addWidget(self.fit_Cp)
        mode_l.addRow("参数:", row)
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
        self.inp_Rp = QLineEdit("0.01")
        self.inp_I = QLineEdit("0.005")
        self.lbl_R = QLabel("R (Ω):")
        self.lbl_Rp = QLabel("R' = dR/dT (Ω/K):")
        fl.addRow("L (m):", self.inp_L)
        fl.addRow("D (m):", self.inp_D)
        fl.addRow("S (m²):", self.inp_S)
        fl.addRow(self.lbl_R, self.inp_R)
        fl.addRow(self.lbl_Rp, self.inp_Rp)
        fl.addRow("I_rms (A):", self.inp_I)
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
        self.valid_btn = QPushButton("适用性检查")
        self.valid_btn.clicked.connect(self.run_validity)
        self.export_btn = QPushButton("导出结果 JSON")
        self.export_btn.clicked.connect(self.export_results)
        btn_row.addWidget(self.fit_btn)
        btn_row.addWidget(self.valid_btn)
        btn_row.addWidget(self.export_btn)
        layout.addLayout(btn_row)

        # 结果
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

        self._load_fiber_preset(self.fiber_preset.currentText())
        self._load_mat_preset(self.mat_preset.currentText())
        self._on_model_changed()

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
            self.lbl_Rp.setText("R'_镀层 = dR/dT (Ω/K):")
            self.mat_g.setTitle("纤维芯材料初值")
            self.rad_g.setTitle("径向辐射 h_r=4εσT0³（细丝建议开启）")
            self.use_rad.setText("计入辐射 (ε>0，写入 m²)")
            self.rad_hint.setText(
                "绝缘细丝辐射往往不可忽略；R/R' 为镀层标定值，κ/C_p 为纤维芯。")
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
            self.lbl_Rp.setText("R' = dR/dT (Ω/K):")
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
        self.inp_Rp.setText(f"{p['R_prime']:.6g}")
        self.inp_I.setText(f"{p['I_rms']:.6g}")

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
        return FiberGeometry(
            L=L, S=S,
            R=_parse_float(self.inp_R.text(), 'R'),
            R_prime=_parse_float(self.inp_Rp.text(), "R'"),
            I_rms=_parse_float(self.inp_I.text(), 'I_rms'),
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
                'R_prime': fiber.R_prime, 'I_rms': fiber.I_rms,
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
            if not params:
                QMessageBox.warning(self, "提示", "请至少勾选一个拟合参数。")
                return

            model = self.current_model()
            if model == MODEL_INSULATING:
                self.fitter = InsulatingFiberFitter(
                    freqs, V3, fiber, mat0,
                    params_to_fit=params,
                    fit_mode=self._fit_mode_key(),
                    diameter=_parse_float(self.inp_D.text(), 'D'),
                    radiation=self.get_radiation(),
                )
            else:
                self.fitter = FiberModelFitter(
                    freqs, V3, fiber, mat0,
                    params_to_fit=params,
                    fit_mode=self._fit_mode_key(),
                    use_eq24=self.use_eq24.isChecked(),
                    radiation=self.get_radiation(),
                )
            fitted = self.fitter.fit(verbose=0)
            self.fitted_material = fitted

            # 写回初值框便于继续调
            self.inp_k.setText(f"{fitted.kappa:.6g}")
            self.inp_Cp.setText(f"{fitted.C_p:.6g}")

            info = self.fitter.last_info
            alpha = info['alpha']

            self.results_text.append("\n========== 拟合完成 ==========")
            self.results_text.append(
                f"模型: {MODEL_LABELS.get(model, model)}")
            self.results_text.append(f"成功: {self.fitter.success}  |  {self.fitter.message}")
            self.results_text.append(f"cost = {self.fitter.cost:.4e}")
            self.results_text.append(f"κ = {fitted.kappa:.6g} W/m·K")
            self.results_text.append(f"C_p = {fitted.C_p:.6g} J/kg·K")
            self.results_text.append(f"ρ = {fitted.rho:.6g} kg/m³ (固定)")
            self.results_text.append(f"α = {alpha:.4e} m²/s")
            if model == MODEL_INSULATING:
                self.results_text.append(
                    f"h_r = {info.get('hr', 0):.4g} W/m²K, "
                    f"ε = {info.get('emissivity', 0):g}")
                if info.get('V3_lf_rms'):
                    self.results_text.append(
                        f"低频极限 |V3|_rms ≈ {info['V3_lf_rms']:.4e} V")
                self.results_text.append(
                    validity_report_insulating(
                        fiber, fitted, freqs,
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
                    validity_report(fiber, fitted, freqs, self.get_radiation()))
            self.results_text.append("")

            self._fill_table(fitted, info, model)
            # 通知主窗口画图
            mw = self.window()
            if hasattr(mw, 'on_fiber_fit_done'):
                mw.on_fiber_fit_done()

        except Exception as e:
            self.results_text.append(f"\n拟合失败: {e}\n")
            traceback.print_exc()
            QMessageBox.critical(self, "拟合失败", str(e))

    def _fill_table(self, mat: MaterialProps, info: dict, model: str = MODEL_CONDUCTIVE):
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
            ]
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
            ]
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
                'R_prime': fiber.R_prime, 'I_rms_A': fiber.I_rms,
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
# 可视化
# ---------------------------------------------------------------------------
class FiberVizWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.bundle = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("显示:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Amplitude", "Phase", "Amplitude & Phase", "Real+Imag"])
        self.plot_type.currentTextChanged.connect(self.redraw)
        ctrl.addWidget(self.plot_type)
        ctrl.addStretch()
        self.save_btn = QPushButton("保存图")
        self.save_btn.clicked.connect(self.save_fig)
        ctrl.addWidget(self.save_btn)
        layout.addLayout(ctrl)

        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        layout.addWidget(NavigationToolbar(self.canvas, self))

    def set_results(self, freqs, V3_exp, V3_fit, title="Fiber 3ω Fit"):
        self.bundle = {
            'freqs': freqs, 'V3_exp': V3_exp, 'V3_fit': V3_fit, 'title': title,
        }
        self.redraw()

    def redraw(self):
        if not self.bundle:
            return
        freqs = self.bundle['freqs']
        Ve = self.bundle['V3_exp']
        Vm = self.bundle['V3_fit']
        self.figure.clear()
        pt = self.plot_type.currentText()

        if pt == "Amplitude & Phase":
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)
            ax1.loglog(freqs, np.abs(Ve) * 1e6, 'ro', label='Exp', markersize=5)
            ax1.loglog(freqs, np.abs(Vm) * 1e6, 'b-', label='Fit', lw=2)
            ax1.set_ylabel('|V3| (μV)')
            ax1.legend()
            ax1.grid(True, which='both', ls='--', alpha=0.3)
            ax2.semilogx(freqs, np.angle(Ve, deg=True), 'ro', label='Exp', markersize=5)
            ax2.semilogx(freqs, np.angle(Vm, deg=True), 'b-', label='Fit', lw=2)
            ax2.set_ylabel('Phase (deg)')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.legend()
            ax2.grid(True, which='both', ls='--', alpha=0.3)
        elif pt == "Phase":
            ax = self.figure.add_subplot(111)
            ax.semilogx(freqs, np.angle(Ve, deg=True), 'ro', label='Exp')
            ax.semilogx(freqs, np.angle(Vm, deg=True), 'b-', label='Fit', lw=2)
            ax.set_ylabel('Phase (deg)')
            ax.set_xlabel('Frequency (Hz)')
            ax.legend()
            ax.grid(True, which='both', ls='--', alpha=0.3)
        elif pt == "Real+Imag":
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)
            ax1.semilogx(freqs, Ve.real * 1e6, 'ro', label='Exp Re')
            ax1.semilogx(freqs, Vm.real * 1e6, 'b-', label='Fit Re', lw=2)
            ax1.set_ylabel('Re(V3) (μV)')
            ax1.legend()
            ax1.grid(True, which='both', ls='--', alpha=0.3)
            ax2.semilogx(freqs, Ve.imag * 1e6, 'ro', label='Exp Im')
            ax2.semilogx(freqs, Vm.imag * 1e6, 'b-', label='Fit Im', lw=2)
            ax2.set_ylabel('Im(V3) (μV)')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.legend()
            ax2.grid(True, which='both', ls='--', alpha=0.3)
        else:
            ax = self.figure.add_subplot(111)
            ax.loglog(freqs, np.abs(Ve) * 1e6, 'ro', label='Exp', markersize=6)
            ax.loglog(freqs, np.abs(Vm) * 1e6, 'b-', label='Fit', lw=2)
            ax.set_ylabel('|V3| (μV)')
            ax.set_xlabel('Frequency (Hz)')
            ax.legend()
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
        self.inp_Rp = QLineEdit("0.01")
        self.inp_I = QLineEdit("0.005")
        for w in (self.inp_L, self.inp_D, self.inp_S, self.inp_R, self.inp_Rp, self.inp_I):
            w.textChanged.connect(self.schedule_update)
        self.lbl_R = QLabel("R (Ω):")
        self.lbl_Rp = QLabel("R' = dR/dT (Ω/K):")
        fl.addRow("L (m):", self.inp_L)
        fl.addRow("D (m):", self.inp_D)
        fl.addRow("S (m²):", self.inp_S)
        fl.addRow(self.lbl_R, self.inp_R)
        fl.addRow(self.lbl_Rp, self.inp_Rp)
        fl.addRow("I_rms (A):", self.inp_I)
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
            self.lbl_Rp.setText("R'_镀层 = dR/dT (Ω/K):")
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
            self.lbl_Rp.setText("R' = dR/dT (Ω/K):")
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

    def _load_fiber_preset(self, name: str):
        p = FIBER_PRESETS.get(name)
        if not p:
            return
        self.inp_L.blockSignals(True)
        self.inp_D.blockSignals(True)
        self.inp_S.blockSignals(True)
        self.inp_R.blockSignals(True)
        self.inp_Rp.blockSignals(True)
        self.inp_I.blockSignals(True)
        self.inp_L.setText(f"{p['L']:.6g}")
        self.inp_D.setText(f"{p['D']:.6g}")
        self.inp_S.setText("")
        self.inp_R.setText(f"{p['R']:.6g}")
        self.inp_Rp.setText(f"{p['R_prime']:.6g}")
        self.inp_I.setText(f"{p['I_rms']:.6g}")
        for w in (self.inp_L, self.inp_D, self.inp_S, self.inp_R, self.inp_Rp, self.inp_I):
            w.blockSignals(False)
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
        return FiberGeometry(
            L=L, S=S,
            R=_parse_float(self.inp_R.text(), 'R'),
            R_prime=_parse_float(self.inp_Rp.text(), "R'"),
            I_rms=_parse_float(self.inp_I.text(), 'I_rms'),
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
                self.inp_L, self.inp_D, self.inp_S, self.inp_R, self.inp_Rp, self.inp_I,
                self.inp_k, self.inp_rho, self.inp_Cp, self.inp_eps, self.inp_T0,
            ]
            for w in widgets:
                w.blockSignals(True)

            self.inp_L.setText(f"{fib['L']:.6g}")
            self.inp_D.setText(f"{fib.get('D', diameter_from_area(fib['S'])):.6g}")
            self.inp_S.setText(f"{fib['S']:.6g}")
            self.inp_R.setText(f"{fib['R']:.6g}")
            self.inp_Rp.setText(f"{fib['R_prime']:.6g}")
            self.inp_I.setText(f"{fib['I_rms']:.6g}")
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
        self.viz.set_results(
            freqs, V3_exp, V3_fit,
            title=f"{MODEL_LABELS.get(model, model)} Fit ({self.fitting.fit_mode.currentText()})")
        self.tabs.setCurrentWidget(self.viz)
        self.status.showMessage(
            f"拟合完成 [{MODEL_LABELS.get(model)}] "
            f"κ={fitter.material.kappa:.4g}, C_p={fitter.material.C_p:.4g}")

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
            "<p>界面风格对齐主程序 thermal_measurement_gui2.py。</p>"
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
