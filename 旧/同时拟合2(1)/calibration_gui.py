"""
3ω 信号通路校正 GUI 模块

作为独立 Tab 集成到主 GUI 中，提供:
1. 标定数据加载与可视化
2. 相位传递函数拟合
3. 实测数据校正与导出
"""

import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from signal_calibration import SignalCalibrator, PhaseTransferModel


class CalibrationWidget(QWidget):
    """3ω 信号通路校正界面"""

    calibration_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.calibrator = SignalCalibrator()
        self.cal_data_loaded = False
        self.measurement_data = None
        self.corrected_data = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # === 左侧控制面板 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(420)

        # --- 标定数据加载 ---
        cal_group = QGroupBox("1. 标定数据（纯电阻替代传感器）")
        cal_layout = QVBoxLayout()

        self.load_cal_btn = QPushButton("加载标定数据文件")
        self.load_cal_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 8px; font-weight: bold; }")
        self.load_cal_btn.clicked.connect(self.load_calibration_data)
        cal_layout.addWidget(self.load_cal_btn)

        self.cal_info_label = QLabel("未加载")
        self.cal_info_label.setWordWrap(True)
        cal_layout.addWidget(self.cal_info_label)

        # 或者加载已保存的标定参数
        load_param_btn = QPushButton("加载已保存的标定参数(.json)")
        load_param_btn.clicked.connect(self.load_saved_calibration)
        cal_layout.addWidget(load_param_btn)

        cal_group.setLayout(cal_layout)
        left_layout.addWidget(cal_group)

        # --- 模型选择与拟合 ---
        fit_group = QGroupBox("2. 相位模型拟合")
        fit_layout = QVBoxLayout()

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型类型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "双极点 (推荐)", "单极点", "三极点",
            "等效时延", "等效时延+偏置",
            "常数相位 (窄带平坦)",
            "多项式 1 阶", "多项式 2 阶", "多项式 3 阶",
        ])
        model_row.addWidget(self.model_combo)
        fit_layout.addLayout(model_row)

        unwrap_row = QHBoxLayout()
        self.unwrap_phase_check = QCheckBox("拟合前相位展开 (unwrap)")
        self.unwrap_phase_check.setChecked(True)
        self.unwrap_phase_check.setToolTip(
            "将标定相位转弧度后 np.unwrap，减轻锁相 ±180° 跳变对拟合的影响"
        )
        unwrap_row.addWidget(self.unwrap_phase_check)
        unwrap_row.addStretch()
        fit_layout.addLayout(unwrap_row)

        self.fit_btn = QPushButton("拟合相位模型")
        self.fit_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "padding: 8px; font-weight: bold; }")
        self.fit_btn.clicked.connect(self.fit_phase_model)
        self.fit_btn.setEnabled(False)
        fit_layout.addWidget(self.fit_btn)

        self.fit_result_text = QTextEdit()
        self.fit_result_text.setMaximumHeight(160)
        self.fit_result_text.setReadOnly(True)
        self.fit_result_text.setStyleSheet("font-family: Consolas, monospace;")
        fit_layout.addWidget(self.fit_result_text)

        save_param_btn = QPushButton("保存标定参数(.json)")
        save_param_btn.clicked.connect(self.save_calibration_params)
        fit_layout.addWidget(save_param_btn)

        fit_group.setLayout(fit_layout)
        left_layout.addWidget(fit_group)

        # --- 数据校正 ---
        correct_group = QGroupBox("3. 校正实测数据")
        correct_layout = QVBoxLayout()

        self.load_meas_btn = QPushButton("加载实测 V₃ω 数据")
        self.load_meas_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "padding: 8px; font-weight: bold; }")
        self.load_meas_btn.clicked.connect(self.load_measurement_data)
        correct_layout.addWidget(self.load_meas_btn)

        self.meas_info_label = QLabel("未加载")
        self.meas_info_label.setWordWrap(True)
        correct_layout.addWidget(self.meas_info_label)

        # 校正选项
        opt_layout = QHBoxLayout()
        self.baseline_check = QCheckBox("基线扣除")
        self.baseline_check.setChecked(True)
        self.phase_check = QCheckBox("相位校正")
        self.phase_check.setChecked(True)
        opt_layout.addWidget(self.baseline_check)
        opt_layout.addWidget(self.phase_check)
        correct_layout.addLayout(opt_layout)

        self.correct_btn = QPushButton("执行校正")
        self.correct_btn.setStyleSheet(
            "QPushButton { background-color: #E91E63; color: white; "
            "padding: 8px; font-weight: bold; }")
        self.correct_btn.clicked.connect(self.run_correction)
        self.correct_btn.setEnabled(False)
        correct_layout.addWidget(self.correct_btn)

        self.export_btn = QPushButton("导出校正后数据")
        self.export_btn.clicked.connect(self.export_corrected_data)
        self.export_btn.setEnabled(False)
        correct_layout.addWidget(self.export_btn)

        correct_group.setLayout(correct_layout)
        left_layout.addWidget(correct_group)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # === 右侧图形区域 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 图表切换
        plot_ctrl = QHBoxLayout()
        plot_ctrl.addWidget(QLabel("显示:"))
        self.plot_combo = QComboBox()
        self.plot_combo.addItems([
            "1ω 相位标定 & 拟合",
            "3ω 基线",
            "校正前后对比 (幅值)",
            "校正前后对比 (相位)",
            "校正前后对比 (实部+虚部)",
        ])
        self.plot_combo.currentIndexChanged.connect(self.update_plot)
        plot_ctrl.addWidget(self.plot_combo)
        plot_ctrl.addStretch()
        right_layout.addLayout(plot_ctrl)

        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        main_layout.addWidget(right_panel)

    # ===== 标定数据加载 =====
    def load_calibration_data(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载标定数据", "",
            "Data Files (*.xlsx *.csv);;Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if not filename:
            return

        try:
            if filename.endswith('.xlsx'):
                df = pd.read_excel(filename)
            else:
                df = pd.read_csv(filename)

            freqs, phase_1w, x_3w, y_3w = self._parse_cal_dataframe(df)

            self.calibrator.load_calibration_data(
                freqs, phase_1w, x_3w, y_3w
            )
            self.cal_data_loaded = True
            self.fit_btn.setEnabled(True)

            info = (f"已加载: {os.path.basename(filename)}\n"
                    f"频率点数: {len(freqs)}\n"
                    f"频率范围: {freqs.min():.1f} ~ {freqs.max():.1f} Hz\n"
                    f"3ω基线数据: {'有' if x_3w is not None else '无'}")
            self.cal_info_label.setText(info)
            self.cal_info_label.setStyleSheet("color: green;")

            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"标定数据加载失败:\n{str(e)}")

    def _parse_cal_dataframe(self, df):
        """解析标定数据 DataFrame"""
        cols = df.columns.str.lower().str.strip()
        df.columns = cols

        freq_col = None
        phase_col = None
        x_col = None
        y_col = None

        for c in cols:
            if any(k in c for k in ['freq', '频率', 'hz']):
                freq_col = c
            if any(k in c for k in ['1omega', '1w', '相位', 'phase_1']):
                phase_col = c
            if any(k in c for k in ['3omega_x', '3w_x', 'x_3', 'x_3omega']):
                x_col = c
            if any(k in c for k in ['3omega_y', '3w_y', 'y_3', 'y_3omega']):
                y_col = c

        if freq_col is None:
            freq_col = cols[0]
        if phase_col is None:
            phase_col = cols[1]

        freqs = df[freq_col].values.astype(float)
        phase_1w = df[phase_col].values.astype(float)

        x_3w = df[x_col].values.astype(float) if x_col else None
        y_3w = df[y_col].values.astype(float) if y_col else None

        return freqs, phase_1w, x_3w, y_3w

    def load_saved_calibration(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载标定参数", "", "JSON Files (*.json)"
        )
        if not filename:
            return
        try:
            self.calibrator.load_calibration(filename)
            self.cal_data_loaded = True
            self.fit_btn.setEnabled(True)

            info = (f"已加载标定参数: {os.path.basename(filename)}\n"
                    f"{self.calibrator.get_summary()}")
            self.cal_info_label.setText(info)
            self.cal_info_label.setStyleSheet("color: green;")

            self._show_fit_result_from_loaded()
            self.correct_btn.setEnabled(self.measurement_data is not None)
            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _show_fit_result_from_loaded(self):
        self.fit_result_text.setPlainText(
            f"已加载标定参数\n{self.calibrator.get_summary()}"
        )

    # ===== 相位模型拟合 =====
    def fit_phase_model(self):
        model_map = {
            "双极点 (推荐)": "two_pole",
            "单极点": "single_pole",
            "三极点": "three_pole",
            "等效时延": "linear_delay",
            "等效时延+偏置": "linear_delay_bias",
            "常数相位 (窄带平坦)": "constant_phase",
            "多项式 1 阶": "poly_1",
            "多项式 2 阶": "poly_2",
            "多项式 3 阶": "poly_3",
        }
        model_type = model_map[self.model_combo.currentText()]

        try:
            result = self.calibrator.fit_phase_model(
                model_type,
                unwrap_phase=self.unwrap_phase_check.isChecked(),
            )

            uw = "是" if result.get('unwrap_phase') else "否"
            lines = [
                f"模型: {self.model_combo.currentText()}",
                f"拟合前 unwrap: {uw}",
                f"RMS 残差: {result['rms_residual_deg']:.4f}°",
                "",
            ]
            lines.append("拟合参数:")
            for name, val, std in zip(result['param_names'],
                                      result['param_values'],
                                      result['param_stdev']):
                if name.startswith('c') and name[1:].isdigit():
                    lines.append(f"  {name} = {val:.6e} ± {std:.6e} (rad/Hz^n)")
                elif name == 'tau':
                    lines.append(f"  {name} = {val*1e9:.3f} ± "
                                 f"{std*1e9:.3f} ns")
                elif name == 'phi0':
                    lines.append(f"  {name} = {np.rad2deg(val):.4f} ± "
                                 f"{np.rad2deg(std):.4f} °")
                else:
                    lines.append(f"  {name} = {val/1e6:.3f} ± "
                                 f"{std/1e6:.3f} MHz")

            lines.append("")
            if result.get('unwrap_phase'):
                lines.append(
                    "说明: 已 unwrap 时，残差在展开相位上计算；「实测」列仍为锁相原始度值。"
                )
                lines.append("")
            lines.append("各频率残差 (°):")
            lines.append(f"{'频率(Hz)':>10} {'实测':>8} {'拟合':>8} {'残差':>8}")
            for i, f in enumerate(self.calibrator.cal_freqs):
                lines.append(
                    f"{f:>10.0f} "
                    f"{self.calibrator.cal_phase_1omega[i]:>8.3f} "
                    f"{result['phi_fitted_deg'][i]:>8.3f} "
                    f"{result['residual_deg'][i]:>8.4f}"
                )

            self.fit_result_text.setPlainText("\n".join(lines))

            if self.measurement_data is not None:
                self.correct_btn.setEnabled(True)

            self.update_plot()
            self.calibration_ready.emit(self.calibrator)

        except Exception as e:
            QMessageBox.critical(self, "拟合失败", str(e))

    def save_calibration_params(self):
        if not self.calibrator.is_calibrated:
            QMessageBox.warning(self, "提示", "请先完成拟合")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "保存标定参数", "calibration_params.json",
            "JSON Files (*.json)"
        )
        if not filename:
            return
        try:
            self.calibrator.save_calibration(filename)
            QMessageBox.information(self, "成功",
                                    f"标定参数已保存到:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    # ===== 加载实测数据 =====
    def load_measurement_data(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载实测 V₃ω 数据", "",
            "Data Files (*.xlsx *.csv);;Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if not filename:
            return

        try:
            if filename.endswith('.xlsx'):
                df = pd.read_excel(filename)
            else:
                df = pd.read_csv(filename)

            freqs, V3_complex = self._parse_measurement_dataframe(df)

            self.measurement_data = {
                'freqs': freqs,
                'V3_measured': V3_complex,
                'filename': os.path.basename(filename),
                'df': df,
            }

            self.meas_info_label.setText(
                f"已加载: {os.path.basename(filename)}\n"
                f"数据点: {len(freqs)}\n"
                f"频率: {freqs.min():.1f} ~ {freqs.max():.1f} Hz"
            )
            self.meas_info_label.setStyleSheet("color: #FF9800;")

            if self.calibrator.is_calibrated:
                self.correct_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"数据加载失败:\n{str(e)}")

    def _parse_measurement_dataframe(self, df):
        """解析实测 V₃ω 数据"""
        cols_lower = df.columns.str.lower().str.strip()

        freq_col = None
        real_col = None
        imag_col = None
        amp_col = None
        phase_col = None

        for i, c in enumerate(cols_lower):
            if any(k in c for k in ['freq', '频率']):
                freq_col = df.columns[i]
            if any(k in c for k in ['v3_real', 'real', '实部', 'x_3']):
                real_col = df.columns[i]
            if any(k in c for k in ['v3_imag', 'imag', '虚部', 'y_3']):
                imag_col = df.columns[i]
            if any(k in c for k in ['v3_amp', 'amplitude', '幅值', 'amp']):
                amp_col = df.columns[i]
            if any(k in c for k in ['v3_phase', 'phase', '相位']):
                phase_col = df.columns[i]

        if freq_col is None:
            freq_col = df.columns[0]

        freqs = df[freq_col].values.astype(float)

        if real_col and imag_col:
            V3 = (df[real_col].values.astype(float)
                  + 1j * df[imag_col].values.astype(float))
        elif amp_col and phase_col:
            amp = df[amp_col].values.astype(float)
            phase_rad = np.deg2rad(df[phase_col].values.astype(float))
            V3 = amp * np.exp(1j * phase_rad)
        else:
            V3 = (df.iloc[:, 1].values.astype(float)
                  + 1j * df.iloc[:, 2].values.astype(float))

        return freqs, V3

    # ===== 执行校正 =====
    def run_correction(self):
        if self.measurement_data is None:
            QMessageBox.warning(self, "提示", "请先加载实测数据")
            return
        if not self.calibrator.is_calibrated:
            QMessageBox.warning(self, "提示", "请先完成标定拟合")
            return

        try:
            freqs = self.measurement_data['freqs']
            V3_meas = self.measurement_data['V3_measured']

            V3_corr = self.calibrator.correct_v3omega(
                freqs, V3_meas,
                apply_baseline=self.baseline_check.isChecked(),
                apply_phase=self.phase_check.isChecked()
            )

            self.corrected_data = {
                'freqs': freqs,
                'V3_corrected': V3_corr,
                'V3_measured': V3_meas,
            }

            self.export_btn.setEnabled(True)

            amp_before = np.abs(V3_meas)
            amp_after = np.abs(V3_corr)
            phase_before = np.angle(V3_meas, deg=True)
            phase_after = np.angle(V3_corr, deg=True)

            lines = ["校正完成！", ""]
            lines.append(f"{'频率(Hz)':>10} {'幅值变化%':>10} {'相位变化(°)':>12}")
            for i, f in enumerate(freqs):
                amp_change = (amp_after[i] / amp_before[i] - 1) * 100
                phase_change = phase_after[i] - phase_before[i]
                lines.append(f"{f:>10.1f} {amp_change:>10.3f} "
                             f"{phase_change:>12.3f}")

            QMessageBox.information(self, "校正完成",
                                    f"已校正 {len(freqs)} 个数据点\n"
                                    f"可导出或在图表中查看对比")

            self.plot_combo.setCurrentIndex(2)
            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "校正失败", str(e))

    # ===== 导出校正数据 =====
    def export_corrected_data(self):
        if self.corrected_data is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "导出校正后数据", "corrected_v3omega.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if not filename:
            return

        try:
            freqs = self.corrected_data['freqs']
            V3 = self.corrected_data['V3_corrected']

            df = pd.DataFrame({
                '频率(Hz)': freqs,
                'v3_real': V3.real,
                'v3_imag': V3.imag,
                'v3_amp': np.abs(V3),
                'v3_phase(deg)': np.angle(V3, deg=True),
            })

            if filename.endswith('.xlsx'):
                df.to_excel(filename, index=False)
            else:
                df.to_csv(filename, index=False)

            QMessageBox.information(self, "成功", f"数据已导出到:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    # ===== 绘图 =====
    def update_plot(self):
        self.figure.clear()
        idx = self.plot_combo.currentIndex()

        if idx == 0:
            self._plot_phase_calibration()
        elif idx == 1:
            self._plot_3omega_baseline()
        elif idx == 2:
            self._plot_correction_amplitude()
        elif idx == 3:
            self._plot_correction_phase()
        elif idx == 4:
            self._plot_correction_re_im()

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_phase_calibration(self):
        """绘制 1ω 相位标定数据和拟合曲线"""
        if not self.cal_data_loaded:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "请先加载标定数据",
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=ax.transAxes)
            return

        ax = self.figure.add_subplot(111)
        freqs = self.calibrator.cal_freqs
        phase = self.calibrator.cal_phase_1omega

        ax.semilogx(freqs, phase, 'ro', markersize=8, label='Calibration data')

        if self.calibrator.is_calibrated:
            f_dense = np.logspace(
                np.log10(freqs.min() * 0.5),
                np.log10(freqs.max() * 2), 500
            )
            ptype = self.calibrator.phase_model_type
            if ptype.startswith('poly_'):
                c = self.calibrator.phase_params
                phi_fit = np.rad2deg(np.polyval(c, f_dense))
                phi_3_at_fbase = np.rad2deg(np.polyval(c, 3 * freqs))
            else:
                models = {
                    'single_pole': PhaseTransferModel.single_pole,
                    'two_pole': PhaseTransferModel.two_pole,
                    'three_pole': PhaseTransferModel.three_pole,
                    'linear_delay': PhaseTransferModel.linear_delay,
                    'linear_delay_bias': PhaseTransferModel.linear_delay_bias,
                    'constant_phase': PhaseTransferModel.constant_phase,
                }
                func = models[ptype]
                phi_fit = np.rad2deg(
                    func(f_dense, *self.calibrator.phase_params))
                phi_3_at_fbase = np.rad2deg(
                    func(3 * freqs, *self.calibrator.phase_params))
            ax.semilogx(f_dense, phi_fit, 'b-', linewidth=2,
                        label=r'Fitted $\phi(f)$')

            ax.semilogx(freqs, phi_3_at_fbase, 'g^', markersize=8,
                        label=r'$\phi(3f)$ at $3f$ (x = base $f$)')

        ax.set_xlabel('频率 (Hz)', fontsize=12)
        ax.set_ylabel('通路相移 (°)', fontsize=12)
        ax.set_title('1ω 相位标定 & 模型拟合', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, which='both', ls='--', alpha=0.3)

    def _plot_3omega_baseline(self):
        """绘制 3ω 基线"""
        if (not self.cal_data_loaded
                or self.calibrator.cal_baseline_3omega is None):
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "无 3ω 基线数据",
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=ax.transAxes)
            return

        freqs = self.calibrator.cal_freqs
        baseline = self.calibrator.cal_baseline_3omega

        ax1 = self.figure.add_subplot(211)
        ax1.semilogx(freqs, np.abs(baseline) * 1e6, 'ro-', markersize=6)
        ax1.set_ylabel('|V₃ω baseline| (μV)', fontsize=12)
        ax1.set_title('3ω 基线信号（应尽量小）', fontsize=14)
        ax1.grid(True, which='both', ls='--', alpha=0.3)

        ax2 = self.figure.add_subplot(212, sharex=ax1)
        ax2.semilogx(freqs, baseline.real * 1e6, 'b.-',
                      label='X (real)', markersize=6)
        ax2.semilogx(freqs, baseline.imag * 1e6, 'r.-',
                      label='Y (imag)', markersize=6)
        ax2.set_xlabel('频率 (Hz)', fontsize=12)
        ax2.set_ylabel('V₃ω baseline (μV)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, which='both', ls='--', alpha=0.3)

    def _plot_correction_amplitude(self):
        """绘制校正前后幅值对比"""
        if self.corrected_data is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "请先执行校正",
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=ax.transAxes)
            return

        freqs = self.corrected_data['freqs']
        V_before = self.corrected_data['V3_measured']
        V_after = self.corrected_data['V3_corrected']

        ax1 = self.figure.add_subplot(211)
        ax1.loglog(freqs, np.abs(V_before) * 1e3, 'ro-',
                   label='Before correction', markersize=6)
        ax1.loglog(freqs, np.abs(V_after) * 1e3, 'b^-',
                   label='After correction', markersize=6)
        ax1.set_ylabel('|V₃ω| (mV)', fontsize=12)
        ax1.set_title('幅值对比', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, which='both', ls='--', alpha=0.3)

        ax2 = self.figure.add_subplot(212, sharex=ax1)
        amp_change = (np.abs(V_after) / np.abs(V_before) - 1) * 100
        ax2.semilogx(freqs, amp_change, 'ko-', markersize=6)
        ax2.set_xlabel('频率 (Hz)', fontsize=12)
        ax2.set_ylabel('幅值变化 (%)', fontsize=12)
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.grid(True, which='both', ls='--', alpha=0.3)

    def _plot_correction_phase(self):
        """绘制校正前后相位对比"""
        if self.corrected_data is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "请先执行校正",
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=ax.transAxes)
            return

        freqs = self.corrected_data['freqs']
        V_before = self.corrected_data['V3_measured']
        V_after = self.corrected_data['V3_corrected']

        ax1 = self.figure.add_subplot(211)
        ax1.semilogx(freqs, np.angle(V_before, deg=True), 'ro-',
                      label='Before correction', markersize=6)
        ax1.semilogx(freqs, np.angle(V_after, deg=True), 'b^-',
                      label='After correction', markersize=6)
        ax1.set_ylabel('Phase (°)', fontsize=12)
        ax1.set_title('相位对比', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, which='both', ls='--', alpha=0.3)

        ax2 = self.figure.add_subplot(212, sharex=ax1)
        phase_change = (np.angle(V_after, deg=True)
                        - np.angle(V_before, deg=True))
        ax2.semilogx(freqs, phase_change, 'ko-', markersize=6)
        ax2.set_xlabel('频率 (Hz)', fontsize=12)
        ax2.set_ylabel('相位校正量 (°)', fontsize=12)
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.grid(True, which='both', ls='--', alpha=0.3)

    def _plot_correction_re_im(self):
        """绘制校正前后实部和虚部"""
        if self.corrected_data is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "请先执行校正",
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=ax.transAxes)
            return

        freqs = self.corrected_data['freqs']
        V_before = self.corrected_data['V3_measured']
        V_after = self.corrected_data['V3_corrected']

        ax1 = self.figure.add_subplot(211)
        ax1.semilogx(freqs, V_before.real * 1e3, 'ro-',
                      label='Before (real)', markersize=5)
        ax1.semilogx(freqs, V_after.real * 1e3, 'b^-',
                      label='After (real)', markersize=5)
        ax1.set_ylabel('Real(V₃ω) (mV)', fontsize=12)
        ax1.set_title('实部 & 虚部对比', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, which='both', ls='--', alpha=0.3)

        ax2 = self.figure.add_subplot(212, sharex=ax1)
        ax2.semilogx(freqs, V_before.imag * 1e3, 'ro-',
                      label='Before (imag)', markersize=5)
        ax2.semilogx(freqs, V_after.imag * 1e3, 'b^-',
                      label='After (imag)', markersize=5)
        ax2.set_xlabel('频率 (Hz)', fontsize=12)
        ax2.set_ylabel('Imag(V₃ω) (mV)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, which='both', ls='--', alpha=0.3)

    def get_calibrator(self):
        """返回当前校正器实例（供其他模块使用）"""
        if self.calibrator.is_calibrated:
            return self.calibrator
        return None


# ===== 独立运行入口 =====
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = QMainWindow()
    window.setWindowTitle("3ω 信号通路校正工具")
    window.resize(1300, 800)

    widget = CalibrationWidget()
    window.setCentralWidget(widget)
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
