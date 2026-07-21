"""
纤维丝3ω方法可视化GUI界面
基于Lu等人的理论模型
完全兼容 ThermalModelFitter 接口
"""

import sys
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QGroupBox, QTabWidget, QScrollArea, QDoubleSpinBox, QCheckBox,
    QFileDialog, QMessageBox, QSplitter, QTextEdit, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QDoubleValidator, QIcon
import json
from pathlib import Path

from fiber_model import (
    calculate_fiber_V3omega,
    FiberThermalFitter,
    create_fiber_material_library,
    create_typical_fiber_params,
    check_validity_conditions
)


class FiberMaterialsLibrary:
    """纤维材料库管理"""

    def __init__(self):
        self.materials = create_fiber_material_library()
        # 添加自定义材料
        self.materials['Custom'] = {
            'kappa': 100.0,
            'rho': 2000.0,
            'C_p': 1000.0,
            'description': '自定义材料'
        }

    def get_material_names(self):
        return list(self.materials.keys())

    def get_material(self, name):
        return self.materials.get(name, self.materials['Custom']).copy()


class FiberConfigLibrary:
    """纤维配置库管理"""

    def __init__(self):
        self.configs = create_typical_fiber_params()
        # 添加自定义配置
        self.configs['Custom'] = {
            'L': 5e-3,
            'diameter': 10e-6,
            'S': np.pi * (5e-6) ** 2,
            'R': 10.0,
            'R_prime': 0.01,
            'I_rms': 10e-3,
            'description': '自定义纤维'
        }

    def get_config_names(self):
        return list(self.configs.keys())

    def get_config(self, name):
        return self.configs.get(name, self.configs['Custom']).copy()


class MaterialParamsWidget(QWidget):
    """材料参数输入组件"""
    paramChanged = pyqtSignal()

    def __init__(self, materials_lib, parent=None):
        super().__init__(parent)
        self.materials_lib = materials_lib
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout()

        # 材料选择
        layout.addWidget(QLabel("<b>材料选择</b>"), 0, 0, 1, 2)

        layout.addWidget(QLabel("预设材料:"), 1, 0)
        self.material_combo = QComboBox()
        self.material_combo.addItems(self.materials_lib.get_material_names())
        self.material_combo.currentTextChanged.connect(self.on_material_changed)
        layout.addWidget(self.material_combo, 1, 1)

        # 热导率 κ
        layout.addWidget(QLabel("热导率 κ (W/m·K):"), 2, 0)
        self.kappa_input = QLineEdit("71.6")
        self.kappa_input.setValidator(QDoubleValidator(0.01, 10000, 6))
        self.kappa_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.kappa_input, 2, 1)

        # 密度 ρ
        layout.addWidget(QLabel("密度 ρ (kg/m³):"), 3, 0)
        self.rho_input = QLineEdit("21450")
        self.rho_input.setValidator(QDoubleValidator(100, 25000, 2))
        self.rho_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.rho_input, 3, 1)

        # 比热容 Cp
        layout.addWidget(QLabel("比热容 Cₚ (J/kg·K):"), 4, 0)
        self.cp_input = QLineEdit("133")
        self.cp_input.setValidator(QDoubleValidator(50, 5000, 2))
        self.cp_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.cp_input, 4, 1)

        # 材料描述
        layout.addWidget(QLabel("描述:"), 5, 0)
        self.desc_label = QLabel("铂丝 (标准参考材料)")
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label, 5, 1)

        self.setLayout(layout)

    def on_material_changed(self, material_name):
        """材料切换"""
        material = self.materials_lib.get_material(material_name)
        self.kappa_input.setText(str(material['kappa']))
        self.rho_input.setText(str(material['rho']))
        self.cp_input.setText(str(material['C_p']))
        self.desc_label.setText(material.get('description', ''))
        self.paramChanged.emit()

    def get_material_props(self):
        """获取材料属性"""
        try:
            return {
                'kappa': float(self.kappa_input.text()),
                'rho': float(self.rho_input.text()),
                'C_p': float(self.cp_input.text())
            }
        except ValueError:
            return None


class FiberParamsWidget(QWidget):
    """纤维参数输入组件"""
    paramChanged = pyqtSignal()

    def __init__(self, fiber_lib, parent=None):
        super().__init__(parent)
        self.fiber_lib = fiber_lib
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout()

        # 配置选择
        layout.addWidget(QLabel("<b>纤维配置</b>"), 0, 0, 1, 2)

        layout.addWidget(QLabel("预设配置:"), 1, 0)
        self.config_combo = QComboBox()
        self.config_combo.addItems(self.fiber_lib.get_config_names())
        self.config_combo.currentTextChanged.connect(self.on_config_changed)
        layout.addWidget(self.config_combo, 1, 1)

        # 长度 L
        layout.addWidget(QLabel("长度 L (mm):"), 2, 0)
        self.L_input = QLineEdit("8.0")
        self.L_input.setValidator(QDoubleValidator(0.1, 100, 3))
        self.L_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.L_input, 2, 1)

        # 直径 D
        layout.addWidget(QLabel("直径 D (μm):"), 3, 0)
        self.diameter_input = QLineEdit("20")
        self.diameter_input.setValidator(QDoubleValidator(0.1, 1000, 3))
        self.diameter_input.textChanged.connect(self.on_diameter_changed)
        layout.addWidget(self.diameter_input, 3, 1)

        # 横截面积 S (自动计算)
        layout.addWidget(QLabel("横截面积 S (μm²):"), 4, 0)
        self.S_label = QLabel("314.16")
        self.S_label.setStyleSheet("background-color: #f0f0f0; padding: 3px;")
        layout.addWidget(self.S_label, 4, 1)

        # 电阻 R
        layout.addWidget(QLabel("电阻 R (Ω):"), 5, 0)
        self.R_input = QLineEdit("5.0")
        self.R_input.setValidator(QDoubleValidator(0.01, 10000, 4))
        self.R_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.R_input, 5, 1)

        # 温度系数 R'
        layout.addWidget(QLabel("温度系数 R' (Ω/K):"), 6, 0)
        self.R_prime_input = QLineEdit("0.01")
        self.R_prime_input.setValidator(QDoubleValidator(-1.0, 1.0, 6))
        self.R_prime_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.R_prime_input, 6, 1)

        # RMS电流
        layout.addWidget(QLabel("RMS电流 I (mA):"), 7, 0)
        self.I_input = QLineEdit("10.0")
        self.I_input.setValidator(QDoubleValidator(0.1, 1000, 3))
        self.I_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.I_input, 7, 1)

        # 描述
        layout.addWidget(QLabel("描述:"), 8, 0)
        self.desc_label = QLabel("20μm铂丝")
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label, 8, 1)

        self.setLayout(layout)

    def on_config_changed(self, config_name):
        """配置切换"""
        config = self.fiber_lib.get_config(config_name)
        self.L_input.setText(f"{config['L'] * 1e3:.3f}")
        self.diameter_input.setText(f"{config['diameter'] * 1e6:.3f}")
        self.R_input.setText(str(config['R']))
        self.R_prime_input.setText(str(config['R_prime']))
        self.I_input.setText(f"{config['I_rms'] * 1e3:.3f}")
        self.desc_label.setText(config.get('description', ''))
        self.update_cross_section()
        self.paramChanged.emit()

    def on_diameter_changed(self):
        """直径改变时自动更新横截面积"""
        self.update_cross_section()
        self.paramChanged.emit()

    def update_cross_section(self):
        """更新横截面积显示"""
        try:
            diameter_m = float(self.diameter_input.text()) * 1e-6
            S_um2 = np.pi * (diameter_m / 2) ** 2 * 1e12
            self.S_label.setText(f"{S_um2:.2f}")
        except:
            self.S_label.setText("---")

    def get_fiber_params(self):
        """获取纤维参数"""
        try:
            diameter_m = float(self.diameter_input.text()) * 1e-6
            return {
                'L': float(self.L_input.text()) * 1e-3,
                'diameter': diameter_m,
                'S': np.pi * (diameter_m / 2) ** 2,
                'R': float(self.R_input.text()),
                'R_prime': float(self.R_prime_input.text()),
                'I_rms': float(self.I_input.text()) * 1e-3
            }
        except ValueError:
            return None


class FrequencyRangeWidget(QWidget):
    """频率范围设置组件"""
    paramChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout()

        layout.addWidget(QLabel("<b>频率范围设置</b>"), 0, 0, 1, 3)

        # 起始频率
        layout.addWidget(QLabel("起始频率 (Hz):"), 1, 0)
        self.freq_start = QDoubleSpinBox()
        self.freq_start.setRange(0.01, 1e6)
        self.freq_start.setValue(1.0)
        self.freq_start.setDecimals(3)
        self.freq_start.valueChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.freq_start, 1, 1)

        # 终止频率
        layout.addWidget(QLabel("终止频率 (Hz):"), 2, 0)
        self.freq_end = QDoubleSpinBox()
        self.freq_end.setRange(0.01, 1e6)
        self.freq_end.setValue(1000.0)
        self.freq_end.setDecimals(3)
        self.freq_end.valueChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.freq_end, 2, 1)

        # 点数
        layout.addWidget(QLabel("数据点数:"), 3, 0)
        self.num_points = QSpinBox()
        self.num_points.setRange(5, 200)
        self.num_points.setValue(30)
        self.num_points.valueChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.num_points, 3, 1)

        # 对数/线性
        layout.addWidget(QLabel("频率间隔:"), 4, 0)
        self.spacing_combo = QComboBox()
        self.spacing_combo.addItems(['对数间隔', '线性间隔'])
        self.spacing_combo.currentTextChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.spacing_combo, 4, 1)

        self.setLayout(layout)

    def get_frequencies(self):
        """生成频率数组"""
        try:
            start = self.freq_start.value()
            end = self.freq_end.value()
            num = self.num_points.value()

            if self.spacing_combo.currentText() == '对数间隔':
                return np.logspace(np.log10(start), np.log10(end), num)
            else:
                return np.linspace(start, end, num)
        except:
            return np.logspace(0, 3, 30)


class ValidityCheckWidget(QWidget):
    """有效性检查显示组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        title = QLabel("<b>模型有效性检查</b>")
        layout.addWidget(title)

        self.check_text = QTextEdit()
        self.check_text.setReadOnly(True)
        self.check_text.setMaximumHeight(120)
        self.check_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ccc;
                font-family: Consolas, monospace;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.check_text)

        self.setLayout(layout)

    def update_validity(self, fiber_params, material_props, freqs):
        """更新有效性检查"""
        try:
            validity = check_validity_conditions(fiber_params, material_props, freqs)

            text = "模型有效性检查结果:\n"
            text += "-" * 50 + "\n"
            text += f"条件(10) - 功率均匀性: {validity['condition_10_value']:.3e} "
            text += f"[{'✓ 满足' if validity['condition_10_satisfied'] else '✗ 不满足'}]\n"
            text += f"条件(31) - 辐射损失:   {validity['condition_31_value']:.3e} "
            text += f"[{'✓ 满足' if validity['condition_31_satisfied'] else '✗ 不满足'}]\n"
            text += f"频率范围 (2ωγ_max):   {validity['max_2omega_gamma']:.2f} "
            text += f"[{'✓ 合适' if validity['frequency_range_ok'] else '✗ 过高'}]\n"
            text += "-" * 50 + "\n"
            text += f"热时间常数 γ:         {validity['gamma']:.3e} s\n"
            text += f"等效直径:             {validity['equivalent_diameter'] * 1e6:.2f} μm\n"

            self.check_text.setPlainText(text)

            # 设置颜色
            if (validity['condition_10_satisfied'] and
                    validity['condition_31_satisfied'] and
                    validity['frequency_range_ok']):
                self.check_text.setStyleSheet("""
                    QTextEdit {
                        background-color: #e8f5e9;
                        border: 1px solid #4caf50;
                        font-family: Consolas, monospace;
                        font-size: 10pt;
                    }
                """)
            else:
                self.check_text.setStyleSheet("""
                    QTextEdit {
                        background-color: #fff3e0;
                        border: 1px solid #ff9800;
                        font-family: Consolas, monospace;
                        font-size: 10pt;
                    }
                """)
        except Exception as e:
            self.check_text.setPlainText(f"检查失败: {str(e)}")


class PlotCanvas(FigureCanvas):
    """绘图画布"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        # 创建子图
        self.ax_amp = self.fig.add_subplot(221)
        self.ax_phase = self.fig.add_subplot(222)
        self.ax_real = self.fig.add_subplot(223)
        self.ax_imag = self.fig.add_subplot(224)

        self.fig.tight_layout(pad=3.0)

    def plot_V3omega(self, freqs, V3_omega, label='Theory', clear=True):
        """绘制3ω电压"""
        # 验证数据长度一致性
        if len(freqs) != len(V3_omega):
            print(f"警告: 频率数组长度({len(freqs)}) != 电压数组长度({len(V3_omega)})")
            # 截取到较短的长度
            min_len = min(len(freqs), len(V3_omega))
            freqs = freqs[:min_len]
            V3_omega = V3_omega[:min_len]
            print(f"已截取到前 {min_len} 个数据点")

        if clear:
            self.ax_amp.clear()
            self.ax_phase.clear()
            self.ax_real.clear()
            self.ax_imag.clear()

        # 幅值
        self.ax_amp.loglog(freqs, np.abs(V3_omega) * 1e6, '-o', label=label, markersize=4)
        self.ax_amp.set_xlabel('Frequency (Hz)')
        self.ax_amp.set_ylabel('|V_3omega| (uV)')
        self.ax_amp.set_title('Amplitude')
        self.ax_amp.grid(True, which='both', alpha=0.3)
        self.ax_amp.legend()

        # 相位
        phase_deg = np.unwrap(np.angle(V3_omega)) * 180 / np.pi
        self.ax_phase.semilogx(freqs, phase_deg, '-o', label=label, markersize=4)
        self.ax_phase.set_xlabel('Frequency (Hz)')
        self.ax_phase.set_ylabel('Phase (deg)')
        self.ax_phase.set_title('Phase')
        self.ax_phase.grid(True, which='both', alpha=0.3)
        self.ax_phase.legend()

        # 实部
        self.ax_real.loglog(freqs, np.abs(V3_omega.real) * 1e6, '-o', label=label, markersize=4)
        self.ax_real.set_xlabel('Frequency (Hz)')
        self.ax_real.set_ylabel('Re(V_3omega) (uV)')
        self.ax_real.set_title('Real Part')
        self.ax_real.grid(True, which='both', alpha=0.3)
        self.ax_real.legend()

        # 虚部
        self.ax_imag.loglog(freqs, np.abs(V3_omega.imag) * 1e6, '-o', label=label, markersize=4)
        self.ax_imag.set_xlabel('Frequency (Hz)')
        self.ax_imag.set_ylabel('Im(V_3omega) (uV)')
        self.ax_imag.set_title('Imaginary Part')
        self.ax_imag.grid(True, which='both', alpha=0.3)
        self.ax_imag.legend()

        self.draw()

    def clear_plots(self):
        """清空所有图"""
        self.ax_amp.clear()
        self.ax_phase.clear()
        self.ax_real.clear()
        self.ax_imag.clear()
        self.draw()


class FittingControlWidget(QWidget):
    """拟合控制组件"""
    fittingRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout()

        layout.addWidget(QLabel("<b>拟合控制</b>"), 0, 0, 1, 2)

        # 拟合类型
        layout.addWidget(QLabel("拟合类型:"), 1, 0)
        self.fit_type_combo = QComboBox()
        self.fit_type_combo.addItems(['材料参数', '纤维参数', '联合拟合'])
        layout.addWidget(self.fit_type_combo, 1, 1)

        # 拟合模式
        layout.addWidget(QLabel("拟合模式:"), 2, 0)
        self.fit_mode_combo = QComboBox()
        self.fit_mode_combo.addItems(['both (实部+虚部)', 'amp (幅值)', 'phase (相位)',
                                      'real (实部)', 'imag (虚部)'])
        layout.addWidget(self.fit_mode_combo, 2, 1)

        # 最大迭代次数
        layout.addWidget(QLabel("最大迭代:"), 3, 0)
        self.max_iter = QSpinBox()
        self.max_iter.setRange(5, 100)
        self.max_iter.setValue(20)
        layout.addWidget(self.max_iter, 3, 1)

        # 收敛容差
        layout.addWidget(QLabel("收敛容差:"), 4, 0)
        self.tolerance = QLineEdit("1e-5")
        layout.addWidget(self.tolerance, 4, 1)

        # 步长因子
        layout.addWidget(QLabel("步长因子 ζ:"), 5, 0)
        self.zeta = QDoubleSpinBox()
        self.zeta.setRange(0.1, 1.0)
        self.zeta.setValue(0.5)
        self.zeta.setSingleStep(0.1)
        self.zeta.setDecimals(2)
        layout.addWidget(self.zeta, 5, 1)

        # 噪声水平（用于生成模拟数据）
        layout.addWidget(QLabel("模拟噪声 (%):"), 6, 0)
        self.noise_level = QDoubleSpinBox()
        self.noise_level.setRange(0.0, 50.0)
        self.noise_level.setValue(2.0)
        self.noise_level.setSingleStep(0.5)
        self.noise_level.setDecimals(1)
        layout.addWidget(self.noise_level, 6, 1)

        # 拟合参数选择
        layout.addWidget(QLabel("材料参数:"), 7, 0)
        self.material_params_layout = QHBoxLayout()
        self.fit_kappa = QCheckBox("κ")
        self.fit_kappa.setChecked(True)
        self.fit_Cp = QCheckBox("Cₚ")
        self.fit_Cp.setChecked(True)
        self.fit_rho = QCheckBox("ρ")
        self.material_params_layout.addWidget(self.fit_kappa)
        self.material_params_layout.addWidget(self.fit_Cp)
        self.material_params_layout.addWidget(self.fit_rho)
        layout.addLayout(self.material_params_layout, 7, 1)

        # 纤维参数选择
        layout.addWidget(QLabel("纤维参数:"), 8, 0)
        self.fiber_params_layout = QHBoxLayout()
        self.fit_R = QCheckBox("R")
        self.fit_R_prime = QCheckBox("R'")
        self.fit_I = QCheckBox("I")
        self.fiber_params_layout.addWidget(self.fit_R)
        self.fiber_params_layout.addWidget(self.fit_R_prime)
        self.fiber_params_layout.addWidget(self.fit_I)
        layout.addLayout(self.fiber_params_layout, 8, 1)

        # 拟合按钮
        self.fit_button = QPushButton("执行拟合")
        self.fit_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.fit_button.clicked.connect(self.fittingRequested.emit)
        layout.addWidget(self.fit_button, 9, 0, 1, 2)

        self.setLayout(layout)

    def get_fit_params(self):
        """获取拟合参数"""
        material_params = []
        if self.fit_kappa.isChecked():
            material_params.append('kappa')
        if self.fit_Cp.isChecked():
            material_params.append('C_p')
        if self.fit_rho.isChecked():
            material_params.append('rho')

        fiber_params = []
        if self.fit_R.isChecked():
            fiber_params.append('R')
        if self.fit_R_prime.isChecked():
            fiber_params.append('R_prime')
        if self.fit_I.isChecked():
            fiber_params.append('I_rms')

        # 确定拟合类型
        fit_type_text = self.fit_type_combo.currentText()
        if fit_type_text == '材料参数':
            fit_type = 'material'
            params_to_fit = {'material': material_params} if material_params else None
        elif fit_type_text == '纤维参数':
            fit_type = 'sensor'
            params_to_fit = {'fiber': fiber_params} if fiber_params else None
        else:  # 联合拟合
            fit_type = 'combined'
            params_to_fit = {}
            if material_params:
                params_to_fit['material'] = material_params
            if fiber_params:
                params_to_fit['fiber'] = fiber_params

        # 拟合模式
        fit_mode_map = {
            'both (实部+虚部)': 'both',
            'amp (幅值)': 'amp',
            'phase (相位)': 'phase',
            'real (实部)': 'real',
            'imag (虚部)': 'imag'
        }
        fit_mode = fit_mode_map[self.fit_mode_combo.currentText()]

        return {
            'fit_type': fit_type,
            'params_to_fit': params_to_fit,
            'fit_mode': fit_mode,
            'max_iter': self.max_iter.value(),
            'tolerance': float(self.tolerance.text()),
            'zeta': self.zeta.value(),
            'noise_level': self.noise_level.value() / 100.0
        }


class ResultsDisplayWidget(QWidget):
    """结果显示组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        title = QLabel("<b>拟合结果</b>")
        layout.addWidget(title)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f0f8ff;
                border: 1px solid #4682b4;
                font-family: Consolas, monospace;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.results_text)

        # 导出按钮
        btn_layout = QHBoxLayout()
        self.export_json_btn = QPushButton("导出JSON")
        self.export_csv_btn = QPushButton("导出CSV")
        self.export_plot_btn = QPushButton("保存图像")

        btn_layout.addWidget(self.export_json_btn)
        btn_layout.addWidget(self.export_csv_btn)
        btn_layout.addWidget(self.export_plot_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def display_results(self, result_dict, freqs, fiber_params, material_props):
        """显示拟合结果"""
        text = "拟合结果摘要\n"
        text += "=" * 60 + "\n\n"

        # 根据不同的返回类型处理
        if isinstance(result_dict, dict):
            if 'fiber_params' in result_dict and 'material_props' in result_dict:
                # 联合拟合结果
                text += "材料属性:\n"
                for key, val in result_dict['material_props'].items():
                    if key != 'description':
                        text += f"  {key}: {val:.4e}\n"

                text += "\n纤维参数:\n"
                for key, val in result_dict['fiber_params'].items():
                    if key not in ['description', 'diameter']:
                        text += f"  {key}: {val:.4e}\n"

                # 计算导出参数
                _, gamma, alpha = calculate_fiber_V3omega(
                    freqs, result_dict['fiber_params'], result_dict['material_props']
                )
            elif 'kappa' in result_dict:
                # 材料参数拟合结果
                text += "拟合的材料属性:\n"
                for key, val in result_dict.items():
                    if key != 'description':
                        text += f"  {key}: {val:.4e}\n"

                # 计算导出参数
                _, gamma, alpha = calculate_fiber_V3omega(freqs, fiber_params, result_dict)
            else:
                # 纤维参数拟合结果
                text += "拟合的纤维参数:\n"
                for key, val in result_dict.items():
                    if key not in ['description', 'diameter']:
                        text += f"  {key}: {val:.4e}\n"

                # 计算导出参数
                _, gamma, alpha = calculate_fiber_V3omega(freqs, result_dict, material_props)

            text += "\n导出参数:\n"
            text += f"  热扩散率 α: {alpha:.4e} m²/s\n"
            text += f"  热时间常数 γ: {gamma:.4e} s\n"

        self.results_text.setPlainText(text)


class FiberThermal3OmegaGUI(QMainWindow):
    """纤维丝3ω方法主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("纤维丝3ω方法热物性测量系统 v1.0")
        self.setGeometry(100, 100, 1400, 900)

        # 初始化库
        self.materials_lib = FiberMaterialsLibrary()
        self.fiber_lib = FiberConfigLibrary()

        # 数据存储
        self.current_V3_theory = None
        self.current_V3_exp = None
        self.current_freqs = None
        self.fitted_result = None

        self.setup_ui()
        self.calculate_and_plot()

    def setup_ui(self):
        """设置UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # 左侧：参数控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # 标签页
        tabs = QTabWidget()

        # Tab 1: 材料参数
        self.material_widget = MaterialParamsWidget(self.materials_lib)
        self.material_widget.paramChanged.connect(self.on_params_changed)
        tabs.addTab(self.material_widget, "材料参数")

        # Tab 2: 纤维参数
        self.fiber_widget = FiberParamsWidget(self.fiber_lib)
        self.fiber_widget.paramChanged.connect(self.on_params_changed)
        tabs.addTab(self.fiber_widget, "纤维参数")

        # Tab 3: 频率设置
        self.freq_widget = FrequencyRangeWidget()
        self.freq_widget.paramChanged.connect(self.on_params_changed)
        tabs.addTab(self.freq_widget, "频率设置")

        left_layout.addWidget(tabs)

        # 有效性检查
        self.validity_widget = ValidityCheckWidget()
        left_layout.addWidget(self.validity_widget)

        # 拟合控制
        self.fitting_widget = FittingControlWidget()
        self.fitting_widget.fittingRequested.connect(self.perform_fitting)
        left_layout.addWidget(self.fitting_widget)

        # 控制按钮
        button_layout = QHBoxLayout()

        self.calc_button = QPushButton("计算理论曲线")
        self.calc_button.clicked.connect(self.calculate_and_plot)
        button_layout.addWidget(self.calc_button)

        self.load_button = QPushButton("加载实验数据")
        self.load_button.clicked.connect(self.load_experimental_data)
        button_layout.addWidget(self.load_button)

        self.save_config_button = QPushButton("保存配置")
        self.save_config_button.clicked.connect(self.save_configuration)
        button_layout.addWidget(self.save_config_button)

        self.load_config_button = QPushButton("加载配置")
        self.load_config_button.clicked.connect(self.load_configuration)
        button_layout.addWidget(self.load_config_button)

        left_layout.addLayout(button_layout)

        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)

        # 右侧：绘图和结果
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # 绘图区域
        self.canvas = PlotCanvas(width=10, height=8)
        right_layout.addWidget(self.canvas)

        # 结果显示
        self.results_widget = ResultsDisplayWidget()
        self.results_widget.setMaximumHeight(200)
        self.results_widget.export_json_btn.clicked.connect(self.export_json)
        self.results_widget.export_csv_btn.clicked.connect(self.export_csv)
        self.results_widget.export_plot_btn.clicked.connect(self.export_plot)
        right_layout.addWidget(self.results_widget)

        right_panel.setLayout(right_layout)

        # 使用分隔器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

        # 状态栏
        self.statusBar().showMessage("就绪")

    def on_params_changed(self):
        """参数变化时的处理"""
        # 延迟计算，避免频繁更新
        if hasattr(self, '_update_timer'):
            self._update_timer.stop()

        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.update_validity_check)
        self._update_timer.start(500)  # 500ms 延迟

    def update_validity_check(self):
        """更新有效性检查"""
        fiber_params = self.fiber_widget.get_fiber_params()
        material_props = self.material_widget.get_material_props()
        freqs = self.freq_widget.get_frequencies()

        if fiber_params and material_props and len(freqs) > 0:
            self.validity_widget.update_validity(fiber_params, material_props, freqs)

    def calculate_and_plot(self):
        """计算并绘制理论曲线"""
        self.statusBar().showMessage("计算中...")

        try:
            # 获取参数
            fiber_params = self.fiber_widget.get_fiber_params()
            material_props = self.material_widget.get_material_props()
            freqs = self.freq_widget.get_frequencies()

            if not fiber_params or not material_props:
                QMessageBox.warning(self, "参数错误", "请检查输入参数是否有效")
                return

            # 计算3ω电压
            V3_result, gamma, alpha = calculate_fiber_V3omega(freqs, fiber_params, material_props)

            self.current_V3_theory = V3_result
            self.current_freqs = freqs

            # 绘图
            self.canvas.plot_V3omega(freqs, V3_result, 'Theory', clear=True)

            # 更新有效性检查
            self.validity_widget.update_validity(fiber_params, material_props, freqs)

            # 显示计算信息
            info_text = f"计算完成\n"
            info_text += f"热扩散率 α: {alpha:.3e} m²/s\n"
            info_text += f"热时间常数 γ: {gamma:.3e} s\n"
            info_text += f"|V₃ω| 范围: {np.min(np.abs(V3_result)):.3e} - {np.max(np.abs(V3_result)):.3e} V"

            self.results_widget.results_text.setPlainText(info_text)
            self.statusBar().showMessage("计算完成", 3000)

        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"计算失败:\n{str(e)}")
            self.statusBar().showMessage("计算失败")

    def load_experimental_data(self):
        """加载实验数据"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择实验数据文件", "",
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            import pandas as pd

            # 读取文件
            if file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            # 简单的列识别
            freq_col = None
            real_col = None
            imag_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'freq' in col_lower:
                    freq_col = col
                elif 'real' in col_lower and 'v3' in col_lower:
                    real_col = col
                elif 'imag' in col_lower and 'v3' in col_lower:
                    imag_col = col

            if not freq_col or not real_col or not imag_col:
                QMessageBox.warning(self, "数据格式错误",
                                    "无法识别数据列，请确保包含:\n- 频率列 (Frequency)\n- 实部列 (V3_Real)\n- 虚部列 (V3_Imag)")
                return

            freqs = df[freq_col].to_numpy()
            V3_real = df[real_col].to_numpy()
            V3_imag = df[imag_col].to_numpy()

            self.current_V3_exp = V3_real + 1j * V3_imag
            self.current_freqs = freqs

            # 重新计算理论曲线（使用实验数据的频率）
            fiber_params = self.fiber_widget.get_fiber_params()
            material_props = self.material_widget.get_material_props()

            if fiber_params and material_props:
                self.current_V3_theory, _, _ = calculate_fiber_V3omega(
                    freqs, fiber_params, material_props
                )
                self.canvas.plot_V3omega(freqs, self.current_V3_theory, 'Theory', clear=True)

            self.canvas.plot_V3omega(freqs, self.current_V3_exp, 'Experimental Data', clear=False)

            self.statusBar().showMessage(f"成功加载 {len(freqs)} 个数据点", 3000)

        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法加载数据:\n{str(e)}")

    def perform_fitting(self):
        """执行拟合"""
        if self.current_V3_exp is None:
            # 生成模拟数据
            reply = QMessageBox.question(
                self, "生成模拟数据",
                "未加载实验数据，是否生成模拟数据进行测试？",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.generate_simulated_data()
            else:
                return

        self.statusBar().showMessage("拟合中，请稍候...")

        try:
            # 获取拟合参数
            fit_config = self.fitting_widget.get_fit_params()

            if not fit_config['params_to_fit']:
                QMessageBox.warning(self, "参数错误", "请至少选择一个要拟合的参数")
                return

            # 获取当前参数
            fiber_params = self.fiber_widget.get_fiber_params()
            material_props = self.material_widget.get_material_props()

            if not fiber_params or not material_props:
                QMessageBox.warning(self, "参数错误", "请检查输入参数是否有效")
                return

            # 创建拟合器
            fitter = FiberThermalFitter(
                freqs=self.current_freqs,
                V3_exp=self.current_V3_exp,
                fiber_params=fiber_params,
                initial_material_props=material_props,
                params_to_fit=fit_config['params_to_fit'],
                fit_mode=fit_config['fit_mode'],
                fit_type=fit_config['fit_type']
            )

            # 执行拟合（重定向输出）
            import io
            from contextlib import redirect_stdout

            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                result = fitter.fit(
                    max_iter=fit_config['max_iter'],
                    tol=fit_config['tolerance'],
                    zeta=fit_config['zeta']
                )

            # 显示拟合过程输出
            fitting_log = output_buffer.getvalue()
            print(fitting_log)  # 打印到控制台

            # 保存结果
            self.fitted_result = result

            # 显示结果
            self.results_widget.display_results(result, self.current_freqs, fiber_params, material_props)

            # 绘制拟合曲线
            if fit_config['fit_type'] == 'material':
                V3_fitted, _, _ = calculate_fiber_V3omega(self.current_freqs, fiber_params, result)
            elif fit_config['fit_type'] == 'sensor':
                V3_fitted, _, _ = calculate_fiber_V3omega(self.current_freqs, result, material_props)
            else:  # combined
                V3_fitted, _, _ = calculate_fiber_V3omega(
                    self.current_freqs, result['fiber_params'], result['material_props']
                )

            self.canvas.plot_V3omega(self.current_freqs, self.current_V3_exp, 'Experimental Data', clear=True)
            self.canvas.plot_V3omega(self.current_freqs, V3_fitted, 'Fitted Curve', clear=False)

            self.statusBar().showMessage("拟合完成", 3000)

            QMessageBox.information(self, "拟合完成",
                                    f"拟合成功完成！\n迭代次数: {fitter.getIterations() if hasattr(fitter, 'getIterations') else '?'}")

        except Exception as e:
            QMessageBox.critical(self, "拟合失败", f"拟合过程出错:\n{str(e)}")
            self.statusBar().showMessage("拟合失败")
            import traceback
            traceback.print_exc()

    def generate_simulated_data(self):
        """生成模拟数据"""
        try:
            fiber_params = self.fiber_widget.get_fiber_params()
            material_props = self.material_widget.get_material_props()
            freqs = self.freq_widget.get_frequencies()

            # 生成理论数据
            V3_ideal, _, _ = calculate_fiber_V3omega(freqs, fiber_params, material_props)

            # 添加噪声
            noise_level = self.fitting_widget.noise_level.value() / 100.0
            np.random.seed(42)

            noise_real = noise_level * np.random.randn(len(freqs)) * np.mean(np.abs(V3_ideal))
            noise_imag = noise_level * np.random.randn(len(freqs)) * np.mean(np.abs(V3_ideal))
            noise = noise_real + 1j * noise_imag

            self.current_V3_exp = V3_ideal + noise
            self.current_V3_theory = V3_ideal
            self.current_freqs = freqs

            # 绘图
            self.canvas.plot_V3omega(freqs, V3_ideal, 'Theory', clear=True)
            self.canvas.plot_V3omega(freqs, self.current_V3_exp, 'Simulated Data', clear=False)

            self.statusBar().showMessage(f"已生成模拟数据 (噪声 {noise_level * 100:.1f}%)", 3000)

        except Exception as e:
            QMessageBox.critical(self, "生成失败", f"生成模拟数据失败:\n{str(e)}")

    def save_configuration(self):
        """保存当前配置"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存配置", "", "JSON Files (*.json);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            config = {
                'material': self.material_widget.get_material_props(),
                'fiber': self.fiber_widget.get_fiber_params(),
                'frequency': {
                    'start': self.freq_widget.freq_start.value(),
                    'end': self.freq_widget.freq_end.value(),
                    'num_points': self.freq_widget.num_points.value(),
                    'spacing': self.freq_widget.spacing_combo.currentText()
                },
                'fitting': self.fitting_widget.get_fit_params()
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            self.statusBar().showMessage(f"配置已保存到: {file_path}", 3000)

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法保存配置:\n{str(e)}")

    def load_configuration(self):
        """加载配置"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载配置", "", "JSON Files (*.json);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 加载材料参数
            if 'material' in config:
                mat = config['material']
                self.material_widget.kappa_input.setText(str(mat.get('kappa', 71.6)))
                self.material_widget.rho_input.setText(str(mat.get('rho', 21450)))
                self.material_widget.cp_input.setText(str(mat.get('C_p', 133)))

            # 加载纤维参数
            if 'fiber' in config:
                fib = config['fiber']
                self.fiber_widget.L_input.setText(str(fib.get('L', 8e-3) * 1e3))
                if 'diameter' in fib:
                    self.fiber_widget.diameter_input.setText(str(fib['diameter'] * 1e6))
                self.fiber_widget.R_input.setText(str(fib.get('R', 5.0)))
                self.fiber_widget.R_prime_input.setText(str(fib.get('R_prime', 0.01)))
                self.fiber_widget.I_input.setText(str(fib.get('I_rms', 10e-3) * 1e3))

            # 加载频率设置
            if 'frequency' in config:
                freq = config['frequency']
                self.freq_widget.freq_start.setValue(freq.get('start', 1.0))
                self.freq_widget.freq_end.setValue(freq.get('end', 1000.0))
                self.freq_widget.num_points.setValue(freq.get('num_points', 30))

            self.statusBar().showMessage(f"配置已加载: {file_path}", 3000)
            self.calculate_and_plot()

        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法加载配置:\n{str(e)}")

    def export_json(self):
        """导出JSON结果"""
        if self.fitted_result is None:
            QMessageBox.warning(self, "无数据", "请先执行拟合")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出JSON", "", "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            # 准备导出数据
            fiber_params = self.fiber_widget.get_fiber_params()
            material_props = self.material_widget.get_material_props()

            if isinstance(self.fitted_result, dict) and 'fiber_params' in self.fitted_result:
                # 联合拟合
                V3_fitted, gamma, alpha = calculate_fiber_V3omega(
                    self.current_freqs,
                    self.fitted_result['fiber_params'],
                    self.fitted_result['material_props']
                )
            elif isinstance(self.fitted_result, dict) and 'kappa' in self.fitted_result:
                # 材料拟合
                V3_fitted, gamma, alpha = calculate_fiber_V3omega(
                    self.current_freqs, fiber_params, self.fitted_result
                )
            else:
                # 纤维拟合
                V3_fitted, gamma, alpha = calculate_fiber_V3omega(
                    self.current_freqs, self.fitted_result, material_props
                )

            output = {
                'fitted_result': self._convert_to_serializable(self.fitted_result),
                'initial_fiber_params': self._convert_to_serializable(fiber_params),
                'initial_material_props': self._convert_to_serializable(material_props),
                'alpha': float(alpha),
                'gamma': float(gamma)
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            self.statusBar().showMessage(f"结果已导出: {file_path}", 3000)

        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出JSON失败:\n{str(e)}")

    def export_csv(self):
        """导出CSV数据"""
        if self.current_freqs is None or self.current_V3_exp is None:
            QMessageBox.warning(self, "无数据", "请先加载数据或执行拟合")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出CSV", "", "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            import pandas as pd

            data = {
                'Frequency_Hz': self.current_freqs,
                'V3_Real_Exp': self.current_V3_exp.real,
                'V3_Imag_Exp': self.current_V3_exp.imag,
                'V3_Magnitude_Exp': np.abs(self.current_V3_exp),
                'V3_Phase_Exp_deg': np.unwrap(np.angle(self.current_V3_exp)) * 180 / np.pi
            }

            # 如果有拟合结果，添加拟合数据
            if self.fitted_result is not None:
                fiber_params = self.fiber_widget.get_fiber_params()
                material_props = self.material_widget.get_material_props()

                if isinstance(self.fitted_result, dict) and 'kappa' in self.fitted_result:
                    V3_fitted, _, _ = calculate_fiber_V3omega(
                        self.current_freqs, fiber_params, self.fitted_result
                    )
                elif isinstance(self.fitted_result, dict) and 'R' in self.fitted_result:
                    V3_fitted, _, _ = calculate_fiber_V3omega(
                        self.current_freqs, self.fitted_result, material_props
                    )
                else:
                    V3_fitted, _, _ = calculate_fiber_V3omega(
                        self.current_freqs,
                        self.fitted_result['fiber_params'],
                        self.fitted_result['material_props']
                    )

                data.update({
                    'V3_Real_Fitted': V3_fitted.real,
                    'V3_Imag_Fitted': V3_fitted.imag,
                    'V3_Magnitude_Fitted': np.abs(V3_fitted),
                    'V3_Phase_Fitted_deg': np.unwrap(np.angle(V3_fitted)) * 180 / np.pi
                })

            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)

            self.statusBar().showMessage(f"数据已导出: {file_path}", 3000)

        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出CSV失败:\n{str(e)}")

    def export_plot(self):
        """保存图像"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )

        if not file_path:
            return

        try:
            self.canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.statusBar().showMessage(f"图像已保存: {file_path}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存图像失败:\n{str(e)}")

    def _convert_to_serializable(self, obj):
        """转换为可序列化对象"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        else:
            return obj


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 设置应用字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    # 创建主窗口
    window = FiberThermal3OmegaGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

