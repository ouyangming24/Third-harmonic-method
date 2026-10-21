import sys
import os
import json
import time
import traceback
from datetime import datetime
import numpy as np
import pandas as pd

# --- 兼容子目录运行: 把上级目录(主项目根)加入 sys.path -----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from pyqtgraph import PlotWidget
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# 导入项目模块
from olson_model26_3 import ThermalModelFitter, JointThermalModelFitter, calculate_analytical_Z, convert_Z_to_V3
from calibration_gui import CalibrationWidget
from signal_calibration import SignalCalibrator
from 苏黎世锁相适配3 import ZurichLockInControl
from oe1300_lockin import OE1300LockInControl

# 本目录新增模块：两层厚度约束拟合器
from two_layer_thickness_fitter import TwoLayerThicknessFitter

# 扫频时每个频点的滤波时间常数 τ = TC_FACTOR / f(Hz)，苏黎世 demods timeconstant 与 OE1300 OFLT 共用此关系
LOCKIN_TC_FACTOR = 5.0

# OE1300 多次 SNAP 默认（与此前程序内写死值一致）
OE1300_SNAP_SAMPLES_DEFAULT = 6
OE1300_SNAP_DURATION_DEFAULT = 1.2


class EmittingStream(QObject):
    """自定义输出流，将文本发射为信号"""
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class MaterialDatabase:
    """材料数据库管理"""

    def __init__(self):
        self.materials = {
            'Si (硅)': {'k_y': 140.0, 'rho': 2330.0, 'cap': 700.0, 'H': np.inf, 'psi': 1.0, 'Rc': 0.0},
            'SiO2 (二氧化硅)': {'k_y': 1.4, 'rho': 2200.0, 'cap': 730.0, 'H': 1.0e-3, 'psi': 1.0, 'Rc': 1e-8},
            'Al2O3 (氧化铝)': {'k_y': 40.0, 'rho': 3900.0, 'cap': 880.0, 'H': 1.0e-3, 'psi': 1.0, 'Rc': 1e-8},
            'Diamond (金刚石)': {'k_y': 2000.0, 'rho': 3510.0, 'cap': 500.0, 'H': 1.0e-3, 'psi': 1.0, 'Rc': 1e-8},
            'Glass (玻璃)': {'k_y': 1.12, 'rho': 2500.0, 'cap': 840.0, 'H': np.inf, 'psi': 1.0, 'Rc': 1e-8},
            'Air (空气)': {'k_y': 0.03, 'rho': 1.2, 'cap': 1005.0, 'H': 10e-6, 'psi': 1.0, 'Rc': 1e-8},
            'Steel (钢)': {'k_y': 40.0, 'rho': 7850.0, 'cap': 500.0, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8},
            'PI (聚酰亚胺)': {'k_y': 3e-01, 'rho': 1420.0, 'cap': 1090.0, 'H': 25e-6, 'psi': 1.0, 'Rc': 1e-8},
            'MgO (氧化镁)': {'k_y': 50.0, 'rho': 3580.0, 'cap': 900.0, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8},
            'Ag_grease (含银导热硅脂)': {'k_y': 30, 'rho': 1500, 'cap': 1000, 'H': 40e-6, 'psi': 1.0, 'Rc': 1e-8},
            'silicone_grease (导热硅脂)': {'k_y': 18, 'rho': 3000, 'cap': 1250, 'H': 42e-6, 'psi': 1.0, 'Rc': 5.2e-8},
            'glass (玻璃)': {'k_y': 1.3, 'rho': 2200, 'cap': 800, 'H': 1000e-6, 'psi': 1.0, 'Rc': 1e-3},
            'zeolite (石英)': {'k_y': 1.07, 'rho': 2000, 'cap': 1250, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8},
            'steel (钢)': {'k_y': 40, 'rho': 7850, 'cap': 500, 'H': 30e-6, 'psi': 1.0, 'Rc': 1e-8},
            'sensor (传感器)': {'k_y': 0.72, 'rho': 1900, 'cap': 500, 'H': 25e-6, 'psi': 1.0, 'Rc': 1e-8},
            'Al (铝)': {'k_y': 237, 'rho': 2900, 'cap': 1000, 'H': 75e-6, 'psi': 1.0, 'Rc': 1.6E-3},
            'Cu (铜)': {'k_y': 400, 'rho': 8960, 'cap': 385, 'H': 4e-3, 'psi': 1.0, 'Rc': 3.8e-4},
            'yunmu (云母)': {'k_y': 0.55, 'rho': 2800, 'cap': 880, 'H': 100e-6, 'psi': 1.0, 'Rc': 5e-6},
            'CNO (碳纳米层)': {'k_y': 24, 'rho': 460, 'cap': 1660, 'H': 4.2e-6, 'psi': 1.0, 'Rc': 6.6e-6},
            'siN (氮化硅)': {'k_y': 1, 'rho': 700, 'cap': 3200, 'H': 1.5e-6, 'psi': 1.0, 'Rc': 1e-7},
            'picu (金属有机物混合)': {'k_y': 1.4, 'rho': 1850, 'cap': 1E3, 'H': 40e-6, 'psi': 1.0, 'Rc': 2.88e-5},
            'Custom (自定义)': {'k_y': 1.0, 'rho': 2000.0, 'cap': 1000.0, 'H': 1.0e-3, 'psi': 1.0, 'Rc': 1e-8}
        }

    def get_material(self, name):
        return self.materials.get(name, self.materials['Custom (自定义)']).copy()

    def get_material_names(self):
        return list(self.materials.keys())


class SensorDatabase:
    """传感器配置数据库"""

    def __init__(self):
        self.sensors = {
            '大传感器D1 (300μm)': {'b': 300e-6, 'L': 9e-3, 'R': 5.985, 'dR_dT': 0.0094, 'V_rms': 0.264, 'H_e': 100e-9,
                                   'rho_C_e': 2.5116e6},
            '大传感器D2 (250μm)': {'b': 250e-6, 'L': 8e-3, 'R': 6.225, 'dR_dT': 0.0099, 'V_rms': 0.264, 'H_e': 100e-9,
                                   'rho_C_e': 2.5116e6},
            '大传感器D3 (200μm)': {'b': 200e-6, 'L': 7e-3, 'R': 6.664, 'dR_dT': 0.011, 'V_rms': 0.264, 'H_e': 100e-9,
                                   'rho_C_e': 2.5116e6},
            '大传感器D4 (150μm)': {'b': 150e-6, 'L': 6e-3, 'R': 7.29, 'dR_dT': 0.0122, 'V_rms': 0.264, 'H_e': 100e-9,
                                   'rho_C_e': 2.5116e6},
            '小传感器X1 (50μm)': {'b': 50e-6, 'L': 2e-3, 'R': 9.398, 'dR_dT': 0.0149, 'V_rms': 0.4, 'H_e': 100e-9,
                                  'rho_C_e': 2.5116e6},
            '小传感器X2 (20μm)': {'b': 20e-6, 'L': 2e-3, 'R': 6.74, 'dR_dT': 0.014391, 'V_rms': 0.4, 'H_e': 100e-9,
                                  'rho_C_e': 2.5116e6},
            '小传感器X3 (5μm)': {'b': 5e-6, 'L': 2e-3, 'R': 6.74, 'dR_dT': 0.014391, 'V_rms': 0.4, 'H_e': 100e-9,
                                 'rho_C_e': 2.5116e6},
            'Custom (自定义)': {'b': 200e-6, 'L': 5e-3, 'R': 5.6, 'dR_dT': 1.9583e-02, 'V_rms': 0.43, 'H_e': 100e-9,
                                'rho_C_e': 2.5116e6}
        }

    def get_sensor(self, name):
        return self.sensors.get(name, self.sensors['Custom (自定义)']).copy()

    def get_sensor_names(self):
        return list(self.sensors.keys())


# --- 新增模块: 温升计算器对话框 ---
class TemperatureCalculatorDialog(QDialog):
    """温升计算器 (集成原温升计算.py功能)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3ω温升计算器")
        self.resize(500, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 1. 传感器参数
        sensor_group = QGroupBox("传感器参数")
        sensor_layout = QGridLayout()

        sensor_layout.addWidget(QLabel("电阻 R (Ω):"), 0, 0)
        self.r_input = QLineEdit("5.2")
        sensor_layout.addWidget(self.r_input, 0, 1)

        sensor_layout.addWidget(QLabel("温度系数 α_R (/K):"), 1, 0)
        self.alpha_input = QLineEdit("0.0022")
        sensor_layout.addWidget(self.alpha_input, 1, 1)

        sensor_layout.addWidget(QLabel("线长 L (mm):"), 2, 0)
        self.l_input = QLineEdit("8")
        sensor_layout.addWidget(self.l_input, 2, 1)

        sensor_layout.addWidget(QLabel("线宽 2b (μm):"), 3, 0)
        self.width_input = QLineEdit("500")
        sensor_layout.addWidget(self.width_input, 3, 1)

        sensor_group.setLayout(sensor_layout)
        layout.addWidget(sensor_group)

        # 2. 驱动参数
        drive_group = QGroupBox("驱动参数")
        drive_layout = QFormLayout()
        self.v_drive_input = QLineEdit("400")
        drive_layout.addRow("驱动电压 V (mV):", self.v_drive_input)
        drive_group.setLayout(drive_layout)
        layout.addWidget(drive_group)

        # 3. 测量信号
        signal_group = QGroupBox("测量信号")
        signal_layout = QHBoxLayout()
        signal_layout.addWidget(QLabel("V_3ω 信号:"))
        self.v3w_input = QLineEdit("100")
        signal_layout.addWidget(self.v3w_input)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["μV", "mV", "V"])
        signal_layout.addWidget(self.unit_combo)
        signal_group.setLayout(signal_layout)
        layout.addWidget(signal_group)

        # 4. 按钮
        btn_layout = QHBoxLayout()
        self.calc_btn = QPushButton("计算温升")
        self.calc_btn.clicked.connect(self.calculate)
        self.calc_max_btn = QPushButton("计算最大允许V3")
        self.calc_max_btn.clicked.connect(self.calculate_max)
        btn_layout.addWidget(self.calc_btn)
        btn_layout.addWidget(self.calc_max_btn)
        layout.addLayout(btn_layout)

        # 5. 结果显示
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.result_area.setFont(QFont("Consolas", 10))
        layout.addWidget(self.result_area)

        self.setLayout(layout)

    def calculate(self):
        try:
            R = float(self.r_input.text())
            alpha = float(self.alpha_input.text())
            V_drive = float(self.v_drive_input.text()) * 1e-3
            V3_val = float(self.v3w_input.text())

            unit = self.unit_combo.currentText()
            if unit == "μV":
                V3 = V3_val * 1e-6
            elif unit == "mV":
                V3 = V3_val * 1e-3
            else:
                V3 = V3_val

            # 计算
            I = V_drive / R
            V1 = I * R
            delta_T = 2 * V3 / (V1 * alpha)

            # 显示
            self.result_area.clear()
            self.result_area.append(f"=== 计算结果 ===\n")
            self.result_area.append(f"驱动电流 I: {I * 1e3:.2f} mA")
            self.result_area.append(f"温升 ΔT: {delta_T:.4f} K")

            if delta_T < 1.0:
                self.result_area.append("\n✅ 状态: 理想 (温升 < 1K)")
            elif delta_T < 2.0:
                self.result_area.append("\n⚠️ 状态: 可接受 (温升 < 2K)")
            else:
                self.result_area.append("\n❌ 状态: 过高 (建议降低驱动电压)")

        except ValueError:
            self.result_area.setText("输入错误，请检查数值。")

    def calculate_max(self):
        try:
            R = float(self.r_input.text())
            alpha = float(self.alpha_input.text())
            V_drive = float(self.v_drive_input.text()) * 1e-3
            V1 = V_drive  # 近似

            # 计算目标V3
            v3_1k = 1.0 * V1 * alpha / 2
            v3_2k = 2.0 * V1 * alpha / 2

            self.result_area.clear()
            self.result_area.append(f"=== 最大允许 V3w ===\n")
            self.result_area.append(f"目标温升 1K 时 V3w: {v3_1k * 1e6:.1f} μV")
            self.result_area.append(f"目标温升 2K 时 V3w: {v3_2k * 1e6:.1f} μV")

        except ValueError:
            self.result_area.setText("输入错误，请检查数值。")


# --- 新增模块: 正向模拟组件 ---
class SimulationWidget(QWidget):
    """正向模拟 (集成gui_3omega.py功能)"""

    def __init__(self):
        super().__init__()
        self.material_db = MaterialDatabase()
        self.sensor_db = SensorDatabase()
        self.exp_data = None

        # 自动更新定时器
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.run_simulation_silent)

        self.init_ui()

    def schedule_update(self):
        """计划更新模拟结果 (延迟执行以防卡顿)"""
        self.update_timer.start(300)  # 300ms 延迟

    def run_simulation_silent(self):
        """静默运行模拟（不弹出错误对话框，仅在状态栏或日志显示，避免打断输入）"""
        try:
            self.run_simulation(silent=True)
        except Exception:
            pass  # 忽略输入过程中的临时错误

    def init_ui(self):
        layout = QHBoxLayout()

        # 左侧：参数配置 (复用FittingWidget的部分逻辑思想，但重新实现以适配独立性)
        left_panel = QScrollArea()
        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)

        # 传感器参数
        sensor_group = QGroupBox("传感器设置")
        sensor_layout = QGridLayout()
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(self.sensor_db.get_sensor_names())
        self.sensor_combo.currentTextChanged.connect(self.load_sensor)
        sensor_layout.addWidget(QLabel("预设:"), 0, 0)
        sensor_layout.addWidget(self.sensor_combo, 0, 1)

        self.sensor_inputs = {}
        params = ['b', 'L', 'R', 'dR_dT', 'V_rms', 'H_e', 'rho_C_e']
        for i, p in enumerate(params):
            sensor_layout.addWidget(QLabel(p), i + 1, 0)
            inp = QLineEdit()
            inp.textChanged.connect(self.schedule_update)  # 绑定自动更新
            self.sensor_inputs[p] = inp
            sensor_layout.addWidget(inp, i + 1, 1)

        sensor_group.setLayout(sensor_layout)
        self.left_layout.addWidget(sensor_group)

        # 频率设置
        freq_group = QGroupBox("频率范围")
        freq_layout = QFormLayout()
        self.freq_min = QLineEdit("0.1")
        self.freq_min.textChanged.connect(self.schedule_update)
        self.freq_max = QLineEdit("1000")
        self.freq_max.textChanged.connect(self.schedule_update)
        self.freq_pts = QLineEdit("50")
        self.freq_pts.textChanged.connect(self.schedule_update)

        freq_layout.addRow("Min (Hz):", self.freq_min)
        freq_layout.addRow("Max (Hz):", self.freq_max)
        freq_layout.addRow("Points:", self.freq_pts)
        freq_group.setLayout(freq_layout)
        self.left_layout.addWidget(freq_group)

        # 材料层
        layer_group = QGroupBox("材料层结构")
        self.layer_layout = QVBoxLayout()
        self.layer_widgets = []  # 存储层控件引用

        # 默认添加2层
        self.add_layer_ui(0)
        self.add_layer_ui(1)

        layer_btns = QHBoxLayout()
        add_btn = QPushButton("添加层")
        add_btn.clicked.connect(lambda: self.add_layer_ui(len(self.layer_widgets)))
        clear_btn = QPushButton("重置层")
        clear_btn.clicked.connect(self.reset_layers)
        layer_btns.addWidget(add_btn)
        layer_btns.addWidget(clear_btn)

        self.layer_layout.addLayout(layer_btns)
        layer_group.setLayout(self.layer_layout)
        self.left_layout.addWidget(layer_group)

        # 噪声模拟设置
        noise_group = QGroupBox("噪声模拟")
        noise_layout = QGridLayout()

        self.noise_enabled = QCheckBox("添加随机噪声")
        self.noise_enabled.toggled.connect(self.toggle_noise_inputs)
        self.noise_enabled.toggled.connect(self.schedule_update)
        noise_layout.addWidget(self.noise_enabled, 0, 0, 1, 2)

        noise_layout.addWidget(QLabel("幅值噪声 (%):"), 1, 0)
        self.noise_amp = QDoubleSpinBox()
        self.noise_amp.setRange(0, 100)
        self.noise_amp.setValue(0.1)  # 默认0.1%
        self.noise_amp.setSingleStep(0.01)
        self.noise_amp.setEnabled(False)
        self.noise_amp.valueChanged.connect(self.schedule_update)
        noise_layout.addWidget(self.noise_amp, 1, 1)

        noise_layout.addWidget(QLabel("相位噪声 (°):"), 2, 0)
        self.noise_phase = QDoubleSpinBox()
        self.noise_phase.setRange(0, 360)
        self.noise_phase.setValue(0.1)  # 默认0.1度
        self.noise_phase.setSingleStep(0.01)
        self.noise_phase.setEnabled(False)
        self.noise_phase.valueChanged.connect(self.schedule_update)
        noise_layout.addWidget(self.noise_phase, 2, 1)

        noise_group.setLayout(noise_layout)
        self.left_layout.addWidget(noise_group)

        # 实验数据导入
        exp_group = QGroupBox("实验数据对比")
        exp_layout = QVBoxLayout()

        # 按钮布局
        exp_btn_layout = QHBoxLayout()
        self.load_exp_btn = QPushButton("导入实验数据")
        self.load_exp_btn.clicked.connect(self.load_experimental_data)
        exp_btn_layout.addWidget(self.load_exp_btn)

        self.clear_exp_btn = QPushButton("清除数据")
        self.clear_exp_btn.clicked.connect(self.clear_experimental_data)
        exp_btn_layout.addWidget(self.clear_exp_btn)

        exp_layout.addLayout(exp_btn_layout)

        self.exp_file_label = QLabel("未加载")
        self.exp_file_label.setWordWrap(True)
        exp_layout.addWidget(self.exp_file_label)
        exp_group.setLayout(exp_layout)
        self.left_layout.addWidget(exp_group)

        # 导出数据按钮
        self.export_btn = QPushButton("导出模拟数据")
        self.export_btn.clicked.connect(self.export_simulation_data)
        self.left_layout.addWidget(self.export_btn)

        # 计算按钮
        self.calc_btn = QPushButton("开始模拟计算")
        self.calc_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        self.calc_btn.clicked.connect(self.run_simulation)
        self.left_layout.addWidget(self.calc_btn)

        left_panel.setWidget(left_widget)
        left_panel.setWidgetResizable(True)
        left_panel.setMaximumWidth(400)
        layout.addWidget(left_panel)

        # 右侧：绘图
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        layout.addWidget(right_panel)
        self.setLayout(layout)

        # 初始化加载
        self.load_sensor(self.sensor_combo.currentText())

        # 存储最后一次模拟结果
        self.last_sim_data = None

    def toggle_noise_inputs(self, checked):
        """切换噪声输入框的启用状态"""
        self.noise_amp.setEnabled(checked)
        self.noise_phase.setEnabled(checked)

    def load_experimental_data(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载实验数据", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if filename:
            try:
                if filename.endswith('.xlsx'):
                    df = pd.read_excel(filename)
                else:
                    df = pd.read_csv(filename)

                # 简单列名检测逻辑
                if '频率 (Hz)' in df.columns:
                    freq_col = '频率 (Hz)'
                elif 'Frequency' in df.columns:
                    freq_col = 'Frequency'
                else:
                    freq_col = df.columns[0]  # 默认第一列

                # 查找 V3 数据
                v3_amp_col = None
                v3_phase_col = None

                cols = df.columns
                for c in cols:
                    if 'V3_幅值' in c or ('V3' in c and 'amp' in c.lower()) or 'Amplitude_3f' in c:
                        v3_amp_col = c
                    if 'V3_相位' in c or ('V3' in c and 'phase' in c.lower()) or 'Phase_3f' in c:
                        v3_phase_col = c

                # 如果找不到幅值相位，尝试找实部虚部
                v3_real_col = None
                v3_imag_col = None
                if v3_amp_col is None:
                    for c in cols:
                        if 'V3_Real' in c or 'v3_real' in c.lower():
                            v3_real_col = c
                        if 'V3_Imag' in c or 'v3_imag' in c.lower():
                            v3_imag_col = c

                if v3_amp_col is None and (v3_real_col is None or v3_imag_col is None):
                    raise ValueError("无法识别V3数据列，请确保包含 'V3_幅值'/'V3_相位' 或 'V3_Real'/'V3_Imag'")

                self.exp_data = {
                    'freq': df[freq_col].values,
                    'type': 'amp_phase' if v3_amp_col else 'real_imag',
                    'data': (df[v3_amp_col].values, df[v3_phase_col].values) if v3_amp_col else (
                        df[v3_real_col].values, df[v3_imag_col].values)
                }

                self.exp_file_label.setText(os.path.basename(filename))
                QMessageBox.information(self, "成功", "实验数据加载成功")

                # 刷新图表
                self.run_simulation_silent()

            except Exception as e:
                self.exp_data = None
                self.exp_file_label.setText("加载失败")
                QMessageBox.critical(self, "错误", str(e))

    def clear_experimental_data(self):
        """清除已加载的实验数据"""
        if self.exp_data is None:
            return

        self.exp_data = None
        self.exp_file_label.setText("未加载")

        # 刷新图表
        self.run_simulation_silent()
        QMessageBox.information(self, "成功", "实验数据已清除")

    def export_simulation_data(self):
        """导出模拟数据为CSV/Excel，格式兼容拟合输入"""
        if not hasattr(self, 'last_sim_data') or self.last_sim_data is None:
            QMessageBox.warning(self, "提示", "请先运行一次模拟以生成数据。")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "保存模拟数据", "simulation_data.csv", "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if not filename:
            return

        try:
            freqs = self.last_sim_data['freqs']
            V3 = self.last_sim_data['V3']
            # V3 已经是最终结果（包含可能的噪声）

            # 构建 DataFrame，包含拟合所需的标准列名
            # 标准列名: '频率 (Hz)', 'V3_Real', 'V3_Imag', 'V3_幅值', 'V3_相位'
            df = pd.DataFrame()
            df['频率 (Hz)'] = freqs
            df['V3_Real'] = V3.real
            df['V3_Imag'] = V3.imag
            df['V3_幅值'] = np.abs(V3)
            df['V3_相位'] = np.degrees(np.unwrap(np.angle(V3)))  # 保存为展开后的度数，避免跳变

            if filename.endswith('.xlsx'):
                df.to_excel(filename, index=False)
            else:
                df.to_csv(filename, index=False)

            QMessageBox.information(self, "成功", f"数据已保存至:\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存文件时出错:\n{str(e)}")

    def add_layer_ui(self, idx):
        group = QGroupBox(f"Layer {idx}")
        layout = QGridLayout()

        mat_combo = QComboBox()
        mat_combo.addItems(self.material_db.get_material_names())
        layout.addWidget(QLabel("材料:"), 0, 0)
        layout.addWidget(mat_combo, 0, 1)

        inputs = {}
        props = ['k_y', 'rho', 'cap', 'H', 'Rc', 'psi']
        for i, p in enumerate(props):
            layout.addWidget(QLabel(p), i + 1, 0)
            inp = QLineEdit()
            inp.textChanged.connect(self.schedule_update)  # 绑定自动更新
            inputs[p] = inp
            layout.addWidget(inp, i + 1, 1)

        mat_combo.currentTextChanged.connect(lambda t, inps=inputs: self.load_material(t, inps))
        self.load_material(mat_combo.currentText(), inputs)

        group.setLayout(layout)
        # 插入到按钮布局之前
        self.layer_layout.insertWidget(len(self.layer_widgets), group)
        self.layer_widgets.append({'group': group, 'inputs': inputs, 'combo': mat_combo})

        # 触发一次更新
        self.schedule_update()

    def reset_layers(self):
        for w in self.layer_widgets:
            w['group'].deleteLater()
        self.layer_widgets = []
        self.add_layer_ui(0)

    def load_sensor(self, name):
        data = self.sensor_db.get_sensor(name)
        for k, v in data.items():
            if k in self.sensor_inputs:
                self.sensor_inputs[k].setText(str(v))
        # load_sensor 改变了输入框文本，会自动触发 textChanged -> schedule_update

    def load_material(self, name, inputs):
        data = self.material_db.get_material(name)
        for k, v in data.items():
            inputs[k].setText(str(v))
        # load_material 改变了输入框文本，会自动触发 textChanged -> schedule_update

    def run_simulation(self, silent=False):
        try:
            # 1. 频率
            freqs = np.logspace(np.log10(float(self.freq_min.text())),
                                np.log10(float(self.freq_max.text())),
                                int(self.freq_pts.text()))

            # 2. 传感器
            sensor = {k: float(v.text()) for k, v in self.sensor_inputs.items()}

            # 3. 层
            layers = []
            for i, w in enumerate(self.layer_widgets):
                l_data = {}
                for k, v in w['inputs'].items():
                    val = v.text()
                    try:
                        f_val = np.inf if val == 'inf' else float(val)
                        if f_val <= 0 and k in ['k_y', 'rho', 'cap', 'H'] and not (
                                k == 'H' and i == len(self.layer_widgets) - 1):
                            # H 对于最底层(基底)可能不重要，但在GUI逻辑中它是最后一层
                            # 但因为我们要反转，所以GUI的最后一层其实是模型的第0层(基底)
                            # 基底的厚度在Olson模型中未被使用，所以可以容忍
                            # 但对于其他层，H必须大于0
                            if k == 'H' and i == len(self.layer_widgets) - 1:
                                pass
                            elif k == 'Rc' and f_val < 0:
                                raise ValueError(f"Rc 不能为负数")
                            elif k != 'Rc':
                                raise ValueError(f"{k} 必须大于 0")
                        l_data[k] = f_val
                    except ValueError as ve:
                        if not silent:
                            QMessageBox.warning(self, "参数错误", f"第 {i + 1} 层参数 {k} 无效: {str(ve)}")
                        return
                layers.append(l_data)

            # 4. 计算
            # 注意：Olson模型要求layers列表顺序为 [基底, ..., 顶层]
            # 而GUI通常是 [顶层, ..., 基底] (从上到下)
            # 因此这里需要反转layers列表传给模型
            model_layers = layers[::-1]

            # 额外检查：反转后，model_layers[0]是基底，model_layers[1:]是薄膜
            # 确保所有薄膜层的 H > 0
            for idx, layer in enumerate(model_layers[1:], 1):
                if layer['H'] <= 0:
                    if not silent:
                        QMessageBox.warning(self, "参数错误",
                                            f"模型第 {idx} 层 (GUI倒数第 {idx + 1} 层) 厚度 H 必须大于 0")
                    return

            Z = calculate_analytical_Z(freqs, sensor['b'], model_layers)
            V3 = convert_Z_to_V3(Z, freqs, sensor['b'], sensor)

            # 检查结果有效性
            if np.any(np.isnan(V3)) or np.any(np.isinf(V3)):
                if not silent:
                    QMessageBox.critical(self, "计算错误",
                                         "计算结果包含 NaN 或 Inf。\n请检查参数设置，特别是薄膜层厚度不能为0，热导率等参数是否合理。")
                return

            # 5. 处理噪声
            V3_final = V3.copy()
            if hasattr(self, 'noise_enabled') and self.noise_enabled.isChecked():
                # 生成高斯噪声
                amp_noise_percent = self.noise_amp.value() / 100.0
                phase_noise_deg = self.noise_phase.value()

                # 幅值噪声: A' = A * (1 + N(0, sigma))
                amp_noise = np.random.normal(0, amp_noise_percent, size=len(V3))
                current_amp = np.abs(V3)
                noisy_amp = current_amp * (1 + amp_noise)

                # 相位噪声: P' = P + N(0, sigma_deg)
                phase_noise = np.random.normal(0, np.radians(phase_noise_deg), size=len(V3))
                current_phase = np.angle(V3)
                noisy_phase = current_phase + phase_noise

                # 重建 V3 复数
                V3_final = noisy_amp * np.exp(1j * noisy_phase)

            # 存储结果 (供导出使用)
            self.last_sim_data = {
                'freqs': freqs,
                'V3': V3_final,
                'V3_ideal': V3
            }

            # 6. 绘图
            self.figure.clear()
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)

            # 绘制理想曲线 (虚线) - 改为 semilogx 以匹配旧版显示习惯
            ax1.semilogx(freqs, np.abs(V3) * 1e3, 'b--', label='Ideal Amp', alpha=0.7)
            ax1.set_ylabel('|V3| (mV)')
            ax1.grid(True, which='both', ls='--')

            # 绘制理想相位 (使用unwrap避免跳变)
            ideal_phase = np.degrees(np.unwrap(np.angle(V3)))
            ax2.semilogx(freqs, ideal_phase, 'r--', label='Ideal Phase', alpha=0.7)
            ax2.set_ylabel('Phase (deg)')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.grid(True, which='both', ls='--')

            # 如果有噪声，绘制带噪声的数据点
            if hasattr(self, 'noise_enabled') and self.noise_enabled.isChecked():
                ax1.semilogx(freqs, np.abs(V3_final) * 1e3, 'b.', markersize=4, label='Simulated Data')
                sim_phase = np.degrees(np.unwrap(np.angle(V3_final)))
                ax2.semilogx(freqs, sim_phase, 'r.', markersize=4, label='Simulated Data')

            # 绘制实验数据
            if self.exp_data:
                e_freq = self.exp_data['freq']
                if self.exp_data['type'] == 'amp_phase':
                    e_amp, e_phase = self.exp_data['data']
                    # 假设输入是 V，转为 mV
                    e_amp_mv = e_amp * 1e3
                else:
                    e_real, e_imag = self.exp_data['data']
                    e_complex = e_real + 1j * e_imag
                    e_amp_mv = np.abs(e_complex) * 1e3
                    e_phase = np.angle(e_complex, deg=True)

                ax1.semilogx(e_freq, e_amp_mv, 'go', markersize=4, label='Exp Data')
                ax2.semilogx(e_freq, e_phase, 'go', markersize=4, label='Exp Data')

            ax1.legend()
            ax2.legend()

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "计算错误", str(e))


# --- 新增模块: 灵敏度分析组件 ---
class SensitivityWidget(SimulationWidget):
    """灵敏度分析 (继承SimulationWidget的UI布局，修改计算逻辑)"""

    def __init__(self):
        super().__init__()
        # 修改按钮文字
        self.calc_btn.setText("运行灵敏度分析")
        self.sensitivity_data = None  # 存储计算结果
        self.init_control_panel()

    def init_control_panel(self):
        """初始化控制面板"""
        # 参数选择器
        self.param_group = QGroupBox("分析参数选择")
        layout = QVBoxLayout()

        self.param_list = QListWidget()
        layout.addWidget(self.param_list)

        btn_layout = QHBoxLayout()
        sel_all = QPushButton("全选")
        sel_all.clicked.connect(self.select_all)
        sel_none = QPushButton("全不选")
        sel_none.clicked.connect(self.select_none)
        btn_layout.addWidget(sel_all)
        btn_layout.addWidget(sel_none)
        layout.addLayout(btn_layout)

        self.param_group.setLayout(layout)

        # 显示类型选择器
        self.display_group = QGroupBox("显示类型")
        d_layout = QVBoxLayout()
        self.display_combo = QComboBox()
        self.display_combo.addItems([
            "Amplitude Sensitivity (Relative)",
            "Phase Sensitivity (Deg)",
            "Real Part Sensitivity (Norm. to Mag)",
            "Imag Part Sensitivity (Norm. to Mag)",
            "Amplitude & Phase (Dual Plot)"
        ])
        self.display_combo.currentIndexChanged.connect(self.update_plot)
        d_layout.addWidget(self.display_combo)
        self.display_group.setLayout(d_layout)

        # 插入到计算按钮之前
        idx = self.left_layout.indexOf(self.calc_btn)
        self.left_layout.insertWidget(idx, self.param_group)
        self.left_layout.insertWidget(idx + 1, self.display_group)

        self.update_param_list()

    def update_param_list(self):
        """更新参数列表"""
        self.param_list.clear()

        # 传感器参数
        for p in self.sensor_inputs.keys():
            item = QListWidgetItem(f"Sensor: {p}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, ('sensor', p))
            self.param_list.addItem(item)

        # 层参数
        for i, w in enumerate(self.layer_widgets):
            for p in w['inputs'].keys():
                item = QListWidgetItem(f"Layer {i + 1}: {p}")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if p in ['k_y', 'Rc'] else Qt.Unchecked)
                item.setData(Qt.UserRole, ('layer', i, p))
                self.param_list.addItem(item)

    def select_all(self):
        for i in range(self.param_list.count()):
            self.param_list.item(i).setCheckState(Qt.Checked)

    def select_none(self):
        for i in range(self.param_list.count()):
            self.param_list.item(i).setCheckState(Qt.Unchecked)

    def add_layer_ui(self, idx):
        super().add_layer_ui(idx)
        # 延迟更新参数列表，确保 widget 已添加
        if hasattr(self, 'param_list'):
            self.update_param_list()

    def reset_layers(self):
        super().reset_layers()
        if hasattr(self, 'param_list'):
            self.update_param_list()

    def run_simulation(self):
        """覆盖父类方法，执行灵敏度分析"""
        try:
            # 获取选中的参数
            selected_params = []
            for i in range(self.param_list.count()):
                item = self.param_list.item(i)
                if item.checkState() == Qt.Checked:
                    selected_params.append((item.text(), item.data(Qt.UserRole)))

            if not selected_params:
                QMessageBox.warning(self, "提示", "请至少选择一个参数进行分析")
                return

            # 1. 获取基础参数 (同父类)
            freqs = np.logspace(np.log10(float(self.freq_min.text())),
                                np.log10(float(self.freq_max.text())),
                                int(self.freq_pts.text()))
            sensor = {k: float(v.text()) for k, v in self.sensor_inputs.items()}
            layers = []
            for w in self.layer_widgets:
                l_data = {}
                for k, v in w['inputs'].items():
                    val = v.text()
                    l_data[k] = np.inf if val == 'inf' else float(val)
                layers.append(l_data)

            # 2. 计算基准值
            # 注意：Olson模型要求layers列表顺序为 [基底, ..., 顶层]
            # 而GUI通常是 [顶层, ..., 基底] (从上到下)
            # 因此这里需要反转layers列表传给模型
            model_layers_base = layers[::-1]
            Z_base = calculate_analytical_Z(freqs, sensor['b'], model_layers_base)
            V3_base = convert_Z_to_V3(Z_base, freqs, sensor['b'], sensor)
            V3_amp_base = np.abs(V3_base)
            V3_phase_base = np.unwrap(np.angle(V3_base))

            # 3. 分析参数
            perturbation = 0.01  # 1%
            self.sensitivity_data = {
                'freqs': freqs,
                'results': {}
            }

            import copy

            for name, info in selected_params:
                # 准备扰动数据
                mod_sensor = sensor.copy()
                mod_layers = copy.deepcopy(layers)

                type_ = info[0]
                if type_ == 'sensor':
                    param = info[1]
                    mod_sensor[param] *= (1 + perturbation)
                else:  # layer
                    layer_idx, param = info[1], info[2]
                    orig_val = mod_layers[layer_idx][param]

                    # 特殊处理无穷大和零值
                    if np.isinf(orig_val):
                        # 如果是无穷大（例如基底厚度），扰动无意义，跳过或设为0
                        # 但为了保持程序运行，我们不做改变，灵敏度自然为0
                        pass
                    elif orig_val == 0 and param == 'Rc':
                        mod_layers[layer_idx][param] = 1e-9
                    else:
                        mod_layers[layer_idx][param] *= (1 + perturbation)

                # 计算新值 (同样需要反转layers)
                model_layers_mod = mod_layers[::-1]
                Z_new = calculate_analytical_Z(freqs, mod_sensor['b'], model_layers_mod)
                V3_new = convert_Z_to_V3(Z_new, freqs, mod_sensor['b'], mod_sensor)

                # 计算各种灵敏度
                # 幅值灵敏度 (相对): (dA/A) / (dp/p)
                sens_amp = (np.abs(V3_new) - V3_amp_base) / V3_amp_base / perturbation

                # 相位灵敏度 (绝对): (dPhi) / (dp/p) -> 单位: 度
                sens_phase = (np.unwrap(np.angle(V3_new)) - V3_phase_base) * 180 / np.pi / perturbation

                # 实部灵敏度 (相对于幅值归一化): (dR/|A|) / (dp/p)
                # 使用幅值归一化可以避免实部为0时的除零错误，且物理意义是实部变化占总信号的比例
                sens_real = (V3_new.real - V3_base.real) / V3_amp_base / perturbation

                # 虚部灵敏度 (相对于幅值归一化)
                sens_imag = (V3_new.imag - V3_base.imag) / V3_amp_base / perturbation

                self.sensitivity_data['results'][name] = {
                    'amp': sens_amp,
                    'phase': sens_phase,
                    'real': sens_real,
                    'imag': sens_imag
                }

            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "分析错误", str(e))
            traceback.print_exc()

    def update_plot(self):
        """更新图表显示"""
        if not self.sensitivity_data:
            return

        self.figure.clear()

        display_type = self.display_combo.currentText()
        freqs = self.sensitivity_data['freqs']
        results = self.sensitivity_data['results']

        if display_type == "Amplitude & Phase (Dual Plot)":
            # 双图模式
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)

            for name, data in results.items():
                ax1.semilogx(freqs, data['amp'], label=name)
                ax2.semilogx(freqs, data['phase'], label=name)

            ax1.set_ylabel('Amp Sens. (dA/A)/(dp/p)')
            ax1.set_title('Amplitude Sensitivity')
            ax1.legend(fontsize='small')
            ax1.grid(True, which='both', ls='--')

            ax2.set_ylabel('Phase Sens. (deg)/(dp/p)')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_title('Phase Sensitivity')
            ax2.legend(fontsize='small')
            ax2.grid(True, which='both', ls='--')

        else:
            # 单图模式
            ax = self.figure.add_subplot(111)

            # 确定要绘制的数据类型键名
            key_map = {
                "Amplitude Sensitivity (Relative)": 'amp',
                "Phase Sensitivity (Deg)": 'phase',
                "Real Part Sensitivity (Norm. to Mag)": 'real',
                "Imag Part Sensitivity (Norm. to Mag)": 'imag'
            }
            data_key = key_map.get(display_type, 'amp')

            # 绘图
            for name, data in results.items():
                y_vals = data[data_key]
                ax.semilogx(freqs, y_vals, label=name)

            # 设置标签
            ax.set_xlabel('Frequency (Hz)')
            if data_key == 'amp':
                ax.set_ylabel('Sensitivity (dA/A) / (dp/p)')
                ax.set_title('Amplitude Sensitivity')
            elif data_key == 'phase':
                ax.set_ylabel('Sensitivity (deg) / (dp/p)')
                ax.set_title('Phase Sensitivity')
            else:
                ax.set_ylabel('Sensitivity (dV/|V|) / (dp/p)')
                ax.set_title(f'{display_type}')

            ax.legend()
            ax.grid(True, which='both', ls='--')

        self.figure.tight_layout()
        self.canvas.draw()


class DataComparisonWidget(QWidget):
    """多组实验数据对比模块"""

    def __init__(self):
        super().__init__()
        self.datasets = []  # 存储数据字典
        # Matplotlib 默认颜色循环
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # --- 左侧控制面板 ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 按钮区
        btn_group = QGroupBox("数据操作")
        btn_layout = QVBoxLayout()

        self.add_btn = QPushButton("添加数据文件")
        self.add_btn.clicked.connect(self.add_data)
        self.add_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px;")

        self.remove_btn = QPushButton("移除选中")
        self.remove_btn.clicked.connect(self.remove_selected_data)

        self.clear_btn = QPushButton("清空所有")
        self.clear_btn.clicked.connect(self.clear_all_data)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_group.setLayout(btn_layout)
        left_layout.addWidget(btn_group)

        # 数据列表区
        list_group = QGroupBox("已加载数据")
        list_layout = QVBoxLayout()
        self.data_list = QListWidget()
        self.data_list.setSelectionMode(QAbstractItemView.ExtendedSelection)  # 支持多选
        list_layout.addWidget(self.data_list)
        list_group.setLayout(list_layout)
        left_layout.addWidget(list_group)

        # 绘图选项
        opt_group = QGroupBox("显示选项")
        opt_layout = QVBoxLayout()
        self.log_x_check = QCheckBox("X轴对数坐标 (Log Freq)")
        self.log_x_check.setChecked(True)
        self.log_x_check.toggled.connect(self.update_plot)

        self.log_y_check = QCheckBox("Y轴(幅值)对数坐标")
        self.log_y_check.setChecked(False)
        self.log_y_check.toggled.connect(self.update_plot)

        opt_layout.addWidget(self.log_x_check)
        opt_layout.addWidget(self.log_y_check)

        # 频率范围筛选
        range_group = QGroupBox("频率范围筛选")
        range_layout = QGridLayout()
        self.auto_range_check = QCheckBox("自动范围")
        self.auto_range_check.setChecked(True)
        self.auto_range_check.toggled.connect(self.toggle_range_inputs)
        self.auto_range_check.toggled.connect(self.update_plot)

        range_layout.addWidget(self.auto_range_check, 0, 0, 1, 2)

        range_layout.addWidget(QLabel("Min (Hz):"), 1, 0)
        self.min_freq_spin = QDoubleSpinBox()
        self.min_freq_spin.setRange(0.001, 1e9)
        self.min_freq_spin.setValue(1)
        self.min_freq_spin.setDecimals(1)
        self.min_freq_spin.setEnabled(False)
        self.min_freq_spin.editingFinished.connect(self.update_plot)
        range_layout.addWidget(self.min_freq_spin, 1, 1)

        range_layout.addWidget(QLabel("Max (Hz):"), 2, 0)
        self.max_freq_spin = QDoubleSpinBox()
        self.max_freq_spin.setRange(0.001, 1e9)
        self.max_freq_spin.setValue(100000)
        self.max_freq_spin.setDecimals(1)
        self.max_freq_spin.setEnabled(False)
        self.max_freq_spin.editingFinished.connect(self.update_plot)
        range_layout.addWidget(self.max_freq_spin, 2, 1)

        range_group.setLayout(range_layout)
        opt_layout.addWidget(range_group)

        opt_group.setLayout(opt_layout)
        left_layout.addWidget(opt_group)

        left_layout.addStretch()

        # 限制左侧宽度
        left_panel.setMaximumWidth(300)
        layout.addWidget(left_panel)

        # --- 右侧绘图区域 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        layout.addWidget(right_panel)

    def toggle_range_inputs(self, checked):
        self.min_freq_spin.setEnabled(not checked)
        self.max_freq_spin.setEnabled(not checked)

    def add_data(self):
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "加载实验数据", "", "Data Files (*.xlsx *.csv);;Excel Files (*.xlsx);;CSV Files (*.csv)"
        )

        if not filenames:
            return

        for filename in filenames:
            try:
                self.load_single_file(filename)
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"文件 {os.path.basename(filename)} 加载失败:\n{str(e)}")

        self.update_list()
        self.update_plot()

    def load_single_file(self, filename):
        if filename.endswith('.xlsx'):
            df = pd.read_excel(filename)
        else:
            df = pd.read_csv(filename)

        # 1. 频率
        if '频率 (Hz)' in df.columns:
            freq_col = '频率 (Hz)'
        elif 'Frequency' in df.columns:
            freq_col = 'Frequency'
        else:
            freq_col = df.columns[0]

        # 2. V3 数据
        v3_amp_col = None
        v3_phase_col = None
        v3_real_col = None
        v3_imag_col = None

        cols = df.columns
        for c in cols:
            c_lower = c.lower()
            if 'v3_幅值' in c or ('v3' in c and 'amp' in c_lower) or 'amplitude_3f' in c:
                v3_amp_col = c
            if 'v3_相位' in c or ('v3' in c and 'phase' in c_lower) or 'phase_3f' in c:
                v3_phase_col = c
            if 'v3_real' in c or 'v3_real' in c_lower:
                v3_real_col = c
            if 'v3_imag' in c or 'v3_imag' in c_lower:
                v3_imag_col = c

        freq = df[freq_col].values

        if v3_amp_col and v3_phase_col:
            amp = df[v3_amp_col].values
            phase = df[v3_phase_col].values
        elif v3_real_col and v3_imag_col:
            real = df[v3_real_col].values
            imag = df[v3_imag_col].values
            complex_v3 = real + 1j * imag
            amp = np.abs(complex_v3)
            phase = np.degrees(np.angle(complex_v3))
        else:
            raise ValueError("无法识别 V3 数据列 (需包含 幅值/相位 或 实部/虚部)")

        # 分配颜色
        color_idx = len(self.datasets) % len(self.colors)
        color = self.colors[color_idx]

        self.datasets.append({
            'filename': os.path.basename(filename),
            'full_path': filename,
            'freq': freq,
            'amp': amp,
            'phase': phase,
            'color': color
        })

    def remove_selected_data(self):
        selected_items = self.data_list.selectedItems()
        if not selected_items:
            return

        # 获取要删除的索引
        indices_to_remove = []
        for item in selected_items:
            row = self.data_list.row(item)
            indices_to_remove.append(row)

        indices_to_remove.sort(reverse=True)

        for idx in indices_to_remove:
            if 0 <= idx < len(self.datasets):
                self.datasets.pop(idx)

        self.update_list()
        self.update_plot()

    def clear_all_data(self):
        self.datasets = []
        self.update_list()
        self.update_plot()

    def update_list(self):
        self.data_list.clear()
        for i, data in enumerate(self.datasets):
            item = QListWidgetItem(f"{i + 1}. {data['filename']}")
            item.setForeground(QColor(data['color']))
            self.data_list.addItem(item)

    def update_plot(self):
        self.figure.clear()

        if not self.datasets:
            self.canvas.draw()
            return

        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212, sharex=ax1)

        is_log_x = self.log_x_check.isChecked()
        is_log_y = self.log_y_check.isChecked()

        for data in self.datasets:
            f = data['freq']
            amp = data['amp']
            phase = data['phase']
            c = data['color']
            label = data['filename']

            # Plot Amplitude
            if is_log_x and is_log_y:
                ax1.loglog(f, amp, 'o-', color=c, label=label, markersize=4, linewidth=1.5)
            elif is_log_x:
                ax1.semilogx(f, amp, 'o-', color=c, label=label, markersize=4, linewidth=1.5)
            elif is_log_y:
                ax1.semilogy(f, amp, 'o-', color=c, label=label, markersize=4, linewidth=1.5)
            else:
                ax1.plot(f, amp, 'o-', color=c, label=label, markersize=4, linewidth=1.5)

            # Plot Phase
            if is_log_x:
                ax2.semilogx(f, phase, 'o-', color=c, markersize=4, linewidth=1.5)
            else:
                ax2.plot(f, phase, 'o-', color=c, markersize=4, linewidth=1.5)

        ax1.set_ylabel('V3 Amplitude (V)')
        ax1.grid(True, which="both", ls="-", alpha=0.5)
        ax1.legend(loc='best', fontsize='small')

        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (deg)')
        ax2.grid(True, which="both", ls="-", alpha=0.5)

        # 应用频率范围限制
        if not self.auto_range_check.isChecked():
            min_f = self.min_freq_spin.value()
            max_f = self.max_freq_spin.value()
            if min_f < max_f:
                ax1.set_xlim(min_f, max_f)
                # ax2 sharex with ax1, so it updates automatically

        self.figure.tight_layout()
        self.canvas.draw()


class DataAcquisitionThread(QThread):
    """数据采集线程"""
    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)
    finished_acquisition = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.lockin = None
        self.frequencies = []
        self.output_amplitude = 0.4
        self.is_running = False
        self.device_id = 'dev3791'
        self.zi_host = '192.168.59.62'
        self.zi_port = 8004
        # 锁相类型: zurich | oe1300_tcp | oe1300_serial
        self.lockin_backend = 'zurich'
        self.oe_tcp_host = '192.168.1.1'
        self.oe_tcp_port = 10001
        self.oe_serial_port = 'COM3'
        self.oe_input_range_index = None  # None 则不下发 OE1300 IRNG
        # 滤波稳定等待：False=自动（τ×倍数，与苏黎世逻辑一致）；True=固定秒数
        self.wait_use_manual = False
        self.wait_manual_seconds = 5.0
        self.integration_multiplier = 5  # 自动模式下 wait = τ * multiplier
        self.oe_snap_n_samples = OE1300_SNAP_SAMPLES_DEFAULT
        self.oe_snap_duration = OE1300_SNAP_DURATION_DEFAULT

    def setup(
            self,
            device_id,
            frequencies,
            output_amplitude,
            zi_host='192.168.59.62',
            zi_port=8004,
            lockin_backend='zurich',
            oe_tcp_host='192.168.1.1',
            oe_tcp_port=10001,
            oe_serial_port='COM3',
            oe_input_range_index=None,
            wait_use_manual=False,
            wait_manual_seconds=5.0,
            integration_multiplier=5,
            oe_snap_n_samples=None,
            oe_snap_duration=None,
    ):
        self.device_id = device_id
        self.frequencies = frequencies
        self.output_amplitude = output_amplitude
        self.zi_host = zi_host
        self.zi_port = int(zi_port)
        self.lockin_backend = (lockin_backend or 'zurich').lower()
        self.oe_tcp_host = oe_tcp_host
        self.oe_tcp_port = int(oe_tcp_port)
        self.oe_serial_port = oe_serial_port
        self.oe_input_range_index = oe_input_range_index
        self.wait_use_manual = bool(wait_use_manual)
        self.wait_manual_seconds = float(wait_manual_seconds)
        self.integration_multiplier = int(integration_multiplier)
        self.oe_snap_n_samples = (
            int(oe_snap_n_samples)
            if oe_snap_n_samples is not None
            else OE1300_SNAP_SAMPLES_DEFAULT
        )
        self.oe_snap_duration = (
            float(oe_snap_duration)
            if oe_snap_duration is not None
            else OE1300_SNAP_DURATION_DEFAULT
        )

    def run(self):
        try:
            self.is_running = True
            backend = self.lockin_backend

            if backend == 'zurich':
                self.progress_update.emit(5, f"连接苏黎世锁相 {self.device_id}...")
                self.lockin = ZurichLockInControl(self.device_id, server_host=self.zi_host, server_port=self.zi_port)
                self.lockin.configure_signal_inputs(
                    input_channel=0, range=1, ac=False, diff=True
                )
                self.lockin.configure_signal_outputs(
                    output_channel=0, amplitude=self.output_amplitude,
                    on=True, range_val=10.0, offset=0.0
                )
            elif backend == 'oe1300_tcp':
                self.lockin = OE1300LockInControl(
                    connection='tcp',
                    tcp_host=self.oe_tcp_host,
                    tcp_port=self.oe_tcp_port,
                )
                if self.oe_input_range_index is not None:
                    self.lockin.set_input_range_index(int(self.oe_input_range_index))
                self.lockin.configure_signal_inputs(
                    input_channel=0, range=1, ac=False, diff=True
                )
                self.lockin.configure_signal_outputs(
                    output_channel=0, amplitude=self.output_amplitude,
                    on=True, range_val=10.0, offset=0.0
                )
            elif backend == 'oe1300_serial':
                self.lockin = OE1300LockInControl(
                    connection='serial',
                    serial_port=self.oe_serial_port,
                )
                if self.oe_input_range_index is not None:
                    self.lockin.set_input_range_index(int(self.oe_input_range_index))
                self.lockin.configure_signal_inputs(
                    input_channel=0, range=1, ac=False, diff=True
                )
                self.lockin.configure_signal_outputs(
                    output_channel=0, amplitude=self.output_amplitude,
                    on=True, range_val=10.0, offset=0.0
                )
            else:
                raise ValueError(f"未知锁相类型: {backend}")

            total_freqs = len(self.frequencies)

            for idx, freq in enumerate(self.frequencies):
                if not self.is_running:
                    break

                progress = int((idx + 1) / total_freqs * 100)
                self.progress_update.emit(progress, f"测量频率: {freq:.2f} Hz")

                # 苏黎世与 OE1300：每个频点 τ 随 f 变化（与苏黎世 timeconstant / OE1300 OFLT 一致）
                time_constant = (
                    LOCKIN_TC_FACTOR / freq if freq > 0 else 0.1
                )

                self.lockin.configure_lockin(
                    freq=freq, order_time=4, time_constant=time_constant
                )

                if self.wait_use_manual:
                    w = max(0.0, self.wait_manual_seconds)
                    self.progress_update.emit(
                        progress, f"固定等待 {w:.2f} s (频率 {freq:.2f} Hz)"
                    )
                    time.sleep(w)
                else:
                    self.lockin.wait_for_integration(
                        time_constant,
                        duration_multiplier=self.integration_multiplier,
                    )

                if backend == 'zurich':
                    x1, y1, x3, y3 = self.lockin.read_demod_data(duration=2)
                else:
                    x1, y1, x3, y3 = self.lockin.read_demod_data(
                        duration=max(0.05, self.oe_snap_duration),
                        n_samples=max(1, self.oe_snap_n_samples),
                    )

                amp1, phase1, _, _ = self.lockin.calculate_amplitude_phase(x1, y1)
                amp3, phase3, x3_mean, y3_mean = self.lockin.calculate_amplitude_phase(
                    x3, y3
                )

                if all(v is not None for v in [amp3, phase3, x3_mean, y3_mean]):
                    data = {
                        'frequency': freq,
                        'v3_real': x3_mean,
                        'v3_imag': y3_mean,
                        'v3_amp': amp3,
                        'v3_phase': phase3,
                        'v1_amp': amp1,
                        'v1_phase': phase1
                    }
                    self.data_received.emit(data)

            self.finished_acquisition.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.lockin:
                self.lockin.disconnect()
            self.is_running = False

    def stop(self):
        self.is_running = False


class LockInControlWidget(QWidget):
    """锁相放大器控制界面"""

    def __init__(self):
        super().__init__()
        self.acquisition_thread = DataAcquisitionThread()
        self.collected_data = []
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout()

        # 设备连接组
        device_group = QGroupBox("设备连接")
        device_layout = QGridLayout()

        device_layout.addWidget(QLabel("锁相类型:"), 0, 0)
        self.lockin_type_combo = QComboBox()
        self.lockin_type_combo.addItems([
            "苏黎世 (zhinst)",
            "OE1300 网口 (TCP)",
            "OE1300 串口 (RS232)",
        ])
        self.lockin_type_combo.setToolTip(
            "OE1300 使用手册 ASCII 指令；网口默认 192.168.1.1:10001"
        )
        device_layout.addWidget(self.lockin_type_combo, 0, 1)

        device_layout.addWidget(QLabel("苏黎世 IP/Device:"), 1, 0)

        zi_box = QHBoxLayout()
        self.zi_host_input = QLineEdit("192.168.59.62")
        self.zi_port_spin = QSpinBox()
        self.zi_port_spin.setRange(1, 65535)
        self.zi_port_spin.setValue(8004)
        self.device_id_input = QLineEdit("dev3791")

        zi_box.addWidget(self.zi_host_input)
        zi_box.addWidget(QLabel(":"))
        zi_box.addWidget(self.zi_port_spin)
        zi_box.addWidget(self.device_id_input)

        device_layout.addLayout(zi_box, 1, 1, 1, 3)

        device_layout.addWidget(QLabel("OE1300 IP:"), 2, 0)
        self.oe_host_input = QLineEdit("192.168.1.1")
        device_layout.addWidget(self.oe_host_input, 2, 1)
        device_layout.addWidget(QLabel("端口:"), 2, 2)
        self.oe_port_spin = QSpinBox()
        self.oe_port_spin.setRange(1, 65535)
        self.oe_port_spin.setValue(10001)
        device_layout.addWidget(self.oe_port_spin, 2, 3)

        device_layout.addWidget(QLabel("OE1300 串口:"), 3, 0)
        self.oe_serial_input = QLineEdit("COM3")
        device_layout.addWidget(self.oe_serial_input, 3, 1)
        device_layout.addWidget(QLabel("IRNG索引(可选):"), 3, 2)
        self.oe_irng_spin = QSpinBox()
        self.oe_irng_spin.setRange(-1, 255)
        self.oe_irng_spin.setValue(-1)
        self.oe_irng_spin.setSpecialValueText("不下发")
        self.oe_irng_spin.setToolTip("-1 表示不通过指令设置 IRNG，请在仪器面板选量程")
        device_layout.addWidget(self.oe_irng_spin, 3, 3)

        self.connect_btn = QPushButton("连接/检测")
        self.connect_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        device_layout.addWidget(self.connect_btn, 1, 2, 1, 2)

        self.status_label = QLabel("状态: 未连接")
        self.status_label.setStyleSheet("QLabel { color: red; }")
        device_layout.addWidget(self.status_label, 4, 0, 1, 4)

        self.lockin_type_combo.currentIndexChanged.connect(self._on_lockin_type_changed)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # 测量参数组
        param_group = QGroupBox("测量参数")
        param_layout = QGridLayout()

        # 频率设置
        param_layout.addWidget(QLabel("起始频率 (Hz):"), 0, 0)
        self.freq_start = QDoubleSpinBox()
        self.freq_start.setRange(0.1, 100000000)
        self.freq_start.setValue(10)
        param_layout.addWidget(self.freq_start, 0, 1)

        param_layout.addWidget(QLabel("结束频率 (Hz):"), 0, 2)
        self.freq_end = QDoubleSpinBox()
        self.freq_end.setRange(0.1, 100000000)
        self.freq_end.setValue(1000)
        param_layout.addWidget(self.freq_end, 0, 3)

        param_layout.addWidget(QLabel("频率点数:"), 1, 0)
        self.freq_points = QSpinBox()
        self.freq_points.setRange(1, 100)
        self.freq_points.setValue(30)
        param_layout.addWidget(self.freq_points, 1, 1)

        param_layout.addWidget(QLabel("频率分布:"), 1, 2)
        self.freq_scale = QComboBox()
        self.freq_scale.addItems(["对数分布", "线性分布"])
        param_layout.addWidget(self.freq_scale, 1, 3)

        # 输出参数
        param_layout.addWidget(QLabel("输出电压 (V):"), 2, 0)
        self.output_voltage = QDoubleSpinBox()
        self.output_voltage.setRange(0.001, 10.0)
        self.output_voltage.setDecimals(3)
        self.output_voltage.setSingleStep(0.001)
        self.output_voltage.setValue(0.400)
        self.output_voltage.setToolTip("精确到小数点后 3 位（例如 0.430 V）")
        param_layout.addWidget(self.output_voltage, 2, 1)

        param_layout.addWidget(QLabel("测量次数:"), 2, 2)
        self.measurement_count = QSpinBox()
        self.measurement_count.setRange(1, 10)
        self.measurement_count.setValue(1)
        param_layout.addWidget(self.measurement_count, 2, 3)

        # 滤波器稳定等待：默认自动 = 时间常数×倍数（τ=5/f Hz，与当前苏黎世/OE1300 共用逻辑）
        param_layout.addWidget(QLabel("稳定等待:"), 3, 0)
        self.wait_manual_check = QCheckBox("手动指定等待时间 (秒)")
        self.wait_manual_check.setChecked(False)
        self.wait_manual_check.setToolTip(
            f"不勾选：每个频点等待时间 = 时间常数 × 5；"
            f"时间常数 τ = {LOCKIN_TC_FACTOR:g}/频率(Hz)（苏黎世 timeconstant / OE1300 OFLT）。"
            "勾选：每个频点使用下方固定秒数。"
        )
        param_layout.addWidget(self.wait_manual_check, 3, 1)
        param_layout.addWidget(QLabel("固定等待 (s):"), 3, 2)
        self.wait_manual_spin = QDoubleSpinBox()
        self.wait_manual_spin.setRange(0.0, 7200.0)
        self.wait_manual_spin.setDecimals(2)
        self.wait_manual_spin.setSingleStep(0.5)
        self.wait_manual_spin.setValue(5.0)
        self.wait_manual_spin.setEnabled(False)
        param_layout.addWidget(self.wait_manual_spin, 3, 3)

        self.wait_auto_hint = QLabel(
            f"自动：等待 = τ×5，τ = {LOCKIN_TC_FACTOR:g}/f(Hz)；"
            "苏黎世与 OE1300 每个频点均按此设置滤波积分（OE1300 为 OFLT）"
        )
        self.wait_auto_hint.setWordWrap(True)
        self.wait_auto_hint.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
        param_layout.addWidget(self.wait_auto_hint, 4, 0, 1, 4)

        self.wait_manual_check.toggled.connect(self._on_wait_manual_toggled)
        self._on_wait_manual_toggled(self.wait_manual_check.isChecked())

        param_layout.addWidget(QLabel("信号缩放倍数:"), 5, 0)
        self.signal_scale_spin = QDoubleSpinBox()
        self.signal_scale_spin.setRange(1e-6, 1e6)
        self.signal_scale_spin.setDecimals(6)
        self.signal_scale_spin.setSingleStep(0.000001)
        self.signal_scale_spin.setValue(1.0)
        self.signal_scale_spin.setToolTip(
            "对 V3 实部/虚部/幅值、V1 幅值乘以该系数（<1 为缩小）；"
            "频率与相位(°)不变。测量结束后改倍数会刷新表格与曲线；"
            "保存文件为当前倍数下的数值。倍数可精确到小数点后 6 位。"
        )
        param_layout.addWidget(self.signal_scale_spin, 5, 1)
        self.signal_scale_hint = QLabel(
            "作用于采集电压量；相位不缩放。实时显示与「保存数据」均为缩放后数值。"
        )
        self.signal_scale_hint.setWordWrap(True)
        self.signal_scale_hint.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
        param_layout.addWidget(self.signal_scale_hint, 5, 2, 1, 2)

        param_layout.addWidget(QLabel("OE1300 SNAP 次数:"), 6, 0)
        self.oe_snap_samples_spin = QSpinBox()
        self.oe_snap_samples_spin.setRange(1, 500)
        self.oe_snap_samples_spin.setValue(OE1300_SNAP_SAMPLES_DEFAULT)
        self.oe_snap_samples_spin.setToolTip(
            "仅 OE1300 测量时生效；每个频点重复 SNAP? 的次数，用于平均降噪"
        )
        param_layout.addWidget(self.oe_snap_samples_spin, 6, 1)
        param_layout.addWidget(QLabel("OE1300 读数时长 (s):"), 6, 2)
        self.oe_snap_duration_spin = QDoubleSpinBox()
        self.oe_snap_duration_spin.setRange(0.05, 600.0)
        self.oe_snap_duration_spin.setDecimals(2)
        self.oe_snap_duration_spin.setSingleStep(0.1)
        self.oe_snap_duration_spin.setValue(OE1300_SNAP_DURATION_DEFAULT)
        self.oe_snap_duration_spin.setToolTip(
            "仅 OE1300；相邻 SNAP 间隔约为 max(读数时长/次数, 0.05)s"
        )
        param_layout.addWidget(self.oe_snap_duration_spin, 6, 3)

        self.oe_snap_hint = QLabel(
            "上两项仅在选择 OE1300（网口/串口）时参与测量；苏黎世仍用内部 poll 取数。"
        )
        self.oe_snap_hint.setWordWrap(True)
        self.oe_snap_hint.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
        param_layout.addWidget(self.oe_snap_hint, 7, 0, 1, 4)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 控制按钮
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始测量")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-size: 14px; padding: 5px; }")
        self.stop_btn = QPushButton("停止测量")
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-size: 14px; padding: 5px; }")
        self.stop_btn.setEnabled(False)
        self.save_btn = QPushButton("保存数据")
        self.save_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-size: 14px; padding: 5px; }")

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.save_btn)
        layout.addLayout(control_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("准备就绪")
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        # 实时数据显示
        data_group = QGroupBox("实时数据显示")
        data_layout = QVBoxLayout()

        # 创建实时图表
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setMinimumHeight(400)

        # V3幅值图
        self.amp_plot = self.plot_widget.addPlot(title="V3 幅值", row=0, col=0)
        self.amp_plot.setLabel('left', '幅值', units='V')
        self.amp_plot.setLabel('bottom', '频率', units='Hz')
        self.amp_plot.setLogMode(x=True, y=False)
        self.amp_curve = self.amp_plot.plot(pen='r', symbol='o', symbolSize=6)

        # V3相位图
        self.phase_plot = self.plot_widget.addPlot(title="V3 相位", row=0, col=1)
        self.phase_plot.setLabel('left', '相位', units='deg')
        self.phase_plot.setLabel('bottom', '频率', units='Hz')
        self.phase_plot.setLogMode(x=True, y=False)
        self.phase_curve = self.phase_plot.plot(pen='b', symbol='o', symbolSize=6)

        data_layout.addWidget(self.plot_widget)

        # 数据表格
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(7)
        self.data_table.setHorizontalHeaderLabels(
            ["频率 (Hz)", "V3_Real", "V3_Imag", "V3_幅值", "V3_相位", "V1_幅值", "V1_相位"]
        )
        self.data_table.setMaximumHeight(200)
        data_layout.addWidget(self.data_table)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        self.setLayout(layout)
        self._on_lockin_type_changed()

    def connect_signals(self):
        self.connect_btn.clicked.connect(self.connect_device)
        self.start_btn.clicked.connect(self.start_measurement)
        self.stop_btn.clicked.connect(self.stop_measurement)
        self.save_btn.clicked.connect(self.save_data)

        self.acquisition_thread.data_received.connect(self.update_display)
        self.acquisition_thread.progress_update.connect(self.update_progress)
        self.acquisition_thread.error_occurred.connect(self.handle_error)
        self.acquisition_thread.finished_acquisition.connect(self.measurement_finished)
        self.signal_scale_spin.valueChanged.connect(self._on_signal_scale_changed)

    def _on_wait_manual_toggled(self, checked: bool):
        self.wait_manual_spin.setEnabled(checked)
        self.wait_auto_hint.setVisible(not checked)

    def _signal_scale_k(self) -> float:
        return float(self.signal_scale_spin.value())

    def _scaled_data_row(self, raw: dict) -> dict:
        """仪器原始一行 → 缩放后（用于显示与保存）；相位、频率不缩放。"""
        k = self._signal_scale_k()
        return {
            'frequency': float(raw['frequency']),
            'v3_real': float(raw['v3_real']) * k,
            'v3_imag': float(raw['v3_imag']) * k,
            'v3_amp': float(raw['v3_amp']) * k,
            'v3_phase': float(raw['v3_phase']),
            'v1_amp': float(raw['v1_amp']) * k,
            'v1_phase': float(raw['v1_phase']),
        }

    def _refresh_plots_and_table_from_collected(self):
        """按当前缩放倍数重绘表格与 V3 幅相曲线（内存中仍为仪器原始值）。"""
        n = len(self.collected_data)
        self.data_table.setRowCount(n)
        freqs, amps, phases = [], [], []
        for i, raw in enumerate(self.collected_data):
            s = self._scaled_data_row(raw)
            freqs.append(s['frequency'])
            amps.append(s['v3_amp'])
            phases.append(s['v3_phase'])
            self.data_table.setItem(i, 0, QTableWidgetItem(f"{s['frequency']:.2f}"))
            self.data_table.setItem(i, 1, QTableWidgetItem(f"{s['v3_real']:.6e}"))
            self.data_table.setItem(i, 2, QTableWidgetItem(f"{s['v3_imag']:.6e}"))
            self.data_table.setItem(i, 3, QTableWidgetItem(f"{s['v3_amp']:.6e}"))
            self.data_table.setItem(i, 4, QTableWidgetItem(f"{s['v3_phase']:.2f}"))
            self.data_table.setItem(i, 5, QTableWidgetItem(f"{s['v1_amp']:.6e}"))
            self.data_table.setItem(i, 6, QTableWidgetItem(f"{s['v1_phase']:.2f}"))
        if n > 0:
            self.amp_curve.setData(freqs, amps)
            self.phase_curve.setData(freqs, phases)
        else:
            self.amp_curve.setData([], [])
            self.phase_curve.setData([], [])

    def _on_signal_scale_changed(self):
        if self.collected_data:
            self._refresh_plots_and_table_from_collected()

    def _on_lockin_type_changed(self):
        idx = self.lockin_type_combo.currentIndex()
        zurich = idx == 0
        self.zi_host_input.setEnabled(zurich)
        self.zi_port_spin.setEnabled(zurich)
        self.device_id_input.setEnabled(zurich)
        self.oe_host_input.setEnabled(idx == 1)
        self.oe_port_spin.setEnabled(idx == 1)
        self.oe_serial_input.setEnabled(idx == 2)
        self.oe_irng_spin.setEnabled(idx in (1, 2))
        oe = idx in (1, 2)
        self.oe_snap_samples_spin.setEnabled(oe)
        self.oe_snap_duration_spin.setEnabled(oe)

    def connect_device(self):
        idx = self.lockin_type_combo.currentIndex()
        try:
            if idx == 0:
                self.status_label.setText("状态: 苏黎世请运行测量时自动连接")
                self.status_label.setStyleSheet("QLabel { color: #666; }")
                QMessageBox.information(
                    self, "提示",
                    "苏黎世锁相在点击「开始测量」时由 zhinst 连接。\n"
                    f"当前 device_id: {self.device_id_input.text()}",
                )
                return
            if idx == 1:
                li = OE1300LockInControl(
                    connection="tcp",
                    tcp_host=self.oe_host_input.text().strip(),
                    tcp_port=self.oe_port_spin.value(),
                )
            else:
                li = OE1300LockInControl(
                    connection="serial",
                    serial_port=self.oe_serial_input.text().strip(),
                )
            try:
                idn = li.idn()
            finally:
                li.disconnect()
            self.status_label.setText("状态: OE1300 应答正常")
            self.status_label.setStyleSheet("QLabel { color: green; }")
            QMessageBox.information(self, "连接成功", f"*IDN? 返回:\n{idn}")
        except Exception as e:
            self.status_label.setText("状态: 连接失败")
            self.status_label.setStyleSheet("QLabel { color: red; }")
            QMessageBox.critical(self, "连接失败", str(e))

    def start_measurement(self):
        # 生成频率列表
        start = self.freq_start.value()
        end = self.freq_end.value()
        points = self.freq_points.value()

        if self.freq_scale.currentText() == "对数分布":
            frequencies = np.logspace(np.log10(start), np.log10(end), points)
        else:
            frequencies = np.linspace(start, end, points)

        # 清空之前的数据
        self.collected_data = []
        self.data_table.setRowCount(0)

        lt = self.lockin_type_combo.currentIndex()
        if lt == 0:
            backend = 'zurich'
            oe_tcp_host, oe_tcp_port, oe_serial = '192.168.1.1', 10001, 'COM3'
            oe_irng = None
        elif lt == 1:
            backend = 'oe1300_tcp'
            oe_tcp_host = self.oe_host_input.text().strip()
            oe_tcp_port = self.oe_port_spin.value()
            oe_serial = 'COM3'
            oe_irng = self.oe_irng_spin.value() if self.oe_irng_spin.value() >= 0 else None
        else:
            backend = 'oe1300_serial'
            oe_tcp_host, oe_tcp_port = '192.168.1.1', 10001
            oe_serial = self.oe_serial_input.text().strip()
            oe_irng = self.oe_irng_spin.value() if self.oe_irng_spin.value() >= 0 else None

        self.acquisition_thread.setup(
            self.device_id_input.text().strip(),
            frequencies,
            self.output_voltage.value(),
            zi_host=self.zi_host_input.text().strip(),
            zi_port=self.zi_port_spin.value(),
            lockin_backend=backend,
            oe_tcp_host=oe_tcp_host,
            oe_tcp_port=oe_tcp_port,
            oe_serial_port=oe_serial,
            oe_input_range_index=oe_irng,
            wait_use_manual=self.wait_manual_check.isChecked(),
            wait_manual_seconds=self.wait_manual_spin.value(),
            integration_multiplier=5,
            oe_snap_n_samples=self.oe_snap_samples_spin.value(),
            oe_snap_duration=self.oe_snap_duration_spin.value(),
        )

        # 启动测量
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.acquisition_thread.start()

    def stop_measurement(self):
        self.acquisition_thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_display(self, data):
        # 存仪器原始值；表格/曲线按「信号缩放倍数」显示
        self.collected_data.append(data)
        self._refresh_plots_and_table_from_collected()

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def handle_error(self, error_msg):
        QMessageBox.critical(self, "错误", f"测量过程中出现错误:\n{error_msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def measurement_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.information(self, "完成", "数据采集完成！")

    def save_data(self):
        if not self.collected_data:
            QMessageBox.warning(self, "警告", "没有数据可保存！")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "保存数据", f"measurement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "Excel Files (*.xlsx)"
        )

        if filename:
            scaled_rows = [self._scaled_data_row(d) for d in self.collected_data]
            df = pd.DataFrame(scaled_rows)
            df.to_excel(filename, index=False)
            QMessageBox.information(self, "成功", f"数据已保存到:\n{filename}")


class FittingWidget(QWidget):
    """数据拟合界面"""

    def __init__(self):
        super().__init__()
        self.material_db = MaterialDatabase()
        self.sensor_db = SensorDatabase()
        self.layer_widgets = []
        self.fitting_data = None
        self.joint_datasets_ui = []
        self._calibrator = None
        self.init_ui()

    def set_calibrator(self, calibrator):
        """设置信号校正器（由主窗口传入）"""
        self._calibrator = calibrator

    def _get_calibrator(self):
        """获取已就绪的校正器"""
        if self._calibrator is not None and self._calibrator.is_calibrated:
            return self._calibrator
        return None

    def init_ui(self):
        layout = QVBoxLayout()

        # 顶部控制区
        top_layout = QHBoxLayout()

        # 数据加载
        load_group = QGroupBox("数据加载")
        load_layout = QVBoxLayout()
        self.load_btn = QPushButton("加载数据文件")
        self.load_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        load_layout.addWidget(self.load_btn)
        self.data_file_label = QLabel("未加载数据")
        load_layout.addWidget(self.data_file_label)
        load_group.setLayout(load_layout)
        top_layout.addWidget(load_group)

        # 拟合模式选择
        mode_group = QGroupBox("拟合模式")
        mode_layout = QVBoxLayout()
        self.fit_mode = QComboBox()
        self.fit_mode.addItems([
            "材料参数拟合", "传感器参数拟合", "材料+传感器联合拟合",
            "多线宽联合拟合", "两层厚度约束拟合"
        ])
        mode_layout.addWidget(self.fit_mode)
        self.fit_method = QComboBox()
        self.fit_method.addItems([
            "Amplitude", "Phase", "Real+Imaginary", "Amplitude+Phase",
            "Real Part Only", "Imaginary Part Only"
        ])
        mode_layout.addWidget(self.fit_method)

        self.apply_calibration_cb = QCheckBox("应用信号校正 (如果已标定)")
        self.apply_calibration_cb.setChecked(False)  # 默认不开启
        self.apply_calibration_cb.setToolTip(
            "选中此项后，如果在'信号校正'页加载了标定参数，将会自动对当前拟合数据应用基线扣除和相位校正。")
        mode_layout.addWidget(self.apply_calibration_cb)

        mode_group.setLayout(mode_layout)
        top_layout.addWidget(mode_group)

        # 频率筛选
        freq_group = QGroupBox("频率筛选")
        freq_layout = QGridLayout()
        freq_layout.addWidget(QLabel("最小频率 (Hz):"), 0, 0)
        self.freq_min_selector = QDoubleSpinBox()
        self.freq_min_selector.setDecimals(3)
        self.freq_min_selector.setRange(0.0, 1e8)
        self.freq_min_selector.setEnabled(False)
        freq_layout.addWidget(self.freq_min_selector, 0, 1)

        freq_layout.addWidget(QLabel("最大频率 (Hz):"), 1, 0)
        self.freq_max_selector = QDoubleSpinBox()
        self.freq_max_selector.setDecimals(3)
        self.freq_max_selector.setRange(0.0, 1e8)
        self.freq_max_selector.setEnabled(False)
        freq_layout.addWidget(self.freq_max_selector, 1, 1)

        self.freq_min_selector.valueChanged.connect(self.on_freq_min_changed)
        self.freq_max_selector.valueChanged.connect(self.on_freq_max_changed)

        freq_group.setLayout(freq_layout)
        top_layout.addWidget(freq_group)

        layout.addLayout(top_layout)

        # 参数设置区域
        self.param_tabs = QTabWidget()

        # 传感器参数标签页
        self.sensor_widget = self.create_sensor_widget()
        self.param_tabs.addTab(self.sensor_widget, "传感器参数")

        # 材料层设置标签页
        self.layers_widget = self.create_layers_widget()
        self.param_tabs.addTab(self.layers_widget, "材料层设置")

        # 拟合设置标签页
        self.fitting_settings_widget = self.create_fitting_settings()
        self.param_tabs.addTab(self.fitting_settings_widget, "拟合设置")

        # 联合拟合数据集标签页
        self.joint_datasets_widget = self.create_joint_datasets_widget()
        self.param_tabs.addTab(self.joint_datasets_widget, "联合拟合数据集")

        # 两层厚度约束设置标签页 (新功能)
        self.thickness_constraint_widget = self.create_thickness_constraint_widget()
        self.param_tabs.addTab(self.thickness_constraint_widget, "两层厚度约束")

        layout.addWidget(self.param_tabs)

        # 控制按钮
        control_layout = QHBoxLayout()
        self.fit_btn = QPushButton("开始拟合")
        self.fit_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-size: 16px; padding: 10px; }")
        self.export_btn = QPushButton("导出结果")
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-size: 16px; padding: 10px; }")
        control_layout.addWidget(self.fit_btn)
        control_layout.addWidget(self.export_btn)
        layout.addLayout(control_layout)

        # 结果显示区
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setReadOnly(True)
        layout.addWidget(QLabel("拟合结果:"))
        layout.addWidget(self.results_text)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["类别", "参数", "拟合值"])
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setMaximumHeight(220)
        layout.addWidget(QLabel("拟合结果表:"))
        layout.addWidget(self.results_table)

        self.setLayout(layout)

        # 连接信号
        self.load_btn.clicked.connect(self.load_data)
        self.fit_btn.clicked.connect(self.run_fitting)
        self.export_btn.clicked.connect(self.export_results)
        self.fit_mode.currentTextChanged.connect(self.update_fitting_mode)
        self.update_fitting_mode(self.fit_mode.currentText())

    def create_sensor_widget(self):
        widget = QWidget()
        layout = QGridLayout()

        # 传感器预设选择
        layout.addWidget(QLabel("传感器预设:"), 0, 0)
        self.sensor_preset = QComboBox()
        self.sensor_preset.addItems(self.sensor_db.get_sensor_names())
        layout.addWidget(self.sensor_preset, 0, 1, 1, 2)
        self.sensor_preset.currentTextChanged.connect(self.load_sensor_preset)

        # 传感器参数输入
        self.sensor_inputs = {}
        params = ['b', 'L', 'R', 'dR_dT', 'V_rms', 'H_e', 'rho_C_e']
        labels = ['半宽度b (m)', '长度L (m)', '电阻R (Ω)', 'dR/dT (Ω/K)',
                  'RMS电压 (V)', '加热器厚度H_e (m)', '热容rho_C_e (J/m³K)']

        for i, (param, label) in enumerate(zip(params, labels)):
            row = i + 1
            layout.addWidget(QLabel(label + ":"), row, 0)
            input_widget = QLineEdit()
            input_widget.setText("1.0e-3" if 'b' in param or 'L' in param else "1.0")
            self.sensor_inputs[param] = input_widget
            layout.addWidget(input_widget, row, 1)

            # 添加是否拟合复选框（传感器拟合 / 材料+传感器联合时显示）
            checkbox = QCheckBox("拟合")
            checkbox.setObjectName(f"sensor_{param}_fit")
            layout.addWidget(checkbox, row, 2)

            bounds_widget = QWidget()
            bounds_widget.setObjectName(f"sensor_{param}_bounds")
            bounds_layout = QHBoxLayout(bounds_widget)
            bounds_layout.setContentsMargins(0, 0, 0, 0)
            bounds_layout.addWidget(QLabel("范围:"))
            min_input = QLineEdit()
            min_input.setObjectName(f"sensor_{param}_min")
            min_input.setPlaceholderText("最小")
            min_input.setMaximumWidth(80)
            bounds_layout.addWidget(min_input)
            bounds_layout.addWidget(QLabel("-"))
            max_input = QLineEdit()
            max_input.setObjectName(f"sensor_{param}_max")
            max_input.setPlaceholderText("最大")
            max_input.setMaximumWidth(80)
            bounds_layout.addWidget(max_input)
            layout.addWidget(bounds_widget, row, 3)

        widget.setLayout(layout)
        return widget

    def create_layers_widget(self):
        widget = QWidget()
        main_layout = QVBoxLayout()

        order_hint = QLabel(
            "层顺序约定：从上到下 = 顶层 → 基底。"
            "「第 1 层」= layer_0 = 顶层（靠近 3ω 传感器），"
            "最后一层 = 基底（厚度 H 填 inf）。"
        )
        order_hint.setWordWrap(True)
        order_hint.setStyleSheet(
            "color:#1565C0; font-size:11px; padding:4px; "
            "background:#E3F2FD; border:1px solid #90CAF9;"
        )
        main_layout.addWidget(order_hint)

        # 层数控制
        layer_control = QHBoxLayout()
        layer_control.addWidget(QLabel("材料层数:"))
        self.layer_count = QSpinBox()
        self.layer_count.setRange(1, 10)
        self.layer_count.setValue(2)
        layer_control.addWidget(self.layer_count)
        self.add_layer_btn = QPushButton("添加层")
        self.add_layer_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        layer_control.addWidget(self.add_layer_btn)
        self.remove_layer_btn = QPushButton("删除层")
        self.remove_layer_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        layer_control.addWidget(self.remove_layer_btn)
        layer_control.addStretch()
        main_layout.addLayout(layer_control)

        # 层参数滚动区域
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.layers_layout = QVBoxLayout(scroll_widget)

        # 初始化两层
        for i in range(2):
            self.add_layer_widget(i)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        widget.setLayout(main_layout)

        # 连接信号
        self.add_layer_btn.clicked.connect(self.add_layer)
        self.remove_layer_btn.clicked.connect(self.remove_layer)

        return widget

    def add_layer_widget(self, layer_index):
        """添加单层材料设置界面"""
        layer_group = QGroupBox(f"第 {layer_index + 1} 层")
        layout = QGridLayout()

        # 材料预设选择
        layout.addWidget(QLabel("材料预设:"), 0, 0)
        material_combo = QComboBox()
        material_combo.addItems(self.material_db.get_material_names())
        material_combo.setObjectName(f"layer_{layer_index}_preset")
        layout.addWidget(material_combo, 0, 1, 1, 3)

        # 参数输入
        params = ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']
        labels = ['热导率 k_y (W/mK)', '密度 ρ (kg/m³)', '比热容 c_p (J/kg·K)',
                  '厚度 H (m)', '各向异性因子 ψ', '界面热阻 Rc (K·m²/W)']
        units = ['W/mK', 'kg/m³', 'J/kg·K', 'm', '', 'K·m²/W']

        inputs = {}
        for i, (param, label, unit) in enumerate(zip(params, labels, units)):
            row = i + 1
            layout.addWidget(QLabel(label + ":"), row, 0)

            # 参数输入框
            input_widget = QLineEdit()
            input_widget.setObjectName(f"layer_{layer_index}_{param}")
            inputs[param] = input_widget
            layout.addWidget(input_widget, row, 1)

            # 是否拟合复选框
            fit_checkbox = QCheckBox("拟合")
            fit_checkbox.setObjectName(f"layer_{layer_index}_{param}_fit")
            layout.addWidget(fit_checkbox, row, 2)

            # 参数边界设置
            bounds_layout = QHBoxLayout()
            bounds_layout.addWidget(QLabel("范围:"))

            min_input = QLineEdit()
            min_input.setObjectName(f"layer_{layer_index}_{param}_min")
            min_input.setPlaceholderText("最小值")
            min_input.setMaximumWidth(80)
            bounds_layout.addWidget(min_input)

            bounds_layout.addWidget(QLabel("-"))

            max_input = QLineEdit()
            max_input.setObjectName(f"layer_{layer_index}_{param}_max")
            max_input.setPlaceholderText("最大值")
            max_input.setMaximumWidth(80)
            bounds_layout.addWidget(max_input)

            layout.addLayout(bounds_layout, row, 3)

        # 连接预设选择信号
        material_combo.currentTextChanged.connect(
            lambda text, idx=layer_index: self.load_material_preset(idx, text)
        )

        layer_group.setLayout(layout)
        self.layers_layout.addWidget(layer_group)
        self.layer_widgets.append({'group': layer_group, 'inputs': inputs})

        # 加载默认材料
        material_combo.setCurrentIndex(0)

    def load_material_preset(self, layer_index, material_name):
        """加载材料预设值"""
        material = self.material_db.get_material(material_name)

        for param, value in material.items():
            input_name = f"layer_{layer_index}_{param}"
            input_widget = self.findChild(QLineEdit, input_name)
            if input_widget:
                if param == 'H' and np.isinf(value):
                    input_widget.setText("inf")
                else:
                    input_widget.setText(f"{value:.3e}")

    def load_sensor_preset(self, sensor_name):
        """加载传感器预设值"""
        sensor = self.sensor_db.get_sensor(sensor_name)
        for param, value in sensor.items():
            if param in self.sensor_inputs:
                self.sensor_inputs[param].setText(f"{value:.3e}")

    def add_layer(self):
        layer_count = len(self.layer_widgets)
        self.add_layer_widget(layer_count)
        self.layer_count.setValue(layer_count + 1)
        self._refresh_tc_layer_combos()

    def remove_layer(self):
        if len(self.layer_widgets) > 1:
            widget_info = self.layer_widgets.pop()
            widget_info['group'].deleteLater()
            self.layer_count.setValue(len(self.layer_widgets))
            self._refresh_tc_layer_combos()

    def create_fitting_settings(self):
        widget = QWidget()
        layout = QGridLayout()

        # 迭代设置
        layout.addWidget(QLabel("最大迭代次数:"), 0, 0)
        self.max_iter = QSpinBox()
        self.max_iter.setRange(1, 100)
        self.max_iter.setValue(30)
        layout.addWidget(self.max_iter, 0, 1)

        layout.addWidget(QLabel("收敛容差:"), 1, 0)
        self.tolerance = QLineEdit("1e-6")
        layout.addWidget(self.tolerance, 1, 1)

        layout.addWidget(QLabel("阻尼因子 ζ:"), 2, 0)
        self.zeta = QDoubleSpinBox()
        self.zeta.setRange(0.1, 1.0)
        self.zeta.setValue(0.5)
        self.zeta.setSingleStep(0.1)
        layout.addWidget(self.zeta, 2, 1)

        # 添加空白
        layout.setRowStretch(3, 1)

        widget.setLayout(layout)
        return widget

    def create_joint_datasets_widget(self):
        """创建联合拟合数据集管理界面"""
        widget = QWidget()
        main_layout = QVBoxLayout()

        info_label = QLabel(
            "多线宽联合拟合：加载多个不同线宽传感器的数据，共享材料参数同时拟合。\n"
            "每个数据集需指定各自的传感器参数和数据文件。"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #1565C0; font-size: 12px; padding: 5px;")
        main_layout.addWidget(info_label)

        btn_layout = QHBoxLayout()
        self.add_dataset_btn = QPushButton("添加数据集")
        self.add_dataset_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        self.add_dataset_btn.clicked.connect(self.add_joint_dataset)
        btn_layout.addWidget(self.add_dataset_btn)

        self.remove_dataset_btn = QPushButton("删除最后一个数据集")
        self.remove_dataset_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.remove_dataset_btn.clicked.connect(self.remove_joint_dataset)
        btn_layout.addWidget(self.remove_dataset_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.joint_datasets_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        widget.setLayout(main_layout)
        return widget

    def add_joint_dataset(self):
        """添加一个联合拟合数据集条目"""
        ds_index = len(self.joint_datasets_ui)
        group = QGroupBox(f"数据集 {ds_index + 1}")
        layout = QGridLayout()

        layout.addWidget(QLabel("传感器预设:"), 0, 0)
        sensor_combo = QComboBox()
        sensor_combo.addItems(self.sensor_db.get_sensor_names())
        layout.addWidget(sensor_combo, 0, 1, 1, 2)

        sensor_inputs = {}
        params = ['b', 'L', 'R', 'dR_dT', 'V_rms', 'H_e', 'rho_C_e']
        labels = ['半宽度b (m)', '长度L (m)', '电阻R (Ω)', 'dR/dT (Ω/K)',
                  'RMS电压 (V)', '加热器厚度H_e (m)', '热容rho_C_e (J/m³K)']

        for i, (param, label) in enumerate(zip(params, labels)):
            row = i + 1
            layout.addWidget(QLabel(label + ":"), row, 0)
            inp = QLineEdit()
            sensor_inputs[param] = inp
            layout.addWidget(inp, row, 1, 1, 2)

        def load_preset(name, inputs=sensor_inputs):
            sensor = self.sensor_db.get_sensor(name)
            for p, v in sensor.items():
                if p in inputs:
                    inputs[p].setText(f"{v:.3e}")

        sensor_combo.currentTextChanged.connect(load_preset)
        sensor_combo.setCurrentIndex(0)

        row_data = len(params) + 1
        layout.addWidget(QLabel("数据文件:"), row_data, 0)
        file_label = QLabel("未加载")
        layout.addWidget(file_label, row_data, 1)
        load_btn = QPushButton("加载")
        load_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        layout.addWidget(load_btn, row_data, 2)

        ds_info = {
            'group': group,
            'sensor_combo': sensor_combo,
            'sensor_inputs': sensor_inputs,
            'file_label': file_label,
            'data': None,
        }

        def on_load(checked=False, info=ds_info):
            filename, _ = QFileDialog.getOpenFileName(
                self, "加载数据文件", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
            )
            if filename:
                try:
                    if filename.endswith('.xlsx'):
                        df = pd.read_excel(filename)
                    else:
                        df = pd.read_csv(filename)
                    info['data'] = df
                    info['file_label'].setText(os.path.basename(filename))
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"加载失败:\n{str(e)}")

        load_btn.clicked.connect(on_load)

        group.setLayout(layout)
        self.joint_datasets_layout.addWidget(group)
        self.joint_datasets_ui.append(ds_info)

    def remove_joint_dataset(self):
        """删除最后一个联合拟合数据集"""
        if self.joint_datasets_ui:
            ds_info = self.joint_datasets_ui.pop()
            ds_info['group'].deleteLater()

    # ------------------------------------------------------------------ #
    # 新增：两层厚度约束拟合 UI 配置
    # ------------------------------------------------------------------ #
    def create_thickness_constraint_widget(self):
        """两层厚度约束拟合的参数面板（H_total 已知，反推 H_top / H_bot）"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        tip = QLabel(
            "适用场景：相邻两层样品<b>总厚度 H_total 已知</b>，但每层各自厚度未知。\n"
            "拟合时引入归一化变量 f = H_top/H_total ∈ (0,1)，自动满足总厚度约束。\n"
            "可在「材料层设置」和「传感器参数」中勾选其它要联合拟合的参数（如 k_y、Rc、R、dR_dT 等）。\n"
            "<b>层索引约定：layer_0 = 顶层（最靠近 3ω 传感器），索引越大越靠近基底，"
            "最后一层 = 基底（厚度填 inf）。「材料层设置」里的「第 1 层」就是 layer_0，"
            "依此类推。</b>"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#1565C0; font-size:12px; padding:6px; "
                          "background:#E3F2FD; border:1px solid #90CAF9;")
        layout.addWidget(tip)

        grp = QGroupBox("厚度耦合层选择 (layer_0 = 顶层；最大索引 = 基底)")
        gl = QGridLayout()

        gl.addWidget(QLabel("上层 (更靠近表面):"), 0, 0)
        self.tc_top_combo = QComboBox()
        self.tc_top_combo.setToolTip(
            "GUI 顺序索引：layer_0 是顶层；索引越大越靠近基底。\n"
            "这里选你想拆分的两层中『更靠上/更靠近表面』的那一层。"
        )
        gl.addWidget(self.tc_top_combo, 0, 1)

        gl.addWidget(QLabel("下层 (更靠近基底):"), 0, 2)
        self.tc_bot_combo = QComboBox()
        self.tc_bot_combo.setToolTip(
            "自动列出与「上层」相邻的候选层（通常就是 上层索引 + 1）。\n"
            "拟合输出的 H_bot 对应这一层。"
        )
        gl.addWidget(self.tc_bot_combo, 0, 3)

        gl.addWidget(QLabel("H_total (m):"), 1, 0)
        self.tc_H_total = QLineEdit("4e-6")
        self.tc_H_total.setToolTip("两层总厚度，单位：米。例如 4 μm 填 4e-6")
        gl.addWidget(self.tc_H_total, 1, 1)

        gl.addWidget(QLabel("f 初值:"), 1, 2)
        self.tc_f_init = QDoubleSpinBox()
        self.tc_f_init.setRange(0.001, 0.999)
        self.tc_f_init.setSingleStep(0.05)
        self.tc_f_init.setDecimals(3)
        self.tc_f_init.setValue(0.5)
        gl.addWidget(self.tc_f_init, 1, 3)

        gl.addWidget(QLabel("f 下限:"), 2, 0)
        self.tc_f_lo = QDoubleSpinBox()
        self.tc_f_lo.setRange(0.001, 0.999)
        self.tc_f_lo.setDecimals(3)
        self.tc_f_lo.setValue(0.05)
        gl.addWidget(self.tc_f_lo, 2, 1)
        gl.addWidget(QLabel("f 上限:"), 2, 2)
        self.tc_f_hi = QDoubleSpinBox()
        self.tc_f_hi.setRange(0.001, 0.999)
        self.tc_f_hi.setDecimals(3)
        self.tc_f_hi.setValue(0.95)
        gl.addWidget(self.tc_f_hi, 2, 3)

        gl.addWidget(QLabel("最大迭代:"), 3, 0)
        self.tc_max_iter = QSpinBox()
        self.tc_max_iter.setRange(20, 5000)
        self.tc_max_iter.setValue(400)
        gl.addWidget(self.tc_max_iter, 3, 1)

        self.tc_scan_f_cb = QCheckBox("拟合前扫描 cost(f) 曲线（可视化解唯一性）")
        self.tc_scan_f_cb.setChecked(True)
        gl.addWidget(self.tc_scan_f_cb, 3, 2, 1, 2)

        grp.setLayout(gl)
        layout.addWidget(grp)

        warn = QLabel(
            "提示：在此模式下，<b>请不要</b>勾选「材料层设置」里这两层的 H 作为拟合参数，"
            "厚度由 H_total + f 完全约束。其它参数 (k_y/Rc/rho/cap/psi) 和传感器参数可正常勾选。"
        )
        warn.setWordWrap(True)
        warn.setStyleSheet("color:#B71C1C; font-size:11px; padding:4px;")
        layout.addWidget(warn)

        layout.addStretch()

        # 初始化下拉
        self._refresh_tc_layer_combos()
        self.tc_top_combo.currentIndexChanged.connect(self._sync_tc_bot_combo)

        return widget

    def _refresh_tc_layer_combos(self):
        """根据当前层数刷新两层厚度约束的层下拉框"""
        if not hasattr(self, 'tc_top_combo'):
            return
        n = len(self.layer_widgets)
        cur = self.tc_top_combo.currentIndex()
        self.tc_top_combo.blockSignals(True)
        self.tc_top_combo.clear()
        for i in range(n):
            self.tc_top_combo.addItem(f"layer_{i}")
        if 0 <= cur < n:
            self.tc_top_combo.setCurrentIndex(cur)
        else:
            self.tc_top_combo.setCurrentIndex(min(1, max(0, n - 2)))
        self.tc_top_combo.blockSignals(False)
        self._sync_tc_bot_combo()

    def _sync_tc_bot_combo(self):
        if not hasattr(self, 'tc_bot_combo') or not hasattr(self, 'tc_top_combo'):
            return
        n = len(self.layer_widgets)
        top = self.tc_top_combo.currentIndex()
        candidates = []
        if top - 1 >= 0:
            candidates.append(top - 1)
        if top + 1 < n:
            candidates.append(top + 1)
        self.tc_bot_combo.blockSignals(True)
        self.tc_bot_combo.clear()
        for c in candidates:
            self.tc_bot_combo.addItem(f"layer_{c}", userData=c)
        self.tc_bot_combo.blockSignals(False)

    def update_fitting_mode(self, mode):
        """更新拟合模式相关的UI"""
        is_sensor_only = mode == "传感器参数拟合"
        is_combined = mode == "材料+传感器联合拟合"
        is_joint_mode = mode == "多线宽联合拟合"
        is_thickness_mode = mode == "两层厚度约束拟合"

        # 厚度约束模式下也允许同时勾选传感器参数（可选）
        show_sensor_fit = is_sensor_only or is_combined or is_thickness_mode
        for param in self.sensor_inputs.keys():
            checkbox = self.sensor_widget.findChild(QCheckBox, f"sensor_{param}_fit")
            if checkbox:
                checkbox.setVisible(show_sensor_fit)
                checkbox.setChecked(False)
            bw = self.sensor_widget.findChild(QWidget, f"sensor_{param}_bounds")
            if bw:
                bw.setVisible(show_sensor_fit)
            if not show_sensor_fit:
                for suffix in ('_min', '_max'):
                    w = self.sensor_widget.findChild(QLineEdit, f"sensor_{param}{suffix}")
                    if w:
                        w.clear()

        # 厚度约束模式下显示材料层拟合框（可选 k_y、Rc 等同时拟合）
        show_material_fit = (not is_sensor_only) or is_combined or is_thickness_mode
        for layer_idx in range(len(self.layer_widgets)):
            for param in ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']:
                checkbox = self.findChild(QCheckBox, f"layer_{layer_idx}_{param}_fit")
                if checkbox:
                    checkbox.setVisible(show_material_fit)
                    checkbox.setChecked(False)

        if is_joint_mode:
            for i in range(self.param_tabs.count()):
                if self.param_tabs.tabText(i) == "联合拟合数据集":
                    self.param_tabs.setCurrentIndex(i)
                    break

        if is_thickness_mode:
            for i in range(self.param_tabs.count()):
                if self.param_tabs.tabText(i) == "两层厚度约束":
                    self.param_tabs.setCurrentIndex(i)
                    break
            self._refresh_tc_layer_combos()

    def initialize_frequency_filters(self, freqs):
        """根据导入数据初始化频率筛选范围"""
        if freqs.size == 0:
            self.freq_min_selector.setEnabled(False)
            self.freq_max_selector.setEnabled(False)
            return

        min_freq = float(np.min(freqs))
        max_freq = float(np.max(freqs))
        if min_freq == max_freq:
            max_freq += 1e-9  # 防止范围为零导致报错

        for widget in (self.freq_min_selector, self.freq_max_selector):
            widget.blockSignals(True)

        self.freq_min_selector.setRange(min_freq, max_freq)
        self.freq_max_selector.setRange(min_freq, max_freq)
        self.freq_min_selector.setValue(min_freq)
        self.freq_max_selector.setValue(max_freq)
        self.freq_min_selector.setEnabled(True)
        self.freq_max_selector.setEnabled(True)

        for widget in (self.freq_min_selector, self.freq_max_selector):
            widget.blockSignals(False)

    def on_freq_min_changed(self, value):
        if self.freq_max_selector.isEnabled() and value > self.freq_max_selector.value():
            self.freq_max_selector.blockSignals(True)
            self.freq_max_selector.setValue(value)
            self.freq_max_selector.blockSignals(False)

    def on_freq_max_changed(self, value):
        if self.freq_min_selector.isEnabled() and value < self.freq_min_selector.value():
            self.freq_min_selector.blockSignals(True)
            self.freq_min_selector.setValue(value)
            self.freq_min_selector.blockSignals(False)

    def _append_result_row(self, category, name, value):
        """在结果表中追加一行"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(category))
        self.results_table.setItem(row, 1, QTableWidgetItem(name))
        display_val = self._format_result_value(value)
        self.results_table.setItem(row, 2, QTableWidgetItem(display_val))

    def _format_result_value(self, value):
        if isinstance(value, str):
            return value
        if isinstance(value, (float, int, np.floating, np.integer)):
            if np.isinf(value):
                return "inf"
            return f"{value:.6e}"
        return str(value)

    def update_results_table(self, fitted_result, result_kind):
        """刷新拟合结果表格。result_kind: 'sensor' | 'material' | 'combined'"""
        self.results_table.setRowCount(0)
        if fitted_result is None:
            return

        if result_kind == 'sensor':
            for param, value in fitted_result.items():
                self._append_result_row("传感器参数", param, value)
        elif result_kind == 'combined':
            for param, value in fitted_result['sensor_params'].items():
                self._append_result_row("传感器参数", param, value)
            for idx, layer in enumerate(fitted_result['layers']):
                for param, value in layer.items():
                    self._append_result_row(f"第 {idx + 1} 层", param, value)
        else:
            for idx, layer in enumerate(fitted_result):
                for param, value in layer.items():
                    self._append_result_row(f"第 {idx + 1} 层", param, value)

    def load_data(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载数据文件", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )

        if filename:
            try:
                if filename.endswith('.xlsx'):
                    self.fitting_data = pd.read_excel(filename)
                else:
                    self.fitting_data = pd.read_csv(filename)

                self.data_file_label.setText(f"已加载: {os.path.basename(filename)}")
                freqs = self.fitting_data.iloc[:, 0].values.astype(float)
                self.initialize_frequency_filters(freqs)
                QMessageBox.information(self, "成功", f"数据加载成功!\n共 {len(self.fitting_data)} 个数据点")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载数据失败:\n{str(e)}")

    def _parse_v3_from_dataframe(self, df):
        """从 DataFrame 中提取 V3 复数数据"""
        if 'v3_real' in df.columns and 'v3_imag' in df.columns:
            real_vals = df['v3_real'].values.astype(float)
            imag_vals = df['v3_imag'].values.astype(float)
        elif 'V3_Real' in df.columns:
            real_vals = df['V3_Real'].values.astype(float)
            imag_vals = df['V3_Imag'].values.astype(float)
        else:
            real_vals = df.iloc[:, 1].values.astype(float)
            imag_vals = df.iloc[:, 2].values.astype(float)
        return real_vals + 1j * imag_vals

    def run_joint_fitting(self):
        """执行多线宽联合拟合"""
        # 每次拟合前清除上次的拟合结果引用，防止失败时显示旧数据
        if hasattr(self, 'fitter'):
            del self.fitter
        if hasattr(self, 'fitted_result'):
            del self.fitted_result

        if not self.joint_datasets_ui:
            QMessageBox.warning(self, "警告", "请先添加至少一个数据集！")
            return

        for i, ds_info in enumerate(self.joint_datasets_ui):
            if ds_info['data'] is None:
                QMessageBox.warning(self, "警告", f"数据集 {i + 1} 尚未加载数据文件！")
                return

        try:
            self.results_table.setRowCount(0)
            self.results_text.clear()

            # 准备所有数据集
            datasets = []
            calibrator = self._get_calibrator()
            apply_cal = self.apply_calibration_cb.isChecked() and calibrator is not None and calibrator.is_calibrated

            for ds_info in self.joint_datasets_ui:
                df = ds_info['data']
                freqs = df.iloc[:, 0].values.astype(float)
                V3_exp = self._parse_v3_from_dataframe(df)

                if apply_cal:
                    V3_exp = calibrator.correct_v3omega(freqs, V3_exp)

                sp = {}
                for param, inp in ds_info['sensor_inputs'].items():
                    sp[param] = float(inp.text())

                datasets.append({
                    'freqs': freqs,
                    'V3_exp': V3_exp,
                    'b': sp['b'],
                    'sensor_params': sp,
                })

            # 获取共享材料层参数
            layers = []
            for i in range(len(self.layer_widgets)):
                layer = {}
                for param in ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']:
                    input_widget = self.findChild(QLineEdit, f"layer_{i}_{param}")
                    if input_widget:
                        text = input_widget.text()
                        if text.lower() == 'inf':
                            layer[param] = np.inf
                        else:
                            layer[param] = float(text)
                layers.append(layer)

            # 获取拟合参数
            params_to_fit = {}
            param_bounds = {}
            for i in range(len(self.layer_widgets)):
                layer_params_to_fit = []
                layer_bounds = {}
                for param in ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']:
                    checkbox = self.findChild(QCheckBox, f"layer_{i}_{param}_fit")
                    if checkbox and checkbox.isChecked():
                        layer_params_to_fit.append(param)
                        min_input = self.findChild(QLineEdit, f"layer_{i}_{param}_min")
                        max_input = self.findChild(QLineEdit, f"layer_{i}_{param}_max")
                        if min_input and max_input and min_input.text() and max_input.text():
                            layer_bounds[param] = (float(min_input.text()), float(max_input.text()))
                        else:
                            if param == 'k_y':
                                layer_bounds[param] = (0.01, 5000)
                            elif param == 'rho':
                                layer_bounds[param] = (100, 20000)
                            elif param == 'cap':
                                layer_bounds[param] = (100, 5000)
                            elif param == 'Rc':
                                layer_bounds[param] = (0, 1e-3)
                            elif param == 'psi':
                                layer_bounds[param] = (0.01, 100)
                            else:
                                current_val = layers[i][param]
                                layer_bounds[param] = (current_val * 0.1, current_val * 10)
                if layer_params_to_fit:
                    params_to_fit[f'layer_{i}'] = layer_params_to_fit
                    param_bounds[f'layer_{i}'] = layer_bounds

            if not params_to_fit:
                QMessageBox.warning(self, "警告", "请在材料层设置中选择至少一个参数进行拟合！")
                return

            fit_method_map = {
                "Amplitude": "amp", "Phase": "phase",
                "Real+Imaginary": "both", "Amplitude+Phase": "both",
                "Real Part Only": "real", "Imaginary Part Only": "imag"
            }
            fit_method = fit_method_map[self.fit_method.currentText()]

            # 层顺序反转（GUI 是 [顶层→基底]，模型是 [基底→顶层]）
            N = len(layers)
            fitter_initial_layers = [l.copy() for l in layers[::-1]]

            fitter_params_to_fit = {}
            fitter_param_bounds = {}
            for key, val in params_to_fit.items():
                if key.startswith('layer_'):
                    gui_idx = int(key.split('_')[1])
                    model_idx = N - 1 - gui_idx
                    fitter_params_to_fit[f'layer_{model_idx}'] = val
            for key, val in param_bounds.items():
                if key.startswith('layer_'):
                    gui_idx = int(key.split('_')[1])
                    model_idx = N - 1 - gui_idx
                    fitter_param_bounds[f'layer_{model_idx}'] = val

            fitter = JointThermalModelFitter(
                datasets=datasets,
                initial_layers=fitter_initial_layers,
                params_to_fit=fitter_params_to_fit,
                fit_mode=fit_method,
                param_bounds=fitter_param_bounds,
            )

            original_stdout = sys.stdout
            sys.stdout = EmittingStream()

            def append_text(text):
                cursor = self.results_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.insertText(text)
                self.results_text.setTextCursor(cursor)
                self.results_text.ensureCursorVisible()
                QApplication.processEvents()

            sys.stdout.textWritten.connect(append_text)

            try:
                self.results_text.append("开始多线宽联合拟合...\n")
                QApplication.processEvents()

                fitted_result_raw = fitter.fit(
                    max_iter=self.max_iter.value(),
                    tol=float(self.tolerance.text()),
                    zeta=self.zeta.value()
                )

                fitted_result = fitted_result_raw[::-1]

                self.results_text.append("\n联合拟合完成！\n")
                self.results_text.append("拟合的材料参数:")
                for i, layer in enumerate(fitted_result):
                    self.results_text.append(f"\n第 {i} 层:")
                    for param, value in layer.items():
                        self.results_text.append(f"  {param}: {value:.6e}")

                self.update_results_table(fitted_result, 'material')
                self.fitted_result = fitted_result
                self.fitter = fitter
            finally:
                sys.stdout = original_stdout

        except Exception as e:
            self.results_text.append(f"\n联合拟合失败: {str(e)}\n")
            traceback.print_exc()
            if hasattr(self, 'ax1'):
                self.ax1.clear()
            if hasattr(self, 'ax2'):
                self.ax2.clear()
            if hasattr(self, 'figure') and hasattr(self.figure, 'canvas'):
                self.figure.canvas.draw()

    # ------------------------------------------------------------------ #
    # 新增：两层厚度约束拟合执行流程
    # ------------------------------------------------------------------ #
    def run_two_layer_thickness_fitting(self):
        """两层厚度约束拟合：H_total 已知 -> 反推 H_top / H_bot；可同时拟合传感器和其它材料参数"""
        if hasattr(self, 'fitter'):
            del self.fitter
        if hasattr(self, 'fitted_result'):
            del self.fitted_result

        if self.fitting_data is None:
            QMessageBox.warning(self, "警告", "请先加载数据文件！")
            return

        try:
            self.results_table.setRowCount(0)
            self.results_text.clear()

            # --- 1. 数据 ---
            freqs = self.fitting_data.iloc[:, 0].values.astype(float)
            use_filter = (self.freq_min_selector.isEnabled()
                          and self.freq_max_selector.isEnabled())
            if use_filter:
                mask = ((freqs >= self.freq_min_selector.value()) &
                        (freqs <= self.freq_max_selector.value()))
            else:
                mask = np.ones(freqs.shape, dtype=bool)
            if not np.any(mask):
                QMessageBox.warning(self, "警告", "选定频率区间没有数据点！")
                return
            freqs = freqs[mask]

            if ('v3_real' in self.fitting_data.columns
                    and 'v3_imag' in self.fitting_data.columns):
                real_vals = self.fitting_data['v3_real'].values.astype(float)
                imag_vals = self.fitting_data['v3_imag'].values.astype(float)
            elif 'V3_Real' in self.fitting_data.columns:
                real_vals = self.fitting_data['V3_Real'].values.astype(float)
                imag_vals = self.fitting_data['V3_Imag'].values.astype(float)
            else:
                real_vals = self.fitting_data.iloc[:, 1].values.astype(float)
                imag_vals = self.fitting_data.iloc[:, 2].values.astype(float)
            V3_exp = real_vals[mask] + 1j * imag_vals[mask]

            # 信号校正
            calibrator = self._get_calibrator()
            if (self.apply_calibration_cb.isChecked()
                    and calibrator is not None and calibrator.is_calibrated):
                V3_exp = calibrator.correct_v3omega(freqs, V3_exp)
                self.results_text.append("✓ 已应用信号通路校正\n")

            # --- 2. 传感器与材料层参数 ---
            sensor_params = {}
            for param, w in self.sensor_inputs.items():
                sensor_params[param] = float(w.text())

            layers = []
            for i in range(len(self.layer_widgets)):
                layer = {}
                for param in ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']:
                    inp = self.findChild(QLineEdit, f"layer_{i}_{param}")
                    if inp:
                        t = inp.text()
                        layer[param] = np.inf if t.lower() == 'inf' else float(t)
                layers.append(layer)

            # --- 3. 厚度约束参数 ---
            top_idx = self.tc_top_combo.currentIndex()
            bot_idx = self.tc_bot_combo.currentData()
            if bot_idx is None:
                QMessageBox.warning(self, "警告", "请正确设置下层索引（必须与顶层相邻）")
                return
            if abs(top_idx - bot_idx) != 1:
                QMessageBox.warning(self, "警告", "顶层与下层索引必须相邻")
                return

            try:
                H_total = float(self.tc_H_total.text())
            except ValueError:
                QMessageBox.warning(self, "警告", "H_total 必须是数字（米）")
                return
            if H_total <= 0:
                QMessageBox.warning(self, "警告", "H_total 必须为正数")
                return
            f_init = self.tc_f_init.value()
            f_lo = self.tc_f_lo.value()
            f_hi = self.tc_f_hi.value()
            if not (0 < f_lo < f_hi < 1):
                QMessageBox.warning(self, "警告", "f 边界必须满足 0 < lo < hi < 1")
                return

            # --- 4. 收集其它要联合拟合的参数 (材料 + 传感器) ---
            extra_params_to_fit = {}
            extra_param_bounds = {}
            for i in range(len(self.layer_widgets)):
                layer_params_to_fit = []
                layer_bounds = {}
                for param in ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']:
                    cb = self.findChild(QCheckBox, f"layer_{i}_{param}_fit")
                    if cb and cb.isChecked():
                        if param == 'H' and i in (top_idx, bot_idx):
                            QMessageBox.warning(
                                self, "警告",
                                f"layer_{i} 是厚度耦合层，请不要勾选 H 作为拟合参数。"
                                "厚度已由 H_total + f 完全约束。"
                            )
                            return
                        if param == 'H':
                            QMessageBox.warning(
                                self, "警告",
                                f"layer_{i} 的 H 不能与厚度约束拟合同时启用（请取消勾选）。"
                            )
                            return
                        layer_params_to_fit.append(param)
                        min_input = self.findChild(QLineEdit, f"layer_{i}_{param}_min")
                        max_input = self.findChild(QLineEdit, f"layer_{i}_{param}_max")
                        if (min_input and max_input
                                and min_input.text().strip()
                                and max_input.text().strip()):
                            layer_bounds[param] = (
                                float(min_input.text()),
                                float(max_input.text()),
                            )
                if layer_params_to_fit:
                    extra_params_to_fit[f'layer_{i}'] = layer_params_to_fit
                    if layer_bounds:
                        extra_param_bounds[f'layer_{i}'] = layer_bounds

            # 传感器
            sensor_params_to_fit = []
            sensor_bounds_dict = {}
            for param in self.sensor_inputs.keys():
                cb = self.sensor_widget.findChild(QCheckBox, f"sensor_{param}_fit")
                if cb and cb.isChecked():
                    sensor_params_to_fit.append(param)
                    min_input = self.sensor_widget.findChild(
                        QLineEdit, f"sensor_{param}_min")
                    max_input = self.sensor_widget.findChild(
                        QLineEdit, f"sensor_{param}_max")
                    if (min_input and max_input
                            and min_input.text().strip()
                            and max_input.text().strip()):
                        sensor_bounds_dict[param] = (
                            float(min_input.text()),
                            float(max_input.text()),
                        )
            if sensor_params_to_fit:
                extra_params_to_fit['sensor'] = sensor_params_to_fit
                if sensor_bounds_dict:
                    extra_param_bounds['sensor'] = sensor_bounds_dict

            # --- 5. 拟合方法 ---
            fit_method_map = {
                "Amplitude": "amp", "Phase": "phase",
                "Real+Imaginary": "both", "Amplitude+Phase": "both",
                "Real Part Only": "real", "Imaginary Part Only": "imag",
            }
            fit_method = fit_method_map[self.fit_method.currentText()]

            # --- 6. 启动拟合 ---
            fitter = TwoLayerThicknessFitter(
                freqs=freqs, V3_exp=V3_exp,
                b=sensor_params['b'], sensor_params=sensor_params,
                initial_layers=layers,
                layer_top_idx=top_idx, layer_bot_idx=bot_idx,
                H_total=H_total,
                extra_params_to_fit=extra_params_to_fit,
                extra_param_bounds=extra_param_bounds,
                fit_mode=fit_method,
                f_init=f_init,
                f_bounds=(f_lo, f_hi),
            )

            self.results_text.append(
                f"开始两层厚度约束拟合\n"
                f"  H_total = {H_total*1e6:.4f} μm\n"
                f"  layer_top = {top_idx}, layer_bot = {bot_idx}\n"
                f"  额外拟合参数: {extra_params_to_fit if extra_params_to_fit else '无'}\n"
            )
            QApplication.processEvents()

            # 可选 cost(f) 扫描，画在结果可视化页面（保存到 fitter 上供外部读取）
            f_scan = None
            if self.tc_scan_f_cb.isChecked():
                self.results_text.append("正在扫描 cost(f) ...")
                QApplication.processEvents()
                f_scan = fitter.scan_f(n=81)
                self.results_text.append("扫描完成。\n")

            # 把 print 重定向到结果框
            original_stdout = sys.stdout
            sys.stdout = EmittingStream()

            def append_text(text):
                cursor = self.results_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.insertText(text)
                self.results_text.setTextCursor(cursor)
                self.results_text.ensureCursorVisible()
                QApplication.processEvents()
            sys.stdout.textWritten.connect(append_text)

            try:
                result = fitter.fit(
                    max_iter=self.tc_max_iter.value(),
                    verbose=1,
                )
            finally:
                sys.stdout = original_stdout

            # --- 7. 整理结果到表格 ---
            self.results_table.setRowCount(0)
            self._append_result_row("厚度约束", "H_total (m)", result.H_total)
            self._append_result_row("厚度约束", "f = H_top/H_total", result.f_fraction)
            self._append_result_row("厚度约束",
                                    f"H_top  (layer_{top_idx}) [m]", result.H_top)
            self._append_result_row("厚度约束",
                                    f"H_bot  (layer_{bot_idx}) [m]", result.H_bot)

            if 'sensor' in result.extra_params:
                for param, val in result.extra_params['sensor'].items():
                    self._append_result_row("传感器参数", param, val)
            for key, props in result.extra_params.items():
                if key == 'sensor':
                    continue
                idx = int(key.split('_', 1)[1])
                for prop, val in props.items():
                    self._append_result_row(f"第 {idx + 1} 层 (GUI layer_{idx})",
                                            prop, val)

            self.results_text.append(
                "\n拟合完成！\n"
                f"  H_top = {result.H_top*1e6:.4f} μm\n"
                f"  H_bot = {result.H_bot*1e6:.4f} μm\n"
                f"  f     = {result.f_fraction:.6f}\n"
                f"  cost  = {result.cost:.4e}, n_iter = {result.n_iter}\n"
            )

            # 保存为内部状态，方便结果可视化页绘图与导出
            self.fitted_result = result
            self.fitter = fitter
            # 给可视化页/导出用一些“通用接口”一致的属性
            fitter.freqs = freqs
            fitter.V3_exp = V3_exp
            fitter.b = sensor_params['b']
            fitter.sensor_params = result_sensor_for_plot = dict(fitter.sensor_params)
            fitter.layers = [dict(l) for l in fitter.layers]
            fitter.f_scan = f_scan
            fitter.is_two_layer_thickness = True

        except Exception as e:
            self.results_text.append(f"\n两层厚度约束拟合失败: {e}\n")
            traceback.print_exc()

    def run_fitting(self):
        """执行拟合"""
        # 每次拟合前清除上次的拟合结果引用，防止失败时显示旧数据
        if hasattr(self, 'fitter'):
            del self.fitter
        if hasattr(self, 'fitted_result'):
            del self.fitted_result

        if self.fit_mode.currentText() == "多线宽联合拟合":
            self.run_joint_fitting()
            return

        if self.fit_mode.currentText() == "两层厚度约束拟合":
            self.run_two_layer_thickness_fitting()
            return

        if self.fitting_data is None:
            QMessageBox.warning(self, "警告", "请先加载数据文件！")
            return

        try:
            self.results_table.setRowCount(0)
            self.results_text.clear()

            # 准备数据
            freqs = self.fitting_data.iloc[:, 0].values.astype(float)  # 假设第一列是频率

            use_filter = self.freq_min_selector.isEnabled() and self.freq_max_selector.isEnabled()
            if use_filter:
                min_freq = self.freq_min_selector.value()
                max_freq = self.freq_max_selector.value()
                mask = (freqs >= min_freq) & (freqs <= max_freq)
            else:
                mask = np.ones(freqs.shape, dtype=bool)

            if not np.any(mask):
                QMessageBox.warning(self, "警告", "选定的频率区间内没有数据点！")
                return

            freqs = freqs[mask]

            # 查找V3数据列
            if 'v3_real' in self.fitting_data.columns and 'v3_imag' in self.fitting_data.columns:
                real_vals = self.fitting_data['v3_real'].values.astype(float)
                imag_vals = self.fitting_data['v3_imag'].values.astype(float)
            elif 'V3_Real' in self.fitting_data.columns:
                real_vals = self.fitting_data['V3_Real'].values.astype(float)
                imag_vals = self.fitting_data['V3_Imag'].values.astype(float)
            else:
                # 尝试从第2、3列获取
                real_vals = self.fitting_data.iloc[:, 1].values.astype(float)
                imag_vals = self.fitting_data.iloc[:, 2].values.astype(float)

            V3_exp = real_vals[mask] + 1j * imag_vals[mask]

            # 应用信号校正（如果已标定且用户勾选）
            calibrator = self._get_calibrator()
            if self.apply_calibration_cb.isChecked() and calibrator is not None and calibrator.is_calibrated:
                V3_exp_uncorrected = V3_exp.copy()
                V3_exp = calibrator.correct_v3omega(freqs, V3_exp)
                self.results_text.append("✓ 已应用信号通路校正（基线扣除 + 相位校正）\n")
                if len(V3_exp) > 0:
                    self.results_text.append(f"  [Debug] 数据点1 校正前: {V3_exp_uncorrected[0]:.4e}")
                    self.results_text.append(f"  [Debug] 数据点1 校正后: {V3_exp[0]:.4e}")
                    self.results_text.append(
                        f"  [Debug] 相位变化: {np.angle(V3_exp_uncorrected[0], deg=True):.1f}° -> {np.angle(V3_exp[0], deg=True):.1f}°\n")
            elif calibrator is not None and calibrator.is_calibrated:
                self.results_text.append("⚠ 已加载信号校正参数，但未勾选应用校正，本次拟合使用原始数据\n")

            # 获取传感器参数
            sensor_params = {}
            for param, input_widget in self.sensor_inputs.items():
                try:
                    value = float(input_widget.text())
                    sensor_params[param] = value
                except:
                    QMessageBox.warning(self, "警告", f"传感器参数 {param} 输入无效")
                    return

            # 获取材料层参数
            layers = []
            for i in range(len(self.layer_widgets)):
                layer = {}
                for param in ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']:
                    input_widget = self.findChild(QLineEdit, f"layer_{i}_{param}")
                    if input_widget:
                        text = input_widget.text()
                        if text.lower() == 'inf':
                            layer[param] = np.inf
                        else:
                            layer[param] = float(text)
                layers.append(layer)

            # 确定拟合参数
            mode = self.fit_mode.currentText()
            is_sensor_only = mode == "传感器参数拟合"
            is_combined = mode == "材料+传感器联合拟合"
            params_to_fit = {}
            param_bounds = {}

            if not is_sensor_only:
                for i in range(len(self.layer_widgets)):
                    layer_params_to_fit = []
                    layer_bounds = {}

                    for param in ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']:
                        checkbox = self.findChild(QCheckBox, f"layer_{i}_{param}_fit")
                        if checkbox and checkbox.isChecked():
                            layer_params_to_fit.append(param)

                            min_input = self.findChild(QLineEdit, f"layer_{i}_{param}_min")
                            max_input = self.findChild(QLineEdit, f"layer_{i}_{param}_max")

                            if min_input and max_input and min_input.text() and max_input.text():
                                layer_bounds[param] = (float(min_input.text()), float(max_input.text()))
                            else:
                                if param == 'k_y':
                                    layer_bounds[param] = (0.01, 5000)
                                elif param == 'rho':
                                    layer_bounds[param] = (100, 20000)
                                elif param == 'cap':
                                    layer_bounds[param] = (100, 5000)
                                elif param == 'Rc':
                                    layer_bounds[param] = (0, 1e-3)
                                else:
                                    current_val = layers[i][param]
                                    layer_bounds[param] = (current_val * 0.1, current_val * 10)

                    if layer_params_to_fit:
                        params_to_fit[f'layer_{i}'] = layer_params_to_fit
                        param_bounds[f'layer_{i}'] = layer_bounds

            if is_sensor_only or is_combined:
                sensor_params_to_fit = []
                sensor_bounds = {}

                for param in self.sensor_inputs.keys():
                    checkbox = self.sensor_widget.findChild(QCheckBox, f"sensor_{param}_fit")
                    if checkbox and checkbox.isChecked():
                        sensor_params_to_fit.append(param)
                        min_input = self.sensor_widget.findChild(QLineEdit, f"sensor_{param}_min")
                        max_input = self.sensor_widget.findChild(QLineEdit, f"sensor_{param}_max")
                        if (min_input and max_input and min_input.text().strip()
                                and max_input.text().strip()):
                            sensor_bounds[param] = (
                                float(min_input.text()),
                                float(max_input.text()),
                            )
                        else:
                            current_val = sensor_params[param]
                            if param == 'dR_dT':
                                sensor_bounds[param] = (current_val * 0.1, current_val * 10)
                            else:
                                sensor_bounds[param] = (current_val * 0.5, current_val * 2.0)

                if sensor_params_to_fit:
                    params_to_fit['sensor'] = sensor_params_to_fit
                    param_bounds['sensor'] = sensor_bounds

            if is_combined:
                has_mat = any(k.startswith('layer_') for k in params_to_fit)
                has_sens = 'sensor' in params_to_fit
                if not has_mat:
                    QMessageBox.warning(self, "警告", "材料+传感器联合拟合需至少勾选一项材料层参数。")
                    return
                if not has_sens:
                    QMessageBox.warning(self, "警告", "材料+传感器联合拟合需至少勾选一项传感器参数。")
                    return

            if not params_to_fit:
                QMessageBox.warning(self, "警告", "请选择至少一个参数进行拟合！")
                return

            # 确定拟合方法
            fit_method_map = {
                "Amplitude": "amp",
                "Phase": "phase",
                "Real+Imaginary": "both",
                "Amplitude+Phase": "both",
                "Real Part Only": "real",
                "Imaginary Part Only": "imag"
            }
            fit_method = fit_method_map[self.fit_method.currentText()]

            # --- 层顺序处理逻辑开始 ---
            # GUI 是 [顶层, ..., 基底]，但模型需要 [基底, ..., 顶层]
            # 1. 初始层反转
            fitter_initial_layers = [l.copy() for l in layers[::-1]]

            # 2. 拟合参数键名映射 (layer_i -> layer_{N-1-i})
            N = len(layers)
            fitter_params_to_fit = {}
            if 'sensor' in params_to_fit:
                fitter_params_to_fit['sensor'] = params_to_fit['sensor']

            for key, val in params_to_fit.items():
                if key.startswith('layer_'):
                    # 提取 GUI 中的层索引
                    gui_idx = int(key.split('_')[1])
                    # 计算对应的模型层索引
                    model_idx = N - 1 - gui_idx
                    # 构建新的键名
                    fitter_params_to_fit[f'layer_{model_idx}'] = val

            # 3. 边界键名映射
            fitter_param_bounds = {}
            if 'sensor' in param_bounds:
                fitter_param_bounds['sensor'] = param_bounds['sensor']

            for key, val in param_bounds.items():
                if key.startswith('layer_'):
                    gui_idx = int(key.split('_')[1])
                    model_idx = N - 1 - gui_idx
                    fitter_param_bounds[f'layer_{model_idx}'] = val
            # --- 层顺序处理逻辑结束 ---

            # 创建拟合器
            if is_sensor_only:
                fit_type = 'sensor'
            elif is_combined:
                fit_type = 'combined'
            else:
                fit_type = 'material'

            fitter = ThermalModelFitter(
                freqs=freqs,
                V3_exp=V3_exp,
                b=sensor_params['b'],
                sensor_params=sensor_params,
                initial_layers=fitter_initial_layers,  # 传入反转后的层
                params_to_fit=fitter_params_to_fit,  # 传入映射后的拟合参数
                fit_mode=fit_method,
                param_bounds=fitter_param_bounds,  # 传入映射后的边界
                fit_type=fit_type
            )

            # 重定向标准输出到文本框
            original_stdout = sys.stdout
            sys.stdout = EmittingStream()

            # 使用 QTextCursor 来添加文本
            def append_text(text):
                cursor = self.results_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.insertText(text)
                self.results_text.setTextCursor(cursor)
                self.results_text.ensureCursorVisible()
                QApplication.processEvents()

            sys.stdout.textWritten.connect(append_text)

            try:
                # 执行拟合
                self.results_text.append("开始拟合...\n")
                QApplication.processEvents()

                fitted_result_raw = fitter.fit(
                    max_iter=self.max_iter.value(),
                    tol=float(self.tolerance.text()),
                    zeta=self.zeta.value()
                )

                # --- 拟合结果还原逻辑 ---
                if is_sensor_only:
                    fitted_result = fitted_result_raw
                    result_kind = 'sensor'
                elif is_combined:
                    # olson 中 combined 成功时返回 {'layers', 'sensor_params'}；矩阵奇异时
                    # 与 sensor 模式相同仅返回 sensor_params，需在此区分避免 KeyError
                    if isinstance(fitted_result_raw, dict) and 'layers' in fitted_result_raw:
                        fitted_result = {
                            'layers': fitted_result_raw['layers'][::-1],
                            'sensor_params': dict(fitted_result_raw['sensor_params']),
                        }
                        result_kind = 'combined'
                    else:
                        fitted_result = None
                        result_kind = None
                        self.results_text.append(
                            "\n拟合失败: 联合拟合未正常完成（例如矩阵奇异、灵敏度过低）。"
                            "请减少同时拟合的参数或调整初值/边界后重试。\n"
                        )
                else:
                    fitted_result = fitted_result_raw[::-1]
                    result_kind = 'material'

                if fitted_result is not None:
                    self.results_text.append("\n拟合完成！\n")

                    if is_sensor_only:
                        self.results_text.append("拟合的传感器参数:")
                        for param, value in fitted_result.items():
                            self.results_text.append(f"  {param}: {value:.6e}")
                    elif is_combined:
                        self.results_text.append("拟合的传感器参数:")
                        for param, value in fitted_result['sensor_params'].items():
                            self.results_text.append(f"  {param}: {value:.6e}")
                        self.results_text.append("\n拟合的材料参数:")
                        for i, layer in enumerate(fitted_result['layers']):
                            self.results_text.append(f"\n第 {i + 1} 层 (GUI Layer {i}):")
                            for param, value in layer.items():
                                self.results_text.append(f"  {param}: {value:.6e}")
                    else:
                        self.results_text.append("拟合的材料参数:")
                        for i, layer in enumerate(fitted_result):
                            self.results_text.append(f"\n第 {i + 1} 层 (GUI Layer {i}):")
                            for param, value in layer.items():
                                self.results_text.append(f"  {param}: {value:.6e}")

                    self.update_results_table(fitted_result, result_kind)
                    self.fitted_result = fitted_result
                    self.fitter = fitter

            finally:
                # 恢复标准输出
                sys.stdout = original_stdout

        except Exception as e:
            self.results_text.append(f"\n拟合失败: {str(e)}\n")
            traceback.print_exc()
            # 即使拟合失败，也尝试展示当前使用的实测数据（可能已被校正），以便用户排查问题
            try:
                if hasattr(self, 'ax1'):
                    self.ax1.clear()
                if hasattr(self, 'ax2'):
                    self.ax2.clear()

                # 尝试用假数据填充拟合结果，以便复用现有的绘图函数只画出实验散点
                if 'freqs' in locals() and 'V3_exp' in locals():
                    dummy_fit = np.full_like(V3_exp, np.nan + 1j * np.nan)
                    self.viz_widget.plot_fitting_results(
                        freqs, V3_exp, dummy_fit,
                        title=f"Experimental Data (Fit Failed: {str(e)})"
                    )
                    self.tabs.setCurrentWidget(self.viz_widget)
                elif hasattr(self, 'figure') and hasattr(self.figure, 'canvas'):
                    self.figure.canvas.draw()
            except Exception as plot_e:
                print(f"Failed to plot experimental data after fit error: {plot_e}")
                if hasattr(self, 'figure') and hasattr(self.figure, 'canvas'):
                    self.figure.canvas.draw()

    def export_results(self):
        """导出拟合结果"""
        if not hasattr(self, 'fitted_result'):
            QMessageBox.warning(self, "警告", "没有拟合结果可导出！")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "导出结果", f"fitting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )

        if filename:
            try:
                # 准备导出数据
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'fit_mode': self.fit_mode.currentText(),
                    'fit_method': self.fit_method.currentText(),
                    'results': {}
                }

                fit_mode = self.fit_mode.currentText()
                if fit_mode == "两层厚度约束拟合":
                    r = self.fitted_result
                    export_data['results'] = {
                        'H_total_m':   r.H_total,
                        'f_fraction':  r.f_fraction,
                        'H_top_m':     r.H_top,
                        'H_bot_m':     r.H_bot,
                        'extra_params': r.extra_params,
                        'success':     bool(r.success),
                        'message':     str(r.message),
                        'cost':        float(r.cost) if not np.isnan(r.cost) else None,
                        'n_iter':      int(r.n_iter),
                    }
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=4, ensure_ascii=False)
                    QMessageBox.information(self, "成功", f"结果已导出到:\n{filename}")
                    return
                if fit_mode == "传感器参数拟合":
                    export_data['results'] = {}
                    for k, v in self.fitted_result.items():
                        if isinstance(v, (float, int, np.floating)) and np.isinf(v):
                            export_data['results'][k] = 'inf'
                        else:
                            export_data['results'][k] = float(v)
                elif fit_mode == "材料+传感器联合拟合":
                    export_data['results'] = {'layers': {}, 'sensor_params': {}}
                    for i, layer in enumerate(self.fitted_result['layers']):
                        export_data['results']['layers'][f'layer_{i}'] = {}
                        for key, value in layer.items():
                            if np.isinf(value):
                                export_data['results']['layers'][f'layer_{i}'][key] = 'inf'
                            else:
                                export_data['results']['layers'][f'layer_{i}'][key] = float(value)
                    for key, value in self.fitted_result['sensor_params'].items():
                        if isinstance(value, (float, int, np.floating)) and np.isinf(value):
                            export_data['results']['sensor_params'][key] = 'inf'
                        else:
                            export_data['results']['sensor_params'][key] = float(value)
                else:
                    for i, layer in enumerate(self.fitted_result):
                        export_data['results'][f'layer_{i}'] = {}
                        for key, value in layer.items():
                            if np.isinf(value):
                                export_data['results'][f'layer_{i}'][key] = 'inf'
                            else:
                                export_data['results'][f'layer_{i}'][key] = float(value)

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=4, ensure_ascii=False)

                QMessageBox.information(self, "成功", f"结果已导出到:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")


class VisualizationWidget(QWidget):
    """结果可视化界面"""

    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self):
        super().__init__()
        self.current_data = None
        self.joint_data = None
        self.curve_checkboxes = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # --- 左侧曲线开关面板 ---
        self.curve_panel = QWidget()
        curve_panel_layout = QVBoxLayout(self.curve_panel)

        curve_panel_layout.addWidget(QLabel("曲线显示开关:"))

        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(lambda: self._set_all_checkboxes(True))
        self.deselect_all_btn = QPushButton("全不选")
        self.deselect_all_btn.clicked.connect(lambda: self._set_all_checkboxes(False))
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.select_all_btn)
        btn_row.addWidget(self.deselect_all_btn)
        curve_panel_layout.addLayout(btn_row)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(scroll_widget)
        self.checkbox_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        curve_panel_layout.addWidget(scroll_area)

        self.curve_panel.setMaximumWidth(260)
        self.curve_panel.setVisible(False)
        main_layout.addWidget(self.curve_panel)

        # --- 右侧图形区域 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        plot_control = QHBoxLayout()
        plot_control.addWidget(QLabel("显示图表:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Amplitude", "Phase", "Real Part", "Imaginary Part", "Amplitude & Phase", "All (2x2)"])
        self.plot_type.setCurrentText("Amplitude")
        self.plot_type.currentTextChanged.connect(self.update_plot_type)
        plot_control.addWidget(self.plot_type)
        plot_control.addStretch()
        right_layout.addLayout(plot_control)

        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        control_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新图表")
        self.refresh_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        self.save_fig_btn = QPushButton("保存图表")
        self.save_fig_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        control_layout.addWidget(self.refresh_btn)
        control_layout.addWidget(self.save_fig_btn)
        control_layout.addStretch()
        right_layout.addLayout(control_layout)

        main_layout.addWidget(right_panel)
        self.setLayout(main_layout)

        self.refresh_btn.clicked.connect(self.refresh_plot)
        self.save_fig_btn.clicked.connect(self.save_figure)

    def _set_all_checkboxes(self, checked):
        for cb in self.curve_checkboxes:
            cb.setChecked(checked)
        self._redraw_joint()

    def _clear_checkboxes(self):
        for cb in self.curve_checkboxes:
            cb.deleteLater()
        self.curve_checkboxes = []

    def plot_joint_fitting_results(self, datasets, layers, title="Joint Fitting Results"):
        """绘制联合拟合结果，每个数据集的实验值和拟合值各自可开关"""
        self.current_data = None
        self.joint_data = {'datasets': datasets, 'layers': layers, 'title': title}

        self._clear_checkboxes()

        computed = []
        for i, ds in enumerate(datasets):
            Z_fit = calculate_analytical_Z(ds['freqs'], ds['b'], layers)
            V3_fit = convert_Z_to_V3(Z_fit, ds['freqs'], ds['b'], ds['sensor_params'])
            computed.append(V3_fit)

        self.joint_data['V3_fitted_list'] = computed

        for i, ds in enumerate(datasets):
            color = self.COLORS[i % len(self.COLORS)]
            b_um = ds['b'] * 1e6

            cb_exp = QCheckBox(f"DS{i + 1} 实验 (b={b_um:.0f}μm)")
            cb_exp.setChecked(True)
            cb_exp.setStyleSheet(f"color: {color}; font-weight: bold;")
            cb_exp.toggled.connect(self._redraw_joint)
            self.checkbox_layout.insertWidget(self.checkbox_layout.count() - 1, cb_exp)
            self.curve_checkboxes.append(cb_exp)

            cb_fit = QCheckBox(f"DS{i + 1} 拟合 (b={b_um:.0f}μm)")
            cb_fit.setChecked(True)
            cb_fit.setStyleSheet(f"color: {color};")
            cb_fit.toggled.connect(self._redraw_joint)
            self.checkbox_layout.insertWidget(self.checkbox_layout.count() - 1, cb_fit)
            self.curve_checkboxes.append(cb_fit)

        self.curve_panel.setVisible(True)
        self._redraw_joint()

    def _redraw_joint(self):
        """根据复选框状态重绘联合拟合图"""
        if not self.joint_data:
            return

        self.figure.clear()
        plot_type = self.plot_type.currentText()
        datasets = self.joint_data['datasets']
        fitted_list = self.joint_data['V3_fitted_list']
        title = self.joint_data['title']

        if plot_type in ("Amplitude & Phase", "All (2x2)"):
            if plot_type == "All (2x2)":
                ax_amp = self.figure.add_subplot(221)
                ax_pha = self.figure.add_subplot(222)
                ax_re = self.figure.add_subplot(223)
                ax_im = self.figure.add_subplot(224)
                axes_map = {'amp': ax_amp, 'pha': ax_pha, 're': ax_re, 'im': ax_im}
            else:
                ax_amp = self.figure.add_subplot(211)
                ax_pha = self.figure.add_subplot(212, sharex=ax_amp)
                axes_map = {'amp': ax_amp, 'pha': ax_pha}
        else:
            ax_single = self.figure.add_subplot(111)
            axes_map = {'single': ax_single}

        for i, ds in enumerate(datasets):
            color = self.COLORS[i % len(self.COLORS)]
            cb_exp = self.curve_checkboxes[i * 2]
            cb_fit = self.curve_checkboxes[i * 2 + 1]
            freqs = ds['freqs']
            V3_exp = ds['V3_exp']
            V3_fit = fitted_list[i]
            b_um = ds['b'] * 1e6
            label_exp = f"DS{i + 1} Exp (b={b_um:.0f}μm)"
            label_fit = f"DS{i + 1} Fit (b={b_um:.0f}μm)"

            self._plot_curves_on_axes(
                axes_map, plot_type, freqs, V3_exp, V3_fit,
                color, label_exp, label_fit,
                show_exp=cb_exp.isChecked(), show_fit=cb_fit.isChecked()
            )

        for ax in self.figure.get_axes():
            ax.legend(fontsize='small', loc='best')
            ax.grid(True, which="both", ls="--", alpha=0.3)

        self.figure.suptitle(title, fontsize=14)
        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_curves_on_axes(self, axes_map, plot_type, freqs, V3_exp, V3_fit,
                             color, label_exp, label_fit, show_exp=True, show_fit=True):
        """在给定的 axes 上绘制一组 (实验+拟合) 曲线"""
        import matplotlib.colors as mcolors
        fit_color = color
        try:
            rgb = mcolors.to_rgb(color)
            fit_color = tuple(min(1, c * 0.7) for c in rgb)
        except Exception:
            pass

        def draw_on(ax, exp_y, fit_y, log_x=True, log_y=False):
            plot_fn = ax.loglog if (log_x and log_y) else (ax.semilogx if log_x else ax.plot)
            if show_exp:
                plot_fn(freqs, exp_y, 'o', color=color, label=label_exp, markersize=5, alpha=0.8)
            if show_fit:
                plot_fn(freqs, fit_y, '-', color=fit_color, label=label_fit, linewidth=2)

        if 'single' in axes_map:
            ax = axes_map['single']
            if plot_type == "Amplitude":
                draw_on(ax, np.abs(V3_exp) * 1e3, np.abs(V3_fit) * 1e3, log_x=True, log_y=True)
                ax.set_ylabel('|V3| (mV)')
                ax.set_xlabel('Frequency (Hz)')
            elif plot_type == "Phase":
                draw_on(ax, np.angle(V3_exp, deg=True), np.angle(V3_fit, deg=True))
                ax.set_ylabel('Phase (deg)')
                ax.set_xlabel('Frequency (Hz)')
            elif plot_type == "Real Part":
                draw_on(ax, V3_exp.real * 1e3, V3_fit.real * 1e3)
                ax.set_ylabel('Real(V3) (mV)')
                ax.set_xlabel('Frequency (Hz)')
            elif plot_type == "Imaginary Part":
                draw_on(ax, V3_exp.imag * 1e3, V3_fit.imag * 1e3)
                ax.set_ylabel('Imag(V3) (mV)')
                ax.set_xlabel('Frequency (Hz)')
        else:
            if 'amp' in axes_map:
                draw_on(axes_map['amp'], np.abs(V3_exp) * 1e3, np.abs(V3_fit) * 1e3, log_x=True, log_y=True)
                axes_map['amp'].set_ylabel('|V3| (mV)')
                axes_map['amp'].set_title('Amplitude')
            if 'pha' in axes_map:
                draw_on(axes_map['pha'], np.angle(V3_exp, deg=True), np.angle(V3_fit, deg=True))
                axes_map['pha'].set_ylabel('Phase (deg)')
                axes_map['pha'].set_xlabel('Frequency (Hz)')
                axes_map['pha'].set_title('Phase')
            if 're' in axes_map:
                draw_on(axes_map['re'], V3_exp.real * 1e3, V3_fit.real * 1e3)
                axes_map['re'].set_ylabel('Real(V3) (mV)')
                axes_map['re'].set_xlabel('Frequency (Hz)')
                axes_map['re'].set_title('Real Part')
            if 'im' in axes_map:
                draw_on(axes_map['im'], V3_exp.imag * 1e3, V3_fit.imag * 1e3)
                axes_map['im'].set_ylabel('Imag(V3) (mV)')
                axes_map['im'].set_xlabel('Frequency (Hz)')
                axes_map['im'].set_title('Imaginary Part')

    def plot_fitting_results(self, freqs, V3_exp, V3_fitted, V3_initial=None, title="Fitting Results"):
        """绘制单数据集拟合结果"""
        self.joint_data = None
        self._clear_checkboxes()
        self.curve_panel.setVisible(False)

        self.current_data = {
            'freqs': freqs,
            'V3_exp': V3_exp,
            'V3_fitted': V3_fitted,
            'V3_initial': V3_initial,
            'title': title
        }

        self.figure.clear()

        plot_type = self.plot_type.currentText()

        if plot_type == "All (2x2)":
            # 显示所有四个图
            ax1 = self.figure.add_subplot(221)
            ax2 = self.figure.add_subplot(222)
            ax3 = self.figure.add_subplot(223)
            ax4 = self.figure.add_subplot(224)

            # 幅值图
            ax1.loglog(freqs, np.abs(V3_exp) * 1e3, 'ro', label='Experimental Data', markersize=6)
            ax1.loglog(freqs, np.abs(V3_fitted) * 1e3, 'b-', label='Fitted Model', linewidth=2)
            if V3_initial is not None:
                ax1.loglog(freqs, np.abs(V3_initial) * 1e3, 'g--', label='Initial Guess', linewidth=1)
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('|V3| (mV)')
            ax1.set_title('V3 Amplitude')
            ax1.legend()
            ax1.grid(True, which="both", ls="--", alpha=0.3)

            # 相位图
            ax2.semilogx(freqs, np.angle(V3_exp, deg=True), 'ro', label='Experimental Data', markersize=6)
            ax2.semilogx(freqs, np.angle(V3_fitted, deg=True), 'b-', label='Fitted Model', linewidth=2)
            if V3_initial is not None:
                ax2.semilogx(freqs, np.angle(V3_initial, deg=True), 'g--', label='Initial Guess', linewidth=1)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Phase (deg)')
            ax2.set_title('V3 Phase')
            ax2.legend()
            ax2.grid(True, which="both", ls="--", alpha=0.3)

            # 实部图
            ax3.semilogx(freqs, V3_exp.real * 1e3, 'ro', label='Experimental Data', markersize=6)
            ax3.semilogx(freqs, V3_fitted.real * 1e3, 'b-', label='Fitted Model', linewidth=2)
            if V3_initial is not None:
                ax3.semilogx(freqs, V3_initial.real * 1e3, 'g--', label='Initial Guess', linewidth=1)
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Real(V3) (mV)')
            ax3.set_title('V3 Real Part')
            ax3.legend()
            ax3.grid(True, which="both", ls="--", alpha=0.3)

            # 虚部图
            ax4.semilogx(freqs, V3_exp.imag * 1e3, 'ro', label='Experimental Data', markersize=6)
            ax4.semilogx(freqs, V3_fitted.imag * 1e3, 'b-', label='Fitted Model', linewidth=2)
            if V3_initial is not None:
                ax4.semilogx(freqs, V3_initial.imag * 1e3, 'g--', label='Initial Guess', linewidth=1)
            ax4.set_xlabel('Frequency (Hz)')
            ax4.set_ylabel('Imag(V3) (mV)')
            ax4.set_title('V3 Imaginary Part')
            ax4.legend()
            ax4.grid(True, which="both", ls="--", alpha=0.3)

        elif plot_type == "Amplitude & Phase":
            # 双图模式 (幅值 + 相位)
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)

            # 幅值图
            ax1.loglog(freqs, np.abs(V3_exp) * 1e3, 'ro', label='Experimental Data', markersize=6)
            ax1.loglog(freqs, np.abs(V3_fitted) * 1e3, 'b-', label='Fitted Model', linewidth=2)
            if V3_initial is not None:
                ax1.loglog(freqs, np.abs(V3_initial) * 1e3, 'g--', label='Initial Guess', linewidth=1)
            ax1.set_ylabel('|V3| (mV)')
            ax1.set_title('V3 Amplitude')
            ax1.legend()
            ax1.grid(True, which="both", ls="--", alpha=0.3)

            # 相位图
            ax2.semilogx(freqs, np.angle(V3_exp, deg=True), 'ro', label='Experimental Data', markersize=6)
            ax2.semilogx(freqs, np.angle(V3_fitted, deg=True), 'b-', label='Fitted Model', linewidth=2)
            if V3_initial is not None:
                ax2.semilogx(freqs, np.angle(V3_initial, deg=True), 'g--', label='Initial Guess', linewidth=1)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Phase (deg)')
            ax2.set_title('V3 Phase')
            ax2.legend()
            ax2.grid(True, which="both", ls="--", alpha=0.3)

        else:
            # 显示单个图
            ax = self.figure.add_subplot(111)

            if plot_type == "Amplitude":
                ax.loglog(freqs, np.abs(V3_exp) * 1e3, 'ro', label='Experimental Data', markersize=8)
                ax.loglog(freqs, np.abs(V3_fitted) * 1e3, 'b-', label='Fitted Model', linewidth=2.5)
                if V3_initial is not None:
                    ax.loglog(freqs, np.abs(V3_initial) * 1e3, 'g--', label='Initial Guess', linewidth=1.5)
                ax.set_ylabel('|V3| (mV)', fontsize=12)
                ax.set_title('V3 Amplitude', fontsize=14)

            elif plot_type == "Phase":
                ax.semilogx(freqs, np.angle(V3_exp, deg=True), 'ro', label='Experimental Data', markersize=8)
                ax.semilogx(freqs, np.angle(V3_fitted, deg=True), 'b-', label='Fitted Model', linewidth=2.5)
                if V3_initial is not None:
                    ax.semilogx(freqs, np.angle(V3_initial, deg=True), 'g--', label='Initial Guess', linewidth=1.5)
                ax.set_ylabel('Phase (deg)', fontsize=12)
                ax.set_title('V3 Phase', fontsize=14)

            elif plot_type == "Real Part":
                ax.semilogx(freqs, V3_exp.real * 1e3, 'ro', label='Experimental Data', markersize=8)
                ax.semilogx(freqs, V3_fitted.real * 1e3, 'b-', label='Fitted Model', linewidth=2.5)
                if V3_initial is not None:
                    ax.semilogx(freqs, V3_initial.real * 1e3, 'g--', label='Initial Guess', linewidth=1.5)
                ax.set_ylabel('Real(V3) (mV)', fontsize=12)
                ax.set_title('V3 Real Part', fontsize=14)

            elif plot_type == "Imaginary Part":
                ax.semilogx(freqs, V3_exp.imag * 1e3, 'ro', label='Experimental Data', markersize=8)
                ax.semilogx(freqs, V3_fitted.imag * 1e3, 'b-', label='Fitted Model', linewidth=2.5)
                if V3_initial is not None:
                    ax.semilogx(freqs, V3_initial.imag * 1e3, 'g--', label='Initial Guess', linewidth=1.5)
                ax.set_ylabel('Imag(V3) (mV)', fontsize=12)
                ax.set_title('V3 Imaginary Part', fontsize=14)

            ax.set_xlabel('Frequency (Hz)', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, which="both", ls="--", alpha=0.3)

        self.figure.suptitle(title, fontsize=16)
        self.figure.tight_layout()
        self.canvas.draw()

    def update_plot_type(self):
        """更新图表类型"""
        if self.joint_data:
            self._redraw_joint()
        elif self.current_data:
            self.plot_fitting_results(
                self.current_data['freqs'],
                self.current_data['V3_exp'],
                self.current_data['V3_fitted'],
                self.current_data['V3_initial'],
                self.current_data['title']
            )

    def refresh_plot(self):
        """刷新图表（需要从其他窗口获取数据）"""
        QMessageBox.information(self, "提示", "请在拟合完成后查看结果")

    def save_figure(self):
        """保存图表"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存图表", f"fitting_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )

        if filename:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "成功", f"图表已保存到:\n{filename}")


class ThermalMeasurementGUI(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("3ω热物性测量系统 v1.1 (含两层厚度约束拟合)")
        self.setWindowIcon(QIcon('icon.png'))

        # 设置窗口大小
        self.resize(1400, 900)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建标签页
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background: white;
            }
            QTabBar::tab {
                padding: 10px 20px;
                margin: 2px;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background: #2196F3;
                color: white;
            }
        """)

        # 添加各个功能模块
        self.lockin_widget = LockInControlWidget()
        self.tabs.addTab(self.lockin_widget, "🔬 锁相控制")

        self.fitting_widget = FittingWidget()
        self.tabs.addTab(self.fitting_widget, "📊 数据拟合")

        self.viz_widget = VisualizationWidget()
        self.tabs.addTab(self.viz_widget, "📈 结果可视化")

        # --- 新增标签页 ---
        self.simulation_widget = SimulationWidget()
        self.tabs.addTab(self.simulation_widget, "🔁 正向模拟")

        self.sensitivity_widget = SensitivityWidget()
        self.tabs.addTab(self.sensitivity_widget, "🎯 灵敏度分析")

        self.data_comparison_widget = DataComparisonWidget()
        self.tabs.addTab(self.data_comparison_widget, "📈 数据对比")

        self.calibration_widget = CalibrationWidget()
        self.tabs.addTab(self.calibration_widget, "🔧 信号校正")

        main_layout.addWidget(self.tabs)

        # 创建菜单栏
        self.create_menu_bar()

        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        # 连接信号
        self.fitting_widget.fit_btn.clicked.connect(self.on_fitting_completed)
        self.calibration_widget.calibration_ready.connect(
            self.fitting_widget.set_calibrator
        )

    def create_menu_bar(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')

        new_action = QAction('新建项目', self)
        new_action.setShortcut('Ctrl+N')
        file_menu.addAction(new_action)

        open_action = QAction('打开项目', self)
        open_action.setShortcut('Ctrl+O')
        file_menu.addAction(open_action)

        save_action = QAction('保存项目', self)
        save_action.setShortcut('Ctrl+S')
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 工具菜单
        tools_menu = menubar.addMenu('工具(&T)')

        sensitivity_action = QAction('温升计算器', self)
        sensitivity_action.triggered.connect(self.show_temp_calculator)
        tools_menu.addAction(sensitivity_action)

        calibration_action = QAction('传感器校准', self)
        tools_menu.addAction(calibration_action)

        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')

        manual_action = QAction('使用手册', self)
        help_menu.addAction(manual_action)

        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_temp_calculator(self):
        """显示温升计算器"""
        dialog = TemperatureCalculatorDialog(self)
        dialog.exec_()

    def on_fitting_completed(self):
        """拟合完成后的处理"""
        if not (hasattr(self.fitting_widget, 'fitter') and hasattr(self.fitting_widget, 'fitted_result')):
            return

        try:
            fitter = self.fitting_widget.fitter

            # 两层厚度约束拟合：单独走绘图流程（包含 cost(f) 扫描子图）
            if getattr(fitter, 'is_two_layer_thickness', False):
                self._plot_two_layer_thickness_result(fitter, self.fitting_widget.fitted_result)
                self.tabs.setCurrentWidget(self.viz_widget)
                self.status_bar.showMessage(
                    f"两层厚度约束拟合完成: "
                    f"H_top={self.fitting_widget.fitted_result.H_top*1e6:.3f} μm, "
                    f"H_bot={self.fitting_widget.fitted_result.H_bot*1e6:.3f} μm"
                )
                return

            is_joint = isinstance(fitter, JointThermalModelFitter)

            if is_joint:
                if not fitter.datasets:
                    return
                self.viz_widget.plot_joint_fitting_results(
                    fitter.datasets, fitter.layers,
                    title=f"Joint Fit ({len(fitter.datasets)} datasets)"
                )
                self.tabs.setCurrentWidget(self.viz_widget)
                self.status_bar.showMessage(
                    f"联合拟合完成 (共 {len(fitter.datasets)} 个数据集)，可在左侧面板切换曲线显示"
                )
            else:
                # 单数据集拟合
                freqs = fitter.freqs
                V3_exp = fitter.V3_exp

                Z_fitted = calculate_analytical_Z(freqs, fitter.b, fitter.layers)
                V3_fitted = convert_Z_to_V3(Z_fitted, freqs, fitter.b, fitter.sensor_params)

                self.viz_widget.plot_fitting_results(
                    freqs, V3_exp, V3_fitted,
                    title=f"Fitting Results ({self.fitting_widget.fit_method.currentText()})"
                )
                self.tabs.setCurrentWidget(self.viz_widget)
                self.status_bar.showMessage("拟合完成，结果已显示在可视化页面")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示拟合结果时出错:\n{str(e)}")

    def _plot_two_layer_thickness_result(self, fitter, result):
        """在 VisualizationWidget 的 figure 上画两层厚度约束拟合结果（含 cost(f) 扫描）"""
        viz = self.viz_widget
        viz.joint_data = None
        viz._clear_checkboxes()
        viz.curve_panel.setVisible(False)
        viz.current_data = None

        viz.figure.clear()
        freqs = fitter.freqs
        V3_exp = fitter.V3_exp
        V3_fit = result.V3_fit
        V3_init = result.V3_initial

        has_scan = getattr(fitter, 'f_scan', None) is not None
        if has_scan:
            gs = viz.figure.add_gridspec(2, 2, width_ratios=[2, 1])
            ax_amp = viz.figure.add_subplot(gs[0, 0])
            ax_ph = viz.figure.add_subplot(gs[1, 0], sharex=ax_amp)
            ax_sc = viz.figure.add_subplot(gs[:, 1])
        else:
            ax_amp = viz.figure.add_subplot(211)
            ax_ph = viz.figure.add_subplot(212, sharex=ax_amp)
            ax_sc = None

        ax_amp.loglog(freqs, np.abs(V3_exp) * 1e3, 'ro', ms=4, label='Experimental')
        if V3_init is not None:
            ax_amp.loglog(freqs, np.abs(V3_init) * 1e3, 'g--', lw=1.0, label='Initial')
        if V3_fit is not None:
            ax_amp.loglog(freqs, np.abs(V3_fit) * 1e3, 'b-', lw=2.0, label='Fit')
        ax_amp.set_ylabel('|V3| (mV)')
        ax_amp.set_title(
            f"H_total={result.H_total*1e6:.2f} μm | "
            f"H_top={result.H_top*1e6:.3f}μm  H_bot={result.H_bot*1e6:.3f}μm "
            f"(f={result.f_fraction:.3f})"
        )
        ax_amp.grid(True, which='both', ls='--', alpha=0.3)
        ax_amp.legend(fontsize='small')

        ph_exp = np.unwrap(np.angle(V3_exp)) * 180 / np.pi
        ax_ph.semilogx(freqs, ph_exp, 'ro', ms=4, label='Experimental')
        if V3_init is not None:
            ax_ph.semilogx(freqs, np.unwrap(np.angle(V3_init)) * 180 / np.pi,
                           'g--', lw=1.0, label='Initial')
        if V3_fit is not None:
            ax_ph.semilogx(freqs, np.unwrap(np.angle(V3_fit)) * 180 / np.pi,
                           'b-', lw=2.0, label='Fit')
        ax_ph.set_xlabel('Frequency (Hz)')
        ax_ph.set_ylabel('Phase (deg)')
        ax_ph.grid(True, which='both', ls='--', alpha=0.3)
        ax_ph.legend(fontsize='small')

        if ax_sc is not None:
            f_arr, cost_arr = fitter.f_scan
            ax_sc.semilogy(f_arr, cost_arr, 'k-')
            ax_sc.axvline(result.f_fraction, color='r', ls='--',
                          label=f'fit f={result.f_fraction:.3f}')
            ax_sc.set_xlabel('f = H_top / H_total')
            ax_sc.set_ylabel('cost')
            ax_sc.set_title('Cost vs f (scan)')
            ax_sc.grid(True, which='both', ls='--', alpha=0.3)
            ax_sc.legend(fontsize='small')

        viz.figure.suptitle("Two-Layer Thickness Constrained Fit", fontsize=13)
        viz.figure.tight_layout()
        viz.canvas.draw()

    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h2>3ω热物性测量系统 v1.1 (含两层厚度约束拟合)</h2>
        <p>基于3-omega方法的多层材料热物性测量与分析软件</p>
        <p><b>功能特性:</b></p>
        <ul>
            <li>苏黎世或 OE1300 锁相放大器自动控制（OE1300 串口/网口 ASCII 指令）</li>
            <li>实时数据采集与显示</li>
            <li>多层材料热物性参数拟合</li>
            <li><b>两层厚度约束拟合（新功能）：H_total 已知时反推各自厚度，
                可同时与传感器/其它材料参数联合拟合</b></li>
            <li>正向模型模拟与分析</li>
            <li>灵敏度分析</li>
            <li>温升计算工具</li>
        </ul>
        <p><b>开发者:</b> 热物性测量实验室</p>
        <p><b>版本:</b> 1.1.0</p>
        <p><b>日期:</b> 2026</p>
        """
        QMessageBox.about(self, "关于", about_text)


def main():
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 设置应用程序调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, Qt.black)
    palette.setColor(QPalette.Base, Qt.white)
    palette.setColor(QPalette.AlternateBase, QColor(230, 230, 230))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.black)
    palette.setColor(QPalette.Text, Qt.black)
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, Qt.black)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    # 创建并显示主窗口
    window = ThermalMeasurementGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
