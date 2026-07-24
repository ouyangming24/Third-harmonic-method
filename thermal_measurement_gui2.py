import sys
import os
import json
import re
import time
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
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


def _configure_matplotlib_chinese_font():
    """为 matplotlib 图例/标题配置可显示中文的字体（Windows 优先微软雅黑）。"""
    from matplotlib import font_manager
    candidates = [
        'Microsoft YaHei',
        'SimHei',
        'SimSun',
        'KaiTi',
        'FangSong',
        'PingFang SC',
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'Arial Unicode MS',
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break
    if chosen is None:
        # 按常见 Windows 字体文件路径再试一次
        win_paths = [
            r'C:\Windows\Fonts\msyh.ttc',
            r'C:\Windows\Fonts\msyh.ttf',
            r'C:\Windows\Fonts\simhei.ttf',
            r'C:\Windows\Fonts\simsun.ttc',
        ]
        for path in win_paths:
            if os.path.isfile(path):
                try:
                    font_manager.fontManager.addfont(path)
                    prop = font_manager.FontProperties(fname=path)
                    chosen = prop.get_name()
                    break
                except Exception:
                    continue
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen] + list(plt.rcParams.get('font.sans-serif', []))
    plt.rcParams['axes.unicode_minus'] = False  # 负号用 ASCII，避免变成方框
    return chosen


_MATPLOTLIB_CN_FONT = _configure_matplotlib_chinese_font()

# 导入项目模块
from olson_model26_3 import (
    ThermalModelFitter, JointThermalModelFitter, calculate_analytical_Z, convert_Z_to_V3,
    FitCancelledError,
)
from calibration_gui import CalibrationWidget
from signal_calibration import SignalCalibrator
from 苏黎世锁相适配3 import ZurichLockInControl
from oe1300_lockin import OE1300LockInControl

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

    def __init__(self, parent=None, sensor_params=None):
        super().__init__(parent)
        self.setWindowTitle("3ω温升计算器")
        self.resize(500, 620)
        self._fitting_source = None
        self.init_ui()
        if sensor_params:
            self.apply_sensor_params(sensor_params)

    def init_ui(self):
        layout = QVBoxLayout()

        sync_row = QHBoxLayout()
        self.sync_btn = QPushButton("从数据拟合同步传感器参数")
        self.sync_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        self.sync_btn.setToolTip("读取「数据拟合」页当前传感器参数并填入下方初值")
        self.sync_btn.clicked.connect(self.sync_from_fitting)
        sync_row.addWidget(self.sync_btn)
        layout.addLayout(sync_row)

        # 1. 传感器参数
        sensor_group = QGroupBox("传感器参数")
        sensor_layout = QGridLayout()

        sensor_layout.addWidget(QLabel("电阻 R (Ω):"), 0, 0)
        self.r_input = QLineEdit("5.2")
        sensor_layout.addWidget(self.r_input, 0, 1)

        sensor_layout.addWidget(QLabel("温度系数 α_R (/K):"), 1, 0)
        self.alpha_input = QLineEdit("0.0022")
        self.alpha_input.setToolTip("相对温度系数 α_R = (dR/dT) / R；由拟合页 dR_dT 与 R 换算")
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
        self.v_drive_input.setToolTip("默认同步拟合页 V_rms（换算为 mV）")
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

    def set_fitting_source(self, fitting_widget):
        self._fitting_source = fitting_widget

    def apply_sensor_params(self, sensor_params):
        """用拟合页传感器参数（SI）填充温升计算器初值。"""
        if not sensor_params:
            return
        try:
            R = float(sensor_params.get('R', self.r_input.text()))
            self.r_input.setText(f"{R:.6g}")

            if 'dR_dT' in sensor_params and R != 0:
                alpha_R = float(sensor_params['dR_dT']) / R
                self.alpha_input.setText(f"{alpha_R:.6g}")

            if 'L' in sensor_params:
                self.l_input.setText(f"{float(sensor_params['L']) * 1e3:.6g}")

            if 'b' in sensor_params:
                # 拟合页 b 为半宽 (m) → 线宽 2b (μm)
                self.width_input.setText(f"{2.0 * float(sensor_params['b']) * 1e6:.6g}")

            if 'V_rms' in sensor_params:
                self.v_drive_input.setText(f"{float(sensor_params['V_rms']) * 1e3:.6g}")
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    def sync_from_fitting(self):
        src = self._fitting_source
        if src is None and self.parent() is not None:
            src = getattr(self.parent(), 'fitting_widget', None)
        if src is None or not hasattr(src, 'sensor_inputs'):
            QMessageBox.warning(self, "提示", "未关联数据拟合页，无法同步。")
            return
        try:
            sp = {k: float(inp.text()) for k, inp in src.sensor_inputs.items()}
            self.apply_sensor_params(sp)
            self.result_area.append(
                f"\n已从数据拟合同步: R={sp.get('R')}, "
                f"dR/dT={sp.get('dR_dT')}, L={sp.get('L')}, b={sp.get('b')}, "
                f"V_rms={sp.get('V_rms')}\n"
            )
        except Exception as e:
            QMessageBox.warning(self, "同步失败", f"读取传感器参数失败:\n{e}")

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
        clear_btn = QPushButton("重置层")
        clear_btn.clicked.connect(self.reset_layers)
        layer_btns.addWidget(clear_btn)
        layer_btns.addStretch()

        self.layer_layout.addLayout(layer_btns)
        layer_group.setLayout(self.layer_layout)
        self.left_layout.addWidget(layer_group)

        self.sync_from_fitting_btn = QPushButton("同步数据拟合参数")
        self.sync_from_fitting_btn.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background-color: #7B1FA2; }"
        )
        self.sync_from_fitting_btn.setToolTip(
            "一键将「数据拟合」页中的传感器参数与材料层结构同步到本页"
        )
        self.sync_from_fitting_btn.clicked.connect(self.sync_from_fitting)
        self._fitting_source = None
        self.left_layout.addWidget(self.sync_from_fitting_btn)

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

    def _create_layer_op_button(self, text, tooltip, container, is_add):
        btn = QPushButton(text)
        btn.setFixedSize(26, 26)
        btn.setToolTip(tooltip)
        btn.setProperty('layer_container', container)
        if is_add:
            btn.setStyleSheet(
                "QPushButton { color: #4CAF50; font-weight: bold; font-size: 16px; "
                "border: 1px solid #4CAF50; border-radius: 3px; padding: 0; }"
                "QPushButton:hover { background-color: #E8F5E9; }"
            )
            btn.clicked.connect(self._on_layer_add_clicked)
        else:
            btn.setStyleSheet(
                "QPushButton { color: #f44336; font-weight: bold; font-size: 16px; "
                "border: 1px solid #f44336; border-radius: 3px; padding: 0; }"
                "QPushButton:hover { background-color: #FFEBEE; }"
            )
            btn.clicked.connect(self._on_layer_remove_clicked)
        return btn

    def _layer_index_of(self, container):
        for i, w in enumerate(self.layer_widgets):
            if w.get('container') == container:
                return i
        return -1

    def _on_layer_add_clicked(self):
        container = self.sender().property('layer_container')
        idx = self._layer_index_of(container)
        if idx < 0:
            return
        if len(self.layer_widgets) >= 10:
            QMessageBox.warning(self, "提示", "材料层数最多为 10 层。")
            return
        self.add_layer_ui(idx)

    def _on_layer_remove_clicked(self):
        container = self.sender().property('layer_container')
        idx = self._layer_index_of(container)
        if idx >= 0:
            self._remove_layer_at(idx)

    def _renumber_layer_ui_titles(self):
        """按当前顺序重排层标题（Layer 0, Layer 1, ...）。"""
        for i, w in enumerate(self.layer_widgets):
            w['title_label'].setText(f"Layer {i}")

    def _remove_layer_at(self, pos):
        if len(self.layer_widgets) <= 1:
            QMessageBox.warning(self, "提示", "至少保留一层材料结构。")
            return
        if pos < 0 or pos >= len(self.layer_widgets):
            return
        w = self.layer_widgets.pop(pos)
        self.layer_layout.removeWidget(w['container'])
        w['container'].deleteLater()
        self._renumber_layer_ui_titles()
        self.schedule_update()

    def add_layer_ui(self, idx=None):
        if idx is None:
            idx = len(self.layer_widgets)
        idx = max(0, min(int(idx), len(self.layer_widgets)))

        container = QWidget()
        container.setObjectName("layerBlock")
        container.setStyleSheet(
            "#layerBlock { border: 1px solid #bdbdbd; border-radius: 4px; "
            "background: #fafafa; margin-top: 2px; }"
        )
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(8, 6, 8, 8)

        header = QHBoxLayout()
        title_label = QLabel(f"Layer {idx}")
        title_label.setStyleSheet("font-weight: bold;")
        header.addWidget(title_label)
        header.addStretch()
        btn_add = self._create_layer_op_button('+', "在该层上方插入新层", container, is_add=True)
        btn_remove = self._create_layer_op_button('−', "删除该层", container, is_add=False)
        header.addWidget(btn_add)
        header.addWidget(btn_remove)
        container_layout.addLayout(header)

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
            inp.textChanged.connect(self.schedule_update)
            inputs[p] = inp
            layout.addWidget(inp, i + 1, 1)

        mat_combo.currentTextChanged.connect(lambda t, inps=inputs: self.load_material(t, inps))
        self.load_material(mat_combo.currentText(), inputs)

        content = QWidget()
        content.setLayout(layout)
        container_layout.addWidget(content)

        self.layer_layout.insertWidget(idx, container)
        self.layer_widgets.insert(idx, {
            'container': container,
            'title_label': title_label,
            'inputs': inputs,
            'combo': mat_combo,
        })

        self._renumber_layer_ui_titles()
        self.schedule_update()

    def set_fitting_source(self, fitting_widget):
        """设置参数来源（数据拟合页）。"""
        self._fitting_source = fitting_widget

    def _clear_all_layers(self):
        """移除全部材料层控件。"""
        for w in self.layer_widgets:
            self.layer_layout.removeWidget(w['container'])
            w['container'].deleteLater()
        self.layer_widgets.clear()

    def apply_model_config(self, config):
        """应用传感器参数与材料层结构配置。"""
        self.update_timer.stop()
        try:
            sensor = config.get('sensor', {})
            for param, value in sensor.items():
                if param in self.sensor_inputs:
                    self.sensor_inputs[param].setText(str(value))

            layers = config.get('layers', [])
            self._clear_all_layers()
            if not layers:
                self.add_layer_ui(0)
            else:
                for i, layer_data in enumerate(layers):
                    self.add_layer_ui(i)
                    for param, value in layer_data.items():
                        if param in self.layer_widgets[i]['inputs']:
                            self.layer_widgets[i]['inputs'][param].setText(str(value))

            if hasattr(self, 'param_list'):
                self.update_param_list()
        finally:
            self.schedule_update()

    def sync_from_fitting(self):
        """从数据拟合页同步传感器参数与材料层结构。"""
        if self._fitting_source is None:
            QMessageBox.warning(self, "提示", "未关联数据拟合模块。")
            return
        try:
            config = self._fitting_source.get_model_config()
            self.apply_model_config(config)
            n_layers = len(config.get('layers', []))
            QMessageBox.information(
                self, "同步成功",
                f"已从数据拟合同步参数：\n"
                f"  传感器参数 {len(config.get('sensor', {}))} 项\n"
                f"  材料层 {n_layers} 层"
            )
        except Exception as e:
            QMessageBox.critical(self, "同步失败", f"同步参数时出错:\n{str(e)}")

    def reset_layers(self):
        for w in self.layer_widgets:
            w['container'].deleteLater()
        self.layer_widgets = []
        self.add_layer_ui(0)
        self.add_layer_ui(1)

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

    def add_layer_ui(self, idx=None):
        super().add_layer_ui(idx)
        if hasattr(self, 'param_list'):
            self.update_param_list()

    def _remove_layer_at(self, pos):
        super()._remove_layer_at(pos)
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
        self.freq_points.setRange(1, 10000)
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
        self._fit_cancel_requested = False
        self._last_fitted_param_spec = None
        self.init_ui()

    def set_calibrator(self, calibrator):
        """设置信号校正器（由主窗口传入）"""
        self._calibrator = calibrator

    def _get_calibrator(self):
        """获取已就绪的校正器"""
        if self._calibrator is not None and self._calibrator.is_calibrated:
            return self._calibrator
        return None

    def get_model_config(self):
        """导出当前传感器参数与材料层结构（供正向模拟/灵敏度分析同步）。"""
        sensor = {}
        for param, inp in self.sensor_inputs.items():
            sensor[param] = inp.text().strip()

        layers = []
        layer_params = ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']
        for w in self.layer_widgets:
            layer = {}
            for param in layer_params:
                if param in w['inputs']:
                    layer[param] = w['inputs'][param].text().strip()
            layers.append(layer)
        return {'sensor': sensor, 'layers': layers}

    def _set_last_fitted_param_spec(self, params_to_fit, thickness_gui_layers=None):
        """记录本次拟合实际参与的参数（用于结果表，不依赖事后复选框状态）。"""
        spec = {
            'sensor': list(params_to_fit.get('sensor', [])),
            'layers': {},
        }
        for key, vals in params_to_fit.items():
            if key.startswith('layer_'):
                spec['layers'][int(key.split('_')[1])] = list(vals)
        if thickness_gui_layers:
            spec['thickness_layers'] = list(thickness_gui_layers)
        self._last_fitted_param_spec = spec

    def _was_layer_param_fitted(self, layer_idx, param):
        """判断该材料层参数是否参与了最近一次拟合。"""
        spec = getattr(self, '_last_fitted_param_spec', None)
        if spec:
            thickness_layers = spec.get('thickness_layers', [])
            if param == 'H' and layer_idx in thickness_layers:
                return True
            return param in spec.get('layers', {}).get(layer_idx, [])
        return self._is_layer_param_fitted(layer_idx, param)

    def _was_sensor_param_fitted(self, param):
        """判断该传感器参数是否参与了最近一次拟合。"""
        spec = getattr(self, '_last_fitted_param_spec', None)
        if spec:
            return param in spec.get('sensor', [])
        return self._is_sensor_param_fitted(param)

    def _begin_fitting_ui(self):
        self._fit_cancel_requested = False
        self.fit_btn.setEnabled(False)
        self.stop_fit_btn.setEnabled(True)

    def _end_fitting_ui(self):
        self.fit_btn.setEnabled(True)
        self.stop_fit_btn.setEnabled(False)
        self._fit_cancel_requested = False

    def _should_cancel_fit(self):
        return self._fit_cancel_requested

    def _request_stop_fitting(self):
        self._fit_cancel_requested = True
        self.stop_fit_btn.setEnabled(False)
        self.results_text.append("\n正在中止拟合...\n")
        QApplication.processEvents()

    def _extract_result_from_fitter(self, fitter):
        """从 fitter 当前状态提取结果（正常结束或中止后均可用）。"""
        if isinstance(fitter, JointThermalModelFitter):
            return fitter.layers[::-1], 'material'

        mode = self.fit_mode.currentText()
        if mode == "传感器参数拟合":
            return dict(fitter.sensor_params), 'sensor'
        if mode == "材料+传感器联合拟合":
            return {
                'layers': [l.copy() for l in fitter.layers[::-1]],
                'sensor_params': dict(fitter.sensor_params),
            }, 'combined'
        return [l.copy() for l in fitter.layers[::-1]], 'material'

    def _display_fitted_result(self, fitted_result, result_kind, cancelled=False):
        """在结果区展示拟合/中止后的参数。"""
        if cancelled:
            self.results_text.append("\n拟合已中止，保留当前中间结果。\n")
        else:
            self.results_text.append("\n拟合完成！\n")

        if result_kind == 'sensor':
            self.results_text.append("拟合的传感器参数:")
            for param, value in fitted_result.items():
                self.results_text.append(f"  {param}: {value:.6e}")
        elif result_kind == 'combined':
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
                self.results_text.append(f"\n第 {i} 层:")
                for param, value in layer.items():
                    self.results_text.append(f"  {param}: {value:.6e}")

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
            "材料参数拟合", "传感器参数拟合", "材料+传感器联合拟合", "多线宽联合拟合"
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

        # 参数设置区域（占据主要垂直空间，材料层设置更宽裕）
        self.param_tabs = QTabWidget()
        self.param_tabs.setMinimumHeight(420)
        self.param_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

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

        layout.addWidget(self.param_tabs, 1)

        # 控制按钮
        control_layout = QHBoxLayout()
        self.fit_btn = QPushButton("开始拟合")
        self.fit_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-size: 16px; padding: 10px; }")
        self.stop_fit_btn = QPushButton("中止拟合")
        self.stop_fit_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-size: 16px; padding: 10px; }")
        self.stop_fit_btn.setEnabled(False)
        self.stop_fit_btn.setToolTip("拟合进行中点击可中止迭代，并保留当前中间结果")
        self.export_btn = QPushButton("导出结果")
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-size: 16px; padding: 10px; }")
        self.export_fitted_data_btn = QPushButton("导出拟合数据")
        self.export_fitted_data_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 16px; padding: 10px; }")
        self.export_fitted_data_btn.setToolTip(
            "将拟合模型计算的 V3 数据导出为 CSV/Excel，频率与点数与导入数据一致，不含 V1 列")
        self.prefit_corr_btn = QPushButton("拟合前相关性")
        self.prefit_corr_btn.setStyleSheet(
            "QPushButton { background-color: #607D8B; color: white; font-size: 16px; padding: 10px; }")
        self.prefit_corr_btn.setToolTip(
            "基于当前初值与已勾选的拟合参数，预判参数耦合程度（无需先运行拟合）")
        control_layout.addWidget(self.fit_btn)
        control_layout.addWidget(self.stop_fit_btn)
        control_layout.addWidget(self.prefit_corr_btn)
        control_layout.addWidget(self.export_btn)
        control_layout.addWidget(self.export_fitted_data_btn)
        layout.addLayout(control_layout)

        # 底部结果区：四块横向并列
        results_container = QWidget()
        results_container.setMaximumHeight(300)
        results_row = QHBoxLayout(results_container)
        results_row.setContentsMargins(0, 0, 0, 0)
        results_row.setSpacing(8)

        # 1. 拟合结果文本
        results_text_group = QGroupBox("拟合结果")
        results_text_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        results_text_layout.addWidget(self.results_text)
        results_text_group.setLayout(results_text_layout)
        results_row.addWidget(results_text_group, 1)

        # 2. 拟合结果表
        results_table_group = QGroupBox("拟合结果表")
        results_table_layout = QVBoxLayout()
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["类别", "参数", "拟合值", "标准误差 (±σ)"])
        rt_header = self.results_table.horizontalHeader()
        rt_header.setSectionResizeMode(QHeaderView.Stretch)
        results_table_layout.addWidget(self.results_table)
        results_table_group.setLayout(results_table_layout)
        results_row.addWidget(results_table_group, 1)

        # 相关性分析区文字样式（提示 + 摘要）
        corr_hint_style = (
            "QLabel { color: #2c3e50; font-size: 12px; font-weight: 600; padding: 2px 0; }"
        )
        corr_summary_style = (
            "QLabel { color: #1a1a1a; font-size: 12px; line-height: 140%; padding: 2px 0; }"
        )

        # 3. 拟合前相关性
        prefit_group = QGroupBox("相关性 (拟合前)")
        prefit_layout = QVBoxLayout()
        prefit_hint = QLabel("|ρ|≥0.7 为强耦合")
        prefit_hint.setStyleSheet(corr_hint_style)
        prefit_layout.addWidget(prefit_hint)
        self.prefit_correlation_summary = QLabel("点击「拟合前相关性」")
        self.prefit_correlation_summary.setWordWrap(True)
        self.prefit_correlation_summary.setStyleSheet(corr_summary_style)
        prefit_layout.addWidget(self.prefit_correlation_summary)
        self.prefit_correlation_table = QTableWidget()
        prefit_layout.addWidget(self.prefit_correlation_table)
        prefit_group.setLayout(prefit_layout)
        results_row.addWidget(prefit_group, 1)

        # 4. 拟合后相关性
        postfit_group = QGroupBox("相关性 (拟合后)")
        postfit_layout = QVBoxLayout()
        postfit_hint = QLabel("|ρ|≥0.7 为强耦合")
        postfit_hint.setStyleSheet(corr_hint_style)
        postfit_layout.addWidget(postfit_hint)
        self.correlation_summary = QLabel("拟合完成后自动显示")
        self.correlation_summary.setWordWrap(True)
        self.correlation_summary.setStyleSheet(corr_summary_style)
        postfit_layout.addWidget(self.correlation_summary)
        self.correlation_table = QTableWidget()
        postfit_layout.addWidget(self.correlation_table)
        postfit_group.setLayout(postfit_layout)
        results_row.addWidget(postfit_group, 1)

        layout.addWidget(results_container, 0)

        self.setLayout(layout)

        # 连接信号
        self.load_btn.clicked.connect(self.load_data)
        self.fit_btn.clicked.connect(self.run_fitting)
        self.stop_fit_btn.clicked.connect(self._request_stop_fitting)
        self.export_btn.clicked.connect(self.export_results)
        self.export_fitted_data_btn.clicked.connect(self.export_fitted_data)
        self.prefit_corr_btn.clicked.connect(self.run_prefit_correlation_analysis)
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

        # 相邻两层总厚度约束（已知总厚、未知分配）
        constraint_group = QGroupBox("相邻两层总厚度约束")
        constraint_layout = QGridLayout()

        self.tc_enable = QCheckBox("启用：已知相邻两层总厚度，自动拟合厚度分配")
        self.tc_enable.setToolTip(
            "勾选后，将固定所选相邻两层的厚度之和为给定总厚度。\n"
            "上层厚度作为单一自由度，与其余勾选参数一起进入\n"
            "同一个高斯-牛顿迭代中同时拟合（下层厚度 = 总厚度 - 上层厚度）。"
        )
        constraint_layout.addWidget(self.tc_enable, 0, 0, 1, 4)

        constraint_layout.addWidget(QLabel("上层(第i层):"), 1, 0)
        self.tc_upper = QSpinBox()
        self.tc_upper.setRange(1, max(1, len(self.layer_widgets) if hasattr(self, 'layer_widgets') else 1))
        self.tc_upper.setValue(1)
        self.tc_upper.setToolTip("约束作用于第 i 层与第 i+1 层（界面从上到下编号）")
        constraint_layout.addWidget(self.tc_upper, 1, 1)
        self.tc_pair_label = QLabel("→ 约束第 1 层 + 第 2 层")
        constraint_layout.addWidget(self.tc_pair_label, 1, 2, 1, 2)
        self.tc_upper.valueChanged.connect(self._update_tc_pair_label)

        constraint_layout.addWidget(QLabel("总厚度 H_total (m):"), 2, 0)
        self.tc_total = QLineEdit()
        self.tc_total.setPlaceholderText("例如 2.0e-3")
        constraint_layout.addWidget(self.tc_total, 2, 1)

        constraint_layout.addWidget(QLabel("上层占比范围:"), 2, 2)
        frac_widget = QWidget()
        frac_layout = QHBoxLayout(frac_widget)
        frac_layout.setContentsMargins(0, 0, 0, 0)
        self.tc_fmin = QLineEdit("0.01")
        self.tc_fmin.setMaximumWidth(60)
        self.tc_fmin.setToolTip("上层厚度占总厚度的最小比例")
        frac_layout.addWidget(self.tc_fmin)
        frac_layout.addWidget(QLabel("-"))
        self.tc_fmax = QLineEdit("0.99")
        self.tc_fmax.setMaximumWidth(60)
        self.tc_fmax.setToolTip("上层厚度占总厚度的最大比例")
        frac_layout.addWidget(self.tc_fmax)
        constraint_layout.addWidget(frac_widget, 2, 3)

        hint = QLabel(
            "说明：启用后，所选两层的『厚度 H』勾选框将被忽略，"
            "厚度由总厚度约束自动确定；其余勾选的参数仍按常规拟合。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        constraint_layout.addWidget(hint, 3, 0, 1, 4)

        constraint_group.setLayout(constraint_layout)
        main_layout.addWidget(constraint_group)

        # 层参数滚动区域
        scroll_area = QScrollArea()
        scroll_area.setMinimumHeight(360)
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.layers_layout = QVBoxLayout(scroll_widget)

        # 初始化两层
        for i in range(2):
            self.add_layer_widget(i)

        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area, 1)

        widget.setLayout(main_layout)

        # 初始化总厚度约束选择器范围
        self._sync_tc_range()

        return widget

    def _create_fitting_layer_op_button(self, text, tooltip, container, is_add):
        btn = QPushButton(text)
        btn.setFixedSize(26, 26)
        btn.setToolTip(tooltip)
        btn.setProperty('layer_container', container)
        if is_add:
            btn.setStyleSheet(
                "QPushButton { color: #4CAF50; font-weight: bold; font-size: 16px; "
                "border: 1px solid #4CAF50; border-radius: 3px; padding: 0; }"
                "QPushButton:hover { background-color: #E8F5E9; }"
            )
            btn.clicked.connect(self._on_fitting_layer_add_clicked)
        else:
            btn.setStyleSheet(
                "QPushButton { color: #f44336; font-weight: bold; font-size: 16px; "
                "border: 1px solid #f44336; border-radius: 3px; padding: 0; }"
                "QPushButton:hover { background-color: #FFEBEE; }"
            )
            btn.clicked.connect(self._on_fitting_layer_remove_clicked)
        return btn

    def _fitting_layer_index_of(self, container):
        for i, w in enumerate(self.layer_widgets):
            if w.get('container') == container:
                return i
        return -1

    def _on_fitting_layer_add_clicked(self):
        container = self.sender().property('layer_container')
        idx = self._fitting_layer_index_of(container)
        if idx < 0:
            return
        if len(self.layer_widgets) >= 10:
            QMessageBox.warning(self, "提示", "材料层数最多为 10 层。")
            return
        self.add_layer_widget(idx)

    def _on_fitting_layer_remove_clicked(self):
        container = self.sender().property('layer_container')
        idx = self._fitting_layer_index_of(container)
        if idx < 0:
            return
        if len(self.layer_widgets) <= 1:
            QMessageBox.warning(self, "提示", "至少保留一层材料结构。")
            return
        widget_info = self.layer_widgets.pop(idx)
        self.layers_layout.removeWidget(widget_info['container'])
        widget_info['container'].deleteLater()
        self._renumber_layer_widgets()
        self._sync_tc_range()

    def add_layer_widget(self, layer_index):
        """在指定位置插入单层材料设置界面。"""
        layer_index = max(0, min(int(layer_index), len(self.layer_widgets)))

        container = QWidget()
        container.setObjectName("fittingLayerBlock")
        container.setStyleSheet(
            "#fittingLayerBlock { border: 1px solid #bdbdbd; border-radius: 4px; "
            "background: #fafafa; margin-top: 2px; }"
        )
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(8, 6, 8, 8)

        header = QHBoxLayout()
        title_label = QLabel(f"第 {layer_index + 1} 层")
        title_label.setStyleSheet("font-weight: bold;")
        header.addWidget(title_label)
        header.addStretch()
        btn_add = self._create_fitting_layer_op_button('+', "在该层上方插入新层", container, is_add=True)
        btn_remove = self._create_fitting_layer_op_button('−', "删除该层", container, is_add=False)
        header.addWidget(btn_add)
        header.addWidget(btn_remove)
        container_layout.addLayout(header)

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
            lambda text, combo=material_combo: self._on_material_preset_changed(combo, text)
        )

        content = QWidget()
        content.setLayout(layout)
        container_layout.addWidget(content)

        self.layers_layout.insertWidget(layer_index, container)
        self.layer_widgets.insert(layer_index, {
            'container': container,
            'title_label': title_label,
            'inputs': inputs,
        })

        material_combo.setCurrentIndex(0)
        self._renumber_layer_widgets()

    def _on_material_preset_changed(self, combo, material_name):
        name = combo.objectName()
        match = re.match(r'^layer_(\d+)_preset$', name)
        if match:
            self.load_material_preset(int(match.group(1)), material_name)

    @staticmethod
    def _set_layer_widget_index(container, title_label, new_index):
        """更新单层控件标题及 objectName（供 findChild 按层索引查找）。"""
        title_label.setText(f"第 {new_index + 1} 层")
        for child in container.findChildren(QWidget):
            name = child.objectName()
            if name.startswith('layer_'):
                child.setObjectName(re.sub(r'^layer_\d+', f'layer_{new_index}', name))

    def _renumber_layer_widgets(self):
        """按当前顺序重排各层编号与控件 objectName。"""
        for i, w in enumerate(self.layer_widgets):
            self._set_layer_widget_index(w['container'], w['title_label'], i)
        self._sync_tc_range()

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

    def _update_tc_pair_label(self, *args):
        """更新总厚度约束所作用层对的提示文字"""
        if hasattr(self, 'tc_pair_label') and hasattr(self, 'tc_upper'):
            i = self.tc_upper.value()
            self.tc_pair_label.setText(f"→ 约束第 {i} 层 + 第 {i + 1} 层")

    def _sync_tc_range(self):
        """根据当前层数同步总厚度约束上层选择器的取值范围"""
        if not hasattr(self, 'tc_upper'):
            return
        n = len(self.layer_widgets)
        # 上层索引最多为 n-1，保证存在相邻的下一层
        max_upper = max(1, n - 1)
        self.tc_upper.setMaximum(max_upper)
        if self.tc_upper.value() > max_upper:
            self.tc_upper.setValue(max_upper)
        self._update_tc_pair_label()

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

    def update_fitting_mode(self, mode):
        """更新拟合模式相关的UI"""
        is_sensor_only = mode == "传感器参数拟合"
        is_combined = mode == "材料+传感器联合拟合"
        is_joint_mode = mode == "多线宽联合拟合"

        show_sensor_fit = is_sensor_only or is_combined
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

        show_material_fit = (not is_sensor_only) or is_combined
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
        self.results_table.setItem(row, 3, QTableWidgetItem("—"))

    def _update_uncertainty_column(self, fitter):
        """拟合完成后：用残差估计噪声，把各参数标准误差填入结果表第4列。"""
        try:
            result = fitter.compute_parameter_uncertainty(gui_order=True)
        except Exception as e:
            self.results_text.append(f"\n参数标准误差估计失败: {e}\n")
            return

        if result.get('error'):
            self.results_text.append(f"\n参数标准误差: {result['error']}\n")
            return

        # fitter 标签 -> (绝对σ, 相对σ)；标签形如 "第3层.k_y" / "传感器.R"
        sigma_map = {}
        for label, s_abs, s_rel in zip(
                result['labels'], result['sigma_abs'], result['sigma_rel']):
            m = re.match(r"第(\d+)层\.(.+)", label)
            if m:
                key = (f"第 {m.group(1)} 层", m.group(2))
            elif label.startswith("传感器."):
                key = ("传感器参数", label.split('.', 1)[1])
            else:
                continue  # 厚度约束等特殊自由度不进常规行
            sigma_map[key] = (s_abs, s_rel)

        for row in range(self.results_table.rowCount()):
            cat_item = self.results_table.item(row, 0)
            name_item = self.results_table.item(row, 1)
            if cat_item is None or name_item is None:
                continue
            entry = sigma_map.get((cat_item.text(), name_item.text()))
            if entry is None:
                continue
            s_abs, s_rel = entry
            text = f"±{s_abs:.3e} ({s_rel * 100:.1f}%)"
            self.results_table.setItem(row, 3, QTableWidgetItem(text))

        self.results_text.append(
            f"\n噪声水平估计 s = {result['noise_std']:.3e}"
            f"（残差自由度 {result['dof']}），标准误差已填入结果表。\n"
        )

    def _format_result_value(self, value):
        if isinstance(value, str):
            return value
        if isinstance(value, (float, int, np.floating, np.integer)):
            if np.isinf(value):
                return "inf"
            return f"{value:.6e}"
        return str(value)

    def _is_layer_param_fitted(self, layer_idx, param):
        """判断材料层参数是否参与本次拟合。"""
        checkbox = self.findChild(QCheckBox, f"layer_{layer_idx}_{param}_fit")
        return checkbox is not None and checkbox.isVisible() and checkbox.isChecked()

    def _is_sensor_param_fitted(self, param):
        """判断传感器参数是否参与本次拟合。"""
        checkbox = self.sensor_widget.findChild(QCheckBox, f"sensor_{param}_fit")
        return checkbox is not None and checkbox.isVisible() and checkbox.isChecked()

    def update_results_table(self, fitted_result, result_kind):
        """刷新拟合结果表格，仅显示参与拟合的参数。result_kind: 'sensor' | 'material' | 'combined'"""
        self.results_table.setRowCount(0)
        if fitted_result is None:
            return

        if result_kind == 'sensor':
            for param, value in fitted_result.items():
                if self._was_sensor_param_fitted(param):
                    self._append_result_row("传感器参数", param, value)
        elif result_kind == 'combined':
            for param, value in fitted_result['sensor_params'].items():
                if self._was_sensor_param_fitted(param):
                    self._append_result_row("传感器参数", param, value)
            for idx, layer in enumerate(fitted_result['layers']):
                for param, value in layer.items():
                    if self._was_layer_param_fitted(idx, param):
                        self._append_result_row(f"第 {idx + 1} 层", param, value)
        else:
            for idx, layer in enumerate(fitted_result):
                for param, value in layer.items():
                    if self._was_layer_param_fitted(idx, param):
                        self._append_result_row(f"第 {idx + 1} 层", param, value)

    def _get_fit_method_code(self):
        fit_method_map = {
            "Amplitude": "amp",
            "Phase": "phase",
            "Real+Imaginary": "both",
            "Amplitude+Phase": "both",
            "Real Part Only": "real",
            "Imaginary Part Only": "imag",
        }
        return fit_method_map[self.fit_method.currentText()]

    def _collect_layers_from_gui(self):
        layers = []
        for i in range(len(self.layer_widgets)):
            layer = {}
            for param in ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']:
                input_widget = self.findChild(QLineEdit, f"layer_{i}_{param}")
                if input_widget:
                    text = input_widget.text()
                    layer[param] = np.inf if text.lower() == 'inf' else float(text)
            layers.append(layer)
        return layers

    def _collect_sensor_params_from_gui(self):
        sensor_params = {}
        for param, input_widget in self.sensor_inputs.items():
            try:
                sensor_params[param] = float(input_widget.text())
            except Exception as exc:
                raise ValueError(f"传感器参数 {param} 输入无效") from exc
        return sensor_params

    def _collect_params_to_fit_from_gui(self, layers, mode, skip_h_layers=None):
        """收集 GUI 中勾选的拟合参数。skip_h_layers 为 GUI 层索引集合（厚度约束时跳过 H）。"""
        skip_h_layers = skip_h_layers or set()
        is_sensor_only = mode == "传感器参数拟合"
        is_combined = mode == "材料+传感器联合拟合"
        params_to_fit = {}
        param_bounds = {}

        if not is_sensor_only:
            for i, layer in enumerate(layers):
                layer_params_to_fit = []
                layer_bounds = {}
                for param in ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']:
                    if param == 'H' and i in skip_h_layers:
                        continue
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
                                current_val = layer[param]
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
                        try:
                            current_val = float(self.sensor_inputs[param].text())
                        except Exception:
                            current_val = 1.0
                        if param == 'dR_dT':
                            sensor_bounds[param] = (current_val * 0.1, current_val * 10)
                        else:
                            sensor_bounds[param] = (current_val * 0.5, current_val * 2.0)
            if sensor_params_to_fit:
                params_to_fit['sensor'] = sensor_params_to_fit
                param_bounds['sensor'] = sensor_bounds

        return params_to_fit, param_bounds

    @staticmethod
    def _map_gui_params_to_model(params_to_fit, param_bounds, n_layers):
        fitter_params = {}
        fitter_bounds = {}
        if 'sensor' in params_to_fit:
            fitter_params['sensor'] = params_to_fit['sensor']
            fitter_bounds['sensor'] = param_bounds['sensor']
        for key, val in params_to_fit.items():
            if key.startswith('layer_'):
                gui_idx = int(key.split('_')[1])
                model_idx = n_layers - 1 - gui_idx
                fitter_params[f'layer_{model_idx}'] = val
        for key, val in param_bounds.items():
            if key.startswith('layer_'):
                gui_idx = int(key.split('_')[1])
                model_idx = n_layers - 1 - gui_idx
                fitter_bounds[f'layer_{model_idx}'] = val
        return fitter_params, fitter_bounds

    def _build_joint_datasets_from_gui(self):
        datasets = []
        calibrator = self._get_calibrator()
        apply_cal = self.apply_calibration_cb.isChecked() and calibrator is not None and calibrator.is_calibrated
        for ds_info in self.joint_datasets_ui:
            df = ds_info['data']
            if df is None:
                raise ValueError("联合拟合数据集中存在未加载的数据文件")
            freqs = df.iloc[:, 0].values.astype(float)
            V3_exp = self._parse_v3_from_dataframe(df)
            if apply_cal:
                V3_exp = calibrator.correct_v3omega(freqs, V3_exp)
            sp = {param: float(inp.text()) for param, inp in ds_info['sensor_inputs'].items()}
            datasets.append({
                'freqs': freqs,
                'V3_exp': V3_exp,
                'b': sp['b'],
                'sensor_params': sp,
            })
        return datasets

    def _build_correlation_probe_fitter(self):
        """基于当前初值与勾选参数构建探针 fitter（不执行拟合）。"""
        mode = self.fit_mode.currentText()
        fit_method = self._get_fit_method_code()

        if mode == "多线宽联合拟合":
            if not self.joint_datasets_ui:
                return None, "请先添加至少一个联合拟合数据集！"
            try:
                datasets = self._build_joint_datasets_from_gui()
                layers = self._collect_layers_from_gui()
                params_to_fit, param_bounds = self._collect_params_to_fit_from_gui(layers, mode)
            except ValueError as e:
                return None, str(e)
            if not params_to_fit:
                return None, "请在材料层设置中至少勾选一个拟合参数！"
            N = len(layers)
            fitter_params, fitter_bounds = self._map_gui_params_to_model(params_to_fit, param_bounds, N)
            fitter = JointThermalModelFitter(
                datasets=datasets,
                initial_layers=[l.copy() for l in layers[::-1]],
                params_to_fit=fitter_params,
                fit_mode=fit_method,
                param_bounds=fitter_bounds,
            )
            return fitter, None

        if self.fitting_data is None:
            return None, "请先加载数据文件！"

        freqs, V3_exp = self._build_experiment_v3(silent=True)
        if freqs is None:
            return None, "选定的频率区间内没有数据点！"

        try:
            sensor_params = self._collect_sensor_params_from_gui()
            layers = self._collect_layers_from_gui()
        except ValueError as e:
            return None, str(e)

        N = len(layers)
        thickness_link = None
        skip_h_layers = set()
        use_tc = (hasattr(self, 'tc_enable') and self.tc_enable.isChecked()
                  and mode != "传感器参数拟合")

        if use_tc:
            gui_upper = self.tc_upper.value() - 1
            gui_lower = gui_upper + 1
            if gui_lower >= N:
                return None, "厚度约束所选层超出当前层数范围！"
            try:
                H_total = float(self.tc_total.text())
            except Exception:
                return None, "总厚度 H_total 输入无效！"
            if not (H_total > 0):
                return None, "总厚度 H_total 必须为正数！"
            try:
                f_min = float(self.tc_fmin.text())
                f_max = float(self.tc_fmax.text())
            except Exception:
                f_min, f_max = 0.01, 0.99
            f_min = min(max(f_min, 1e-4), 1 - 1e-4)
            f_max = min(max(f_max, 1e-4), 1 - 1e-4)
            if f_min >= f_max:
                f_min, f_max = 0.01, 0.99
            skip_h_layers = {gui_upper, gui_lower}
            m_upper = N - 1 - gui_upper
            m_lower = N - 1 - gui_lower
            thickness_link = {
                'upper_idx': m_upper,
                'lower_idx': m_lower,
                'H_total': H_total,
                'bounds': (f_min * H_total, f_max * H_total),
            }

        is_sensor_only = mode == "传感器参数拟合"
        is_combined = mode == "材料+传感器联合拟合"
        params_to_fit, param_bounds = self._collect_params_to_fit_from_gui(
            layers, mode, skip_h_layers=skip_h_layers)

        if not params_to_fit and thickness_link is None:
            return None, "请至少勾选一个拟合参数！"

        if is_combined:
            has_mat = any(k.startswith('layer_') for k in params_to_fit)
            has_sens = 'sensor' in params_to_fit
            if not has_mat:
                return None, "材料+传感器联合拟合需至少勾选一项材料层参数"
            if not has_sens:
                return None, "材料+传感器联合拟合需至少勾选一项传感器参数"

        fitter_params, fitter_bounds = self._map_gui_params_to_model(params_to_fit, param_bounds, N)
        fit_type = 'sensor' if is_sensor_only else ('combined' if is_combined else 'material')
        model_layers = [l.copy() for l in layers[::-1]]

        if thickness_link is not None:
            H_total = thickness_link['H_total']
            cur_up = layers[gui_upper].get('H', np.nan)
            cur_lo = layers[gui_lower].get('H', np.nan)
            if (np.isfinite(cur_up) and np.isfinite(cur_lo) and cur_up > 0 and cur_lo > 0):
                init_frac = cur_up / (cur_up + cur_lo)
            else:
                init_frac = 0.5
            f_min = thickness_link['bounds'][0] / H_total
            f_max = thickness_link['bounds'][1] / H_total
            init_frac = float(np.clip(init_frac, f_min, f_max))
            model_layers[m_upper]['H'] = init_frac * H_total
            model_layers[m_lower]['H'] = (1.0 - init_frac) * H_total

        fitter = ThermalModelFitter(
            freqs=freqs,
            V3_exp=V3_exp,
            b=sensor_params['b'],
            sensor_params=sensor_params,
            initial_layers=model_layers,
            params_to_fit=fitter_params,
            fit_mode=fit_method,
            param_bounds=fitter_bounds,
            fit_type=fit_type,
            thickness_link=thickness_link,
        )
        return fitter, None

    def run_prefit_correlation_analysis(self):
        """拟合前：基于当前初值与勾选参数预判参数耦合。"""
        self.prefit_correlation_summary.setText("正在计算拟合前相关性，请稍候...")
        QApplication.processEvents()
        try:
            fitter, err = self._build_correlation_probe_fitter()
            if err:
                self._display_correlation_result(
                    {'error': err, 'labels': [], 'matrix': None, 'strong_pairs': [], 'condition_number': np.nan},
                    target='prefit',
                )
                QMessageBox.warning(self, "提示", err)
                return
            result = fitter.compute_parameter_correlation(gui_order=True)
            self._display_correlation_result(result, target='prefit')
        except Exception as e:
            self.prefit_correlation_table.setRowCount(0)
            self.prefit_correlation_table.setColumnCount(0)
            self.prefit_correlation_summary.setText(f"拟合前相关性分析失败: {e}")
            QMessageBox.critical(self, "错误", f"拟合前相关性分析失败:\n{e}")

    def update_correlation_analysis(self, fitter=None):
        """拟合完成后，基于收敛解处的灵敏度矩阵计算并显示参数相关系数。"""
        fitter = fitter if fitter is not None else getattr(self, 'fitter', None)
        if fitter is None:
            self.correlation_table.setRowCount(0)
            self.correlation_table.setColumnCount(0)
            self.correlation_summary.setText("拟合完成后自动显示")
            return

        try:
            self.correlation_summary.setText("[拟合后] 正在计算参数相关性...")
            QApplication.processEvents()
            result = fitter.compute_parameter_correlation(gui_order=True)
            self._display_correlation_result(result, target='post')
        except Exception as e:
            self.correlation_table.setRowCount(0)
            self.correlation_table.setColumnCount(0)
            self.correlation_summary.setText(f"相关性分析失败: {e}")

        self._update_uncertainty_column(fitter)

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

        summary_parts = [f"条件数 κ = {cond:.2e}"]
        if cond > 1e6:
            summary_parts.append("整体病态，参数辨识性较差")
        if strong:
            pairs = [f"{a} ↔ {b} (ρ={rho:+.3f})" for a, b, rho in strong[:6]]
            summary_parts.append("强耦合: " + "; ".join(pairs))
        else:
            summary_parts.append("未发现强耦合参数对 (|ρ|≥0.7)")
        summary.setText(prefix + " | ".join(summary_parts))

        table.setRowCount(n)
        table.setColumnCount(n)
        table.setHorizontalHeaderLabels(labels)
        table.setVerticalHeaderLabels(labels)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

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
        self._last_fitted_param_spec = None

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
            self.correlation_table.setRowCount(0)
            self.correlation_table.setColumnCount(0)
            self.correlation_summary.setText("拟合中...")

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

            self._set_last_fitted_param_spec(params_to_fit)

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

            self._begin_fitting_ui()
            try:
                self.results_text.append("开始多线宽联合拟合...\n")
                QApplication.processEvents()

                fitted_result_raw = fitter.fit(
                    max_iter=self.max_iter.value(),
                    tol=float(self.tolerance.text()),
                    zeta=self.zeta.value(),
                    should_cancel=self._should_cancel_fit,
                )

                fitted_result = fitted_result_raw[::-1]

                self._display_fitted_result(fitted_result, 'material')
                self.update_results_table(fitted_result, 'material')
                self.fitted_result = fitted_result
                self.fitter = fitter
                self.update_correlation_analysis(fitter)
            except FitCancelledError:
                fitted_result, result_kind = self._extract_result_from_fitter(fitter)
                self._display_fitted_result(fitted_result, result_kind, cancelled=True)
                self.update_results_table(fitted_result, result_kind)
                self.fitted_result = fitted_result
                self.fitter = fitter
                self.update_correlation_analysis(fitter)
            finally:
                sys.stdout = original_stdout
                self._end_fitting_ui()

        except FitCancelledError:
            pass
        except Exception as e:
            self.results_text.append(f"\n联合拟合失败: {str(e)}\n")
            traceback.print_exc()
            if hasattr(self, 'ax1'):
                self.ax1.clear()
            if hasattr(self, 'ax2'):
                self.ax2.clear()
            if hasattr(self, 'figure') and hasattr(self.figure, 'canvas'):
                self.figure.canvas.draw()

    def run_fitting(self):
        """执行拟合"""
        # 每次拟合前清除上次的拟合结果引用，防止失败时显示旧数据
        if hasattr(self, 'fitter'):
            del self.fitter
        if hasattr(self, 'fitted_result'):
            del self.fitted_result
        self._last_fitted_param_spec = None

        if self.fit_mode.currentText() == "多线宽联合拟合":
            self.run_joint_fitting()
            return

        if self.fitting_data is None:
            QMessageBox.warning(self, "警告", "请先加载数据文件！")
            return

        # 若启用了相邻两层总厚度约束（仅适用于含材料层的拟合模式），走专用流程
        if (hasattr(self, 'tc_enable') and self.tc_enable.isChecked()
                and self.fit_mode.currentText() != "传感器参数拟合"):
            self.run_constrained_fitting()
            return

        try:
            self.results_table.setRowCount(0)
            self.results_text.clear()
            self.correlation_table.setRowCount(0)
            self.correlation_table.setColumnCount(0)
            self.correlation_summary.setText("拟合中...")

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

            self._set_last_fitted_param_spec(params_to_fit)

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

            self._begin_fitting_ui()
            try:
                # 执行拟合
                self.results_text.append("开始拟合...\n")
                QApplication.processEvents()

                fitted_result_raw = fitter.fit(
                    max_iter=self.max_iter.value(),
                    tol=float(self.tolerance.text()),
                    zeta=self.zeta.value(),
                    should_cancel=self._should_cancel_fit,
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
                    self._display_fitted_result(fitted_result, result_kind)
                    self.update_results_table(fitted_result, result_kind)
                    self.fitted_result = fitted_result
                    self.fitter = fitter
                    self.update_correlation_analysis(fitter)
                else:
                    self.correlation_summary.setText("拟合未成功，无相关性结果")
            except FitCancelledError:
                fitted_result, result_kind = self._extract_result_from_fitter(fitter)
                self._display_fitted_result(fitted_result, result_kind, cancelled=True)
                self.update_results_table(fitted_result, result_kind)
                self.fitted_result = fitted_result
                self.fitter = fitter
                self.update_correlation_analysis(fitter)
            finally:
                # 恢复标准输出
                sys.stdout = original_stdout
                self._end_fitting_ui()

        except FitCancelledError:
            pass
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

    def _build_experiment_v3(self, silent=False):
        """根据已加载数据、频率筛选与信号校正返回 (freqs, V3_exp)。失败返回 (None, None)。"""
        freqs = self.fitting_data.iloc[:, 0].values.astype(float)

        use_filter = self.freq_min_selector.isEnabled() and self.freq_max_selector.isEnabled()
        if use_filter:
            min_freq = self.freq_min_selector.value()
            max_freq = self.freq_max_selector.value()
            mask = (freqs >= min_freq) & (freqs <= max_freq)
        else:
            mask = np.ones(freqs.shape, dtype=bool)

        if not np.any(mask):
            if not silent:
                QMessageBox.warning(self, "警告", "选定的频率区间内没有数据点！")
            return None, None
        freqs = freqs[mask]

        if 'v3_real' in self.fitting_data.columns and 'v3_imag' in self.fitting_data.columns:
            real_vals = self.fitting_data['v3_real'].values.astype(float)
            imag_vals = self.fitting_data['v3_imag'].values.astype(float)
        elif 'V3_Real' in self.fitting_data.columns:
            real_vals = self.fitting_data['V3_Real'].values.astype(float)
            imag_vals = self.fitting_data['V3_Imag'].values.astype(float)
        else:
            real_vals = self.fitting_data.iloc[:, 1].values.astype(float)
            imag_vals = self.fitting_data.iloc[:, 2].values.astype(float)

        V3_exp = real_vals[mask] + 1j * imag_vals[mask]

        calibrator = self._get_calibrator()
        if self.apply_calibration_cb.isChecked() and calibrator is not None and calibrator.is_calibrated:
            V3_exp = calibrator.correct_v3omega(freqs, V3_exp)
        return freqs, V3_exp

    @staticmethod
    def _residual_norm(V3_exp, V3_model, fit_method):
        """按拟合模式计算实验与模型的残差平方和。"""
        if V3_model is None or not np.all(np.isfinite(V3_model)):
            return np.inf
        if fit_method == 'real':
            err = V3_exp.real - V3_model.real
        elif fit_method == 'imag':
            err = V3_exp.imag - V3_model.imag
        elif fit_method == 'amp':
            err = np.abs(V3_exp) - np.abs(V3_model)
        elif fit_method == 'phase':
            err = np.unwrap(np.angle(V3_exp)) - np.unwrap(np.angle(V3_model))
        else:  # both
            err = np.concatenate([V3_exp.real - V3_model.real,
                                  V3_exp.imag - V3_model.imag])
        return float(np.sum(err ** 2))

    def run_constrained_fitting(self):
        """相邻两层总厚度约束拟合（厚度参与高斯-牛顿迭代）。

        已知相邻两层 (i, i+1) 的总厚度 H_total，但不知道各自厚度。
        将上层厚度 H_upper 作为单一自由度，下层厚度由 H_lower = H_total - H_upper
        自动推出。该自由度与其余勾选的材料/传感器参数一起，进入同一个
        ThermalModelFitter 高斯-牛顿迭代中同时拟合（通过 thickness_link 机制）。
        """
        if hasattr(self, 'fitter'):
            del self.fitter
        if hasattr(self, 'fitted_result'):
            del self.fitted_result
        self._last_fitted_param_spec = None

        try:
            self.results_table.setRowCount(0)
            self.results_text.clear()
            self.correlation_table.setRowCount(0)
            self.correlation_table.setColumnCount(0)
            self.correlation_summary.setText("拟合中...")

            N = len(self.layer_widgets)
            gui_upper = self.tc_upper.value() - 1  # 0-based GUI 上层索引
            gui_lower = gui_upper + 1
            if gui_lower >= N:
                QMessageBox.warning(self, "警告", "约束所选层超出当前层数范围！")
                return

            try:
                H_total = float(self.tc_total.text())
            except Exception:
                QMessageBox.warning(self, "警告", "总厚度 H_total 输入无效！")
                return
            if not (H_total > 0):
                QMessageBox.warning(self, "警告", "总厚度 H_total 必须为正数！")
                return

            try:
                f_min = float(self.tc_fmin.text())
                f_max = float(self.tc_fmax.text())
            except Exception:
                f_min, f_max = 0.01, 0.99
            f_min = min(max(f_min, 1e-4), 1 - 1e-4)
            f_max = min(max(f_max, 1e-4), 1 - 1e-4)
            if f_min >= f_max:
                f_min, f_max = 0.01, 0.99

            # 实验数据
            freqs, V3_exp = self._build_experiment_v3()
            if freqs is None:
                return

            # 传感器参数
            sensor_params = {}
            for param, input_widget in self.sensor_inputs.items():
                try:
                    sensor_params[param] = float(input_widget.text())
                except Exception:
                    QMessageBox.warning(self, "警告", f"传感器参数 {param} 输入无效")
                    return

            # 材料层参数（GUI 顺序：顶层→基底）
            layers = []
            for i in range(N):
                layer = {}
                for param in ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']:
                    input_widget = self.findChild(QLineEdit, f"layer_{i}_{param}")
                    if input_widget:
                        text = input_widget.text()
                        layer[param] = np.inf if text.lower() == 'inf' else float(text)
                layers.append(layer)

            mode = self.fit_mode.currentText()
            is_combined = mode == "材料+传感器联合拟合"

            # 收集其余待拟合参数（排除被约束两层的厚度 H，厚度由 thickness_link 处理）
            params_to_fit = {}
            param_bounds = {}
            for i in range(N):
                layer_params_to_fit = []
                layer_bounds = {}
                for param in ['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']:
                    if param == 'H' and i in (gui_upper, gui_lower):
                        continue  # 厚度由总厚度约束自由度处理，忽略勾选
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
                                cv = layers[i][param]
                                layer_bounds[param] = (cv * 0.1, cv * 10)
                if layer_params_to_fit:
                    params_to_fit[f'layer_{i}'] = layer_params_to_fit
                    param_bounds[f'layer_{i}'] = layer_bounds

            if is_combined:
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
                            sensor_bounds[param] = (float(min_input.text()), float(max_input.text()))
                        else:
                            cv = sensor_params[param]
                            if param == 'dR_dT':
                                sensor_bounds[param] = (cv * 0.1, cv * 10)
                            else:
                                sensor_bounds[param] = (cv * 0.5, cv * 2.0)
                if sensor_params_to_fit:
                    params_to_fit['sensor'] = sensor_params_to_fit
                    param_bounds['sensor'] = sensor_bounds

            fit_method_map = {
                "Amplitude": "amp", "Phase": "phase",
                "Real+Imaginary": "both", "Amplitude+Phase": "both",
                "Real Part Only": "real", "Imaginary Part Only": "imag"
            }
            fit_method = fit_method_map[self.fit_method.currentText()]

            # 模型层顺序（基底→顶层）与索引映射
            base_model_layers = [l.copy() for l in layers[::-1]]
            m_upper = N - 1 - gui_upper
            m_lower = N - 1 - gui_lower  # == m_upper - 1

            # 上层厚度初值：依据当前两层厚度的比例，否则取总厚度一半
            cur_up = layers[gui_upper].get('H', np.nan)
            cur_lo = layers[gui_lower].get('H', np.nan)
            if (np.isfinite(cur_up) and np.isfinite(cur_lo)
                    and cur_up > 0 and cur_lo > 0):
                init_frac = cur_up / (cur_up + cur_lo)
            else:
                init_frac = 0.5
            init_frac = float(np.clip(init_frac, f_min, f_max))
            base_model_layers[m_upper]['H'] = init_frac * H_total
            base_model_layers[m_lower]['H'] = (1.0 - init_frac) * H_total

            # 拟合参数键名映射到模型层索引
            fitter_params_to_fit = {}
            fitter_param_bounds = {}
            if 'sensor' in params_to_fit:
                fitter_params_to_fit['sensor'] = params_to_fit['sensor']
                fitter_param_bounds['sensor'] = param_bounds['sensor']
            for key, val in params_to_fit.items():
                if key.startswith('layer_'):
                    gidx = int(key.split('_')[1])
                    midx = N - 1 - gidx
                    fitter_params_to_fit[f'layer_{midx}'] = val
                    fitter_param_bounds[f'layer_{midx}'] = param_bounds[key]

            # 厚度约束自由度（参与高斯-牛顿迭代）
            thickness_link = {
                'upper_idx': m_upper,
                'lower_idx': m_lower,
                'H_total': H_total,
                'bounds': (f_min * H_total, f_max * H_total),
            }

            self._set_last_fitted_param_spec(
                params_to_fit, thickness_gui_layers=(gui_upper, gui_lower)
            )

            fit_type = 'combined' if is_combined else 'material'

            fitter = ThermalModelFitter(
                freqs=freqs, V3_exp=V3_exp, b=sensor_params['b'],
                sensor_params=sensor_params, initial_layers=base_model_layers,
                params_to_fit=fitter_params_to_fit, fit_mode=fit_method,
                param_bounds=fitter_param_bounds, fit_type=fit_type,
                thickness_link=thickness_link,
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

            cancelled = False
            self._begin_fitting_ui()
            try:
                self.results_text.append(
                    "开始相邻两层总厚度约束拟合（厚度作为单一自由度参与高斯-牛顿迭代）...")
                self.results_text.append(
                    f"约束：第 {gui_upper + 1} 层 + 第 {gui_lower + 1} 层 = {H_total:.4e} m\n")
                QApplication.processEvents()

                fitter.fit(
                    max_iter=self.max_iter.value(),
                    tol=float(self.tolerance.text()),
                    zeta=self.zeta.value(),
                    should_cancel=self._should_cancel_fit,
                )
            except FitCancelledError:
                cancelled = True
            finally:
                sys.stdout = original_stdout
                self._end_fitting_ui()

            # 提取最终厚度（模型层顺序）
            H_upper = fitter.layers[m_upper]['H']
            H_lower = fitter.layers[m_lower]['H']
            best_frac = H_upper / H_total if H_total > 0 else float('nan')

            # 结果（GUI 顺序：顶层→基底）
            gui_layers = [l.copy() for l in fitter.layers[::-1]]

            if cancelled:
                self.results_text.append("\n厚度约束拟合已中止，保留当前中间结果。")
            else:
                self.results_text.append("\n厚度约束拟合完成！")
            self.results_text.append(f"上层厚度占比 f = {best_frac:.4f}")
            self.results_text.append(
                f"  第 {gui_upper + 1} 层 H = {H_upper:.4e} m  ({best_frac * 100:.2f}%)")
            self.results_text.append(
                f"  第 {gui_lower + 1} 层 H = {H_lower:.4e} m  ({(1 - best_frac) * 100:.2f}%)")

            if is_combined:
                fitted_result = {
                    'layers': gui_layers,
                    'sensor_params': dict(fitter.sensor_params),
                }
                result_kind = 'combined'
                self.results_text.append("\n拟合的传感器参数:")
                for param, value in fitted_result['sensor_params'].items():
                    self.results_text.append(f"  {param}: {value:.6e}")
                self.results_text.append("\n拟合的材料参数:")
                for i, layer in enumerate(fitted_result['layers']):
                    self.results_text.append(f"\n第 {i + 1} 层 (GUI Layer {i}):")
                    for param, value in layer.items():
                        self.results_text.append(f"  {param}: {value:.6e}")
            else:
                fitted_result = gui_layers
                result_kind = 'material'
                self.results_text.append("\n拟合的材料参数:")
                for i, layer in enumerate(fitted_result):
                    self.results_text.append(f"\n第 {i + 1} 层 (GUI Layer {i}):")
                    for param, value in layer.items():
                        self.results_text.append(f"  {param}: {value:.6e}")

            self.update_results_table(fitted_result, result_kind)
            self._append_result_row("厚度约束", f"第{gui_upper + 1}层 H", H_upper)
            self._append_result_row("厚度约束", f"第{gui_lower + 1}层 H", H_lower)
            self._append_result_row("厚度约束", "上层占比 f", best_frac)

            self.fitted_result = fitted_result
            self.fitter = fitter
            self.update_correlation_analysis(fitter)

        except FitCancelledError:
            pass
        except Exception as e:
            self.results_text.append(f"\n厚度约束拟合失败: {str(e)}\n")
            traceback.print_exc()

    @staticmethod
    def _get_v3_export_columns(source_df):
        """从导入数据推断 V3 导出列名（保留频率与 V3 列，排除 V1 列）。"""
        default_cols = ['频率 (Hz)', 'V3_Real', 'V3_Imag', 'V3_幅值', 'V3_相位']
        if source_df is None or source_df.empty:
            return default_cols

        cols = list(source_df.columns)
        non_v1 = [c for c in cols if 'v1' not in str(c).lower()]
        if len(non_v1) >= 5:
            return non_v1[:5]
        if len(non_v1) >= 3:
            result = list(non_v1[:3])
            for name in ['V3_幅值', 'V3_相位']:
                if len(result) >= 5:
                    break
                result.append(name)
            return result
        return default_cols

    @staticmethod
    def _build_fitted_export_dataframe(freqs, V3, source_df):
        """构建与导入格式一致的拟合 V3 导出表（不含 V1 列）。"""
        cols = FittingWidget._get_v3_export_columns(source_df)
        amp = np.abs(V3)
        phase = np.degrees(np.unwrap(np.angle(V3)))
        values = [freqs, V3.real, V3.imag, amp, phase]
        return pd.DataFrame({cols[i]: values[i] for i in range(min(len(cols), 5))})

    @staticmethod
    def _write_export_dataframe(df, filename):
        if filename.endswith('.xlsx'):
            df.to_excel(filename, index=False)
        else:
            if not filename.endswith('.csv'):
                filename = filename + '.csv'
            df.to_csv(filename, index=False)
        return filename

    @staticmethod
    def _append_path_suffix(path, suffix):
        root, ext = os.path.splitext(path)
        if not ext:
            ext = '.csv'
        return f"{root}{suffix}{ext}"

    def _compute_v3_from_fitter(self, freqs, fitter, b=None, sensor_params=None):
        """用拟合后的层参数与传感器参数计算指定频率上的 V3。"""
        b = fitter.b if b is None else b
        sensor_params = fitter.sensor_params if sensor_params is None else sensor_params
        Z = calculate_analytical_Z(freqs, b, fitter.layers)
        return convert_Z_to_V3(Z, freqs, b, sensor_params)

    def export_fitted_data(self):
        """导出拟合模型 V3 数据，频率/点数与导入数据一致，格式不含 V1 列。"""
        if not hasattr(self, 'fitter') or not hasattr(self, 'fitted_result'):
            QMessageBox.warning(self, "警告", "没有拟合结果可导出！请先完成拟合。")
            return

        fitter = self.fitter
        is_joint = isinstance(fitter, JointThermalModelFitter)

        try:
            if is_joint:
                self._export_joint_fitted_data(fitter)
            else:
                self._export_single_fitted_data(fitter)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")

    def _export_single_fitted_data(self, fitter):
        if self.fitting_data is None:
            QMessageBox.warning(self, "警告", "未找到原始导入数据！")
            return

        source_df = self.fitting_data
        freqs = source_df.iloc[:, 0].values.astype(float)
        V3_fit = self._compute_v3_from_fitter(freqs, fitter)
        export_df = self._build_fitted_export_dataframe(freqs, V3_fit, source_df)

        default_name = f"fitted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出拟合数据", default_name,
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if not filename:
            return

        saved_path = self._write_export_dataframe(export_df, filename)
        QMessageBox.information(
            self, "成功",
            f"拟合数据已导出到:\n{saved_path}\n共 {len(export_df)} 个数据点"
        )

    def _export_joint_fitted_data(self, fitter):
        if not self.joint_datasets_ui:
            QMessageBox.warning(self, "警告", "未找到联合拟合数据集！")
            return

        default_name = f"fitted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出拟合数据", default_name,
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if not filename:
            return

        saved_paths = []
        for i, ds_info in enumerate(self.joint_datasets_ui):
            source_df = ds_info['data']
            if source_df is None:
                continue

            freqs = source_df.iloc[:, 0].values.astype(float)
            ds = fitter.datasets[i]
            V3_fit = self._compute_v3_from_fitter(
                freqs, fitter, b=ds['b'], sensor_params=ds['sensor_params']
            )
            export_df = self._build_fitted_export_dataframe(freqs, V3_fit, source_df)

            if len(self.joint_datasets_ui) == 1:
                out_path = filename
            else:
                out_path = self._append_path_suffix(filename, f"_ds{i + 1}")

            saved_paths.append(self._write_export_dataframe(export_df, out_path))

        if not saved_paths:
            QMessageBox.warning(self, "警告", "没有可导出的数据集！")
            return

        if len(saved_paths) == 1:
            msg = f"拟合数据已导出到:\n{saved_paths[0]}"
        else:
            msg = "拟合数据已导出到:\n" + "\n".join(saved_paths)
        QMessageBox.information(self, "成功", msg)

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


# --- 穿透深度 / 线热源近似分析 ---
def thermal_penetration_depth(alpha, freq_hz):
    """热穿透深度 δ = √(α / (4π f))，f 为电频率 (Hz)。"""
    f = np.asarray(freq_hz, dtype=float)
    out = np.full_like(f, np.nan, dtype=float)
    valid = (f > 0) & np.isfinite(alpha) & (alpha > 0)
    out[valid] = np.sqrt(alpha / (4.0 * np.pi * f[valid]))
    return out


def freq_from_penetration_depth(alpha, delta):
    """由穿透深度反推电频率：f = α / (4π δ²)。"""
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(delta) or delta <= 0:
        return np.nan
    return alpha / (4.0 * np.pi * delta ** 2)


class PenetrationAnalysisWidget(QWidget):
    """穿透深度与线热源近似分析：判断当前频率是否合适并给出建议范围。"""

    def __init__(self):
        super().__init__()
        self._fitting_source = None
        self.material_db = MaterialDatabase()
        self.sensor_db = SensorDatabase()
        self.layer_widgets = []
        self.last_result = None
        self.init_ui()

    def set_fitting_source(self, fitting_widget):
        self._fitting_source = fitting_widget

    def init_ui(self):
        main = QHBoxLayout(self)

        # ---- 左侧参数 ----
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(340)
        left_scroll.setMaximumWidth(420)
        left_w = QWidget()
        left = QVBoxLayout(left_w)

        sync_row = QHBoxLayout()
        self.sync_btn = QPushButton("从数据拟合同步参数")
        self.sync_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        self.sync_btn.clicked.connect(self.sync_from_fitting)
        sync_row.addWidget(self.sync_btn)
        left.addLayout(sync_row)

        # 传感器
        sensor_group = QGroupBox("传感器参数")
        sensor_layout = QGridLayout()
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(self.sensor_db.get_sensor_names())
        self.sensor_combo.currentTextChanged.connect(self._load_sensor_preset)
        sensor_layout.addWidget(QLabel("预设:"), 0, 0)
        sensor_layout.addWidget(self.sensor_combo, 0, 1)
        self.sensor_inputs = {}
        for i, p in enumerate(['b', 'L', 'R', 'dR_dT', 'V_rms', 'H_e', 'rho_C_e']):
            sensor_layout.addWidget(QLabel(p), i + 1, 0)
            inp = QLineEdit()
            self.sensor_inputs[p] = inp
            sensor_layout.addWidget(inp, i + 1, 1)
        sensor_group.setLayout(sensor_layout)
        left.addWidget(sensor_group)
        self._load_sensor_preset(self.sensor_combo.currentText())

        # 材料层
        layer_group = QGroupBox("材料层（顶层→基底，最后一层为基底）")
        layer_v = QVBoxLayout()
        self.layer_layout = QVBoxLayout()
        layer_v.addLayout(self.layer_layout)
        layer_btn_row = QHBoxLayout()
        add_btn = QPushButton("添加层")
        add_btn.clicked.connect(lambda: self._add_layer_ui(len(self.layer_widgets)))
        rem_btn = QPushButton("删除顶层")
        rem_btn.clicked.connect(self._remove_top_layer)
        layer_btn_row.addWidget(add_btn)
        layer_btn_row.addWidget(rem_btn)
        layer_v.addLayout(layer_btn_row)
        layer_group.setLayout(layer_v)
        left.addWidget(layer_group)
        self._add_layer_ui(0)
        self._add_layer_ui(1)

        # 判据与当前频率
        crit_group = QGroupBox("判据与待评估频率")
        crit = QFormLayout()
        self.line_factor = QDoubleSpinBox()
        self.line_factor.setRange(1.0, 50.0)
        self.line_factor.setValue(5.0)
        self.line_factor.setDecimals(1)
        self.line_factor.setToolTip("线热源近似：要求 δ ≥ N·b（Cahill 常用 N=5，斜率误差约 1%）")
        crit.addRow("线热源因子 N (δ≥N·b):", self.line_factor)

        self.sub_factor = QDoubleSpinBox()
        self.sub_factor.setRange(1.0, 50.0)
        self.sub_factor.setValue(5.0)
        self.sub_factor.setDecimals(1)
        self.sub_factor.setToolTip("半无限基底近似：要求 δ ≤ H_sub/N（常用 N=5）")
        crit.addRow("基底因子 M (δ≤H/M):", self.sub_factor)

        self.eval_fmin = QLineEdit("1")
        self.eval_fmax = QLineEdit("1000")
        crit.addRow("当前扫频下限 f_min (Hz):", self.eval_fmin)
        crit.addRow("当前扫频上限 f_max (Hz):", self.eval_fmax)
        crit_group.setLayout(crit)
        left.addWidget(crit_group)

        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 8px; }"
        )
        self.analyze_btn.clicked.connect(self.run_analysis)
        left.addWidget(self.analyze_btn)

        self.apply_lockin_btn = QPushButton("将建议范围写入锁相扫频")
        self.apply_lockin_btn.setEnabled(False)
        self.apply_lockin_btn.setToolTip("把建议 f_min / f_max 填入「锁相控制」页的频率起止（若已关联主窗口）")
        self.apply_lockin_btn.clicked.connect(self.apply_to_lockin)
        left.addWidget(self.apply_lockin_btn)

        left.addStretch()
        left_scroll.setWidget(left_w)
        main.addWidget(left_scroll)

        # ---- 右侧：报告 + 图 ----
        right = QVBoxLayout()
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setFont(QFont("Consolas", 10))
        self.report.setMinimumHeight(220)
        right.addWidget(self.report, 1)

        self.figure = Figure(figsize=(8, 4.5))
        self.canvas = FigureCanvas(self.figure)
        right.addWidget(self.canvas, 2)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right.addWidget(self.toolbar)

        main.addLayout(right, 1)

    def _load_sensor_preset(self, name):
        sens = self.sensor_db.get_sensor(name)
        for k, inp in self.sensor_inputs.items():
            if k in sens:
                inp.setText(f"{sens[k]:.6g}")

    def _add_layer_ui(self, idx):
        box = QGroupBox(f"第 {idx} 层" + (" (基底)" if False else ""))
        lay = QGridLayout()
        combo = QComboBox()
        combo.addItems(self.material_db.get_material_names())
        lay.addWidget(QLabel("材料:"), 0, 0)
        lay.addWidget(combo, 0, 1)
        inputs = {}
        for i, p in enumerate(['k_y', 'rho', 'cap', 'H', 'psi', 'Rc']):
            lay.addWidget(QLabel(p), i + 1, 0)
            inp = QLineEdit()
            inputs[p] = inp
            lay.addWidget(inp, i + 1, 1)
        box.setLayout(lay)

        def on_mat(text, inps=inputs):
            mat = self.material_db.get_material(text)
            for k, v in mat.items():
                if k in inps:
                    inps[k].setText("inf" if np.isinf(v) else f"{v:.6g}")

        combo.currentTextChanged.connect(on_mat)
        on_mat(combo.currentText())

        self.layer_layout.addWidget(box)
        self.layer_widgets.append({'box': box, 'combo': combo, 'inputs': inputs})
        self._refresh_layer_titles()

    def _remove_top_layer(self):
        if len(self.layer_widgets) <= 1:
            QMessageBox.warning(self, "提示", "至少保留一层。")
            return
        w = self.layer_widgets.pop(0)
        w['box'].deleteLater()
        self._refresh_layer_titles()

    def _refresh_layer_titles(self):
        n = len(self.layer_widgets)
        for i, w in enumerate(self.layer_widgets):
            tag = " (基底)" if i == n - 1 else (" (顶层)" if i == 0 else "")
            w['box'].setTitle(f"第 {i} 层{tag}")

    def sync_from_fitting(self):
        if self._fitting_source is None:
            QMessageBox.warning(self, "提示", "未关联数据拟合模块。")
            return
        try:
            config = self._fitting_source.get_model_config()
            sensor = config.get('sensor', {})
            for k, v in sensor.items():
                if k in self.sensor_inputs:
                    self.sensor_inputs[k].setText(str(v))

            layers = config.get('layers', [])
            while self.layer_widgets:
                w = self.layer_widgets.pop()
                w['box'].deleteLater()
            if not layers:
                self._add_layer_ui(0)
            else:
                for i, layer in enumerate(layers):
                    self._add_layer_ui(i)
                    for p, val in layer.items():
                        if p in self.layer_widgets[i]['inputs']:
                            self.layer_widgets[i]['inputs'][p].setText(str(val))

            # 若拟合页已有数据频率，填入评估范围
            fd = getattr(self._fitting_source, 'fitting_data', None)
            if fd is not None and len(fd) > 0:
                try:
                    freqs = fd.iloc[:, 0].values.astype(float)
                    freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
                    if len(freqs):
                        self.eval_fmin.setText(f"{float(np.min(freqs)):.6g}")
                        self.eval_fmax.setText(f"{float(np.max(freqs)):.6g}")
                except Exception:
                    pass

            QMessageBox.information(
                self, "同步成功",
                f"已同步传感器与 {len(self.layer_widgets)} 层材料参数。"
            )
        except Exception as e:
            QMessageBox.critical(self, "同步失败", str(e))

    def _parse_layers(self):
        layers = []
        for w in self.layer_widgets:
            layer = {}
            for p, inp in w['inputs'].items():
                t = inp.text().strip().lower()
                if t in ('inf', 'infinity'):
                    layer[p] = np.inf
                else:
                    layer[p] = float(t)
            layers.append(layer)
        return layers

    def run_analysis(self):
        try:
            b = float(self.sensor_inputs['b'].text())
            if b <= 0:
                raise ValueError("半线宽 b 必须 > 0")

            layers = self._parse_layers()
            if not layers:
                raise ValueError("请至少设置一层材料")

            N = float(self.line_factor.value())
            M = float(self.sub_factor.value())
            f_lo = float(self.eval_fmin.text())
            f_hi = float(self.eval_fmax.text())
            if f_lo <= 0 or f_hi <= 0 or f_hi < f_lo:
                raise ValueError("请检查当前扫频上下限（需 >0 且 f_max≥f_min）")

            # 基底 = 最后一层
            sub = layers[-1]
            alpha_sub = sub['k_y'] / (sub['rho'] * sub['cap'])
            H_sub = sub['H']

            # 线热源：δ ≥ N·b → f ≤ f_max_line
            delta_line = N * b
            f_max_line = freq_from_penetration_depth(alpha_sub, delta_line)

            # 半无限：δ ≤ H_sub/M → f ≥ f_min_semi
            if np.isfinite(H_sub) and H_sub > 0:
                delta_semi = H_sub / M
                f_min_semi = freq_from_penetration_depth(alpha_sub, delta_semi)
            else:
                delta_semi = np.inf
                f_min_semi = 0.0

            # 建议窗口
            f_rec_min = f_min_semi
            f_rec_max = f_max_line
            window_ok = np.isfinite(f_rec_max) and (f_rec_min < f_rec_max)

            # 各薄膜层特征频率（δ = H）
            film_info = []
            for i, layer in enumerate(layers[:-1]):
                alpha_i = layer['k_y'] / (layer['rho'] * layer['cap'])
                H_i = layer['H']
                if np.isfinite(H_i) and H_i > 0 and alpha_i > 0:
                    f_H = freq_from_penetration_depth(alpha_i, H_i)
                    film_info.append({
                        'idx': i, 'alpha': alpha_i, 'H': H_i, 'f_H': f_H,
                        'k_y': layer['k_y'],
                    })

            # 评估当前扫频两端
            def judge_freq(f):
                delta = float(thermal_penetration_depth(alpha_sub, f))
                line_ok = delta >= delta_line
                if np.isfinite(H_sub) and H_sub > 0:
                    semi_ok = delta <= delta_semi
                else:
                    semi_ok = True  # 半无限基底无下界约束
                return delta, line_ok, semi_ok

            d_lo, line_lo, semi_lo = judge_freq(f_lo)
            d_hi, line_hi, semi_hi = judge_freq(f_hi)

            # 在区间内均匀抽若干点检查是否都落在建议窗
            probe = np.logspace(np.log10(f_lo), np.log10(f_hi), 21)
            in_window = np.ones_like(probe, dtype=bool)
            if window_ok:
                in_window &= (probe >= f_rec_min) & (probe <= f_rec_max)
            elif np.isfinite(f_rec_max):
                in_window &= (probe <= f_rec_max)
            frac_ok = float(np.mean(in_window)) if len(probe) else 0.0

            self.last_result = {
                'alpha_sub': alpha_sub, 'b': b, 'H_sub': H_sub,
                'N': N, 'M': M, 'delta_line': delta_line, 'delta_semi': delta_semi,
                'f_max_line': f_max_line, 'f_min_semi': f_min_semi,
                'f_rec_min': f_rec_min, 'f_rec_max': f_rec_max,
                'window_ok': window_ok, 'film_info': film_info,
                'f_lo': f_lo, 'f_hi': f_hi,
            }
            self.apply_lockin_btn.setEnabled(window_ok and np.isfinite(f_rec_max))

            # ---- 文本报告 ----
            lines = []
            lines.append("=" * 56)
            lines.append("  3ω 穿透深度 / 线热源近似分析")
            lines.append("=" * 56)
            lines.append("")
            lines.append("【公式】")
            lines.append("  α = k_y / (ρ · c)")
            lines.append("  δ(f) = √(α / (4π f))   （电频率 f，热波频率 2ω）")
            lines.append(f"  线热源近似: δ ≥ {N:g}·b  →  f ≤ α / (4π ({N:g}b)²)")
            lines.append(f"  半无限基底: δ ≤ H_sub/{M:g} →  f ≥ α / (4π (H_sub/{M:g})²)")
            lines.append("")
            lines.append("【基底 / 传感器】")
            lines.append(f"  半线宽 b = {b * 1e6:.4g} μm")
            lines.append(f"  基底 α_sub = {alpha_sub:.4e} m²/s")
            if np.isfinite(H_sub):
                lines.append(f"  基底厚度 H_sub = {H_sub * 1e6:.4g} μm")
            else:
                lines.append("  基底厚度 H_sub = ∞ （半无限，无下界频率约束）")
            lines.append(f"  线热源临界穿透 δ_line = {N:g}·b = {delta_line * 1e6:.4g} μm")
            if np.isfinite(delta_semi):
                lines.append(f"  半无限临界穿透 δ_semi = H/{M:g} = {delta_semi * 1e6:.4g} μm")
            lines.append("")
            lines.append("【建议频率窗口（基于基底 α）】")
            if window_ok:
                lines.append(f"  ✅ 建议范围: {f_rec_min:.4g} Hz  ≤  f  ≤  {f_rec_max:.4g} Hz")
                lines.append(f"     (约 {f_rec_min:.3e} ~ {f_rec_max:.3e} Hz)")
            else:
                lines.append("  ❌ 不存在同时满足两判据的频率窗口")
                lines.append(f"     半无限下界 f_min = {f_min_semi:.4g} Hz")
                lines.append(f"     线热源上界 f_max = {f_max_line:.4g} Hz")
                if np.isfinite(H_sub) and H_sub > 0:
                    lines.append(f"     H_sub/b = {H_sub / b:.3g}；若过小则斜率法难同时满足两条件")
                lines.append("     建议：加宽加热线、加厚基底，或改用全模型拟合（本软件多层模型）")
            lines.append("")

            if film_info:
                lines.append("【薄膜特征频率】δ = H 时 f = α/(4π H²)")
                for fi in film_info:
                    lines.append(
                        f"  层{fi['idx']}: H={fi['H']*1e6:.4g} μm, α={fi['alpha']:.3e} m²/s"
                        f"  →  f_H ≈ {fi['f_H']:.4g} Hz"
                    )
                lines.append("  （扫频宜覆盖各层 f_H 附近，薄膜信息才更敏感）")
                lines.append("")

            lines.append("【当前扫频评估】")
            lines.append(f"  设定范围: {f_lo:.4g} ~ {f_hi:.4g} Hz")
            lines.append(f"  @ f_min={f_lo:.4g} Hz:  δ={d_lo*1e6:.4g} μm")
            lines.append(f"      线热源: {'✅ OK' if line_lo else '❌ 偏高频(δ过小)'}"
                         f"   半无限: {'✅ OK' if semi_lo else '❌ 偏低频(δ过大)'}")
            lines.append(f"  @ f_max={f_hi:.4g} Hz:  δ={d_hi*1e6:.4g} μm")
            lines.append(f"      线热源: {'✅ OK' if line_hi else '❌ 偏高频(δ过小)'}"
                         f"   半无限: {'✅ OK' if semi_hi else '❌ 偏低频(δ过大)'}")
            lines.append(f"  区间内落在建议窗的比例: {frac_ok*100:.0f}%")

            overall = line_lo and line_hi and semi_lo and semi_hi and (frac_ok >= 0.9)
            lines.append("")
            if overall and window_ok:
                lines.append("【总评】✅ 当前频率范围整体合适。")
            elif window_ok:
                lines.append("【总评】⚠️ 当前频率部分不合适，建议改用上方建议窗口。")
                # 给出裁剪建议
                new_lo = max(f_lo, f_rec_min)
                new_hi = min(f_hi, f_rec_max)
                if new_lo < new_hi:
                    lines.append(f"  → 与当前范围的交集: {new_lo:.4g} ~ {new_hi:.4g} Hz")
                else:
                    lines.append(f"  → 建议直接改为: {f_rec_min:.4g} ~ {f_rec_max:.4g} Hz")
            else:
                lines.append("【总评】❌ 几何/物性下无经典斜率法安全窗；请用本软件全模型拟合，并谨慎选频。")

            lines.append("")
            lines.append("注: 上述判据针对「斜率法 / 线热源 + 半无限基底」近似；")
            lines.append("    本软件 Olson 多层模型可在更宽频段工作，但仍建议 δ 与结构尺度可比。")

            self.report.setPlainText("\n".join(lines))
            self._plot_analysis(alpha_sub, b, H_sub, N, M, f_lo, f_hi,
                                f_rec_min, f_rec_max, window_ok, film_info)

        except Exception as e:
            self.report.setPlainText(f"分析失败:\n{e}")
            traceback.print_exc()
            QMessageBox.warning(self, "分析失败", str(e))

    def _plot_analysis(self, alpha_sub, b, H_sub, N, M, f_lo, f_hi,
                       f_rec_min, f_rec_max, window_ok, film_info):
        # 确保中文字体仍生效（其它模块可能改过 rcParams）
        _configure_matplotlib_chinese_font()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 绘图频率跨度
        f_plot_min = min(f_lo, f_rec_min if f_rec_min > 0 else f_lo) / 10
        f_plot_max = max(f_hi, f_rec_max if np.isfinite(f_rec_max) else f_hi) * 10
        f_plot_min = max(f_plot_min, 1e-4)
        freqs = np.logspace(np.log10(f_plot_min), np.log10(f_plot_max), 300)
        delta = thermal_penetration_depth(alpha_sub, freqs) * 1e6  # μm

        ax.loglog(freqs, delta, 'b-', lw=2, label='δ(f) substrate')

        # 判据水平线
        ax.axhline(N * b * 1e6, color='#ff7f0e', ls='--', lw=1.5,
                   label=f'Line source: {N:g}·b = {N*b*1e6:.3g} μm')
        if np.isfinite(H_sub) and H_sub > 0:
            ax.axhline(H_sub / M * 1e6, color='#2ca02c', ls='--', lw=1.5,
                       label=f'Semi-inf: H/{M:g} = {H_sub/M*1e6:.3g} μm')
            ax.axhline(H_sub * 1e6, color='#2ca02c', ls=':', lw=1, alpha=0.7,
                       label=f'H_sub = {H_sub*1e6:.3g} μm')

        # 建议频率窗
        if window_ok:
            ax.axvspan(f_rec_min, f_rec_max, color='#4CAF50', alpha=0.15, label='Recommended band')

        # 当前扫频
        ax.axvspan(f_lo, f_hi, color='#2196F3', alpha=0.12, label='Current sweep')

        # 薄膜 H 标记
        for fi in film_info:
            ax.axhline(fi['H'] * 1e6, color='#9467bd', ls=':', alpha=0.5)
            if np.isfinite(fi['f_H']) and fi['f_H'] > 0:
                ax.axvline(fi['f_H'], color='#9467bd', ls=':', alpha=0.5)
                ax.plot([fi['f_H']], [fi['H'] * 1e6], 'D', color='#9467bd',
                        label=f"Layer{fi['idx']} δ=H")

        ax.set_xlabel('Frequency f (Hz)')
        ax.set_ylabel('Penetration depth δ (μm)')
        ax.set_title('Thermal penetration depth vs frequency')
        ax.grid(True, which='both', ls='--', alpha=0.3)
        ax.legend(fontsize='small', loc='best')
        self.figure.tight_layout()
        self.canvas.draw()

    def apply_to_lockin(self):
        """将建议频率写入锁相页（由主窗口注入回调）。"""
        if not self.last_result or not self.last_result.get('window_ok'):
            QMessageBox.warning(self, "提示", "没有有效的建议频率范围。")
            return
        cb = getattr(self, '_apply_freq_callback', None)
        if cb is None:
            QMessageBox.information(
                self, "建议范围",
                f"f_min = {self.last_result['f_rec_min']:.6g} Hz\n"
                f"f_max = {self.last_result['f_rec_max']:.6g} Hz\n\n"
                "（未关联锁相写入接口，请手动填写）"
            )
            return
        try:
            cb(self.last_result['f_rec_min'], self.last_result['f_rec_max'])
            QMessageBox.information(self, "已写入", "建议频率范围已写入锁相扫频设置。")
        except Exception as e:
            QMessageBox.warning(self, "写入失败", str(e))


class VisualizationWidget(QWidget):
    """结果可视化界面"""

    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # 各拟合参数的 ±20% 曲线配色（按参数序号循环）
    PM20_PLUS_COLORS = ['#ff7f0e', '#d62728', '#9467bd', '#e377c2', '#8c564b']
    PM20_MINUS_COLORS = ['#2ca02c', '#17becf', '#bcbd22', '#1f77b4', '#7f7f7f']

    def __init__(self):
        super().__init__()
        self.current_data = None
        self.joint_data = None
        self.curve_checkboxes = []
        self.pm20_checkboxes = []
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

        self.curve_panel.setMaximumWidth(280)
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

        self.show_pm20_cb = QCheckBox("显示参数 ±20%")
        self.show_pm20_cb.setChecked(True)
        self.show_pm20_cb.setToolTip(
            "对每个已拟合参数单独 ×1.2 / ×0.8 后重新正演理论曲线；"
            "左侧可勾选要显示的参数（默认全部）。"
        )
        self.show_pm20_cb.toggled.connect(self.refresh_current_plot)
        plot_control.addWidget(self.show_pm20_cb)
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
        for cb in self.curve_checkboxes + self.pm20_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self.refresh_current_plot()

    def _clear_checkboxes(self):
        # 清空左侧面板（保留末尾 stretch）
        while self.checkbox_layout.count() > 1:
            item = self.checkbox_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.curve_checkboxes = []
        self.pm20_checkboxes = []

    @staticmethod
    def _param_map_label(p_map):
        """生成拟合参数显示名（模型层序：L0=基底）。"""
        ptype = p_map.get('type', 'material')
        if ptype == 'sensor':
            return f"sensor.{p_map['prop']}"
        if ptype == 'thickness_link':
            return f"H_link(L{p_map['upper_idx']}/L{p_map['lower_idx']})"
        # JointThermalModelFitter 的 map 无 type 字段
        if 'layer_idx' in p_map:
            return f"L{p_map['layer_idx']}.{p_map['prop']}"
        return str(p_map.get('prop', '?'))

    @staticmethod
    def _v3_with_param_scale(freqs, b, layers, sensor_params, p_map, factor):
        """将指定参数乘以 factor 后正演 V3；无法扰动时返回 None。"""
        layers_p = [dict(l) for l in layers]
        sensor_p = dict(sensor_params)
        ptype = p_map.get('type', 'material')

        try:
            if ptype == 'sensor':
                val = sensor_p.get(p_map['prop'])
                if val is None or not np.isfinite(val):
                    return None
                sensor_p[p_map['prop']] = float(val) * factor
            elif ptype == 'thickness_link':
                Ht = float(p_map['H_total'])
                h_up = float(layers_p[p_map['upper_idx']]['H']) * factor
                h_up = float(np.clip(h_up, Ht * 1e-6, Ht * (1.0 - 1e-6)))
                layers_p[p_map['upper_idx']]['H'] = h_up
                layers_p[p_map['lower_idx']]['H'] = Ht - h_up
            else:
                idx = p_map['layer_idx']
                prop = p_map['prop']
                val = layers_p[idx].get(prop)
                if val is None or not np.isfinite(val):
                    return None
                layers_p[idx][prop] = float(val) * factor

            Z = calculate_analytical_Z(freqs, b, layers_p)
            return convert_Z_to_V3(Z, freqs, b, sensor_p)
        except Exception:
            return None

    def _build_pm20_series_single(self, freqs, model_ctx):
        """单数据集：对每个已拟合参数预计算 ±20% 正演曲线。"""
        series = []
        if not model_ctx:
            return series
        param_maps = model_ctx.get('param_maps') or []
        b = model_ctx['b']
        layers = model_ctx['layers']
        sensor = model_ctx['sensor_params']
        for i, p_map in enumerate(param_maps):
            label = self._param_map_label(p_map)
            V_plus = self._v3_with_param_scale(freqs, b, layers, sensor, p_map, 1.2)
            V_minus = self._v3_with_param_scale(freqs, b, layers, sensor, p_map, 0.8)
            if V_plus is None and V_minus is None:
                continue
            series.append({
                'label': label,
                'V3_plus': V_plus,
                'V3_minus': V_minus,
                'color_plus': self.PM20_PLUS_COLORS[i % len(self.PM20_PLUS_COLORS)],
                'color_minus': self.PM20_MINUS_COLORS[i % len(self.PM20_MINUS_COLORS)],
            })
        return series

    def _build_pm20_series_joint(self, datasets, layers, param_maps):
        """联合拟合：材料参数共享，对各数据集分别正演。"""
        series = []
        if not param_maps:
            return series
        for i, p_map in enumerate(param_maps):
            # 联合拟合 param_map 无 type，按材料处理
            p = dict(p_map)
            p.setdefault('type', 'material')
            label = self._param_map_label(p)
            per_ds = []
            any_ok = False
            for ds in datasets:
                V_plus = self._v3_with_param_scale(
                    ds['freqs'], ds['b'], layers, ds['sensor_params'], p, 1.2)
                V_minus = self._v3_with_param_scale(
                    ds['freqs'], ds['b'], layers, ds['sensor_params'], p, 0.8)
                if V_plus is not None or V_minus is not None:
                    any_ok = True
                per_ds.append({'V3_plus': V_plus, 'V3_minus': V_minus})
            if not any_ok:
                continue
            series.append({
                'label': label,
                'per_ds': per_ds,
                'color_plus': self.PM20_PLUS_COLORS[i % len(self.PM20_PLUS_COLORS)],
                'color_minus': self.PM20_MINUS_COLORS[i % len(self.PM20_MINUS_COLORS)],
            })
        return series

    def _setup_pm20_checkboxes(self, pm20_series, prefix=""):
        """左侧面板：每个已拟合参数一对 ±20% 开关（默认全开）。"""
        if not pm20_series:
            return
        hdr = QLabel(prefix or "参数 ±20% 正演:")
        hdr.setStyleSheet("font-weight: bold; margin-top: 6px;")
        self.checkbox_layout.insertWidget(self.checkbox_layout.count() - 1, hdr)
        # 用 QLabel 占位不当作 checkbox；仅 pm20_checkboxes 参与逻辑
        for item in pm20_series:
            cb = QCheckBox(f"{item['label']} ±20%")
            cb.setChecked(True)
            cb.setStyleSheet(
                f"color: {item['color_plus']};"
            )
            cb.toggled.connect(self.refresh_current_plot)
            self.checkbox_layout.insertWidget(self.checkbox_layout.count() - 1, cb)
            self.pm20_checkboxes.append(cb)
        self.curve_panel.setVisible(True)

    def _selected_pm20_indices(self):
        if not self.show_pm20_cb.isChecked():
            return []
        return [i for i, cb in enumerate(self.pm20_checkboxes) if cb.isChecked()]

    def plot_joint_fitting_results(self, datasets, layers, title="Joint Fitting Results",
                                   param_maps=None):
        """绘制联合拟合结果，每个数据集的实验值和拟合值各自可开关"""
        self.current_data = None
        self._clear_checkboxes()

        computed = []
        for ds in datasets:
            Z_fit = calculate_analytical_Z(ds['freqs'], ds['b'], layers)
            V3_fit = convert_Z_to_V3(Z_fit, ds['freqs'], ds['b'], ds['sensor_params'])
            computed.append(V3_fit)

        pm20_series = self._build_pm20_series_joint(datasets, layers, param_maps or [])

        self.joint_data = {
            'datasets': datasets,
            'layers': layers,
            'title': title,
            'V3_fitted_list': computed,
            'pm20_series': pm20_series,
            'param_maps': param_maps or [],
        }

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

        self._setup_pm20_checkboxes(pm20_series)
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
        pm20_series = self.joint_data.get('pm20_series') or []
        sel_pm = self._selected_pm20_indices()

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
                show_exp=cb_exp.isChecked(), show_fit=cb_fit.isChecked(),
            )

            # 对该数据集叠加选中参数的 ±20% 正演
            for j in sel_pm:
                if j >= len(pm20_series):
                    continue
                item = pm20_series[j]
                per = item['per_ds'][i]
                tag = f"DS{i + 1} {item['label']}"
                self._plot_pm20_on_axes(
                    axes_map, plot_type, freqs,
                    per.get('V3_plus'), per.get('V3_minus'),
                    item['color_plus'], item['color_minus'], tag,
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
                draw_on(axes_map['amp'], np.abs(V3_exp) * 1e3, np.abs(V3_fit) * 1e3,
                        log_x=True, log_y=True)
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

    def _plot_pm20_on_axes(self, axes_map, plot_type, freqs, V3_plus, V3_minus,
                           color_plus, color_minus, tag):
        """将某参数 ±20% 正演曲线画到 axes 上（含相位）。"""

        def draw_pair(ax, y_plus, y_minus, log_x=True, log_y=False):
            plot_fn = ax.loglog if (log_x and log_y) else (ax.semilogx if log_x else ax.plot)
            if y_plus is not None and np.any(np.isfinite(y_plus)):
                plot_fn(freqs, y_plus, '--', color=color_plus,
                        label=f'{tag} +20%', linewidth=1.4, alpha=0.9)
            if y_minus is not None and np.any(np.isfinite(y_minus)):
                plot_fn(freqs, y_minus, '--', color=color_minus,
                        label=f'{tag} -20%', linewidth=1.4, alpha=0.9)

        def y_of(V3, kind):
            if V3 is None:
                return None
            if kind == 'amp':
                return np.abs(V3) * 1e3
            if kind == 'pha':
                return np.angle(V3, deg=True)
            if kind == 're':
                return V3.real * 1e3
            if kind == 'im':
                return V3.imag * 1e3
            return None

        if 'single' in axes_map:
            ax = axes_map['single']
            if plot_type == "Amplitude":
                draw_pair(ax, y_of(V3_plus, 'amp'), y_of(V3_minus, 'amp'), True, True)
            elif plot_type == "Phase":
                draw_pair(ax, y_of(V3_plus, 'pha'), y_of(V3_minus, 'pha'))
            elif plot_type == "Real Part":
                draw_pair(ax, y_of(V3_plus, 're'), y_of(V3_minus, 're'))
            elif plot_type == "Imaginary Part":
                draw_pair(ax, y_of(V3_plus, 'im'), y_of(V3_minus, 'im'))
        else:
            if 'amp' in axes_map:
                draw_pair(axes_map['amp'], y_of(V3_plus, 'amp'), y_of(V3_minus, 'amp'), True, True)
            if 'pha' in axes_map:
                draw_pair(axes_map['pha'], y_of(V3_plus, 'pha'), y_of(V3_minus, 'pha'))
            if 're' in axes_map:
                draw_pair(axes_map['re'], y_of(V3_plus, 're'), y_of(V3_minus, 're'))
            if 'im' in axes_map:
                draw_pair(axes_map['im'], y_of(V3_plus, 'im'), y_of(V3_minus, 'im'))

    def plot_fitting_results(self, freqs, V3_exp, V3_fitted, V3_initial=None,
                             title="Fitting Results", model_ctx=None):
        """绘制单数据集拟合结果；若提供 model_ctx 则对每个已拟合参数 ±20% 正演。"""
        self.joint_data = None
        self._clear_checkboxes()

        pm20_series = self._build_pm20_series_single(freqs, model_ctx)

        self.current_data = {
            'freqs': freqs,
            'V3_exp': V3_exp,
            'V3_fitted': V3_fitted,
            'V3_initial': V3_initial,
            'title': title,
            'model_ctx': model_ctx,
            'pm20_series': pm20_series,
        }

        self._setup_pm20_checkboxes(pm20_series)
        if not pm20_series:
            self.curve_panel.setVisible(False)

        self._redraw_single()

    def _redraw_single(self):
        """按当前选项重绘单数据集拟合图。"""
        if not self.current_data:
            return

        freqs = self.current_data['freqs']
        V3_exp = self.current_data['V3_exp']
        V3_fitted = self.current_data['V3_fitted']
        V3_initial = self.current_data.get('V3_initial')
        title = self.current_data['title']
        pm20_series = self.current_data.get('pm20_series') or []
        sel_pm = self._selected_pm20_indices()

        self.figure.clear()
        plot_type = self.plot_type.currentText()

        def add_base_and_pm20(ax, kind):
            """kind: amp|pha|re|im"""
            if kind == 'amp':
                ax.loglog(freqs, np.abs(V3_exp) * 1e3, 'ro', label='Experimental Data', markersize=6)
                ax.loglog(freqs, np.abs(V3_fitted) * 1e3, 'b-', label='Fitted Model', linewidth=2)
                if V3_initial is not None:
                    ax.loglog(freqs, np.abs(V3_initial) * 1e3, 'g--', label='Initial Guess', linewidth=1)
                for j in sel_pm:
                    item = pm20_series[j]
                    if item['V3_plus'] is not None:
                        ax.loglog(freqs, np.abs(item['V3_plus']) * 1e3, '--',
                                  color=item['color_plus'], label=f"{item['label']} +20%",
                                  linewidth=1.5, alpha=0.9)
                    if item['V3_minus'] is not None:
                        ax.loglog(freqs, np.abs(item['V3_minus']) * 1e3, '--',
                                  color=item['color_minus'], label=f"{item['label']} -20%",
                                  linewidth=1.5, alpha=0.9)
                ax.set_ylabel('|V3| (mV)')
            elif kind == 'pha':
                ax.semilogx(freqs, np.angle(V3_exp, deg=True), 'ro', label='Experimental Data', markersize=6)
                ax.semilogx(freqs, np.angle(V3_fitted, deg=True), 'b-', label='Fitted Model', linewidth=2)
                if V3_initial is not None:
                    ax.semilogx(freqs, np.angle(V3_initial, deg=True), 'g--', label='Initial Guess', linewidth=1)
                for j in sel_pm:
                    item = pm20_series[j]
                    if item['V3_plus'] is not None:
                        ax.semilogx(freqs, np.angle(item['V3_plus'], deg=True), '--',
                                    color=item['color_plus'], label=f"{item['label']} +20%",
                                    linewidth=1.5, alpha=0.9)
                    if item['V3_minus'] is not None:
                        ax.semilogx(freqs, np.angle(item['V3_minus'], deg=True), '--',
                                    color=item['color_minus'], label=f"{item['label']} -20%",
                                    linewidth=1.5, alpha=0.9)
                ax.set_ylabel('Phase (deg)')
            elif kind == 're':
                ax.semilogx(freqs, V3_exp.real * 1e3, 'ro', label='Experimental Data', markersize=6)
                ax.semilogx(freqs, V3_fitted.real * 1e3, 'b-', label='Fitted Model', linewidth=2)
                if V3_initial is not None:
                    ax.semilogx(freqs, V3_initial.real * 1e3, 'g--', label='Initial Guess', linewidth=1)
                for j in sel_pm:
                    item = pm20_series[j]
                    if item['V3_plus'] is not None:
                        ax.semilogx(freqs, item['V3_plus'].real * 1e3, '--',
                                    color=item['color_plus'], label=f"{item['label']} +20%",
                                    linewidth=1.5, alpha=0.9)
                    if item['V3_minus'] is not None:
                        ax.semilogx(freqs, item['V3_minus'].real * 1e3, '--',
                                    color=item['color_minus'], label=f"{item['label']} -20%",
                                    linewidth=1.5, alpha=0.9)
                ax.set_ylabel('Real(V3) (mV)')
            elif kind == 'im':
                ax.semilogx(freqs, V3_exp.imag * 1e3, 'ro', label='Experimental Data', markersize=6)
                ax.semilogx(freqs, V3_fitted.imag * 1e3, 'b-', label='Fitted Model', linewidth=2)
                if V3_initial is not None:
                    ax.semilogx(freqs, V3_initial.imag * 1e3, 'g--', label='Initial Guess', linewidth=1)
                for j in sel_pm:
                    item = pm20_series[j]
                    if item['V3_plus'] is not None:
                        ax.semilogx(freqs, item['V3_plus'].imag * 1e3, '--',
                                    color=item['color_plus'], label=f"{item['label']} +20%",
                                    linewidth=1.5, alpha=0.9)
                    if item['V3_minus'] is not None:
                        ax.semilogx(freqs, item['V3_minus'].imag * 1e3, '--',
                                    color=item['color_minus'], label=f"{item['label']} -20%",
                                    linewidth=1.5, alpha=0.9)
                ax.set_ylabel('Imag(V3) (mV)')
            ax.set_xlabel('Frequency (Hz)')
            ax.legend(fontsize='small')
            ax.grid(True, which="both", ls="--", alpha=0.3)

        if plot_type == "All (2x2)":
            ax1 = self.figure.add_subplot(221)
            ax2 = self.figure.add_subplot(222)
            ax3 = self.figure.add_subplot(223)
            ax4 = self.figure.add_subplot(224)
            add_base_and_pm20(ax1, 'amp')
            ax1.set_title('V3 Amplitude')
            add_base_and_pm20(ax2, 'pha')
            ax2.set_title('V3 Phase')
            add_base_and_pm20(ax3, 're')
            ax3.set_title('V3 Real Part')
            add_base_and_pm20(ax4, 'im')
            ax4.set_title('V3 Imaginary Part')
        elif plot_type == "Amplitude & Phase":
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)
            add_base_and_pm20(ax1, 'amp')
            ax1.set_title('V3 Amplitude')
            add_base_and_pm20(ax2, 'pha')
            ax2.set_title('V3 Phase')
        else:
            ax = self.figure.add_subplot(111)
            kind_map = {
                "Amplitude": 'amp',
                "Phase": 'pha',
                "Real Part": 're',
                "Imaginary Part": 'im',
            }
            add_base_and_pm20(ax, kind_map.get(plot_type, 'amp'))
            ax.set_title(f'V3 {plot_type}', fontsize=14)

        self.figure.suptitle(title, fontsize=16)
        self.figure.tight_layout()
        self.canvas.draw()

    def update_plot_type(self):
        """更新图表类型"""
        self.refresh_current_plot()

    def refresh_current_plot(self):
        """按当前选项重绘已有拟合结果（不重新正演）。"""
        if self.joint_data:
            self._redraw_joint()
        elif self.current_data:
            self._redraw_single()

    def refresh_plot(self):
        """刷新图表"""
        if self.joint_data or self.current_data:
            self.refresh_current_plot()
        else:
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
        self.setWindowTitle("3ω热物性测量系统 v1.0")
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

        self.penetration_widget = PenetrationAnalysisWidget()
        self.tabs.addTab(self.penetration_widget, "📐 穿透深度分析")

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
        self.simulation_widget.set_fitting_source(self.fitting_widget)
        self.sensitivity_widget.set_fitting_source(self.fitting_widget)
        self.penetration_widget.set_fitting_source(self.fitting_widget)
        self.penetration_widget._apply_freq_callback = self._apply_recommended_freqs

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

        penetration_action = QAction('穿透深度分析', self)
        penetration_action.triggered.connect(self.show_penetration_analysis)
        tools_menu.addAction(penetration_action)

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
        """显示温升计算器（初值取自数据拟合页当前传感器参数）"""
        sensor_params = None
        try:
            sensor_params = {
                k: float(inp.text())
                for k, inp in self.fitting_widget.sensor_inputs.items()
            }
        except Exception:
            sensor_params = None

        dialog = TemperatureCalculatorDialog(self, sensor_params=sensor_params)
        dialog.set_fitting_source(self.fitting_widget)
        dialog.exec_()

    def show_penetration_analysis(self):
        """切换到穿透深度分析标签页"""
        self.tabs.setCurrentWidget(self.penetration_widget)
        self.status_bar.showMessage("穿透深度 / 线热源近似分析")

    def _apply_recommended_freqs(self, f_min, f_max):
        """将建议频率写入锁相控制页扫频起止。"""
        lo = max(0.1, float(f_min))
        hi = max(lo, float(f_max))
        # QDoubleSpinBox 有范围上限
        hi = min(hi, self.lockin_widget.freq_end.maximum())
        lo = min(lo, hi)
        self.lockin_widget.freq_start.setValue(lo)
        self.lockin_widget.freq_end.setValue(hi)
        self.tabs.setCurrentWidget(self.lockin_widget)
        self.status_bar.showMessage(f"已写入锁相扫频: {lo:.4g} ~ {hi:.4g} Hz")

    def on_fitting_completed(self):
        """拟合完成后的处理"""
        if not (hasattr(self.fitting_widget, 'fitter') and hasattr(self.fitting_widget, 'fitted_result')):
            return

        try:
            fitter = self.fitting_widget.fitter
            is_joint = isinstance(fitter, JointThermalModelFitter)

            if is_joint:
                if not fitter.datasets:
                    return
                param_maps = [dict(p) for p in getattr(fitter, '_param_map', [])]
                self.viz_widget.plot_joint_fitting_results(
                    fitter.datasets, fitter.layers,
                    title=f"Joint Fit ({len(fitter.datasets)} datasets)",
                    param_maps=param_maps,
                )
                self.tabs.setCurrentWidget(self.viz_widget)
                n_p = len(param_maps)
                self.status_bar.showMessage(
                    f"联合拟合完成 (共 {len(fitter.datasets)} 个数据集)，"
                    f"已对 {n_p} 个拟合参数分别 ±20% 正演"
                )
            else:
                # 单数据集拟合
                freqs = fitter.freqs
                V3_exp = fitter.V3_exp

                Z_fitted = calculate_analytical_Z(freqs, fitter.b, fitter.layers)
                V3_fitted = convert_Z_to_V3(Z_fitted, freqs, fitter.b, fitter.sensor_params)

                param_maps = [dict(p) for p in getattr(fitter, '_param_map', [])]
                model_ctx = {
                    'b': fitter.b,
                    'layers': [dict(l) for l in fitter.layers],
                    'sensor_params': dict(fitter.sensor_params),
                    'param_maps': param_maps,
                }

                self.viz_widget.plot_fitting_results(
                    freqs, V3_exp, V3_fitted,
                    title=f"Fitting Results ({self.fitting_widget.fit_method.currentText()})",
                    model_ctx=model_ctx,
                )
                self.tabs.setCurrentWidget(self.viz_widget)
                self.status_bar.showMessage(
                    f"拟合完成，已对 {len(param_maps)} 个拟合参数分别 ±20% 正演"
                )

        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示拟合结果时出错:\n{str(e)}")

    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h2>3ω热物性测量系统 v1.0</h2>
        <p>基于3-omega方法的多层材料热物性测量与分析软件</p>
        <p><b>功能特性:</b></p>
        <ul>
            <li>苏黎世或 OE1300 锁相放大器自动控制（OE1300 串口/网口 ASCII 指令）</li>
            <li>实时数据采集与显示</li>
            <li>多层材料热物性参数拟合</li>
            <li>正向模型模拟与分析</li>
            <li>灵敏度分析</li>
            <li>穿透深度 / 线热源近似分析</li>
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
