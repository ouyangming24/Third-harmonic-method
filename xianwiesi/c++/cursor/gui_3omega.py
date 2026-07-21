import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                             QPushButton, QComboBox, QGroupBox, QTabWidget,
                             QScrollArea, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QFileDialog, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QDoubleValidator
import json
from olson_model26 import calculate_analytical_Z, convert_Z_to_V3


class MaterialsLibrary:
    """材料库管理类"""
    def __init__(self):
        self.materials = {
            'Al2O3': {'k_y': 40.0, 'rho': 3965, 'cap': 1000, 'H': 1e-6, 'psi': 1.0, 'Rc': 5E-5},
            'SiO2': {'k_y': 1.12, 'rho': 2200, 'cap': 31, 'H': 1.47e-6, 'psi': 1.0, 'Rc': 1.e-4},
            'Si': {'k_y': 140.0, 'rho': 2329, 'cap': 700, 'H': 100e-6, 'psi': 1.0, 'Rc': 5.1e-5},
            'Diamond': {'k_y': 1000, 'rho': 3500, 'cap': 500, 'H': 50e-6, 'psi': 1.0, 'Rc': 6e-9},
            'Glass': {'k_y': 1.4, 'rho': 2200, 'cap': 800, 'H': 10e-6, 'psi': 1.0, 'Rc': 1e-4},
            'Ag_Grease': {'k_y': 30, 'rho': 1500, 'cap': 1000, 'H': 40e-6, 'psi': 1.0, 'Rc': 1e-8},
            'Al': {'k_y': 237, 'rho': 2700, 'cap': 800, 'H': 4e-3, 'psi': 1.0, 'Rc': 3.85E-5},
            'Cu': {'k_y': 400, 'rho': 8960, 'cap': 385, 'H': 4e-3, 'psi': 1.0, 'Rc': 3.8e-4},
            'PI': {'k_y': 0.155, 'rho': 1894, 'cap': 1100, 'H': 25e-6, 'psi': 1.0, 'Rc': 6.88e-5},
            'Air': {'k_y': 0.026, 'rho': 1.2, 'cap': 1000, 'H': 0.2e-6, 'psi': 1.0, 'Rc': 1e-8},
            'Custom': {'k_y': 1.0, 'rho': 2000, 'cap': 1000, 'H': 10e-6, 'psi': 1.0, 'Rc': 1e-8},
        }
    
    def get_material_names(self):
        return list(self.materials.keys())
    
    def get_material(self, name):
        return self.materials.get(name, self.materials['Custom']).copy()


class LayerWidget(QWidget):
    """单层材料参数输入组件"""
    paramChanged = pyqtSignal()
    removeRequested = pyqtSignal(object)
    
    def __init__(self, layer_index, materials_lib, parent=None):
        super().__init__(parent)
        self.layer_index = layer_index
        self.materials_lib = materials_lib
        self.setup_ui()
        
    def setup_ui(self):
        layout = QGridLayout()
        
        # 层标题
        title_label = QLabel(f"<b>第 {self.layer_index} 层</b>")
        layout.addWidget(title_label, 0, 0, 1, 2)
        
        # 材料选择下拉框
        layout.addWidget(QLabel("材料:"), 1, 0)
        self.material_combo = QComboBox()
        self.material_combo.addItems(self.materials_lib.get_material_names())
        self.material_combo.currentTextChanged.connect(self.on_material_changed)
        layout.addWidget(self.material_combo, 1, 1)
        
        # 热导率 k_y
        layout.addWidget(QLabel("热导率 k_y (W/m·K):"), 2, 0)
        self.k_y_input = QLineEdit("1.0")
        self.k_y_input.setValidator(QDoubleValidator(0.001, 10000, 6))
        self.k_y_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.k_y_input, 2, 1)
        
        # 密度 rho
        layout.addWidget(QLabel("密度 ρ (kg/m³):"), 3, 0)
        self.rho_input = QLineEdit("2000")
        self.rho_input.setValidator(QDoubleValidator(100, 20000, 2))
        self.rho_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.rho_input, 3, 1)
        
        # 比热容 cap
        layout.addWidget(QLabel("比热容 c (J/kg·K):"), 4, 0)
        self.cap_input = QLineEdit("1000")
        self.cap_input.setValidator(QDoubleValidator(100, 5000, 2))
        self.cap_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.cap_input, 4, 1)
        
        # 厚度 H
        layout.addWidget(QLabel("厚度 H (μm):"), 5, 0)
        self.h_input = QLineEdit("10")
        self.h_input.setValidator(QDoubleValidator(0.001, 10000, 6))
        self.h_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.h_input, 5, 1)
        
        # 界面热阻 Rc
        layout.addWidget(QLabel("界面热阻 Rc (K·m²/W):"), 6, 0)
        self.rc_input = QLineEdit("1e-8")
        self.rc_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.rc_input, 6, 1)
        
        # 各向异性因子 psi
        layout.addWidget(QLabel("各向异性 ψ:"), 7, 0)
        self.psi_input = QLineEdit("1.0")
        self.psi_input.setValidator(QDoubleValidator(0.01, 100, 3))
        self.psi_input.textChanged.connect(self.paramChanged.emit)
        layout.addWidget(self.psi_input, 7, 1)
        
        # 删除按钮
        if self.layer_index > 0:  # 基底层不能删除
            self.remove_btn = QPushButton("删除此层")
            self.remove_btn.clicked.connect(lambda: self.removeRequested.emit(self))
            layout.addWidget(self.remove_btn, 8, 0, 1, 2)
        
        self.setLayout(layout)
        
        # 加载默认材料
        self.on_material_changed(self.material_combo.currentText())
    
    def on_material_changed(self, material_name):
        """材料选择改变时更新参数"""
        material = self.materials_lib.get_material(material_name)
        self.k_y_input.setText(str(material['k_y']))
        self.rho_input.setText(str(material['rho']))
        self.cap_input.setText(str(material['cap']))
        self.h_input.setText(f"{material['H']*1e6:.4f}")  # 转换为微米
        self.rc_input.setText(f"{material['Rc']:.2e}")
        self.psi_input.setText(str(material['psi']))
    
    def get_layer_data(self):
        """获取当前层的参数"""
        try:
            return {
                'k_y': float(self.k_y_input.text()),
                'rho': float(self.rho_input.text()),
                'cap': float(self.cap_input.text()),
                'H': float(self.h_input.text()) * 1e-6 if self.layer_index > 0 else np.inf,  # 基底为无穷大
                'psi': float(self.psi_input.text()),
                'Rc': float(self.rc_input.text())
            }
        except ValueError:
            return None


class MplCanvas(FigureCanvas):
    """Matplotlib画布"""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_amp = self.fig.add_subplot(211)
        self.axes_phase = self.fig.add_subplot(212)
        super().__init__(self.fig)
        
        self.axes_amp.set_xlabel('频率 (Hz)')
        self.axes_amp.set_ylabel('幅值 |V3| (mV)')
        self.axes_amp.set_xscale('log')
        self.axes_amp.grid(True, which='both', ls='--', alpha=0.3)
        
        self.axes_phase.set_xlabel('频率 (Hz)')
        self.axes_phase.set_ylabel('相位 (度)')
        self.axes_phase.set_xscale('log')
        self.axes_phase.grid(True, which='both', ls='--', alpha=0.3)
        
        self.fig.tight_layout()


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3ω 方法热物性测量 - 动态参数调整系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 初始化材料库
        self.materials_lib = MaterialsLibrary()
        
        # 初始化频率范围
        self.freqs = np.logspace(np.log10(0.0001), np.log10(1000), 100)
        
        # 实验数据存储
        self.exp_data = None  # 存储加载的实验数据
        self.exp_filename = None  # 实验数据文件名
        
        # 延迟更新定时器(防止输入过程中频繁计算)
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.calculate_and_plot)
        
        # 设置UI
        self.setup_ui()
        
        # 初始计算
        self.calculate_and_plot()
    
    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        
        # 左侧参数面板
        left_panel = self.create_parameter_panel()
        
        # 右侧绘图区域
        right_panel = self.create_plot_panel()
        
        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
    
    def create_parameter_panel(self):
        """创建左侧参数输入面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 标签页
        tab_widget = QTabWidget()
        
        # === 传感器参数标签页 ===
        sensor_tab = QWidget()
        sensor_layout = QGridLayout()
        
        row = 0
        sensor_layout.addWidget(QLabel("<b>传感器几何参数</b>"), row, 0, 1, 2)
        
        row += 1
        sensor_layout.addWidget(QLabel("半宽 b (μm):"), row, 0)
        self.b_input = QLineEdit("200")
        self.b_input.textChanged.connect(self.schedule_update)
        sensor_layout.addWidget(self.b_input, row, 1)
        
        row += 1
        sensor_layout.addWidget(QLabel("长度 L (mm):"), row, 0)
        self.L_input = QLineEdit("5")
        self.L_input.textChanged.connect(self.schedule_update)
        sensor_layout.addWidget(self.L_input, row, 1)
        
        row += 1
        sensor_layout.addWidget(QLabel("<b>传感器电学参数</b>"), row, 0, 1, 2)
        
        row += 1
        sensor_layout.addWidget(QLabel("电阻 R (Ω):"), row, 0)
        self.R_input = QLineEdit("6.74")
        self.R_input.textChanged.connect(self.schedule_update)
        sensor_layout.addWidget(self.R_input, row, 1)
        
        row += 1
        sensor_layout.addWidget(QLabel("温度系数 dR/dT (Ω/K):"), row, 0)
        self.dRdT_input = QLineEdit("0.014391")
        self.dRdT_input.textChanged.connect(self.schedule_update)
        sensor_layout.addWidget(self.dRdT_input, row, 1)
        
        row += 1
        sensor_layout.addWidget(QLabel("RMS电压 V_rms (V):"), row, 0)
        self.Vrms_input = QLineEdit("0.4")
        self.Vrms_input.textChanged.connect(self.schedule_update)
        sensor_layout.addWidget(self.Vrms_input, row, 1)
        
        row += 1
        sensor_layout.addWidget(QLabel("<b>加热器参数</b>"), row, 0, 1, 2)
        
        row += 1
        sensor_layout.addWidget(QLabel("厚度 H_e (nm):"), row, 0)
        self.He_input = QLineEdit("100")
        self.He_input.textChanged.connect(self.schedule_update)
        sensor_layout.addWidget(self.He_input, row, 1)
        
        row += 1
        sensor_layout.addWidget(QLabel("体积热容 ρC_e (J/m³·K):"), row, 0)
        self.rhoCe_input = QLineEdit("2.5116e6")
        self.rhoCe_input.textChanged.connect(self.schedule_update)
        sensor_layout.addWidget(self.rhoCe_input, row, 1)
        
        sensor_layout.setRowStretch(row + 1, 1)
        sensor_tab.setLayout(sensor_layout)
        
        # === 材料层参数标签页 ===
        layers_tab = QWidget()
        layers_main_layout = QVBoxLayout()
        
        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.layers_layout = QVBoxLayout()
        
        # 添加基底层(第0层)
        self.layer_widgets = []
        base_layer = LayerWidget(0, self.materials_lib)
        base_layer.paramChanged.connect(self.schedule_update)
        self.layer_widgets.append(base_layer)
        self.layers_layout.addWidget(base_layer)
        
        # 添加按钮
        add_layer_btn = QPushButton("➕ 添加新层")
        add_layer_btn.clicked.connect(self.add_layer)
        self.layers_layout.addWidget(add_layer_btn)
        
        self.layers_layout.addStretch()
        scroll_content.setLayout(self.layers_layout)
        scroll.setWidget(scroll_content)
        
        layers_main_layout.addWidget(scroll)
        layers_tab.setLayout(layers_main_layout)
        
        # === 频率范围标签页 ===
        freq_tab = QWidget()
        freq_layout = QGridLayout()
        
        freq_layout.addWidget(QLabel("最小频率 (Hz):"), 0, 0)
        self.freq_min_input = QLineEdit("0.0001")
        self.freq_min_input.textChanged.connect(self.schedule_update)
        freq_layout.addWidget(self.freq_min_input, 0, 1)
        
        freq_layout.addWidget(QLabel("最大频率 (Hz):"), 1, 0)
        self.freq_max_input = QLineEdit("1000")
        self.freq_max_input.textChanged.connect(self.schedule_update)
        freq_layout.addWidget(self.freq_max_input, 1, 1)
        
        freq_layout.addWidget(QLabel("数据点数:"), 2, 0)
        self.freq_points_input = QLineEdit("100")
        self.freq_points_input.textChanged.connect(self.schedule_update)
        freq_layout.addWidget(self.freq_points_input, 2, 1)
        
        freq_layout.setRowStretch(3, 1)
        freq_tab.setLayout(freq_layout)
        
        # 添加标签页
        tab_widget.addTab(sensor_tab, "传感器参数")
        tab_widget.addTab(layers_tab, "材料层")
        tab_widget.addTab(freq_tab, "频率范围")
        
        layout.addWidget(tab_widget)
        
        # 底部按钮
        button_layout = QVBoxLayout()
        
        # 第一行按钮
        button_row1 = QHBoxLayout()
        save_btn = QPushButton("💾 保存配置")
        save_btn.clicked.connect(self.save_config)
        button_row1.addWidget(save_btn)
        
        load_btn = QPushButton("📂 加载配置")
        load_btn.clicked.connect(self.load_config)
        button_row1.addWidget(load_btn)
        
        button_layout.addLayout(button_row1)
        
        # 第二行按钮
        button_row2 = QHBoxLayout()
        import_exp_btn = QPushButton("📥 导入实验数据")
        import_exp_btn.clicked.connect(self.import_experimental_data)
        import_exp_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_row2.addWidget(import_exp_btn)
        
        clear_exp_btn = QPushButton("🗑️ 清除实验数据")
        clear_exp_btn.clicked.connect(self.clear_experimental_data)
        button_row2.addWidget(clear_exp_btn)
        
        button_layout.addLayout(button_row2)
        
        # 第三行按钮
        button_row3 = QHBoxLayout()
        export_btn = QPushButton("📊 导出理论数据")
        export_btn.clicked.connect(self.export_data)
        button_row3.addWidget(export_btn)
        
        button_layout.addLayout(button_row3)
        
        # 实验数据文件名标签
        self.exp_file_label = QLabel("实验数据: 未加载")
        self.exp_file_label.setStyleSheet("color: gray; font-size: 9pt; padding: 2px;")
        self.exp_file_label.setWordWrap(True)
        button_layout.addWidget(self.exp_file_label)
        
        layout.addLayout(button_layout)
        
        panel.setLayout(layout)
        return panel
    
    def create_plot_panel(self):
        """创建右侧绘图面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel("<h2>3ω 电压响应曲线</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 画布
        self.canvas = MplCanvas(self, width=8, height=8, dpi=100)
        layout.addWidget(self.canvas)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: green; padding: 5px;")
        layout.addWidget(self.status_label)
        
        panel.setLayout(layout)
        return panel
    
    def add_layer(self):
        """添加新的材料层"""
        layer_index = len(self.layer_widgets)
        new_layer = LayerWidget(layer_index, self.materials_lib)
        new_layer.paramChanged.connect(self.schedule_update)
        new_layer.removeRequested.connect(self.remove_layer)
        
        # 在"添加新层"按钮之前插入
        self.layers_layout.insertWidget(len(self.layer_widgets), new_layer)
        self.layer_widgets.append(new_layer)
        
        self.schedule_update()
    
    def remove_layer(self, layer_widget):
        """移除材料层"""
        if layer_widget in self.layer_widgets and layer_widget.layer_index > 0:
            self.layer_widgets.remove(layer_widget)
            self.layers_layout.removeWidget(layer_widget)
            layer_widget.deleteLater()
            
            # 重新编号
            for i, widget in enumerate(self.layer_widgets):
                widget.layer_index = i
            
            self.schedule_update()
    
    def schedule_update(self):
        """延迟更新(防止频繁计算)"""
        self.update_timer.start(300)  # 300ms 延迟
    
    def get_sensor_params(self):
        """获取传感器参数"""
        try:
            return {
                'b': float(self.b_input.text()) * 1e-6,  # 微米转米
                'L': float(self.L_input.text()) * 1e-3,  # 毫米转米
                'R': float(self.R_input.text()),
                'dR_dT': float(self.dRdT_input.text()),
                'V_rms': float(self.Vrms_input.text()),
                'H_e': float(self.He_input.text()) * 1e-9,  # 纳米转米
                'rho_C_e': float(self.rhoCe_input.text())
            }
        except ValueError as e:
            self.status_label.setText(f"错误: 传感器参数无效 - {str(e)}")
            self.status_label.setStyleSheet("color: red; padding: 5px;")
            return None
    
    def get_layers(self):
        """获取所有材料层参数"""
        layers = []
        for widget in self.layer_widgets:
            layer_data = widget.get_layer_data()
            if layer_data is None:
                self.status_label.setText(f"错误: 第{widget.layer_index}层参数无效")
                self.status_label.setStyleSheet("color: red; padding: 5px;")
                return None
            layers.append(layer_data)
        return layers
    
    def update_frequency_range(self):
        """更新频率范围"""
        try:
            freq_min = float(self.freq_min_input.text())
            freq_max = float(self.freq_max_input.text())
            freq_points = int(self.freq_points_input.text())
            
            if freq_min <= 0 or freq_max <= freq_min or freq_points < 10:
                raise ValueError("频率范围参数无效")
            
            self.freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), freq_points)
            return True
        except ValueError as e:
            self.status_label.setText(f"错误: {str(e)}")
            self.status_label.setStyleSheet("color: red; padding: 5px;")
            return False
    
    def calculate_and_plot(self):
        """计算并绘制曲线"""
        self.status_label.setText("计算中...")
        self.status_label.setStyleSheet("color: orange; padding: 5px;")
        QApplication.processEvents()  # 更新UI
        
        # 获取参数
        sensor_params = self.get_sensor_params()
        layers = self.get_layers()
        
        if sensor_params is None or layers is None:
            return
        
        if not self.update_frequency_range():
            return
        
        try:
            # 计算理论曲线
            Z_analytical = calculate_analytical_Z(self.freqs, sensor_params['b'], layers)
            V3_analytical = convert_Z_to_V3(Z_analytical, self.freqs, sensor_params['b'], sensor_params)
            
            # 提取幅值和相位
            V3_amp = np.abs(V3_analytical) * 1e3  # 转换为 mV
            V3_phase = np.unwrap(np.angle(V3_analytical)) * 180 / np.pi  # 转换为度
            
            # 清空并重绘
            self.canvas.axes_amp.clear()
            self.canvas.axes_phase.clear()
            
            # 绘制理论曲线
            self.canvas.axes_amp.semilogx(self.freqs, V3_amp, 'b-', linewidth=2, label='理论曲线')
            self.canvas.axes_phase.semilogx(self.freqs, V3_phase, 'r-', linewidth=2, label='理论曲线')
            
            # 如果有实验数据,叠加绘制
            if self.exp_data is not None:
                exp_freq = self.exp_data['频率 (Hz)']
                exp_amp = self.exp_data['V3_幅值 (V)'] * 1e3  # 转换为 mV
                exp_phase = self.exp_data['V3_相位 (度)']
                
                # 筛选当前频率范围内的实验数据点
                freq_min = self.freqs.min()
                freq_max = self.freqs.max()
                mask = (exp_freq >= freq_min) & (exp_freq <= freq_max)
                
                exp_freq_filtered = exp_freq[mask]
                exp_amp_filtered = exp_amp[mask]
                exp_phase_filtered = exp_phase[mask]
                
                # 绘制筛选后的实验数据点
                if len(exp_freq_filtered) > 0:
                    self.canvas.axes_amp.semilogx(exp_freq_filtered, exp_amp_filtered, 'go', 
                                                 markersize=6, label='实验数据', alpha=0.7)
                    self.canvas.axes_phase.semilogx(exp_freq_filtered, exp_phase_filtered, 'mo', 
                                                   markersize=6, label='实验数据', alpha=0.7)
            
            # 设置幅值图
            self.canvas.axes_amp.set_xlabel('频率 (Hz)', fontsize=11)
            self.canvas.axes_amp.set_ylabel('幅值 |V3| (mV)', fontsize=11)
            self.canvas.axes_amp.set_title('3ω 电压幅值响应', fontsize=12, fontweight='bold')
            self.canvas.axes_amp.grid(True, which='both', ls='--', alpha=0.3)
            self.canvas.axes_amp.legend(loc='best')
            
            # 设置相位图
            self.canvas.axes_phase.set_xlabel('频率 (Hz)', fontsize=11)
            self.canvas.axes_phase.set_ylabel('相位 (度)', fontsize=11)
            self.canvas.axes_phase.set_title('3ω 电压相位响应', fontsize=12, fontweight='bold')
            self.canvas.axes_phase.grid(True, which='both', ls='--', alpha=0.3)
            self.canvas.axes_phase.legend(loc='best')
            
            self.canvas.fig.tight_layout()
            self.canvas.draw()
            
            # 显示计算信息
            alpha_0 = layers[0]['k_y'] / (layers[0]['rho'] * layers[0]['cap'])
            info = f"✓ 计算完成 | 层数: {len(layers)} | 基底热扩散率: {alpha_0:.2e} m²/s"
            if self.exp_data is not None:
                # 统计当前频率范围内的实验数据点数
                exp_freq = self.exp_data['频率 (Hz)']
                freq_min = self.freqs.min()
                freq_max = self.freqs.max()
                mask = (exp_freq >= freq_min) & (exp_freq <= freq_max)
                n_points_in_range = mask.sum()
                n_points_total = len(self.exp_data)
                info += f" | 实验点: {n_points_in_range}/{n_points_total} (范围内/总数)"
            self.status_label.setText(info)
            self.status_label.setStyleSheet("color: green; padding: 5px;")
            
        except Exception as e:
            self.status_label.setText(f"错误: {str(e)}")
            self.status_label.setStyleSheet("color: red; padding: 5px;")
            import traceback
            traceback.print_exc()
    
    def save_config(self):
        """保存配置到JSON文件"""
        filename, _ = QFileDialog.getSaveFileName(self, "保存配置", "", "JSON Files (*.json)")
        if not filename:
            return
        
        config = {
            'sensor_params': {
                'b': self.b_input.text(),
                'L': self.L_input.text(),
                'R': self.R_input.text(),
                'dR_dT': self.dRdT_input.text(),
                'V_rms': self.Vrms_input.text(),
                'H_e': self.He_input.text(),
                'rho_C_e': self.rhoCe_input.text()
            },
            'layers': [],
            'frequency': {
                'min': self.freq_min_input.text(),
                'max': self.freq_max_input.text(),
                'points': self.freq_points_input.text()
            }
        }
        
        for widget in self.layer_widgets:
            layer_config = {
                'material': widget.material_combo.currentText(),
                'k_y': widget.k_y_input.text(),
                'rho': widget.rho_input.text(),
                'cap': widget.cap_input.text(),
                'H': widget.h_input.text(),
                'Rc': widget.rc_input.text(),
                'psi': widget.psi_input.text()
            }
            config['layers'].append(layer_config)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            QMessageBox.information(self, "成功", "配置已保存!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def load_config(self):
        """从JSON文件加载配置"""
        filename, _ = QFileDialog.getOpenFileName(self, "加载配置", "", "JSON Files (*.json)")
        if not filename:
            return
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 加载传感器参数
            sp = config['sensor_params']
            self.b_input.setText(sp['b'])
            self.L_input.setText(sp['L'])
            self.R_input.setText(sp['R'])
            self.dRdT_input.setText(sp['dR_dT'])
            self.Vrms_input.setText(sp['V_rms'])
            self.He_input.setText(sp['H_e'])
            self.rhoCe_input.setText(sp['rho_C_e'])
            
            # 加载频率参数
            freq = config['frequency']
            self.freq_min_input.setText(freq['min'])
            self.freq_max_input.setText(freq['max'])
            self.freq_points_input.setText(freq['points'])
            
            # 清空现有层
            for widget in self.layer_widgets[1:]:  # 保留基底层
                self.layers_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_widgets = [self.layer_widgets[0]]
            
            # 加载层参数
            for i, layer_cfg in enumerate(config['layers']):
                if i == 0:
                    widget = self.layer_widgets[0]
                else:
                    self.add_layer()
                    widget = self.layer_widgets[-1]
                
                widget.material_combo.setCurrentText(layer_cfg['material'])
                widget.k_y_input.setText(layer_cfg['k_y'])
                widget.rho_input.setText(layer_cfg['rho'])
                widget.cap_input.setText(layer_cfg['cap'])
                widget.h_input.setText(layer_cfg['H'])
                widget.rc_input.setText(layer_cfg['Rc'])
                widget.psi_input.setText(layer_cfg['psi'])
            
            QMessageBox.information(self, "成功", "配置已加载!")
            self.calculate_and_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")
    
    def import_experimental_data(self):
        """导入实验数据"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "导入实验数据", "", 
            "Excel Files (*.xlsx *.xls);;All Files (*.*)"
        )
        if not filename:
            return
        
        try:
            import pandas as pd
            
            # 读取Excel文件
            df = pd.read_excel(filename)
            
            # 检查必要的列是否存在
            required_columns = ['频率 (Hz)', 'V3_幅值 (V)', 'V3_相位 (度)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # 尝试备选列名
                alt_columns = {
                    '频率 (Hz)': ['Frequency (Hz)', 'frequency', 'freq', 'Freq'],
                    'V3_幅值 (V)': ['V3_Magnitude (V)', 'V3_magnitude', 'magnitude', 'amp', 'V3_幅值'],
                    'V3_相位 (度)': ['V3_Phase (deg)', 'V3_phase', 'phase', 'Phase', 'V3_相位']
                }
                
                # 尝试映射列名
                column_map = {}
                for req_col in required_columns:
                    found = False
                    for col in df.columns:
                        if col == req_col:
                            column_map[req_col] = req_col
                            found = True
                            break
                        elif req_col in alt_columns:
                            for alt in alt_columns[req_col]:
                                if alt in col or col in alt:
                                    column_map[req_col] = col
                                    found = True
                                    break
                        if found:
                            break
                    
                    if not found:
                        raise ValueError(f"缺少列: {req_col}")
                
                # 重命名列
                df = df.rename(columns={v: k for k, v in column_map.items()})
            
            # 存储实验数据
            self.exp_data = df[required_columns].copy()
            self.exp_filename = filename.split('/')[-1].split('\\')[-1]  # 只保留文件名
            
            # 更新标签
            self.exp_file_label.setText(f"实验数据: {self.exp_filename} ({len(self.exp_data)}点)")
            self.exp_file_label.setStyleSheet("color: green; font-size: 9pt; padding: 2px; font-weight: bold;")
            
            # 重新绘图
            self.calculate_and_plot()
            
            QMessageBox.information(
                self, "成功", 
                f"成功导入实验数据!\n文件: {self.exp_filename}\n数据点数: {len(self.exp_data)}"
            )
            
        except Exception as e:
            self.exp_data = None
            self.exp_filename = None
            self.exp_file_label.setText("实验数据: 导入失败")
            self.exp_file_label.setStyleSheet("color: red; font-size: 9pt; padding: 2px;")
            QMessageBox.critical(self, "错误", f"导入实验数据失败:\n{str(e)}")
    
    def clear_experimental_data(self):
        """清除实验数据"""
        if self.exp_data is None:
            QMessageBox.information(self, "提示", "当前没有加载实验数据")
            return
        
        reply = QMessageBox.question(
            self, "确认", 
            f"确定要清除实验数据吗?\n文件: {self.exp_filename}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.exp_data = None
            self.exp_filename = None
            self.exp_file_label.setText("实验数据: 未加载")
            self.exp_file_label.setStyleSheet("color: gray; font-size: 9pt; padding: 2px;")
            
            # 重新绘图
            self.calculate_and_plot()
            
            QMessageBox.information(self, "成功", "实验数据已清除")
    
    def export_data(self):
        """导出理论数据到Excel"""
        filename, _ = QFileDialog.getSaveFileName(self, "导出理论数据", "3omega_theory.xlsx", 
                                                   "Excel Files (*.xlsx)")
        if not filename:
            return
        
        sensor_params = self.get_sensor_params()
        layers = self.get_layers()
        
        if sensor_params is None or layers is None:
            return
        
        try:
            import pandas as pd
            
            # 计算数据
            Z_analytical = calculate_analytical_Z(self.freqs, sensor_params['b'], layers)
            V3_analytical = convert_Z_to_V3(Z_analytical, self.freqs, sensor_params['b'], sensor_params)
            
            # 创建DataFrame
            df = pd.DataFrame({
                '频率 (Hz)': self.freqs,
                'V3_实部 (V)': V3_analytical.real,
                'V3_虚部 (V)': V3_analytical.imag,
                'V3_幅值 (V)': np.abs(V3_analytical),
                'V3_相位 (度)': np.angle(V3_analytical) * 180 / np.pi
            })
            
            df.to_excel(filename, index=False)
            QMessageBox.information(self, "成功", f"理论数据已导出到:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")


def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 设置字体
    font = QFont("Microsoft YaHei UI", 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

