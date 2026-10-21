"""
3ω 热物性测量系统 — 信赖域反射拟合入口

不修改 thermal_measurement_gui2.py / olson_model26_3.py。
拆分：
  trf_optimizer.py  优化器（自写 Coleman–Li / SciPy TRF）
  trf_fitter.py     只调用 Olson 正演的拟合适配层
  3omega_gui.py     本文件，界面与算法切换

拟合设置页可切换求解器；也可：
  python 3omega_gui.py --solver custom
  python 3omega_gui.py --solver scipy
旧入口（欠松弛高斯-牛顿）：
  python thermal_measurement_gui2.py
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import thermal_measurement_gui2 as _legacy_gui  # noqa: E402
from trf_fitter import (  # noqa: E402
    install_into_gui_module,
    set_solver,
    solver_label,
)
from trf_optimizer import SOLVER_CUSTOM, SOLVER_LABELS, SOLVER_SCIPY  # noqa: E402

install_into_gui_module(_legacy_gui)


def _install_solver_selector(fitting_widget, on_changed):
    from PyQt5.QtWidgets import QComboBox, QLabel

    settings = getattr(fitting_widget, "fitting_settings_widget", None)
    if settings is None:
        return None
    layout = settings.layout()
    if layout is None:
        return None

    combo = QComboBox()
    for key, label in SOLVER_LABELS.items():
        combo.addItem(label, key)
    combo.setToolTip(
        "自写 Coleman–Li：按 3ω 信号量级改过相对收敛判据，ζ 为初始信赖域半径。\n"
        "SciPy least_squares：库实现的 TRF，同样先按初值归一化并关闭绝对 gtol；ζ 不使用。"
    )
    layout.addWidget(QLabel("优化算法:"), 3, 0)
    layout.addWidget(combo, 3, 1)
    layout.setRowStretch(3, 0)
    layout.setRowStretch(4, 1)
    combo.currentIndexChanged.connect(lambda _: on_changed(combo.currentData()))
    return combo


class Omega3GUI(_legacy_gui.ThermalMeasurementGUI):
    """界面与旧程序一致，拟合走可切换的信赖域反射法。"""

    def init_ui(self):
        super().init_ui()
        self.setWindowTitle("3ω热物性测量系统 v1.1 — 信赖域反射拟合")
        fw = getattr(self, "fitting_widget", None)
        self._solver_combo = None
        if fw is not None:
            self._solver_combo = _install_solver_selector(fw, self._on_solver_changed)
            if hasattr(fw, "zeta"):
                fw.zeta.setToolTip(
                    "自写 TRF：初始信赖域半径 Δ0（归一化空间）。\n"
                    "0.5 表示第一步最多改变约 50% 相对量；SciPy 模式下此项不使用。"
                )
            self._sync_solver_widgets()
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(f"就绪（拟合：{solver_label()}）")

    def apply_solver(self, name):
        set_solver(name)
        if self._solver_combo is not None:
            idx = self._solver_combo.findData(name)
            if idx >= 0 and self._solver_combo.currentIndex() != idx:
                self._solver_combo.blockSignals(True)
                self._solver_combo.setCurrentIndex(idx)
                self._solver_combo.blockSignals(False)
        self._sync_solver_widgets()
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(f"就绪（拟合：{solver_label()}）")

    def _on_solver_changed(self, name):
        if not name:
            return
        set_solver(name)
        self._sync_solver_widgets()
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(f"就绪（拟合：{solver_label()}）")

    def _sync_solver_widgets(self):
        fw = getattr(self, "fitting_widget", None)
        if fw is None or not hasattr(fw, "zeta"):
            return
        from trf_fitter import current_solver
        fw.zeta.setEnabled(current_solver == SOLVER_CUSTOM)

    def show_about(self):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.about(
            self, "关于",
            """
            <h2>3ω热物性测量系统 — 信赖域反射拟合</h2>
            <p>界面、正演与仪器控制复用现有程序，<b>不修改</b>
            thermal_measurement_gui2.py / olson_model26_3.py。</p>
            <p>拟合代码已拆分：</p>
            <ul>
              <li><code>trf_optimizer.py</code> 优化器</li>
              <li><code>trf_fitter.py</code> 只调用 Olson 正演</li>
            </ul>
            <p>「拟合设置」中可切换：自写 Coleman–Li，或 SciPy
            <code>least_squares(method='trf')</code>。</p>
            <p>旧入口（欠松弛高斯-牛顿）：
            <code>python thermal_measurement_gui2.py</code></p>
            """,
        )


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="3ω 信赖域反射拟合入口")
    parser.add_argument(
        "--solver",
        choices=[SOLVER_CUSTOM, SOLVER_SCIPY],
        default=SOLVER_CUSTOM,
        help="优化器：custom=自写 Coleman–Li，scipy=SciPy least_squares",
    )
    args, unknown = parser.parse_known_args(argv)
    return args, unknown


def main(argv=None):
    from PyQt5.QtGui import QColor, QPalette
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    argv = sys.argv[1:] if argv is None else argv
    args, unknown = _parse_args(argv)
    set_solver(args.solver)

    app = QApplication([sys.argv[0]] + unknown)
    app.setStyle("Fusion")
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

    window = Omega3GUI()
    window.apply_solver(args.solver)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
