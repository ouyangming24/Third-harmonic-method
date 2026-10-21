"""TRF 拟合适配层：只调用 Olson 正演，优化器走 trf_optimizer。

不修改 olson_model26_3.py / thermal_measurement_gui2.py。
继承旧 Fitter 仅复用参数打包、边界、相关性分析；fit() 整段替换。
"""

from __future__ import annotations

import numpy as np

from olson_model26_3 import (
    FitCancelledError,
    JointThermalModelFitter,
    ThermalModelFitter,
    _build_residual_vector,
    calculate_analytical_Z,
    convert_Z_to_V3,
)
from trf_optimizer import (
    SOLVER_CUSTOM,
    SOLVER_LABELS,
    FitCancelled,
    build_scaling,
    minimize_trf,
    report_result,
)

# GUI 下拉框写入这里；fit() 开始时读取
current_solver = SOLVER_CUSTOM


def set_solver(name):
    global current_solver
    name = (name or SOLVER_CUSTOM).lower()
    if name not in SOLVER_LABELS:
        raise ValueError(f"未知求解器: {name!r}")
    current_solver = name
    return current_solver


def solver_label(name=None):
    return SOLVER_LABELS.get(name or current_solver, current_solver)


def _run_scaled_fit(residual_physical, p0, lo, hi, *, max_iter, tol, zeta,
                    should_cancel, labels):
    p0, lo, hi, scale = build_scaling(p0, lo, hi)
    solver = current_solver

    def fun(x):
        return residual_physical(x * scale)

    try:
        result = minimize_trf(
            fun, p0 / scale, lo / scale, hi / scale,
            solver=solver, max_iter=max_iter, tol=float(tol),
            delta0=float(zeta), should_cancel=should_cancel,
        )
    except FitCancelled:
        raise FitCancelledError() from None

    p_fit = result["x"] * scale
    report_result(result, p_fit, lo, hi, labels, float(zeta))
    return p_fit


class ThermalModelFitterTRF(ThermalModelFitter):
    """正演仍用 Olson；优化器按 current_solver 切换。"""

    def fit(self, max_iter=20, tol=1e-5, zeta=0.5, should_cancel=None):
        fit_target = "传感器参数" if self.fit_type == "sensor" else "材料热物性"
        print(f"开始信赖域反射拟合 [{solver_label()}] "
              f"(目标: {fit_target}, 模式: {self.fit_mode})...")

        p0 = self._get_params_as_vector()
        lo, hi = self._bounds_vector
        self._update_layers_from_vector(p0)
        labels = self.get_param_labels(gui_order=True)

        z_fixed = None
        if self.fit_type == "sensor":
            z_fixed = calculate_analytical_Z(self.freqs, self.b, self.layers)

        def residual_physical(p):
            if should_cancel is not None and should_cancel():
                print("拟合已由用户中止。")
                raise FitCancelledError()
            self._update_layers_from_vector(p)
            Z = z_fixed if z_fixed is not None else calculate_analytical_Z(
                self.freqs, self.b, self.layers)
            V3, _, _ = self._convert_Z_to_V3(Z)
            return _build_residual_vector(V3, self.V3_exp, self.fit_mode)

        p_fit = _run_scaled_fit(
            residual_physical, p0, lo, hi,
            max_iter=max_iter, tol=tol, zeta=zeta,
            should_cancel=should_cancel, labels=labels,
        )
        self._update_layers_from_vector(p_fit)

        if self.fit_type == "sensor":
            return self.sensor_params
        if self.fit_type == "combined":
            return {"layers": self.layers, "sensor_params": self.sensor_params}
        return self.layers


class JointThermalModelFitterTRF(JointThermalModelFitter):
    """多线宽联合拟合的 TRF 适配。"""

    def fit(self, max_iter=20, tol=1e-5, zeta=0.5, should_cancel=None):
        print(f"开始多线宽联合信赖域反射拟合 [{solver_label()}] "
              f"(共 {len(self.datasets)} 个数据集, 模式: {self.fit_mode})...")
        for ds in self.datasets:
            print(f"  数据集: b={ds['b'] * 1e6:.1f}μm, {len(ds['freqs'])} 个频率点")

        p0 = self._get_params_as_vector()
        lo, hi = self._bounds_vector
        self._update_layers_from_vector(p0)
        labels = self.get_param_labels(gui_order=True)

        def residual_physical(p):
            if should_cancel is not None and should_cancel():
                print("拟合已由用户中止。")
                raise FitCancelledError()
            self._update_layers_from_vector(p)
            parts = []
            for ds in self.datasets:
                Z = calculate_analytical_Z(ds["freqs"], ds["b"], self.layers)
                V3 = convert_Z_to_V3(Z, ds["freqs"], ds["b"], ds["sensor_params"])
                parts.append(_build_residual_vector(V3, ds["V3_exp"], self.fit_mode))
            return np.concatenate(parts)

        p_fit = _run_scaled_fit(
            residual_physical, p0, lo, hi,
            max_iter=max_iter, tol=tol, zeta=zeta,
            should_cancel=should_cancel, labels=labels,
        )
        self._update_layers_from_vector(p_fit)
        return self.layers


def install_into_gui_module(gui_module):
    """只替换本进程里 GUI 绑定的拟合器类。"""
    gui_module.ThermalModelFitter = ThermalModelFitterTRF
    gui_module.JointThermalModelFitter = JointThermalModelFitterTRF
