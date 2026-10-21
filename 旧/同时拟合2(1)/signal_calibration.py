"""
3ω 信号通路校正模块

功能:
1. 加载标定数据（纯电阻替代传感器时的扫频数据），按频率升序排列
2. 拟合通路相位：极点/时延/常数/低阶多项式；可选拟合前 np.unwrap（缓解 ±180° 跳变）
3. 对实测 V₃ω 逐频点基线扣除 + 乘 exp(-j*phi(3f))
4. 导出 JSON 标定参数
"""

import numpy as np
from scipy.optimize import curve_fit


class PhaseTransferModel:
    """通路相位传递函数模型"""

    @staticmethod
    def single_pole(f, fp1):
        return -np.arctan(f / fp1)

    @staticmethod
    def two_pole(f, fp1, fp2):
        return -np.arctan(f / fp1) - np.arctan(f / fp2)

    @staticmethod
    def three_pole(f, fp1, fp2, fp3):
        return -np.arctan(f / fp1) - np.arctan(f / fp2) - np.arctan(f / fp3)

    @staticmethod
    def linear_delay(f, tau):
        return -2 * np.pi * f * tau

    @staticmethod
    def linear_delay_bias(f, phi0, tau):
        """等效时延 + 与频率无关的相位偏置（弧度），适合锁相参考有常数偏移时。"""
        return phi0 - 2 * np.pi * f * tau

    @staticmethod
    def constant_phase(f, phi0):
        """窄带内相位几乎不随 f 变化时的近似：φ(f)≡φ₀（弧度）。校正时 φ(3f) 同为 φ₀。"""
        f = np.asarray(f, dtype=float)
        return np.full_like(f, phi0, dtype=float)


def _poly_model_eval(f: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """多项式 φ(f)，coeffs 与 np.polyval 一致（高次→低次），弧度。"""
    return np.polyval(coeffs, np.asarray(f, dtype=float))


class SignalCalibrator:
    """3ω 信号校正器"""

    def __init__(self):
        self.cal_freqs = None
        self.cal_phase_1omega = None
        self.cal_baseline_3omega = None
        self.phase_model_type = 'two_pole'
        self.phase_params = None
        self.is_calibrated = False
        self.phase_unwrap_used = False

    def load_calibration_data(self, freqs, phase_1omega_deg,
                              x_3omega=None, y_3omega=None):
        """
        加载标定数据

        参数:
            freqs: 频率数组 (Hz)
            phase_1omega_deg: 1ω相位数组 (度)，相对于激励源
            x_3omega: 3ω实部数组 (V)，可选
            y_3omega: 3ω虚部数组 (V)，可选
        """
        self.cal_freqs = np.asarray(freqs, dtype=float)
        self.cal_phase_1omega = np.asarray(phase_1omega_deg, dtype=float)

        if x_3omega is not None and y_3omega is not None:
            self.cal_baseline_3omega = (
                np.asarray(x_3omega, dtype=float)
                + 1j * np.asarray(y_3omega, dtype=float)
            )
        else:
            self.cal_baseline_3omega = None

        order = np.argsort(self.cal_freqs)
        self.cal_freqs = self.cal_freqs[order]
        self.cal_phase_1omega = self.cal_phase_1omega[order]
        if self.cal_baseline_3omega is not None:
            self.cal_baseline_3omega = self.cal_baseline_3omega[order]

    def fit_phase_model(self, model_type='two_pole', unwrap_phase=True):
        """
        拟合相位传递函数模型

        参数:
            model_type: 见代码中 models / poly_1 / poly_2 / poly_3
            unwrap_phase: True 时先将标定相位转弧度再 np.unwrap，再拟合（缓解 ±180° 跳变）

        返回:
            dict: 拟合结果（参数、残差等）
        """
        if self.cal_freqs is None or self.cal_phase_1omega is None:
            raise ValueError("请先加载标定数据")

        self.phase_model_type = model_type
        self.phase_unwrap_used = bool(unwrap_phase)

        phi_raw_rad = np.deg2rad(self.cal_phase_1omega)
        if unwrap_phase:
            phi_target_rad = np.unwrap(phi_raw_rad)
        else:
            phi_target_rad = phi_raw_rad

        # ---------- 多项式（np.polyfit，与 phase_delay 工具思路一致）----------
        if model_type.startswith('poly_'):
            order = int(model_type.split('_')[1])
            npt = len(self.cal_freqs)
            if npt <= order:
                raise ValueError(
                    f"数据点 {npt} 个不足以拟合 {order} 阶多项式（至少需要 {order + 1} 点）"
                )
            # cov=True 要求点数 > order+1，否则 numpy 报错
            if npt > order + 1:
                coeffs, cov = np.polyfit(
                    self.cal_freqs, phi_target_rad, deg=order, cov=True
                )
                if cov is None or np.ndim(cov) == 0:
                    stdev = np.zeros(order + 1, dtype=float)
                else:
                    stdev = np.sqrt(np.maximum(np.diag(cov), 0.0))
            else:
                coeffs = np.polyfit(
                    self.cal_freqs, phi_target_rad, deg=order, cov=False
                )
                stdev = np.zeros(order + 1, dtype=float)
            self.phase_params = np.asarray(coeffs, dtype=float)
            self.is_calibrated = True
            phi_fit = np.polyval(self.phase_params, self.cal_freqs)
            residual_deg = np.rad2deg(phi_target_rad - phi_fit)
            rms_residual = float(np.sqrt(np.mean(residual_deg ** 2)))
            param_names = [f'c{order - i}' for i in range(order + 1)]
            result = {
                'model_type': model_type,
                'param_names': param_names,
                'param_values': self.phase_params,
                'param_stdev': stdev,
                'residual_deg': residual_deg,
                'rms_residual_deg': rms_residual,
                'phi_fitted_deg': np.rad2deg(phi_fit),
                'unwrap_phase': unwrap_phase,
            }
            return result

        # ---------- 解析模型（curve_fit）----------
        models = {
            'single_pole': (PhaseTransferModel.single_pole, [10e6],
                            ([10e3], [1e9])),
            'two_pole': (PhaseTransferModel.two_pole, [10e6, 3e6],
                         ([10e3, 10e3], [1e9, 1e9])),
            'three_pole': (PhaseTransferModel.three_pole, [15e6, 5e6, 2e6],
                           ([10e3, 10e3, 10e3], [1e9, 1e9, 1e9])),
            'linear_delay': (PhaseTransferModel.linear_delay, [1e-8],
                             ([1e-12], [1e-4])),
            'linear_delay_bias': (
                PhaseTransferModel.linear_delay_bias, [0.0, 1e-8],
                ([-0.035, 1e-12], [0.035, 1e-4])),
            'constant_phase': (
                PhaseTransferModel.constant_phase,
                [0.0],
                ([-np.pi / 2], [np.pi / 2])),
        }

        if model_type not in models:
            raise ValueError(f"未知模型类型: {model_type}")

        func, p0, bounds = models[model_type]

        try:
            popt, pcov = curve_fit(
                func, self.cal_freqs, phi_target_rad,
                p0=p0, bounds=bounds, maxfev=10000
            )
        except RuntimeError as e:
            raise RuntimeError(f"拟合失败: {e}")

        self.phase_params = popt
        self.is_calibrated = True

        phi_fit = func(self.cal_freqs, *popt)
        residual_deg = np.rad2deg(phi_target_rad - phi_fit)
        rms_residual = float(np.sqrt(np.mean(residual_deg ** 2)))

        param_names = {
            'single_pole': ['fp1'],
            'two_pole': ['fp1', 'fp2'],
            'three_pole': ['fp1', 'fp2', 'fp3'],
            'linear_delay': ['tau'],
            'linear_delay_bias': ['phi0', 'tau'],
            'constant_phase': ['phi0'],
        }

        result = {
            'model_type': model_type,
            'param_names': param_names[model_type],
            'param_values': popt,
            'param_stdev': np.sqrt(np.diag(pcov)),
            'residual_deg': residual_deg,
            'rms_residual_deg': rms_residual,
            'phi_fitted_deg': np.rad2deg(phi_fit),
            'unwrap_phase': unwrap_phase,
        }

        return result

    def get_phase_correction(self, freqs_base):
        """
        计算给定基频数组对应的 3ω 通路相移

        参数:
            freqs_base: 基频数组 (Hz)

        返回:
            phi_3omega_rad: 3ω处的通路相移 (弧度)
        """
        if not self.is_calibrated:
            raise ValueError("请先完成相位模型拟合")

        f_3omega = 3 * np.asarray(freqs_base, dtype=float)

        if self.phase_model_type.startswith('poly_'):
            return _poly_model_eval(f_3omega, self.phase_params)

        models = {
            'single_pole': PhaseTransferModel.single_pole,
            'two_pole': PhaseTransferModel.two_pole,
            'three_pole': PhaseTransferModel.three_pole,
            'linear_delay': PhaseTransferModel.linear_delay,
            'linear_delay_bias': PhaseTransferModel.linear_delay_bias,
            'constant_phase': PhaseTransferModel.constant_phase,
        }

        func = models[self.phase_model_type]
        return func(f_3omega, *self.phase_params)

    def get_baseline_at_freqs(self, freqs_base):
        """
        通过插值获取给定基频处的 3ω 基线值

        参数:
            freqs_base: 基频数组

        返回:
            baseline_interp: 插值后的复数基线
        """
        if self.cal_baseline_3omega is None:
            return np.zeros(len(freqs_base), dtype=complex)

        real_interp = np.interp(freqs_base, self.cal_freqs,
                                self.cal_baseline_3omega.real)
        imag_interp = np.interp(freqs_base, self.cal_freqs,
                                self.cal_baseline_3omega.imag)
        return real_interp + 1j * imag_interp

    def correct_v3omega(self, freqs_base, V3_measured,
                        apply_baseline=True, apply_phase=True):
        """
        对 V₃ω 数据进行完整校正

        参数:
            freqs_base: 基频数组 (Hz)
            V3_measured: 实测复数 V₃ω 数组
            apply_baseline: 是否扣除基线
            apply_phase: 是否校正相位

        返回:
            V3_corrected: 校正后的复数 V₃ω 数组
        """
        V3 = np.asarray(V3_measured, dtype=complex).copy()

        if apply_baseline:
            baseline = self.get_baseline_at_freqs(freqs_base)
            V3 = V3 - baseline

        if apply_phase and self.is_calibrated:
            phi_path = self.get_phase_correction(freqs_base)
            V3 = V3 * np.exp(-1j * phi_path)

        return V3

    def get_summary(self):
        """返回校正参数摘要"""
        if not self.is_calibrated:
            return "未标定"

        lines = [
            f"模型类型: {self.phase_model_type}",
            f"拟合前 unwrap: {'是' if self.phase_unwrap_used else '否'}",
        ]
        if self.phase_model_type.startswith('poly_'):
            order = int(self.phase_model_type.split('_')[1])
            names = [f'c{order - i}' for i in range(order + 1)]
            for name, val in zip(names, self.phase_params):
                lines.append(f"  {name} (rad 幂次系数) = {val:.6e}")
        else:
            param_names = {
                'single_pole': ['fp1'],
                'two_pole': ['fp1', 'fp2'],
                'three_pole': ['fp1', 'fp2', 'fp3'],
                'linear_delay': ['tau'],
                'linear_delay_bias': ['phi0', 'tau'],
                'constant_phase': ['phi0'],
            }
            names = param_names[self.phase_model_type]
            for name, val in zip(names, self.phase_params):
                if name == 'tau':
                    lines.append(f"  {name} = {val*1e9:.2f} ns")
                elif name == 'phi0':
                    lines.append(f"  {name} = {np.rad2deg(val):.4f}°")
                else:
                    lines.append(f"  {name} = {val/1e6:.2f} MHz")

        has_baseline = self.cal_baseline_3omega is not None
        lines.append(f"3ω基线数据: {'有' if has_baseline else '无'}")
        lines.append(f"标定频率点数: {len(self.cal_freqs)}")

        return "\n".join(lines)

    def save_calibration(self, filepath):
        """保存标定参数到 JSON 文件"""
        import json
        data = {
            'model_type': self.phase_model_type,
            'phase_params': np.asarray(self.phase_params).tolist(),
            'cal_freqs': self.cal_freqs.tolist(),
            'cal_phase_1omega': self.cal_phase_1omega.tolist(),
            'phase_unwrap_used': self.phase_unwrap_used,
        }
        if self.cal_baseline_3omega is not None:
            data['baseline_real'] = self.cal_baseline_3omega.real.tolist()
            data['baseline_imag'] = self.cal_baseline_3omega.imag.tolist()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_calibration(self, filepath):
        """从 JSON 文件加载标定参数"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.cal_freqs = np.array(data['cal_freqs'])
        self.cal_phase_1omega = np.array(data['cal_phase_1omega'])
        self.phase_model_type = data['model_type']
        self.phase_params = np.array(data['phase_params'], dtype=float)
        self.is_calibrated = True
        self.phase_unwrap_used = bool(data.get('phase_unwrap_used', False))

        if 'baseline_real' in data:
            self.cal_baseline_3omega = (
                np.array(data['baseline_real'])
                + 1j * np.array(data['baseline_imag'])
            )
        else:
            self.cal_baseline_3omega = None
