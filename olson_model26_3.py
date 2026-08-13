import numpy as np
from scipy import integrate


class FitCancelledError(Exception):
    """用户主动中止拟合时抛出。"""


def _build_fisher_matrix_A(D_complex, V3_tilde, V3_exp, fit_mode):
    """由灵敏度矩阵构建 Fisher 信息矩阵 A ≈ J^T J（与拟合迭代中一致）。"""
    if fit_mode == 'both':
        rD, iD = D_complex.real, D_complex.imag
        return rD @ rD.T + iD @ iD.T
    if fit_mode == 'real':
        rD = D_complex.real
        return rD @ rD.T
    if fit_mode == 'imag':
        iD = D_complex.imag
        return iD @ iD.T
    if fit_mode == 'amp':
        amp_model = np.abs(V3_tilde)
        D_amp = np.real(np.conj(V3_tilde) * D_complex) / amp_model
        return D_amp @ D_amp.T
    if fit_mode == 'phase':
        phase_model = np.unwrap(np.angle(V3_tilde))
        D_phase = np.imag(np.conj(V3_tilde) * D_complex) / (np.abs(V3_tilde) ** 2)
        return D_phase @ D_phase.T
    raise ValueError(f"Unknown fit_mode: {fit_mode}")


def _build_residual_vector(V3_tilde, V3_exp, fit_mode):
    """按拟合模式构建实数残差向量（与 Fisher 矩阵 A 同一残差空间）。"""
    if fit_mode == 'both':
        return np.concatenate([
            V3_exp.real - V3_tilde.real,
            V3_exp.imag - V3_tilde.imag,
        ])
    if fit_mode == 'real':
        return V3_exp.real - V3_tilde.real
    if fit_mode == 'imag':
        return V3_exp.imag - V3_tilde.imag
    if fit_mode == 'amp':
        return np.abs(V3_exp) - np.abs(V3_tilde)
    if fit_mode == 'phase':
        return np.unwrap(np.angle(V3_exp)) - np.unwrap(np.angle(V3_tilde))
    raise ValueError(f"Unknown fit_mode: {fit_mode}")


def _uncertainty_from_fisher(A, rss, n_res, param_values, labels, reg=1e-12):
    """由 Fisher 矩阵与残差估计各参数标准误差。

    灵敏度矩阵 D 已按 p_k 缩放（D_k = p_k * dV3/dp_k），因此 inv(A) 的对角元
    对应相对（对数）参数空间的方差；绝对标准误差 = |p_k| * 相对标准误差。
    噪声方差用残差估计：s^2 = RSS / (N_res - M)。
    """
    n = A.shape[0]
    base = {
        'labels': labels, 'values': np.asarray(param_values, dtype=float),
        'sigma_abs': None, 'sigma_rel': None,
        'noise_std': np.nan, 'dof': 0, 'error': None,
    }
    if n == 0:
        base['error'] = '无参与拟合的参数'
        return base

    dof = n_res - n
    if dof <= 0:
        base['error'] = f'残差自由度不足（数据点 {n_res} ≤ 参数数 {n}），无法估计噪声'
        return base

    s2 = rss / dof
    try:
        C = np.linalg.inv(A + np.eye(n) * reg) * s2
    except np.linalg.LinAlgError:
        base['error'] = '矩阵奇异，无法估计参数标准误差'
        return base

    sigma_rel = np.sqrt(np.maximum(np.diag(C), 0.0))
    sigma_abs = np.abs(base['values']) * sigma_rel
    base.update({
        'sigma_abs': sigma_abs,
        'sigma_rel': sigma_rel,
        'noise_std': float(np.sqrt(s2)),
        'dof': dof,
    })
    return base


def _correlation_from_fisher_matrix(A, labels, reg=1e-12):
    """由 Fisher 矩阵求参数归一化相关系数矩阵。"""
    n = A.shape[0]
    if n == 0:
        return {
            'labels': [], 'matrix': np.array([]),
            'strong_pairs': [], 'condition_number': np.nan,
            'error': '无参与拟合的参数',
        }

    A_reg = A + np.eye(n) * reg
    try:
        cond = float(np.linalg.cond(A_reg))
        C = np.linalg.inv(A_reg)
    except np.linalg.LinAlgError:
        return {
            'labels': labels, 'matrix': None,
            'strong_pairs': [], 'condition_number': np.inf,
            'error': '矩阵奇异，无法计算参数相关性（可能存在不可辨识参数）',
        }

    std = np.sqrt(np.maximum(np.diag(C), 0.0))
    corr = np.eye(n)
    strong_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if std[i] > 0 and std[j] > 0:
                rho = float(np.clip(C[i, j] / (std[i] * std[j]), -1.0, 1.0))
                corr[i, j] = corr[j, i] = rho
                if abs(rho) >= 0.7:
                    strong_pairs.append((labels[i], labels[j], rho))

    strong_pairs.sort(key=lambda x: -abs(x[2]))
    return {
        'labels': labels,
        'matrix': corr,
        'strong_pairs': strong_pairs,
        'condition_number': cond,
        'error': None,
    }


# Note: This implementation follows the paper by B.W. Olson, S. Graham, and K. Chen,
# "A practical extension of the 3ω method to multilayer structures," Rev. Sci. Instrum. 76, 053901 (2005).
# Equation numbers in comments refer to this paper.
def convert_Z_to_V3(Z_corrected, freqs, b, sensor_params):
    """Helper function to convert theoretical Z* to theoretical V3_rms."""
    sp = sensor_params
    term = 1j * 4 * np.pi * freqs * sp['H_e'] * sp['rho_C_e']
    prefactor = - (4 * sp['R'] ** 2 * b * sp['L']) / (sp['V_rms'] ** 3 * sp['dR_dT'])

    # Inverse of Eq. (3)
    Z_hat_tilde = 1 / (1 / Z_corrected - term)
    # Inverse of Eq. (2)
    V3_tilde = Z_hat_tilde / prefactor
    return V3_tilde


def _calculate_transform_impedance(chi, f, b, layers):
    """
    Recursively calculates the transform impedance z(chi) for a multilayer stack.
    This function computes z_N(chi) for a given chi and frequency.
    This version is robust for single-layer cases and correctly applies psi to all layers.
    Modified: alpha_y is now calculated from k_y/(rho*cap)
    """
    # --- Layer 1 (n=1, substrate), assumed semi-infinite. Eq. (8). ---
    props_1 = layers[0]
    k_y1 = props_1['k_y']
    # 计算 alpha_y1 = k_y1 / (rho1 * cap1)
    rho1 = props_1.get('rho', 2200)  # 默认密度 kg/m³
    cap1 = props_1.get('cap', 800)  # 默认比热容 J/kg·K
    alpha_y1 = k_y1 / (rho1 * cap1)
    psi_1 = props_1.get('psi', 1.0)  # Correctly apply psi to the substrate

    phi_1 = np.sqrt(psi_1 * chi ** 2 + 1j * 4 * np.pi * f * b ** 2 / alpha_y1)

    if phi_1 == 0: return complex(np.inf, np.inf)

    z_1 = -b / (k_y1 * phi_1)
    Rc1 = props_1.get('Rc', 0.0)
    z_tilde_prev = z_1 - Rc1

    if len(layers) == 1:
        return z_tilde_prev

    # --- Recursively calculate for layers n = 2, 3, ..., N. Eq. (5). ---
    for n in range(1, len(layers)):
        layer_props = layers[n]
        k_yn = layer_props['k_y']
        # 计算 alpha_yn = k_yn / (rhon * capn)
        rhon = layer_props.get('rho', 2000)  # 默认密度
        capn = layer_props.get('cap', 1000)  # 默认比热容
        alpha_yn = k_yn / (rhon * capn)
        psi_n = layer_props.get('psi', 1.0)
        H_n = layer_props['H']

        phi_n = np.sqrt(psi_n * chi ** 2 + 1j * 4 * np.pi * f * b ** 2 / alpha_yn)

        if phi_n == 0: return complex(np.inf, np.inf)

        tanh_term = np.tanh(phi_n * H_n / b)

        denominator_term = (k_yn * phi_n / b) * z_tilde_prev * tanh_term
        if abs(denominator_term - 1.0) < 1e-12: return complex(np.inf, np.inf)

        numerator = denominator_term / tanh_term - tanh_term
        denominator = 1 - denominator_term

        z_n = (b / (k_yn * phi_n)) * (numerator / denominator)

        Rcn = layer_props.get('Rc', 0.0)
        z_tilde_n = z_n - Rcn

        z_tilde_prev = z_tilde_n

    return z_tilde_prev


def calculate_analytical_Z(freqs, b, layers):
    """
    Calculates the analytical surface impedance Z_tilde for a range of frequencies.
    """
    Z_analytical = np.zeros(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        def integrand(chi):
            sinc_term = (np.sin(chi) / chi) ** 2 if chi != 0 else 1.0
            z_transform = _calculate_transform_impedance(chi, f, b, layers)
            return sinc_term * z_transform

        integral_real, _ = integrate.quad(lambda chi: integrand(chi).real, 0, 4 * np.pi, epsabs=1e-12, epsrel=1e-12, limit=100)
        integral_imag, _ = integrate.quad(lambda chi: integrand(chi).imag, 0, 4 * np.pi, epsabs=1e-12, epsrel=1e-12, limit=100)

        Z_analytical[i] = (2 / np.pi) * (integral_real + 1j * integral_imag)
    return Z_analytical


def calculate_experimental_Z(V3_rms, R, b, L, V_rms, dR_dT):
    """
    Converts measured 3-omega voltage into experimental surface impedance. Eq. (2).
    """
    prefactor = - (4 * R ** 2 * b * L) / (V_rms ** 3 * dR_dT)
    return prefactor * V3_rms


def correct_experimental_Z(Z_hat, freqs, H_e, rho_C_e):
    """
    Corrects the experimental impedance for the thermal mass of the heater element. Eq. (3).
    """
    term = 1j * 4 * np.pi * freqs * H_e * rho_C_e
    return 1 / (1 / Z_hat + term)


class ThermalModelFitter:
    """
    A class to fit material properties or sensor parameters to 3-omega data using the Olson et al. model.
    """

    def __init__(self, freqs, V3_exp, b, sensor_params, initial_layers, params_to_fit,
                 fit_mode='both', param_bounds=None, fit_type='material', thickness_link=None):
        """
        Args:
            freqs (np.ndarray): Array of frequencies.
            V3_exp (np.ndarray): Array of experimental 3-omega voltages (complex).
            b (float): Heater half-width.
            sensor_params (dict): Dictionary of sensor properties ('R', 'L', 'V_rms', 'dR_dT', 'H_e', 'rho_C_e').
            initial_layers (list): List of dictionaries for initial layer properties.
            params_to_fit (dict): Dictionary specifying which parameters to fit.
                                 For material: {'layer_0': ['k_y', 'Rc'], ...}
                                 For sensor: {'sensor': ['R', 'dR_dT', 'V_rms'], ...}
            fit_mode (str): Fitting mode. Can be 'both', 'real', 'imag', 'amp', or 'phase'.
            param_bounds (dict): Dictionary of parameter bounds.
            fit_type (str): Type of fitting - 'material' or 'sensor'
            thickness_link (dict): Optional. Couples two adjacent layers whose total
                                 thickness is known but split is unknown. The split is
                                 fitted as a SINGLE Gauss-Newton degree of freedom
                                 (the upper layer thickness H_upper), with the lower
                                 layer thickness derived as H_total - H_upper.
                                 Format: {'upper_idx': i, 'lower_idx': j, 'H_total': float,
                                          'bounds': (H_upper_min, H_upper_max)}  (bounds optional)
        """
        self.freqs = freqs
        self.V3_exp = V3_exp
        self.b = b
        self.sensor_params = sensor_params.copy()  # Make a copy to allow modification
        self.layers = [l.copy() for l in initial_layers]  # Use a copy to avoid side effects
        self.params_to_fit = params_to_fit
        self.param_bounds = param_bounds or {}
        self.fit_type = fit_type  # 'material' or 'sensor'
        self.thickness_link = thickness_link  # 相邻两层总厚度约束（单一自由度），参与高斯-牛顿迭代

        # 若启用厚度约束，确保初值满足 H_upper + H_lower = H_total
        if self.thickness_link is not None:
            u = self.thickness_link['upper_idx']
            l = self.thickness_link['lower_idx']
            Ht = self.thickness_link['H_total']
            h_up = self.layers[u].get('H', Ht * 0.5)
            if not np.isfinite(h_up) or h_up <= 0 or h_up >= Ht:
                h_up = Ht * 0.5
            self.layers[u]['H'] = h_up
            self.layers[l]['H'] = Ht - h_up

        # 根据拟合类型设置参数映射
        if self.fit_type == 'sensor':
            # 传感器参数拟合模式
            self.L_dof = len(params_to_fit.get('sensor', []))
            if self.L_dof == 0:
                raise ValueError("在传感器拟合模式下，必须指定要拟合的传感器参数")
        elif self.fit_type == 'combined':
            # 同时拟合模式：添加材料和传感器参数
            self.L_dof = sum(len(p) for p in params_to_fit.values())
        else:
            # 材料参数拟合模式（原有逻辑）
            self.L_dof = sum(len(p) for p in params_to_fit.values())

        # 厚度约束贡献一个自由度
        if self.thickness_link is not None:
            self.L_dof += 1

        self._param_map = self._create_param_map()
        self._bounds_vector = self._create_bounds_vector()

        if fit_mode not in ['both', 'real', 'imag', 'amp', 'phase']:
            raise ValueError("fit_mode must be one of 'both', 'real', 'imag', 'amp', or 'phase'")
        self.fit_mode = fit_mode

    def _create_param_map(self):
        param_map = []

        if self.fit_type == 'sensor':
            # 传感器参数映射
            if 'sensor' in self.params_to_fit:
                for prop in self.params_to_fit['sensor']:
                    if prop in self.sensor_params:
                        param_map.append({'type': 'sensor', 'prop': prop})
                    else:
                        raise ValueError(f"传感器参数 '{prop}' 不存在")
        elif self.fit_type == 'combined':
            # 同时拟合模式：添加材料和传感器参数
            # 先添加材料参数
            for i, layer in enumerate(self.layers):
                layer_key = f'layer_{i}'
                if layer_key in self.params_to_fit:
                    for prop in self.params_to_fit[layer_key]:
                        param_map.append({'type': 'material', 'layer_idx': i, 'prop': prop})

            # 厚度约束自由度（按材料类处理：改变后需重算 Z）
            if self.thickness_link is not None:
                param_map.append(self._make_thickness_link_entry())

            # 再添加传感器参数
            if 'sensor' in self.params_to_fit:
                for prop in self.params_to_fit['sensor']:
                    if prop in self.sensor_params:
                        param_map.append({'type': 'sensor', 'prop': prop})
                    else:
                        raise ValueError(f"传感器参数 '{prop}' 不存在")
        else:
            # 材料参数映射（原有逻辑）
            for i, layer in enumerate(self.layers):
                layer_key = f'layer_{i}'
                if layer_key in self.params_to_fit:
                    for prop in self.params_to_fit[layer_key]:
                        param_map.append({'type': 'material', 'layer_idx': i, 'prop': prop})

            # 厚度约束自由度
            if self.thickness_link is not None:
                param_map.append(self._make_thickness_link_entry())

        return param_map

    def _make_thickness_link_entry(self):
        """构造厚度约束自由度的参数映射条目（拟合上层厚度 H_upper）。"""
        return {
            'type': 'thickness_link',
            'upper_idx': self.thickness_link['upper_idx'],
            'lower_idx': self.thickness_link['lower_idx'],
            'H_total': self.thickness_link['H_total'],
            'prop': 'H',
        }

    def _create_bounds_vector(self):
        """创建参数边界向量"""
        lower_bounds = []
        upper_bounds = []

        for p_map in self._param_map:
            if p_map['type'] == 'sensor':
                # 传感器参数边界
                prop = p_map['prop']
                if 'sensor' in self.param_bounds and prop in self.param_bounds['sensor']:
                    bounds = self.param_bounds['sensor'][prop]
                    lower_bounds.append(bounds[0])
                    upper_bounds.append(bounds[1])
                else:
                    # 默认传感器参数边界
                    current_val = self.sensor_params[prop]
                    if prop == 'R':  # 电阻
                        lower_bounds.append(current_val * 0.5)
                        upper_bounds.append(current_val * 2.0)
                    elif prop == 'dR_dT':  # 温度系数
                        lower_bounds.append(current_val * 0.1)
                        upper_bounds.append(current_val * 10.0)
                    elif prop == 'V_rms':  # RMS电压
                        lower_bounds.append(current_val * 0.5)
                        upper_bounds.append(current_val * 2.0)
                    elif prop == 'L':  # 长度
                        lower_bounds.append(current_val * 0.9)
                        upper_bounds.append(current_val * 1.1)
                    elif prop == 'H_e':  # 加热器厚度
                        lower_bounds.append(10e-9)
                        upper_bounds.append(1e-6)
                    elif prop == 'rho_C_e':  # 热容
                        lower_bounds.append(1e6)
                        upper_bounds.append(1e7)
                    else:
                        lower_bounds.append(current_val * 0.1)
                        upper_bounds.append(current_val * 10.0)
            elif p_map['type'] == 'thickness_link':
                # 厚度约束自由度：H_upper 的边界，默认 (0, H_total)
                Ht = p_map['H_total']
                bnds = self.thickness_link.get('bounds') if self.thickness_link else None
                if bnds is not None:
                    lower_bounds.append(bnds[0])
                    upper_bounds.append(bnds[1])
                else:
                    lower_bounds.append(Ht * 1e-4)
                    upper_bounds.append(Ht * (1 - 1e-4))
            else:
                # 材料参数边界（原有逻辑）
                layer_key = f'layer_{p_map["layer_idx"]}'
                prop = p_map['prop']

                if layer_key in self.param_bounds and prop in self.param_bounds[layer_key]:
                    bounds = self.param_bounds[layer_key][prop]
                    lower_bounds.append(bounds[0])
                    upper_bounds.append(bounds[1])
                else:
                    # 默认材料参数边界
                    current_val = self.layers[p_map['layer_idx']][prop]
                    if prop == 'Rc':
                        lower_bounds.append(0.0)
                        upper_bounds.append(1e-3)
                    elif prop == 'k_y':
                        lower_bounds.append(0.001)
                        upper_bounds.append(5000)
                    elif prop == 'rho':  # 密度边界
                        lower_bounds.append(100)  # 最小 100 kg/m³
                        upper_bounds.append(20000)  # 最大 20000 kg/m³
                    elif prop == 'cap':  # 比热容边界
                        lower_bounds.append(100)  # 最小 100 J/kg·K
                        upper_bounds.append(5000)  # 最大 5000 J/kg·K
                    elif prop == 'alpha_y':  # 保留兼容性（但不应该再使用）
                        lower_bounds.append(1e-9)
                        upper_bounds.append(1e-3)
                    elif prop == 'H':
                        lower_bounds.append(1e-9)
                        upper_bounds.append(1e-2)
                    else:
                        lower_bounds.append(current_val * 0.01 if current_val > 0 else 1e-10)
                        upper_bounds.append(current_val * 100 if current_val > 0 else 1000)

        return np.array(lower_bounds), np.array(upper_bounds)

    def _apply_bounds(self, param_vector):
        """将参数限制在边界内"""
        lower_bounds, upper_bounds = self._bounds_vector
        return np.clip(param_vector, lower_bounds, upper_bounds)

    def _get_params_as_vector(self):
        vec = np.zeros(self.L_dof)
        for i, p_map in enumerate(self._param_map):
            if p_map['type'] == 'sensor':
                vec[i] = self.sensor_params[p_map['prop']]
            elif p_map['type'] == 'thickness_link':
                # 自由度为上层厚度 H_upper
                vec[i] = self.layers[p_map['upper_idx']]['H']
            else:
                vec[i] = self.layers[p_map['layer_idx']][p_map['prop']]
        return vec

    def _update_layers_from_vector(self, param_vector):
        for i, p_map in enumerate(self._param_map):
            if p_map['type'] == 'sensor':
                self.sensor_params[p_map['prop']] = param_vector[i]
            elif p_map['type'] == 'thickness_link':
                # 上层厚度直接取自向量，下层厚度由总厚度约束推出
                h_up = param_vector[i]
                self.layers[p_map['upper_idx']]['H'] = h_up
                self.layers[p_map['lower_idx']]['H'] = p_map['H_total'] - h_up
            else:
                self.layers[p_map['layer_idx']][p_map['prop']] = param_vector[i]

    def _convert_Z_to_V3(self, Z_corrected):
        """Converts theoretical Z* to theoretical V3_rms."""
        sp = self.sensor_params
        term = 1j * 4 * np.pi * self.freqs * sp['H_e'] * sp['rho_C_e']
        prefactor = - (4 * sp['R'] ** 2 * self.b * sp['L']) / (sp['V_rms'] ** 3 * sp['dR_dT'])

        # Inverse of Eq. (3)
        Z_hat_tilde = 1 / (1 / Z_corrected - term)
        # Inverse of Eq. (2)
        V3_tilde = Z_hat_tilde / prefactor
        return V3_tilde, Z_hat_tilde, prefactor

    def _calculate_sensitivity_matrix(self):
        D = np.zeros((self.L_dof, len(self.freqs)), dtype=complex)
        current_params = self._get_params_as_vector()

        # Recalculate base Z with current layer properties
        Z_base = calculate_analytical_Z(self.freqs, self.b, self.layers)
        V3_base, Z_hat_base, prefactor_base = self._convert_Z_to_V3(Z_base)

        if self.fit_type == 'sensor':
            # 传感器参数的灵敏度计算
            for k in range(self.L_dof):
                p_map = self._param_map[k]
                prop = p_map['prop']

                params_perturbed = np.copy(current_params)
                delta = 1e-6 * params_perturbed[k] if params_perturbed[k] != 0 else 1e-8
                params_perturbed[k] += delta

                # 保存原始传感器参数
                original_value = self.sensor_params[prop]

                # 更新传感器参数
                self.sensor_params[prop] = params_perturbed[k]

                # 重新计算V3
                V3_perturbed, _, _ = self._convert_Z_to_V3(Z_base)

                # 恢复原始参数
                self.sensor_params[prop] = original_value

                # 计算导数
                derivative_V3 = (V3_perturbed - V3_base) / delta

                # Store the sensitivity, D_k = p_k * dV3/dp_k
                D[k, :] = current_params[k] * derivative_V3

        elif self.fit_type == 'combined':
            # 同时拟合模式：材料和传感器参数的灵敏度计算
            original_layers = [l.copy() for l in self.layers]  # 保存原始层参数

            for k in range(self.L_dof):
                p_map = self._param_map[k]
                params_perturbed = np.copy(current_params)
                delta = 1e-6 * params_perturbed[k] if params_perturbed[k] != 0 else 1e-8
                params_perturbed[k] += delta

                if p_map['type'] in ('material', 'thickness_link'):
                    # 材料参数（含厚度约束自由度）的灵敏度计算
                    # 注意：厚度约束自由度更新时会同时改变上、下两层厚度，
                    # 因此有限差分自动包含 dH_lower/dH_upper = -1 的耦合项，
                    # 即对 H_upper 的总导数 = ∂/∂H_upper - ∂/∂H_lower。
                    self._update_layers_from_vector(params_perturbed)

                    # 重新计算Z（材料参数改变需要重新计算Z）
                    Z_perturbed = calculate_analytical_Z(self.freqs, self.b, self.layers)
                    V3_perturbed, _, _ = self._convert_Z_to_V3(Z_perturbed)

                    # 恢复原始层参数
                    self.layers = [l.copy() for l in original_layers]

                elif p_map['type'] == 'sensor':
                    # 传感器参数的灵敏度计算
                    prop = p_map['prop']

                    # 保存原始传感器参数
                    original_value = self.sensor_params[prop]

                    # 更新传感器参数
                    self.sensor_params[prop] = params_perturbed[k]

                    # 重新计算V3（使用相同的Z_base，因为材料属性没变）
                    V3_perturbed, _, _ = self._convert_Z_to_V3(Z_base)

                    # 恢复原始参数
                    self.sensor_params[prop] = original_value

                # 计算导数
                derivative_V3 = (V3_perturbed - V3_base) / delta
                D[k, :] = current_params[k] * derivative_V3
        else:
            # 材料参数的灵敏度计算（原有逻辑）
            # --- Chain Rule Conversion for Sensitivity ---
            conversion_factor = (1 / prefactor_base) * (Z_hat_base / Z_base) ** 2
            original_layers = [l.copy() for l in self.layers]  # Deep copy for restoration

            for k in range(self.L_dof):
                params_perturbed = np.copy(current_params)
                delta = 1e-6 * params_perturbed[k] if params_perturbed[k] != 0 else 1e-8
                params_perturbed[k] += delta

                self._update_layers_from_vector(params_perturbed)

                Z_perturbed = calculate_analytical_Z(self.freqs, self.b, self.layers)

                # Restore original layers for the next iteration of this loop
                self.layers = [l.copy() for l in original_layers]

                # Calculate derivative of Z_tilde w.r.t parameter p_k
                derivative_Z = (Z_perturbed - Z_base) / delta

                # Apply chain rule to get derivative of V3 w.r.t parameter p_k
                derivative_V3 = derivative_Z * conversion_factor

                # Store the sensitivity, D_k = p_k * dV3/dp_k
                D[k, :] = current_params[k] * derivative_V3

        return D

    def fit(self, max_iter=20, tol=1e-5, zeta=0.5, should_cancel=None):
        """
        Performs the iterative fitting process against V3_rms data.
        """
        fit_target = "传感器参数" if self.fit_type == 'sensor' else "材料热物性"
        print(f"开始迭代拟合 (目标: {fit_target}, 模式: {self.fit_mode})...")

        for j in range(max_iter):
            if should_cancel is not None and should_cancel():
                print("拟合已由用户中止。")
                raise FitCancelledError()

            # On each iteration, calculate the model prediction with current parameters
            Z_tilde = calculate_analytical_Z(self.freqs, self.b, self.layers)
            V3_tilde, _, _ = self._convert_Z_to_V3(Z_tilde)

            # This calculates D_k = p_k * dV3/dp_k for each parameter p_k
            D_V3_complex = self._calculate_sensitivity_matrix()

            A, B = 0, 0

            if self.fit_mode == 'both':
                rD, iD = D_V3_complex.real, D_V3_complex.imag
                r_err = self.V3_exp.real - V3_tilde.real
                i_err = self.V3_exp.imag - V3_tilde.imag
                A = rD @ rD.T + iD @ iD.T
                B = rD @ r_err + iD @ i_err
            elif self.fit_mode == 'real':
                rD = D_V3_complex.real
                r_err = self.V3_exp.real - V3_tilde.real
                A = rD @ rD.T
                B = rD @ r_err
            elif self.fit_mode == 'imag':
                iD = D_V3_complex.imag
                i_err = self.V3_exp.imag - V3_tilde.imag
                A = iD @ iD.T
                B = iD @ i_err
            elif self.fit_mode == 'amp':
                V3_exp_amp = np.abs(self.V3_exp)
                V3_tilde_amp = np.abs(V3_tilde)
                err = V3_exp_amp - V3_tilde_amp

                # D_amp = p * d|V3|/dp = Re(conj(V3) * (p*dV3/dp)) / |V3|
                D_amp = np.real(np.conj(V3_tilde) * D_V3_complex) / V3_tilde_amp
                A = D_amp @ D_amp.T
                B = D_amp @ err
            elif self.fit_mode == 'phase':
                # Use np.unwrap to handle phase wrapping issues across the frequency spectrum
                V3_exp_phase = np.unwrap(np.angle(self.V3_exp))
                V3_tilde_phase = np.unwrap(np.angle(V3_tilde))
                err = V3_exp_phase - V3_tilde_phase

                # D_phase = p*d(angle(V3))/dp = Im(conj(V3) * (p*dV3/dp)) / |V3|^2
                D_phase = np.imag(np.conj(V3_tilde) * D_V3_complex) / (np.abs(V3_tilde) ** 2)
                A = D_phase @ D_phase.T
                B = D_phase @ err

            try:
                # Add regularization to avoid singular matrix
                A += np.eye(A.shape[0]) * 1e-12
                c = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                raise RuntimeError("矩阵A是奇异矩阵. 灵敏度过低，无法继续拟合。请检查初始值和边界设置。")

            current_params = self._get_params_as_vector()
            new_params = (c * zeta + 1) * current_params

            # 应用边界约束
            new_params = self._apply_bounds(new_params)
            self._update_layers_from_vector(new_params)

            # 计算实际的参数变化（考虑边界约束后的）
            actual_change = new_params - current_params
            epsilon = np.sum(np.abs(actual_change / current_params))

            print(f"迭代 {j + 1}/{max_iter}: Epsilon = {epsilon:.4e}")

            # 检查是否有参数触及边界（标签与 GUI 层编号一致：顶层=第1层）
            lower_bounds, upper_bounds = self._bounds_vector
            at_lower = np.isclose(new_params, lower_bounds, rtol=1e-6)
            at_upper = np.isclose(new_params, upper_bounds, rtol=1e-6)
            if np.any(at_lower) or np.any(at_upper):
                labels = self.get_param_labels(gui_order=True)
                bounded_params = [
                    labels[idx] for idx in range(len(self._param_map))
                    if at_lower[idx] or at_upper[idx]
                ]
                print(f"  警告: 以下参数触及边界: {', '.join(bounded_params)}")

            if epsilon < tol:
                print(f"在 {j + 1} 次迭代后收敛。")
                break
        else:
            print("警告: 已达到最大迭代次数但未收敛。")

        # 输出拟合结果（详细层参数由 GUI 汇总展示，此处避免重复打印）
        if self.fit_type == 'sensor':
            print("\n最终拟合的传感器参数:")
            for p_map in self._param_map:
                prop = p_map['prop']
                val = self.sensor_params[prop]
                if 'sensor' in self.param_bounds and prop in self.param_bounds['sensor']:
                    bounds = self.param_bounds['sensor'][prop]
                    print(f"  {prop}: {val:.4e} (边界: [{bounds[0]:.2e}, {bounds[1]:.2e}])")
                else:
                    print(f"  {prop}: {val:.4e} (无约束)")
            return self.sensor_params
        elif self.fit_type == 'combined':
            return {'layers': self.layers, 'sensor_params': self.sensor_params}
        else:
            return self.layers

    def get_param_labels(self, gui_order=True):
        """返回参与拟合参数的可读标签（默认映射为 GUI 层编号：顶层=第1层）。"""
        N = len(self.layers)
        labels = []
        for p_map in self._param_map:
            if p_map['type'] == 'sensor':
                labels.append(f"传感器.{p_map['prop']}")
            elif p_map['type'] == 'thickness_link':
                u, l = p_map['upper_idx'], p_map['lower_idx']
                if gui_order:
                    gu, gl = N - 1 - u, N - 1 - l
                    labels.append(f"H约束(第{gu + 1}+第{gl + 1}层)")
                else:
                    labels.append(f"H_link(L{u}+L{l})")
            else:
                i, prop = p_map['layer_idx'], p_map['prop']
                if gui_order:
                    labels.append(f"第{N - 1 - i + 1}层.{prop}")
                else:
                    labels.append(f"layer_{i}.{prop}")
        return labels

    def compute_parameter_correlation(self, gui_order=True):
        """拟合收敛后，基于当前灵敏度矩阵计算参数归一化相关系数。"""
        D = self._calculate_sensitivity_matrix()
        Z_tilde = calculate_analytical_Z(self.freqs, self.b, self.layers)
        V3_tilde, _, _ = self._convert_Z_to_V3(Z_tilde)
        A = _build_fisher_matrix_A(D, V3_tilde, self.V3_exp, self.fit_mode)
        labels = self.get_param_labels(gui_order=gui_order)
        return _correlation_from_fisher_matrix(A, labels)

    def compute_parameter_uncertainty(self, gui_order=True):
        """拟合收敛后，用残差估计噪声并给出各参数标准误差（±σ）。"""
        D = self._calculate_sensitivity_matrix()
        Z_tilde = calculate_analytical_Z(self.freqs, self.b, self.layers)
        V3_tilde, _, _ = self._convert_Z_to_V3(Z_tilde)
        A = _build_fisher_matrix_A(D, V3_tilde, self.V3_exp, self.fit_mode)
        resid = _build_residual_vector(V3_tilde, self.V3_exp, self.fit_mode)
        rss = float(np.sum(resid ** 2))
        labels = self.get_param_labels(gui_order=gui_order)
        values = self._get_params_as_vector()
        return _uncertainty_from_fisher(A, rss, len(resid), values, labels)


class JointThermalModelFitter:
    """
    Joint fitting using multiple sensors (different linewidths) on the same sample.
    Material parameters are shared across all datasets; each dataset has its own
    sensor parameters and frequency/V3 data.
    """

    def __init__(self, datasets, initial_layers, params_to_fit,
                 fit_mode='both', param_bounds=None):
        """
        Args:
            datasets: list of dicts, each containing:
                'freqs': np.ndarray of frequencies
                'V3_exp': np.ndarray of complex V3 values
                'b': heater half-width (m)
                'sensor_params': dict with R, L, V_rms, dR_dT, H_e, rho_C_e
            initial_layers: list of layer property dicts (shared)
            params_to_fit: dict, e.g. {'layer_0': ['k_y', 'psi'], 'layer_1': ['k_y']}
            fit_mode: 'both', 'real', 'imag', 'amp', 'phase'
            param_bounds: dict of bounds per layer/param
        """
        self.datasets = datasets
        self.layers = [l.copy() for l in initial_layers]
        self.params_to_fit = params_to_fit
        self.param_bounds = param_bounds or {}
        self.fit_mode = fit_mode

        self.L_dof = sum(len(p) for key, p in params_to_fit.items() if key.startswith('layer_'))
        self._param_map = self._create_param_map()
        self._bounds_vector = self._create_bounds_vector()

    def _create_param_map(self):
        param_map = []
        for i, layer in enumerate(self.layers):
            layer_key = f'layer_{i}'
            if layer_key in self.params_to_fit:
                for prop in self.params_to_fit[layer_key]:
                    param_map.append({'layer_idx': i, 'prop': prop})
        return param_map

    def _create_bounds_vector(self):
        lower_bounds = []
        upper_bounds = []
        for p_map in self._param_map:
            layer_key = f'layer_{p_map["layer_idx"]}'
            prop = p_map['prop']
            if layer_key in self.param_bounds and prop in self.param_bounds[layer_key]:
                bounds = self.param_bounds[layer_key][prop]
                lower_bounds.append(bounds[0])
                upper_bounds.append(bounds[1])
            else:
                current_val = self.layers[p_map['layer_idx']][prop]
                if prop == 'Rc':
                    lower_bounds.append(0.0)
                    upper_bounds.append(1e-3)
                elif prop == 'k_y':
                    lower_bounds.append(0.001)
                    upper_bounds.append(5000)
                elif prop == 'rho':
                    lower_bounds.append(100)
                    upper_bounds.append(20000)
                elif prop == 'cap':
                    lower_bounds.append(100)
                    upper_bounds.append(5000)
                elif prop == 'H':
                    lower_bounds.append(1e-9)
                    upper_bounds.append(1e-2)
                elif prop == 'psi':
                    lower_bounds.append(0.01)
                    upper_bounds.append(100)
                else:
                    lower_bounds.append(current_val * 0.01 if current_val > 0 else 1e-10)
                    upper_bounds.append(current_val * 100 if current_val > 0 else 1000)
        return np.array(lower_bounds), np.array(upper_bounds)

    def _apply_bounds(self, param_vector):
        lower_bounds, upper_bounds = self._bounds_vector
        return np.clip(param_vector, lower_bounds, upper_bounds)

    def _get_params_as_vector(self):
        vec = np.zeros(self.L_dof)
        for i, p_map in enumerate(self._param_map):
            vec[i] = self.layers[p_map['layer_idx']][p_map['prop']]
        return vec

    def _update_layers_from_vector(self, param_vector):
        for i, p_map in enumerate(self._param_map):
            self.layers[p_map['layer_idx']][p_map['prop']] = param_vector[i]

    def _compute_sensitivity_for_dataset(self, freqs, b, sensor_params):
        """Compute D matrix (material sensitivity) for one dataset."""
        Z_base = calculate_analytical_Z(freqs, b, self.layers)
        V3_base = convert_Z_to_V3(Z_base, freqs, b, sensor_params)

        D = np.zeros((self.L_dof, len(freqs)), dtype=complex)
        current_params = self._get_params_as_vector()
        original_layers = [l.copy() for l in self.layers]

        for k in range(self.L_dof):
            params_perturbed = np.copy(current_params)
            delta = 1e-6 * params_perturbed[k] if params_perturbed[k] != 0 else 1e-8
            params_perturbed[k] += delta
            self._update_layers_from_vector(params_perturbed)

            Z_pert = calculate_analytical_Z(freqs, b, self.layers)
            V3_pert = convert_Z_to_V3(Z_pert, freqs, b, sensor_params)

            self.layers = [l.copy() for l in original_layers]

            D[k, :] = current_params[k] * (V3_pert - V3_base) / delta

        return D, V3_base

    def _accumulate_normal_equations(self, D_complex, V3_exp, V3_tilde):
        """Compute A and B contributions from one dataset for the chosen fit_mode."""
        if self.fit_mode == 'both':
            rD, iD = D_complex.real, D_complex.imag
            r_err = V3_exp.real - V3_tilde.real
            i_err = V3_exp.imag - V3_tilde.imag
            A = rD @ rD.T + iD @ iD.T
            B = rD @ r_err + iD @ i_err
        elif self.fit_mode == 'real':
            rD = D_complex.real
            r_err = V3_exp.real - V3_tilde.real
            A = rD @ rD.T
            B = rD @ r_err
        elif self.fit_mode == 'imag':
            iD = D_complex.imag
            i_err = V3_exp.imag - V3_tilde.imag
            A = iD @ iD.T
            B = iD @ i_err
        elif self.fit_mode == 'amp':
            amp_exp = np.abs(V3_exp)
            amp_model = np.abs(V3_tilde)
            err = amp_exp - amp_model
            D_amp = np.real(np.conj(V3_tilde) * D_complex) / amp_model
            A = D_amp @ D_amp.T
            B = D_amp @ err
        elif self.fit_mode == 'phase':
            phase_exp = np.unwrap(np.angle(V3_exp))
            phase_model = np.unwrap(np.angle(V3_tilde))
            err = phase_exp - phase_model
            D_phase = np.imag(np.conj(V3_tilde) * D_complex) / (np.abs(V3_tilde) ** 2)
            A = D_phase @ D_phase.T
            B = D_phase @ err
        else:
            raise ValueError(f"Unknown fit_mode: {self.fit_mode}")
        return A, B

    def fit(self, max_iter=20, tol=1e-5, zeta=0.5, should_cancel=None):
        print(f"开始多线宽联合拟合 (共 {len(self.datasets)} 个数据集, 模式: {self.fit_mode})...")
        for ds in self.datasets:
            print(f"  数据集: b={ds['b']*1e6:.1f}μm, {len(ds['freqs'])} 个频率点")

        for j in range(max_iter):
            if should_cancel is not None and should_cancel():
                print("拟合已由用户中止。")
                raise FitCancelledError()

            A_total = np.zeros((self.L_dof, self.L_dof))
            B_total = np.zeros(self.L_dof)

            for ds in self.datasets:
                D, V3_model = self._compute_sensitivity_for_dataset(
                    ds['freqs'], ds['b'], ds['sensor_params']
                )
                A_ds, B_ds = self._accumulate_normal_equations(
                    D, ds['V3_exp'], V3_model
                )
                A_total += A_ds
                B_total += B_ds

            try:
                A_total += np.eye(self.L_dof) * 1e-12
                c = np.linalg.solve(A_total, B_total)
            except np.linalg.LinAlgError:
                raise RuntimeError("矩阵A是奇异矩阵，灵敏度过低，无法继续拟合。请检查初始值和边界设置。")

            current_params = self._get_params_as_vector()
            new_params = (c * zeta + 1) * current_params
            new_params = self._apply_bounds(new_params)
            self._update_layers_from_vector(new_params)

            actual_change = new_params - current_params
            epsilon = np.sum(np.abs(actual_change / current_params))
            print(f"迭代 {j + 1}/{max_iter}: Epsilon = {epsilon:.4e}")

            lower_bounds, upper_bounds = self._bounds_vector
            at_lower = np.isclose(new_params, lower_bounds, rtol=1e-6)
            at_upper = np.isclose(new_params, upper_bounds, rtol=1e-6)
            if np.any(at_lower) or np.any(at_upper):
                labels = self.get_param_labels(gui_order=True)
                bounded_params = [
                    labels[idx] for idx in range(len(self._param_map))
                    if at_lower[idx] or at_upper[idx]
                ]
                print(f"  警告: 以下参数触及边界: {', '.join(bounded_params)}")

            if epsilon < tol:
                print(f"在 {j + 1} 次迭代后收敛。")
                break
        else:
            print("警告: 已达到最大迭代次数但未收敛。")

        # 详细层参数由 GUI 汇总展示，避免与结果区重复
        return self.layers

    def get_param_labels(self, gui_order=True):
        """返回联合拟合参与参数标签（默认 GUI 层编号）。"""
        N = len(self.layers)
        labels = []
        for p_map in self._param_map:
            i, prop = p_map['layer_idx'], p_map['prop']
            if gui_order:
                labels.append(f"第{N - 1 - i + 1}层.{prop}")
            else:
                labels.append(f"layer_{i}.{prop}")
        return labels

    def compute_parameter_correlation(self, gui_order=True):
        """联合拟合收敛后，汇总各数据集灵敏度计算参数相关系数。"""
        A_total = np.zeros((self.L_dof, self.L_dof))
        for ds in self.datasets:
            D, V3_model = self._compute_sensitivity_for_dataset(
                ds['freqs'], ds['b'], ds['sensor_params']
            )
            A_ds = _build_fisher_matrix_A(D, V3_model, ds['V3_exp'], self.fit_mode)
            A_total += A_ds
        labels = self.get_param_labels(gui_order=gui_order)
        return _correlation_from_fisher_matrix(A_total, labels)

    def compute_parameter_uncertainty(self, gui_order=True):
        """联合拟合收敛后，汇总各数据集残差与灵敏度估计参数标准误差。"""
        A_total = np.zeros((self.L_dof, self.L_dof))
        rss = 0.0
        n_res = 0
        for ds in self.datasets:
            D, V3_model = self._compute_sensitivity_for_dataset(
                ds['freqs'], ds['b'], ds['sensor_params']
            )
            A_total += _build_fisher_matrix_A(D, V3_model, ds['V3_exp'], self.fit_mode)
            resid = _build_residual_vector(V3_model, ds['V3_exp'], self.fit_mode)
            rss += float(np.sum(resid ** 2))
            n_res += len(resid)
        labels = self.get_param_labels(gui_order=gui_order)
        values = self._get_params_as_vector()
        return _uncertainty_from_fisher(A_total, rss, n_res, values, labels)