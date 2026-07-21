import numpy as np
from scipy import integrate

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

        integral_real, _ = integrate.quad(lambda chi: integrand(chi).real, 0, 4 * np.pi)
        integral_imag, _ = integrate.quad(lambda chi: integrand(chi).imag, 0, 4 * np.pi)

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
                 fit_mode='both', param_bounds=None, fit_type='material'):
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
        """
        self.freqs = freqs
        self.V3_exp = V3_exp
        self.b = b
        self.sensor_params = sensor_params.copy()  # Make a copy to allow modification
        self.layers = [l.copy() for l in initial_layers]  # Use a copy to avoid side effects
        self.params_to_fit = params_to_fit
        self.param_bounds = param_bounds or {}
        self.fit_type = fit_type  # 'material' or 'sensor'

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

        return param_map

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
            else:
                vec[i] = self.layers[p_map['layer_idx']][p_map['prop']]
        return vec

    def _update_layers_from_vector(self, param_vector):
        for i, p_map in enumerate(self._param_map):
            if p_map['type'] == 'sensor':
                self.sensor_params[p_map['prop']] = param_vector[i]
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

                if p_map['type'] == 'material':
                    # 材料参数的灵敏度计算
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

    def fit(self, max_iter=20, tol=1e-5, zeta=0.5):
        """
        Performs the iterative fitting process against V3_rms data.
        """
        fit_target = "传感器参数" if self.fit_type == 'sensor' else "材料热物性"
        print(f"开始迭代拟合 (目标: {fit_target}, 模式: {self.fit_mode})...")

        for j in range(max_iter):
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
                print("错误: 矩阵A是奇异矩阵. 灵敏度过低。")
                return self.layers if self.fit_type == 'material' else self.sensor_params

            current_params = self._get_params_as_vector()
            new_params = (c * zeta + 1) * current_params

            # 应用边界约束
            new_params = self._apply_bounds(new_params)
            self._update_layers_from_vector(new_params)

            # 计算实际的参数变化（考虑边界约束后的）
            actual_change = new_params - current_params
            epsilon = np.sum(np.abs(actual_change / current_params))

            print(f"迭代 {j + 1}/{max_iter}: Epsilon = {epsilon:.4e}")

            # 检查是否有参数触及边界
            lower_bounds, upper_bounds = self._bounds_vector
            at_lower = np.isclose(new_params, lower_bounds, rtol=1e-6)
            at_upper = np.isclose(new_params, upper_bounds, rtol=1e-6)
            if np.any(at_lower) or np.any(at_upper):
                bounded_params = []
                for idx, p_map in enumerate(self._param_map):
                    if at_lower[idx] or at_upper[idx]:
                        if p_map['type'] == 'sensor':
                            bounded_params.append(f"sensor.{p_map['prop']}")
                        else:
                            bounded_params.append(f"layer_{p_map['layer_idx']}.{p_map['prop']}")
                print(f"  警告: 以下参数触及边界: {', '.join(bounded_params)}")

            if epsilon < tol:
                print(f"在 {j + 1} 次迭代后收敛。")
                break
        else:
            print("警告: 已达到最大迭代次数但未收敛。")

        # 输出拟合结果
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
            print("\n最终拟合的材料和传感器参数:")
            for i, layer in enumerate(self.layers):
                print(f"--- 第 {i} 层 ---")
                layer_key = f'layer_{i}'
                for prop, val in layer.items():
                    # 检查是否是待拟合参数
                    if layer_key in self.params_to_fit and prop in self.params_to_fit[layer_key]:
                        # 检查是否有边界约束
                        if layer_key in self.param_bounds and prop in self.param_bounds[layer_key]:
                            bounds = self.param_bounds[layer_key][prop]
                            print(f"  {prop}: {val:.4e} (边界: [{bounds[0]:.2e}, {bounds[1]:.2e}])")
                        else:
                            print(f"  {prop}: {val:.4e} (无约束)")
                    else:
                        print(f"  {prop}: {val:.4e}")
            print("\n传感器参数:")
            for prop in self.sensor_params:
                if 'sensor' in self.params_to_fit and prop in self.params_to_fit['sensor']:
                    val = self.sensor_params[prop]
                    if 'sensor' in self.param_bounds and prop in self.param_bounds['sensor']:
                        bounds = self.param_bounds['sensor'][prop]
                        print(f"  {prop}: {val:.4e} (边界: [{bounds[0]:.2e}, {bounds[1]:.2e}])")
                    else:
                        print(f"  {prop}: {val:.4e} (无约束)")
            # 返回包含材料层和传感器参数的字典
            return {'layers': self.layers, 'sensor_params': self.sensor_params}
        else:
            print("\n最终拟合的材料属性:")
            for i, layer in enumerate(self.layers):
                print(f"--- 第 {i} 层 ---")
                layer_key = f'layer_{i}'
                for prop, val in layer.items():
                    # 检查是否是待拟合参数
                    if layer_key in self.params_to_fit and prop in self.params_to_fit[layer_key]:
                        # 检查是否有边界约束
                        if layer_key in self.param_bounds and prop in self.param_bounds[layer_key]:
                            bounds = self.param_bounds[layer_key][prop]
                            print(f"  {prop}: {val:.4e} (边界: [{bounds[0]:.2e}, {bounds[1]:.2e}])")
                        else:
                            print(f"  {prop}: {val:.4e} (无约束)")
                    else:
                        print(f"  {prop}: {val:.4e}")
            return self.layers