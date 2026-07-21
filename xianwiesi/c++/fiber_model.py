import numpy as np
from scipy.optimize import minimize
import copy

# 基于Lu等人论文的纤维丝3ω方法理论模型实现
# "3ω method for specific heat and thermal conductivity measurements"
# Rev. Sci. Instrum. 72, 2996 (2001)

def calculate_fiber_V3omega(freqs, fiber_params, material_props):
    """
    计算纤维丝3ω电压信号
    
    基于论文方程(19): V3ω ≈ (4 I³ L R R')/(π⁴ κ S √(1+(2ωγ)²))
    
    Args:
        freqs (np.ndarray): 频率数组 [Hz]
        fiber_params (dict): 纤维参数
            - L: 电压触点间长度 [m]
            - S: 横截面积 [m²] 
            - R: 电阻 [Ω]
            - R_prime: 温度系数 dR/dT [Ω/K]
            - I_rms: RMS电流 [A]
        material_props (dict): 材料热物性
            - kappa: 热导率 [W/m·K]
            - rho: 密度 [kg/m³]
            - C_p: 比热容 [J/kg·K]
    
    Returns:
        tuple: (V3_complex, gamma, alpha)
            - V3_complex: 复数3ω电压
            - gamma: 热时间常数 [s]
            - alpha: 热扩散率 [m²/s]
    """
    # 提取参数
    L = fiber_params['L']
    S = fiber_params['S']
    R = fiber_params['R']
    R_prime = fiber_params['R_prime']
    I_rms = fiber_params['I_rms']
    
    kappa = material_props['kappa']
    rho = material_props['rho']
    C_p = material_props['C_p']
    
    # 计算热扩散率和热时间常数
    alpha = kappa / (rho * C_p)  # 热扩散率 [m²/s]
    gamma = L**2 / (np.pi**2 * alpha)  # 热时间常数 [s]
    
    # 计算3ω电压幅值 - 方程(19)
    omega = 2 * np.pi * freqs
    denominator = np.sqrt(1 + (2 * omega * gamma)**2)
    
    V3_amplitude = (4 * I_rms**3 * L * R * abs(R_prime)) / (np.pi**4 * kappa * S * denominator)
    
    # 计算相位 - 方程(18)
    tan_phi = 2 * omega * gamma
    phi = np.arctan(tan_phi)
    
    # 根据R_prime的符号确定相位 - 方程(17)
    if R_prime < 0:
        # 负温度系数 (如碳纳米管)
        phase_offset = -phi
    else:
        # 正温度系数 (如铂丝)
        phase_offset = np.pi - phi
    
    # 构建复数3ω电压
    V3_complex = V3_amplitude * np.exp(1j * phase_offset)
    
    return V3_complex, gamma, alpha

def calculate_dc_temperature_rise(fiber_params, material_props):
    """
    计算直流温升 - 方程中的Δ₀
    
    Args:
        fiber_params (dict): 纤维参数
        material_props (dict): 材料热物性
        
    Returns:
        float: 中心点最大直流温升 [K]
    """
    L = fiber_params['L']
    S = fiber_params['S']
    R = fiber_params['R']
    I_rms = fiber_params['I_rms']
    kappa = material_props['kappa']
    
    # Δ₀ = 2I₀²R/(π κS/L) - 论文中的定义
    # 注意：I_rms = I₀/√2，所以 I₀² = 2*I_rms²
    Delta_0 = (2 * 2 * I_rms**2 * R) / (np.pi * kappa * S / L)
    
    return Delta_0

def check_validity_conditions(fiber_params, material_props, freqs):
    """
    检查模型有效性条件
    
    Args:
        fiber_params (dict): 纤维参数
        material_props (dict): 材料热物性
        freqs (np.ndarray): 频率数组
        
    Returns:
        dict: 各种条件的检查结果
    """
    L = fiber_params['L']
    S = fiber_params['S']
    R = fiber_params['R']
    R_prime = fiber_params['R_prime']
    I_rms = fiber_params['I_rms']
    
    kappa = material_props['kappa']
    rho = material_props['rho']
    C_p = material_props['C_p']
    
    # 条件(10): 加热功率均匀性条件
    # I₀²R'L/(n²π²κS) << 1, 这里取n=1
    I0_squared = 2 * I_rms**2  # I₀² = 2*I_rms²
    condition_10 = (I0_squared * abs(R_prime) * L) / (np.pi**2 * kappa * S)
    
    # 条件(31): 辐射热损失条件 (对于室温以下)
    # gγ << 1, 其中 g = 16εσT₀³/(ρC_p D)
    # 假设圆形截面，D = 2√(S/π)
    D = 2 * np.sqrt(S / np.pi)  # 等效直径
    T0 = 300  # 假设室温
    epsilon = 0.1  # 金属表面发射率
    sigma = 5.67e-8  # Stefan-Boltzmann常数
    
    alpha = kappa / (rho * C_p)
    gamma = L**2 / (np.pi**2 * alpha)
    g = 16 * epsilon * sigma * T0**3 / (rho * C_p * D)
    condition_31 = g * gamma
    
    # 频率范围建议: 2ωγ < 4 (即 φ < 76°)
    omega_max = 2 * np.pi * np.max(freqs)
    max_2omega_gamma = 2 * omega_max * gamma
    
    results = {
        'condition_10_value': condition_10,
        'condition_10_satisfied': condition_10 < 0.1,
        'condition_31_value': condition_31,
        'condition_31_satisfied': condition_31 < 0.1,
        'max_2omega_gamma': max_2omega_gamma,
        'frequency_range_ok': max_2omega_gamma < 4,
        'gamma': gamma,
        'equivalent_diameter': D
    }
    
    return results

class FiberThermalFitter:
    """
    纤维丝热物性参数拟合器
    
    基于Lu等人的3ω方法，可以同时拟合热导率κ和比热容C_p
    完全兼容 olson_model26.py 的 ThermalModelFitter 接口
    """
    
    def __init__(self, freqs, V3_exp, fiber_params, initial_material_props, 
                 params_to_fit=None, fit_mode='both',
                 param_bounds=None, fit_type='material'):
        """
        初始化拟合器（与 ThermalModelFitter 兼容的接口）
        
        Args:
            freqs (np.ndarray): 频率数组
            V3_exp (np.ndarray): 实验3ω电压数据 (复数)
            fiber_params (dict): 纤维参数（可作为传感器参数拟合）
            initial_material_props (dict): 材料属性初值
            params_to_fit (dict): 要拟合的参数字典
                - 'material': ['kappa', 'C_p', 'rho'] - 材料参数
                - 'fiber': ['R', 'R_prime', 'I_rms', 'L', 'S'] - 纤维/传感器参数
                例如: {'material': ['kappa', 'C_p'], 'fiber': ['R', 'R_prime']}
            fit_mode (str): 拟合模式 'both', 'real', 'imag', 'amp', 'phase'
            param_bounds (dict): 参数边界约束
                例如: {'material': {'kappa': (0.1, 5000)}, 'fiber': {'R': (1.0, 10.0)}}
            fit_type (str): 拟合类型 - 'material', 'sensor', 'combined'
        """
        self.freqs = freqs
        self.V3_exp = V3_exp
        self.fiber_params = fiber_params.copy()
        self.material_props = initial_material_props.copy()
        self.param_bounds = param_bounds or {}
        self.fit_type = fit_type
        
        # 处理 params_to_fit 参数的兼容性
        if params_to_fit is None:
            params_to_fit = {'material': ['kappa', 'C_p']}
        elif isinstance(params_to_fit, list):
            # 向后兼容：如果是列表，转换为字典格式
            params_to_fit = {'material': params_to_fit}
        
        self.params_to_fit = params_to_fit
        
        # 验证 fit_mode
        if fit_mode not in ['both', 'real', 'imag', 'amp', 'phase']:
            raise ValueError("fit_mode must be one of 'both', 'real', 'imag', 'amp', or 'phase'")
        self.fit_mode = fit_mode
        
        # 根据拟合类型设置自由度
        if self.fit_type == 'sensor':
            # 传感器参数拟合模式（实际上是纤维参数）
            self.L_dof = len(params_to_fit.get('fiber', []))
            if self.L_dof == 0:
                raise ValueError("在传感器拟合模式下，必须指定要拟合的纤维参数 (params_to_fit['fiber'])")
        elif self.fit_type == 'combined':
            # 同时拟合模式
            self.L_dof = sum(len(p) for p in params_to_fit.values())
        else:
            # 材料参数拟合模式
            self.L_dof = len(params_to_fit.get('material', []))
            if self.L_dof == 0:
                raise ValueError("在材料拟合模式下，必须指定要拟合的材料参数 (params_to_fit['material'])")
        
        # 创建参数映射
        self._param_map = self._create_param_map()
        self._bounds_vector = self._create_bounds_vector()
        
    def _create_param_map(self):
        """创建参数映射（与 ThermalModelFitter 兼容）"""
        param_map = []
        
        if self.fit_type == 'sensor':
            # 传感器参数拟合模式（纤维参数）
            if 'fiber' in self.params_to_fit:
                for prop in self.params_to_fit['fiber']:
                    if prop in self.fiber_params:
                        param_map.append({'type': 'fiber', 'prop': prop})
                    else:
                        raise ValueError(f"纤维参数 '{prop}' 不存在于 fiber_params 中")
        
        elif self.fit_type == 'combined':
            # 同时拟合模式：先添加材料参数，再添加纤维参数
            if 'material' in self.params_to_fit:
                for prop in self.params_to_fit['material']:
                    if prop in self.material_props:
                        param_map.append({'type': 'material', 'prop': prop})
                    else:
                        raise ValueError(f"材料参数 '{prop}' 不存在于 material_props 中")
            
            if 'fiber' in self.params_to_fit:
                for prop in self.params_to_fit['fiber']:
                    if prop in self.fiber_params:
                        param_map.append({'type': 'fiber', 'prop': prop})
                    else:
                        raise ValueError(f"纤维参数 '{prop}' 不存在于 fiber_params 中")
        
        else:
            # 材料参数拟合模式（默认）
            if 'material' in self.params_to_fit:
                for prop in self.params_to_fit['material']:
                    if prop in self.material_props:
                        param_map.append({'type': 'material', 'prop': prop})
                    else:
                        raise ValueError(f"材料参数 '{prop}' 不存在于 material_props 中")
        
        return param_map
    
    def _create_bounds_vector(self):
        """创建参数边界向量（与 ThermalModelFitter 兼容）"""
        lower_bounds = []
        upper_bounds = []
        
        for p_map in self._param_map:
            if p_map['type'] == 'fiber':
                # 纤维参数边界
                prop = p_map['prop']
                if 'fiber' in self.param_bounds and prop in self.param_bounds['fiber']:
                    bounds = self.param_bounds['fiber'][prop]
                    lower_bounds.append(bounds[0])
                    upper_bounds.append(bounds[1])
                else:
                    # 默认纤维参数边界
                    current_val = self.fiber_params[prop]
                    if prop == 'R':  # 电阻
                        lower_bounds.append(current_val * 0.5)
                        upper_bounds.append(current_val * 2.0)
                    elif prop == 'R_prime':  # 温度系数
                        # 允许正负值
                        if current_val > 0:
                            lower_bounds.append(current_val * 0.1)
                            upper_bounds.append(current_val * 10.0)
                        else:
                            lower_bounds.append(current_val * 10.0)
                            upper_bounds.append(current_val * 0.1)
                    elif prop == 'I_rms':  # RMS电流
                        lower_bounds.append(current_val * 0.5)
                        upper_bounds.append(current_val * 2.0)
                    elif prop == 'L':  # 长度
                        lower_bounds.append(current_val * 0.9)
                        upper_bounds.append(current_val * 1.1)
                    elif prop == 'S':  # 横截面积
                        lower_bounds.append(current_val * 0.5)
                        upper_bounds.append(current_val * 2.0)
                    else:
                        lower_bounds.append(current_val * 0.1)
                        upper_bounds.append(current_val * 10.0)
            
            else:  # material
                # 材料参数边界
                prop = p_map['prop']
                if 'material' in self.param_bounds and prop in self.param_bounds['material']:
                    bounds = self.param_bounds['material'][prop]
                    lower_bounds.append(bounds[0])
                    upper_bounds.append(bounds[1])
                else:
                    # 默认材料参数边界
                    current_val = self.material_props[prop]
                    if prop == 'kappa':
                        lower_bounds.append(0.1)
                        upper_bounds.append(5000)
                    elif prop == 'C_p':
                        lower_bounds.append(100)
                        upper_bounds.append(5000)
                    elif prop == 'rho':
                        lower_bounds.append(100)
                        upper_bounds.append(20000)
                    else:
                        lower_bounds.append(current_val * 0.01 if current_val > 0 else 1e-10)
                        upper_bounds.append(current_val * 100 if current_val > 0 else 1000)
        
        return np.array(lower_bounds), np.array(upper_bounds)
    
    def _apply_bounds(self, param_vector):
        """将参数限制在边界内"""
        lower_bounds, upper_bounds = self._bounds_vector
        return np.clip(param_vector, lower_bounds, upper_bounds)
    
    def _get_params_as_vector(self):
        """获取当前参数向量"""
        vec = np.zeros(self.L_dof)
        for i, p_map in enumerate(self._param_map):
            if p_map['type'] == 'fiber':
                vec[i] = self.fiber_params[p_map['prop']]
            else:  # material
                vec[i] = self.material_props[p_map['prop']]
        return vec
    
    def _update_params_from_vector(self, param_vector):
        """从参数向量更新材料和纤维属性"""
        for i, p_map in enumerate(self._param_map):
            if p_map['type'] == 'fiber':
                self.fiber_params[p_map['prop']] = param_vector[i]
            else:  # material
                self.material_props[p_map['prop']] = param_vector[i]
        
    def _calculate_sensitivity_matrix(self):
        """计算灵敏度矩阵 D_k = p_k * dV3/dp_k（兼容材料和纤维参数）"""
        D = np.zeros((self.L_dof, len(self.freqs)), dtype=complex)
        current_params = self._get_params_as_vector()
        
        # 计算基准V3
        V3_base, _, _ = calculate_fiber_V3omega(
            self.freqs, self.fiber_params, self.material_props
        )
        
        # 保存原始状态用于恢复
        original_fiber_params = self.fiber_params.copy()
        original_material_props = self.material_props.copy()
        
        # 对每个参数计算灵敏度
        for k in range(self.L_dof):
            p_map = self._param_map[k]
            params_perturbed = current_params.copy()
            delta = 1e-6 * params_perturbed[k] if params_perturbed[k] != 0 else 1e-8
            params_perturbed[k] += delta
            
            # 更新参数
            self._update_params_from_vector(params_perturbed)
            
            # 计算扰动后的V3
            V3_perturbed, _, _ = calculate_fiber_V3omega(
                self.freqs, self.fiber_params, self.material_props
            )
            
            # 恢复原始状态
            self.fiber_params = original_fiber_params.copy()
            self.material_props = original_material_props.copy()
            
            # 计算导数
            derivative_V3 = (V3_perturbed - V3_base) / delta
            
            # 存储灵敏度 D_k = p_k * dV3/dp_k
            D[k, :] = current_params[k] * derivative_V3
        
        return D
    
    def fit(self, max_iter=20, tol=1e-5, zeta=0.5):
        """
        执行迭代拟合（完全兼容 ThermalModelFitter 的接口）
        
        Args:
            max_iter (int): 最大迭代次数
            tol (float): 收敛容差
            zeta (float): 步长因子 (0 < zeta < 1)
            
        Returns:
            根据 fit_type 返回不同内容:
            - 'material': material_props (dict)
            - 'sensor': fiber_params (dict)
            - 'combined': {'fiber_params': dict, 'material_props': dict}
        """
        fit_target = "纤维参数" if self.fit_type == 'sensor' else "材料热物性"
        print(f"开始迭代拟合 (目标: {fit_target}, 模式: {self.fit_mode})...")
        
        # 检查有效性条件
        validity = check_validity_conditions(
            self.fiber_params, self.material_props, self.freqs
        )
        
        print(f"\n模型有效性检查:")
        print(f"  条件(10) - 功率均匀性: {validity['condition_10_value']:.2e} "
              f"({'满足' if validity['condition_10_satisfied'] else '不满足'})")
        print(f"  条件(31) - 辐射损失: {validity['condition_31_value']:.2e} "
              f"({'满足' if validity['condition_31_satisfied'] else '不满足'})")
        print(f"  频率范围 (2ωγ_max): {validity['max_2omega_gamma']:.2f} "
              f"({'合适' if validity['frequency_range_ok'] else '过高'})")
        
        print(f"\n开始迭代 (最大迭代次数: {max_iter}, 容差: {tol:.1e}, 步长因子: {zeta:.2f})")
        
        # 迭代主循环
        for j in range(max_iter):
            # 计算当前理论V3
            V3_tilde, _, _ = calculate_fiber_V3omega(
                self.freqs, self.fiber_params, self.material_props
            )
            
            # 计算灵敏度矩阵 D_k = p_k * dV3/dp_k
            D_V3_complex = self._calculate_sensitivity_matrix()
            
            A, B = 0, 0
            
            # 根据拟合模式构建方程组（完全兼容 ThermalModelFitter）
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
                # Use np.unwrap to handle phase wrapping issues
                V3_exp_phase = np.unwrap(np.angle(self.V3_exp))
                V3_tilde_phase = np.unwrap(np.angle(V3_tilde))
                err = V3_exp_phase - V3_tilde_phase
                
                # D_phase = p*d(angle(V3))/dp = Im(conj(V3) * (p*dV3/dp)) / |V3|^2
                D_phase = np.imag(np.conj(V3_tilde) * D_V3_complex) / (np.abs(V3_tilde) ** 2)
                A = D_phase @ D_phase.T
                B = D_phase @ err
            
            # 求解线性系统
            try:
                # 添加正则化避免奇异矩阵
                A += np.eye(A.shape[0]) * 1e-12
                c = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                print("错误: 矩阵A是奇异矩阵. 灵敏度过低。")
                return self._return_results_by_fit_type()
            
            # 更新参数
            current_params = self._get_params_as_vector()
            new_params = (c * zeta + 1) * current_params
            
            # 应用边界约束
            new_params = self._apply_bounds(new_params)
            self._update_params_from_vector(new_params)
            
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
                        if p_map['type'] == 'fiber':
                            bounded_params.append(f"fiber.{p_map['prop']}")
                        else:
                            bounded_params.append(f"material.{p_map['prop']}")
                print(f"  警告: 以下参数触及边界: {', '.join(bounded_params)}")
            
            # 检查收敛
            if epsilon < tol:
                print(f"在 {j + 1} 次迭代后收敛。")
                break
                
        else:
            print("警告: 已达到最大迭代次数但未收敛。")
        
        # 输出拟合结果（完全兼容 ThermalModelFitter 的输出格式）
        return self._return_results_by_fit_type()
    
    def _return_results_by_fit_type(self):
        """根据拟合类型返回结果（与 ThermalModelFitter 兼容）"""
        if self.fit_type == 'sensor':
            # 传感器参数拟合模式（纤维参数）
            print("\n最终拟合的纤维参数:")
            for p_map in self._param_map:
                prop = p_map['prop']
                val = self.fiber_params[prop]
                if 'fiber' in self.param_bounds and prop in self.param_bounds['fiber']:
                    bounds = self.param_bounds['fiber'][prop]
                    print(f"  {prop}: {val:.4e} (边界: [{bounds[0]:.2e}, {bounds[1]:.2e}])")
                else:
                    print(f"  {prop}: {val:.4e} (无约束)")
            return self.fiber_params
        
        elif self.fit_type == 'combined':
            # 同时拟合模式
            print("\n最终拟合的材料和纤维参数:")
            print("--- 材料属性 ---")
            for p_map in self._param_map:
                if p_map['type'] == 'material':
                    prop = p_map['prop']
                    val = self.material_props[prop]
                    if 'material' in self.param_bounds and prop in self.param_bounds['material']:
                        bounds = self.param_bounds['material'][prop]
                        print(f"  {prop}: {val:.4e} (边界: [{bounds[0]:.2e}, {bounds[1]:.2e}])")
                    else:
                        print(f"  {prop}: {val:.4e} (无约束)")
            
            print("\n--- 纤维参数 ---")
            for p_map in self._param_map:
                if p_map['type'] == 'fiber':
                    prop = p_map['prop']
                    val = self.fiber_params[prop]
                    if 'fiber' in self.param_bounds and prop in self.param_bounds['fiber']:
                        bounds = self.param_bounds['fiber'][prop]
                        print(f"  {prop}: {val:.4e} (边界: [{bounds[0]:.2e}, {bounds[1]:.2e}])")
                    else:
                        print(f"  {prop}: {val:.4e} (无约束)")
            
            # 返回包含材料和纤维参数的字典
            return {'fiber_params': self.fiber_params, 'material_props': self.material_props}
        
        else:
            # 材料参数拟合模式（默认）
            print("\n最终拟合的材料属性:")
            for p_map in self._param_map:
                prop = p_map['prop']
                val = self.material_props[prop]
                if 'material' in self.param_bounds and prop in self.param_bounds['material']:
                    bounds = self.param_bounds['material'][prop]
                    print(f"  {prop}: {val:.4e} (边界: [{bounds[0]:.2e}, {bounds[1]:.2e}])")
                else:
                    print(f"  {prop}: {val:.4e} (无约束)")
            
            # 计算并显示导出参数
            _, gamma_fitted, alpha_fitted = calculate_fiber_V3omega(
                self.freqs, self.fiber_params, self.material_props
            )
            print(f"\n导出参数:")
            print(f"  热扩散率 alpha: {alpha_fitted:.4e} m^2/s")
            print(f"  热时间常数 gamma: {gamma_fitted:.4e} s")
            
            return self.material_props

def create_fiber_material_library():
    """创建纤维材料库"""
    materials = {
        'platinum': {
            'kappa': 71.6,      # W/m·K @ 300K
            'rho': 21450,       # kg/m³
            'C_p': 133,         # J/kg·K @ 300K
            'description': '铂丝 (标准参考材料)'
        },
        'carbon_nanotube': {
            'kappa': 3000,      # W/m·K (轴向)
            'rho': 1300,        # kg/m³
            'C_p': 700,         # J/kg·K
            'description': '碳纳米管束 (高热导率)'
        },
        'copper': {
            'kappa': 401,       # W/m·K
            'rho': 8960,        # kg/m³
            'C_p': 385,         # J/kg·K
            'description': '铜丝'
        },
        'tungsten': {
            'kappa': 173,       # W/m·K
            'rho': 19300,       # kg/m³
            'C_p': 132,         # J/kg·K
            'description': '钨丝'
        },
        'nichrome': {
            'kappa': 11.3,      # W/m·K
            'rho': 8400,        # kg/m³
            'C_p': 450,         # J/kg·K
            'description': '镍铬合金丝'
        }
    }
    
    return materials

def create_typical_fiber_params():
    """创建典型纤维参数配置"""
    configs = {
        'platinum_wire_20um': {
            'L': 8e-3,                    # 长度 8mm
            'diameter': 20e-6,            # 直径 20μm
            'S': np.pi * (10e-6)**2,      # 横截面积
            'R': 5.0,                     # 电阻 5Ω
            'R_prime': 0.01,              # 温度系数 0.01 Ω/K (正)
            'I_rms': 10e-3,               # RMS电流 10mA
            'description': '20μm铂丝 (论文中的配置)'
        },
        # 'carbon_nanotube_bundle': {
        #     'L': 3.2e-3,                    # 长度 1mm
        #     'diameter': 35e-6,            # 直径 10μm
        #     'S': np.pi * (17.5e-6)**2,       # 横截面积
        #     'R': 12.1,                    # 电阻 50Ω
        #     'R_prime': 0.02,             # 温度系数 -0.05 Ω/K (负)
        #     'I_rms': 21.8e-3,                # RMS电流 5mA
        #     'description': '碳纳米管束 (负温度系数)'
        # },
        'carbon_nanotube_bundle': {
            'L': 1e-3,  # 长度 1mm
            'diameter': 10e-6,  # 直径 10μm
            'S': np.pi * (5e-6) ** 2,  # 横截面积
            'R': 50.0,  # 电阻 50Ω
            'R_prime': -0.05,  # 温度系数 -0.05 Ω/K (负)
            'I_rms': 5e-3,  # RMS电流 5mA
            'description': '碳纳米管束 (负温度系数)'
        },
        'thin_wire_1um': {
            'L': 0.5e-3,                  # 长度 0.5mm
            'diameter': 1e-6,             # 直径 1μm
            'S': np.pi * (0.5e-6)**2,     # 横截面积
            'R': 1000.0,                  # 电阻 1000Ω
            'R_prime': 1.0,               # 温度系数 1.0 Ω/K
            'I_rms': 1e-3,                # RMS电流 1mA
            'description': '超细丝 (高阻抗)'
        }
    }
    
    return configs

if __name__ == "__main__":
    # 纤维丝3ω方法模型测试（兼容 ThermalModelFitter 接口）
    print("纤维丝3ω方法模型测试")
    print("="*70)
    
    # 创建测试参数
    freqs = np.logspace(0, 2, 20)  # 1-100 Hz
    
    fiber_params = create_typical_fiber_params()['platinum_wire_20um']
    materials = create_fiber_material_library()
    material_props = materials['platinum']
    
    # 计算理论3ω电压
    V3_theory, gamma, alpha = calculate_fiber_V3omega(freqs, fiber_params, material_props)
    
    print(f"\n基础测试:")
    print(f"  热扩散率: {alpha:.2e} m^2/s")
    print(f"  热时间常数: {gamma:.2e} s")
    print(f"  3ω电压范围: {np.min(np.abs(V3_theory)):.2e} - {np.max(np.abs(V3_theory)):.2e} V")
    
    # 检查有效性条件
    validity = check_validity_conditions(fiber_params, material_props, freqs)
    print(f"\n有效性检查:")
    for key, value in validity.items():
        if isinstance(value, bool):
            print(f"  {key}: {'✓' if value else '✗'}")
        elif isinstance(value, (int, float)):
            print(f"  {key}: {value:.2e}")
    
    # ===== 测试新的拟合接口（兼容 ThermalModelFitter） =====
    print("\n" + "="*70)
    print("测试兼容接口 1: 材料参数拟合 (fit_type='material')")
    print("="*70)
    
    # 生成模拟实验数据（添加噪声）
    np.random.seed(42)
    noise_level = 0.02
    V3_exp = V3_theory * (1 + noise_level * (np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))))
    
    # 材料参数拟合（新接口）
    initial_props = {'kappa': 60.0, 'rho': 21450, 'C_p': 140}
    fitter1 = FiberThermalFitter(
        freqs, V3_exp, fiber_params, initial_props,
        params_to_fit={'material': ['kappa', 'C_p']},
        fit_mode='both',
        param_bounds={'material': {'kappa': (50, 100), 'C_p': (100, 200)}},
        fit_type='material'
    )
    result1 = fitter1.fit(max_iter=10)
    
    print("\n" + "="*70)
    print("测试兼容接口 2: 传感器参数拟合 (fit_type='sensor')")
    print("="*70)
    
    # 传感器参数拟合（纤维参数）
    fitter2 = FiberThermalFitter(
        freqs, V3_exp, fiber_params.copy(), material_props,
        params_to_fit={'fiber': ['R', 'R_prime']},
        fit_mode='both',
        param_bounds={'fiber': {'R': (4.0, 6.0), 'R_prime': (0.005, 0.015)}},
        fit_type='sensor'
    )
    result2 = fitter2.fit(max_iter=10)
    
    print("\n" + "="*70)
    print("测试兼容接口 3: 联合拟合 (fit_type='combined')")
    print("="*70)
    
    # 同时拟合材料和纤维参数
    fitter3 = FiberThermalFitter(
        freqs, V3_exp, fiber_params.copy(), initial_props.copy(),
        params_to_fit={'material': ['kappa'], 'fiber': ['R']},
        fit_mode='both',
        fit_type='combined'
    )
    result3 = fitter3.fit(max_iter=10)
    
    print("\n" + "="*70)
    print("所有测试完成！接口完全兼容 olson_model26.ThermalModelFitter")
    print("="*70)
