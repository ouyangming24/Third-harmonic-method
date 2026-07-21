"""
快速测试脚本 - 检查GUI依赖和基本功能
"""
import sys

def check_dependencies():
    """检查依赖库是否安装"""
    required = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'PyQt5': 'PyQt5'
    }
    
    missing = []
    print("检查依赖库...")
    print("-" * 50)
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name:20s} 已安装")
        except ImportError:
            print(f"✗ {name:20s} 未安装")
            missing.append(name)
    
    print("-" * 50)
    
    if missing:
        print(f"\n缺少 {len(missing)} 个依赖库: {', '.join(missing)}")
        print("\n请运行以下命令安装:")
        print("pip install -r requirements_gui.txt")
        return False
    else:
        print("\n所有依赖库已安装! ✓")
        return True

def test_model():
    """测试模型计算功能"""
    print("\n" + "=" * 50)
    print("测试理论模型计算...")
    print("-" * 50)
    
    try:
        import numpy as np
        from olson_model26 import calculate_analytical_Z, convert_Z_to_V3
        
        # 简单测试
        freqs = np.logspace(-1, 2, 10)
        sensor_params = {
            'b': 200e-6,
            'L': 5e-3,
            'R': 6.74,
            'dR_dT': 0.014,
            'V_rms': 0.4,
            'H_e': 100e-9,
            'rho_C_e': 2.5116e6
        }
        layers = [
            {'k_y': 140, 'rho': 2329, 'cap': 700, 'H': np.inf, 'psi': 1.0, 'Rc': 1e-8}
        ]
        
        Z = calculate_analytical_Z(freqs, sensor_params['b'], layers)
        V3 = convert_Z_to_V3(Z, freqs, sensor_params['b'], sensor_params)
        
        print(f"✓ 计算成功!")
        print(f"  频率点数: {len(freqs)}")
        print(f"  V3幅值范围: {np.abs(V3).min()*1e3:.3f} ~ {np.abs(V3).max()*1e3:.3f} mV")
        print(f"  相位范围: {np.angle(V3, deg=True).min():.1f} ~ {np.angle(V3, deg=True).max():.1f} 度")
        return True
        
    except Exception as e:
        print(f"✗ 计算失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_import():
    """测试GUI模块导入"""
    print("\n" + "=" * 50)
    print("测试GUI模块...")
    print("-" * 50)
    
    try:
        from PyQt5.QtWidgets import QApplication
        print("✓ PyQt5导入成功")
        
        # 测试创建应用
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        print("✓ QApplication创建成功")
        
        # 不实际显示GUI,只测试导入
        print("✓ GUI框架正常")
        return True
        
    except Exception as e:
        print(f"✗ GUI测试失败: {str(e)}")
        return False

def main():
    print("=" * 50)
    print("3ω方法 GUI系统 - 依赖检查")
    print("=" * 50)
    print()
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n[失败] 请先安装缺失的依赖库")
        return
    
    # 测试模型
    model_ok = test_model()
    
    # 测试GUI
    gui_ok = test_gui_import()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"依赖检查: {'✓ 通过' if deps_ok else '✗ 失败'}")
    print(f"模型计算: {'✓ 通过' if model_ok else '✗ 失败'}")
    print(f"GUI框架:  {'✓ 通过' if gui_ok else '✗ 失败'}")
    print("=" * 50)
    
    if deps_ok and model_ok and gui_ok:
        print("\n[成功] 所有测试通过! 可以启动GUI程序了:")
        print("运行命令: python gui_3omega.py")
    else:
        print("\n[警告] 存在问题,请检查上述错误信息")
    
    print()

if __name__ == '__main__':
    main()




