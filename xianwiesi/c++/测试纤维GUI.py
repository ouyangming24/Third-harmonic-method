"""
纤维GUI快速测试脚本
在启动完整GUI前检查依赖和基本功能
"""

import sys

def check_dependencies():
    """检查依赖包"""
    print("检查依赖包...")
    print("=" * 60)
    
    dependencies = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'PyQt5': 'PyQt5',
        'pandas': 'Pandas'
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name:15s} - 已安装")
        except ImportError:
            print(f"✗ {name:15s} - 未安装")
            missing.append(module)
    
    print("=" * 60)
    
    if missing:
        print(f"\n缺少以下依赖包: {', '.join(missing)}")
        print("\n请运行以下命令安装:")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ 所有依赖包已安装")
        return True


def test_fiber_model():
    """测试fiber_model模块"""
    print("\n测试 fiber_model 模块...")
    print("=" * 60)
    
    try:
        from fiber_model import (
            calculate_fiber_V3omega,
            create_fiber_material_library,
            create_typical_fiber_params
        )
        
        import numpy as np
        
        # 创建测试数据
        freqs = np.logspace(0, 2, 10)
        fiber_configs = create_typical_fiber_params()
        materials = create_fiber_material_library()
        
        fiber = fiber_configs['platinum_wire_20um']
        material = materials['platinum']
        
        # 计算
        V3, gamma, alpha = calculate_fiber_V3omega(freqs, fiber, material)
        
        print(f"✓ 计算成功")
        print(f"  频率点数: {len(freqs)}")
        print(f"  V3ω 范围: {np.min(np.abs(V3)):.2e} - {np.max(np.abs(V3)):.2e} V")
        print(f"  热扩散率: {alpha:.2e} m²/s")
        print(f"  热时间常数: {gamma:.2e} s")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_gui_imports():
    """测试GUI相关导入"""
    print("\n测试 GUI 模块...")
    print("=" * 60)
    
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        
        # 创建临时应用（不显示窗口）
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        print("✓ PyQt5 导入成功")
        print(f"  Qt 版本: {Qt.PYQT_VERSION_STR if hasattr(Qt, 'PYQT_VERSION_STR') else 'Unknown'}")
        
        return True
        
    except Exception as e:
        print(f"✗ GUI模块导入失败: {str(e)}")
        return False


def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("纤维丝3ω方法GUI系统 - 依赖检查")
    print("=" * 60 + "\n")
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n请先安装缺少的依赖包，然后重新运行测试")
        input("\n按 Enter 键退出...")
        return 1
    
    # 测试核心模块
    model_ok = test_fiber_model()
    
    if not model_ok:
        print("\nfiber_model 模块测试失败，请检查代码")
        input("\n按 Enter 键退出...")
        return 1
    
    # 测试GUI
    gui_ok = test_gui_imports()
    
    if not gui_ok:
        print("\nGUI 模块测试失败，请重新安装 PyQt5")
        input("\n按 Enter 键退出...")
        return 1
    
    # 所有测试通过
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    print("\n可以启动GUI了:")
    print("  Windows: 双击 启动纤维GUI.bat")
    print("  或运行:  python gui_fiber_3omega.py")
    
    input("\n按 Enter 键退出...")
    return 0


if __name__ == "__main__":
    sys.exit(main())



