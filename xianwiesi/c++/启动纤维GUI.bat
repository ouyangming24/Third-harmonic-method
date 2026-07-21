@echo off
chcp 65001 >nul
echo ========================================
echo 纤维丝3ω方法热物性测量系统 v1.0
echo ========================================
echo.

echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo 检查依赖包...
pip show PyQt5 >nul 2>&1
if errorlevel 1 (
    echo [警告] PyQt5未安装，正在安装依赖...
    pip install -r requirements_fiber_gui.txt
)

echo.
echo 启动GUI程序...
python gui_fiber_3omega.py

if errorlevel 1 (
    echo.
    echo [错误] 程序运行出错
    pause
)



