@echo off
chcp 65001 >nul
echo ====================================
echo  3ω 方法热物性测量 GUI 系统
echo ====================================
echo.
echo 正在启动...
python gui_3omega.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 启动失败! 请检查:
    echo 1. Python 是否已安装
    echo 2. 依赖库是否已安装 (运行: pip install -r requirements_gui.txt)
    echo.
    pause
)




