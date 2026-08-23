@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "QT_API=PySide6"
cd /d "%SCRIPT_DIR%"
rem 优先使用 install_FluoQuant.bat 安装的内置便携环境
if exist "%SCRIPT_DIR%runtime\pythonw.exe" (
    start "" "%SCRIPT_DIR%runtime\pythonw.exe" "%SCRIPT_DIR%fluoquant.py"
    goto :done
)
if exist "%SCRIPT_DIR%runtime\python.exe" (
    "%SCRIPT_DIR%runtime\python.exe" "%SCRIPT_DIR%fluoquant.py"
    if errorlevel 1 pause
    goto :done
)
rem 回退到系统 Python
where pythonw.exe >nul 2>nul
if errorlevel 1 (
    where python.exe >nul 2>nul
    if errorlevel 1 (
        echo 未找到 Python 环境。请先双击 install_FluoQuant.bat 完成一键安装。
        pause
        goto :done
    )
    python "%SCRIPT_DIR%fluoquant.py"
    if errorlevel 1 pause
) else (
    start "" pythonw.exe "%SCRIPT_DIR%fluoquant.py"
)
:done
endlocal
