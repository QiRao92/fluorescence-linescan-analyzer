@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "QT_API=PySide6"
cd /d "%SCRIPT_DIR%"
where pythonw.exe >nul 2>nul
if errorlevel 1 (
    echo Could not find pythonw.exe. Starting with python.exe instead.
    python "%SCRIPT_DIR%fluoquant.py"
    if errorlevel 1 pause
) else (
    start "" pythonw.exe "%SCRIPT_DIR%fluoquant.py"
)
endlocal
