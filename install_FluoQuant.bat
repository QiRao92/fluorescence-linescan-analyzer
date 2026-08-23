@echo off
rem FluoQuant one-click installer: downloads a portable Python into the
rem local "runtime" folder and installs all dependencies there. Nothing
rem outside this folder is modified.
setlocal
cd /d "%~dp0"
set "PYVER=3.12.10"
set "RUNTIME=%~dp0runtime"
set "MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"

echo ==============================================
echo  FluoQuant 环境一键安装
echo  全部文件安装到本目录的 runtime 文件夹，
echo  不会修改系统或其他 Python 环境。
echo ==============================================
echo.

if exist "%RUNTIME%\python.exe" (
    echo [1/4] 检测到已有便携版 Python，跳过下载。
    goto haveruntime
)
echo [1/4] 下载便携版 Python %PYVER% ...
curl.exe -L --retry 3 -o python-embed.zip https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-embed-amd64.zip
if errorlevel 1 goto fail
echo       解压 ...
powershell -NoProfile -Command "Expand-Archive -Force 'python-embed.zip' 'runtime'"
if errorlevel 1 goto fail
del python-embed.zip
powershell -NoProfile -Command "(Get-Content 'runtime\python312._pth') -replace '#import site','import site' | Set-Content 'runtime\python312._pth'; Add-Content 'runtime\python312._pth' -Value '..'"
if errorlevel 1 goto fail

:haveruntime
if exist "%RUNTIME%\Lib\site-packages\pip" (
    echo [2/4] pip 已安装，跳过。
    goto havepip
)
echo [2/4] 安装 pip ...
curl.exe -L --retry 3 -o get-pip.py https://bootstrap.pypa.io/get-pip.py
if errorlevel 1 goto fail
"%RUNTIME%\python.exe" get-pip.py --no-warn-script-location
if errorlevel 1 goto fail
del get-pip.py

:havepip
echo [3/4] 安装依赖（首次约需 5-15 分钟，请耐心等待）...
"%RUNTIME%\python.exe" -m pip install -r requirements.txt -i %MIRROR%
if errorlevel 1 goto fail

echo.
set "CP="
set /p CP=[4/4] 是否安装 Cellpose 深度学习分割（额外下载约 1-2 GB，输入 y 回车确认，直接回车跳过）:
if /i "%CP%"=="y" (
    "%RUNTIME%\python.exe" -m pip install cellpose -i %MIRROR%
    if errorlevel 1 goto fail
)

echo.
echo 正在校验安装 ...
"%RUNTIME%\python.exe" -c "import PySide6, numpy, scipy, skimage, matplotlib, tifffile, czifile; print('OK')"
if errorlevel 1 goto fail

echo.
echo ==============================================
echo  安装完成！双击 open_FluoQuant.bat 启动软件。
echo ==============================================
pause
exit /b 0

:fail
echo.
echo 安装失败。常见原因：网络不通或被防火墙拦截。
echo 请检查网络后重新运行本脚本（已完成的步骤会自动跳过）。
pause
exit /b 1
