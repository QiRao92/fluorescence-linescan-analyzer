#!/bin/bash
# FluoQuant macOS 一键安装：在本目录的 runtime_mac 文件夹里创建独立
# Python 环境并安装全部依赖，不会修改系统或其他 Python 环境。
cd "$(dirname "$0")" || exit 1
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "=============================================="
echo " FluoQuant 环境一键安装 (macOS)"
echo " 全部文件安装到本目录的 runtime_mac 文件夹，"
echo " 不会修改系统或其他 Python 环境。"
echo "=============================================="
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "未找到 python3。请先安装 Python："
    echo "  方法一：从 https://www.python.org/downloads/ 下载安装包"
    echo "  方法二：在终端执行 xcode-select --install"
    echo "安装完成后重新双击本脚本。"
    read -r -p "按回车退出..."
    exit 1
fi
echo "[1/4] 使用 $(python3 --version 2>&1)"

if [ ! -x "runtime_mac/bin/python" ]; then
    echo "[2/4] 创建独立环境 runtime_mac ..."
    python3 -m venv runtime_mac || { echo "创建环境失败"; read -r -p "按回车退出..."; exit 1; }
else
    echo "[2/4] 检测到已有 runtime_mac 环境，跳过创建。"
fi

echo "[3/4] 安装依赖（首次约需 5-15 分钟，请耐心等待）..."
"runtime_mac/bin/python" -m pip install --upgrade pip -i "$MIRROR" >/dev/null
"runtime_mac/bin/python" -m pip install -r requirements.txt -i "$MIRROR" || {
    echo "依赖安装失败。请检查网络后重新运行本脚本（已完成的步骤会自动跳过）。"
    read -r -p "按回车退出..."
    exit 1
}

echo
read -r -p "[4/4] 是否安装 Cellpose 深度学习分割（额外下载约 1-2 GB，输入 y 回车确认，直接回车跳过）: " CP
if [ "$CP" = "y" ] || [ "$CP" = "Y" ]; then
    "runtime_mac/bin/python" -m pip install cellpose -i "$MIRROR" || {
        echo "Cellpose 安装失败（不影响其他功能，可稍后重试）。"
    }
fi

echo
echo "正在校验安装 ..."
"runtime_mac/bin/python" -c "import PySide6, numpy, scipy, skimage, matplotlib, tifffile, czifile; print('OK')" || {
    echo "校验失败，请重新运行本脚本。"
    read -r -p "按回车退出..."
    exit 1
}

echo
echo "=============================================="
echo " 安装完成！双击 open_FluoQuant.command 启动软件。"
echo "=============================================="
read -r -p "按回车退出..."
