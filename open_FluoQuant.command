#!/bin/bash
# FluoQuant macOS 启动器：优先使用 install_FluoQuant.command 安装的
# runtime_mac 独立环境，否则回退到系统 python3。
cd "$(dirname "$0")" || exit 1
export QT_API=PySide6

if [ -x "runtime_mac/bin/python" ]; then
    exec "runtime_mac/bin/python" fluoquant.py "$@"
fi

if command -v python3 >/dev/null 2>&1; then
    exec python3 fluoquant.py "$@"
fi

echo "未找到 Python 环境。请先双击 install_FluoQuant.command 完成一键安装。"
read -r -p "按回车退出..."
exit 1
