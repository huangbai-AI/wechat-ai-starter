#!/bin/sh
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo '请先按教程安装 Python 3.10 或更新版本。'
  read -r answer
  exit 1
fi
python3 start.py
