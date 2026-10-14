#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
itu_python="${ITU_PYTHON:-/home/zhangqingtian/.venv/bin/python}"
if [[ ! -x "$itu_python" ]]; then
  echo "找不到音频服务环境，请设置 ITU_PYTHON 为已安装 server/requirements.txt 的 Python。" >&2
  exit 1
fi
exec "$itu_python" -m uvicorn server.app:app --host 127.0.0.1 --port 8001 "$@"
