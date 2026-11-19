#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
itu_python="${ITU_PYTHON:-/home/zhangqingtian/.venv/bin/python}"
if [[ ! -x "$itu_python" ]]; then
  echo "找不到音频服务环境，请设置 ITU_PYTHON 为已安装 server/requirements.txt 的 Python。" >&2
  exit 1
fi
export ITU_ACCEPTANCE_CASE_WORKERS="${ITU_ACCEPTANCE_CASE_WORKERS:-2}"
export ITU_MODEL_A_ENABLED="${ITU_MODEL_A_ENABLED:-0}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
exec "$itu_python" -m uvicorn server.app:app --host 127.0.0.1 --port 8001 "$@"
