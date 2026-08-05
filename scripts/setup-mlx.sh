#!/usr/bin/env bash
# Create the MLX virtualenv used by `quantpilot bench-mlx`.
# MLX lives in its own venv so quantpilot's core stays dependency-free.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${1:-}"
if [ -z "$PY" ]; then
  for candidate in python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null; then PY="$candidate"; break; fi
  done
fi

echo "creating .venv-mlx with $PY ($($PY --version))"
"$PY" -m venv .venv-mlx
./.venv-mlx/bin/pip install --quiet --upgrade pip
./.venv-mlx/bin/pip install --quiet mlx-lm
./.venv-mlx/bin/python -c "import mlx.core, mlx_lm; print('mlx-lm', mlx_lm.__version__ if hasattr(mlx_lm, '__version__') else 'ok')"
echo "done — quantpilot will find .venv-mlx automatically"
