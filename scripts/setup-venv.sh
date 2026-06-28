#!/usr/bin/env bash
# Create an isolated project venv. Never installs into system Python.
# Usage: ./scripts/setup-venv.sh [extra]   e.g. ./scripts/setup-venv.sh dev,sim
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
EXTRA="${1:-dev}"

if [[ ! -d .venv ]]; then
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[${EXTRA}]"

echo "venv ready at $ROOT/.venv  (activate: source .venv/bin/activate)"
python -c "import pynvml; print('pynvml import OK')" 2>/dev/null \
    || echo "note: pynvml not importable yet (expected if no NVIDIA driver on this host)"
