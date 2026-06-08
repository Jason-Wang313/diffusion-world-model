#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v wslpath >/dev/null 2>&1; then
  export PYTHONPATH="${PWD}/src"
  export WSLENV="PYTHONPATH/p:${WSLENV:-}"
elif command -v cygpath >/dev/null 2>&1; then
  export PYTHONPATH="$(cygpath -w "$PWD/src")"
else
  export PYTHONPATH="$PWD/src"
fi
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python.exe >/dev/null 2>&1; then
    PYTHON_BIN="python.exe"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi
"$PYTHON_BIN" experiments/run_experiments.py --mode full
"$PYTHON_BIN" -m dwm_best_of_n.audit
