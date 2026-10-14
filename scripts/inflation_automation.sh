#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
export MPLCONFIGDIR="$PROJECT_DIR/tmp/matplotlib"
if [ -n "${INFLATION_PYTHON:-}" ]; then
    PYTHON_BIN="$INFLATION_PYTHON"
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi
exec "$PYTHON_BIN" -u -m inflation_engine.automation "$@"
