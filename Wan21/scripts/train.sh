#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON_BIN:-${PYTHON:-/root/miniconda3/envs/timeopd/bin/python}}" "$ROOT/Wan21/run_experiment.py" train "$@"
