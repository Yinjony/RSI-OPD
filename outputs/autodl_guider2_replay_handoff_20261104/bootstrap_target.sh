#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
tar -xzf deployment/code.tar.gz
ENV_ROOT="$ROOT/envs/timeopd"
if [[ ! -f "$ROOT/deployment/environment.ready" ]]; then
  if [[ -e "$ENV_ROOT" ]]; then
    echo 'An incomplete environment exists. Inspect it before unpacking again.' >&2
    exit 1
  fi
  mkdir -p "$ENV_ROOT"
  tar -xzf deployment/environment/timeopd.tar.gz -C "$ENV_ROOT"
  "$ENV_ROOT/bin/python" "$ENV_ROOT/bin/conda-unpack"
  touch "$ROOT/deployment/environment.ready"
fi
"$ENV_ROOT/bin/python" deployment/prepare_target.py
"$ENV_ROOT/bin/python" deployment/check_runtime.py
"$ENV_ROOT/bin/python" Wan21/wan_train_guider_sft.py --config Wan21/configs/guider2_replay_autodl.yaml --check-data
touch "$ROOT/deployment/READY"
