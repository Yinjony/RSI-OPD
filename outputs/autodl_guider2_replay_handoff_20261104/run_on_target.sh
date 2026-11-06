#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/envs/timeopd/bin/python"
CONFIG="$ROOT/Wan21/configs/guider2_replay_autodl.yaml"
MODE="${1:-check}"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export SP_SIZE=1
case "$MODE" in
  all)
    test -f "$ROOT/deployment/READY" || { echo 'Transfer/preparation not complete: deployment/READY is missing' >&2; exit 1; }
    exec 9>"$ROOT/deployment/pipeline.lock"
    flock -n 9 || { echo 'A pipeline is already running' >&2; exit 1; }
    bash deployment/run_on_target.sh check
    bash deployment/run_on_target.sh preencode
    bash deployment/run_on_target.sh train
    ;;
  check)
    "$PYTHON_BIN" deployment/check_runtime.py
    "$PYTHON_BIN" Wan21/wan_train_guider_sft.py --config "$CONFIG" --check-data
    ;;
  preencode)
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" deployment/runtime.py Wan21/scripts/data_preprocessing/preencode_guider_sft.py --config "$CONFIG"
    ;;
  train)
    CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" Wan21/wan_train_guider_sft.py --config "$CONFIG" --check-cache
    CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=2 deployment/runtime.py Wan21/wan_train_guider_sft.py --config "$CONFIG"
    bash deployment/run_on_target.sh export
    ;;
  export)
    "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
root = Path.cwd()
sys.path[:0] = [str(root / 'Wan21'), str(root), str(root / 'shared')]
from wan_utils.guider_sft_config import load_config
from wan_train_guider_sft_continue_lora import export_final_checkpoint
cfg = load_config(sys.argv[1])
latest = json.loads((Path(cfg.output_dir) / 'latest.json').read_text())
if latest['step'] != cfg.max_steps:
    raise RuntimeError('Training is not complete')
if Path(latest['adapter']).with_name('model.pt').exists():
    raise FileExistsError('Final model.pt already exists; refusing to overwrite')
export_final_checkpoint(cfg, [])
PY
    ;;
  *) echo 'Modes: all, check, preencode, train, export' >&2; exit 2 ;;
esac
