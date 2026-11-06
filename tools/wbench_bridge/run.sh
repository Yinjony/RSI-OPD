#!/usr/bin/env bash
set -euo pipefail

BRIDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINWM_ROOT="$(cd "${BRIDGE_DIR}/../.." && pwd)"
WBENCH_DATA_DIR="${WBENCH_DATA_DIR:-/root/data/wbench_handoff/WBench/data}"
WBENCH_WORK_DIR="${WBENCH_WORK_DIR:-/root/data/wbench_handoff/WBench/work_dirs/minwm}"
PYTHON_BIN="${PYTHON_BIN:-/root/data/huangrui/conda_envs/wanmodel/bin/python}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${MINWM_ROOT}/Wan21:${MINWM_ROOT}/shared:${BRIDGE_DIR}:${PYTHONPATH:-}"

MANIFEST="${MANIFEST:-${BRIDGE_DIR}/manifests/wbench_nav_minwm.jsonl}"
CHECKPOINT="${CHECKPOINT:-${MINWM_ROOT}/ckpts/Wan21/Action2V/dmd/model.pt}"
CONFIG="${CONFIG:-${MINWM_ROOT}/Wan21/configs/causal_forcing_dmd_camera.yaml}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

CASE_ARGS=()
if [[ -n "${CASE_ID:-}" ]]; then
  CASE_ARGS=(--case-id "${CASE_ID}")
fi

echo "=== minWM <-> WBench bridge ==="
echo "  python:      ${PYTHON_BIN}"
echo "  wbench data: ${WBENCH_DATA_DIR}"
echo "  work dir:    ${WBENCH_WORK_DIR}"
echo "  checkpoint:  ${CHECKPOINT}"

cd "${MINWM_ROOT}"

echo "--- Step 1: convert WBench navigation cases -> manifest ---"
"${PYTHON_BIN}" "${BRIDGE_DIR}/convert_cases.py" \
  --wbench-data-dir "${WBENCH_DATA_DIR}" \
  --out "${MANIFEST}"

echo "--- Step 2: generate case_{id}_combined.mp4 ---"
"${PYTHON_BIN}" "${BRIDGE_DIR}/infer_wbench.py" \
  --manifest "${MANIFEST}" \
  --project-root "${MINWM_ROOT}" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-root "${WBENCH_WORK_DIR}" \
  "${LIMIT_ARGS[@]}" \
  "${CASE_ARGS[@]}" \
  "$@"
