#!/usr/bin/env bash
# No credentials are stored here. SSH asks for the password once.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --background ]]; then
  exec /root/miniconda3/envs/timeopd/bin/python "$HERE/background_transfer.py"
fi
exec 8>"$HERE/transfer.lock"
flock -n 8 || { echo 'A source transfer is already running' >&2; exit 1; }
SOURCE_ROOT="$(cd -- "$HERE/../.." && pwd)"
DESTINATION=root@connect.westb.seetacloud.com
TARGET_ROOT=/root/autodl-tmp/ltl/minWM
SOCKET="${TMPDIR:-/tmp}/minwm_transfer_${BASHPID}.sock"
SSH=(ssh -p 53280 -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HERE/known_hosts" -S "$SOCKET")
test -f "$HERE/source_manifest.json"
"${SSH[@]}" -M -o ControlPersist=600 -fN "$DESTINATION"
trap '"${SSH[@]}" -O exit "$DESTINATION" >/dev/null 2>&1 || true' EXIT
"${SSH[@]}" "$DESTINATION" "mkdir -p '$TARGET_ROOT/deployment/environment' '$TARGET_ROOT/dataset/guider2_replay/videos' '$TARGET_ROOT/ckpts/Wan21/Action2V/dmd' '$TARGET_ROOT/Wan21/wan_models/Wan2.1-T2V-1.3B'; df -h /root/autodl-tmp; nvidia-smi --query-gpu=index,name,memory.used --format=csv"
RSYNC=(rsync -az --compress-choice=zstd --compress-level=3 --partial --info=progress2 -e "ssh -p 53280 -S $SOCKET -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$HERE/known_hosts")
"${RSYNC[@]}" --exclude environment --exclude 'data/videos' --exclude known_hosts --exclude 'transfer.*' --exclude '*.lock' "$HERE/" "$DESTINATION:$TARGET_ROOT/deployment/"
"${RSYNC[@]}" "$HERE/environment/timeopd.tar.gz" "$DESTINATION:$TARGET_ROOT/deployment/environment/"
"${RSYNC[@]}" --copy-links "$HERE/data/videos/" "$DESTINATION:$TARGET_ROOT/dataset/guider2_replay/videos/"
"${RSYNC[@]}" /root/data/mcy/timeOPD/minWM/ckpts/Wan21/Action2V/dmd/model.pt "$DESTINATION:$TARGET_ROOT/ckpts/Wan21/Action2V/dmd/"
"${RSYNC[@]}" --copy-links --files-from="$HERE/model_files.txt" /root/data/mcy/timeOPD/minWM/Wan21/wan_models/Wan2.1-T2V-1.3B/ "$DESTINATION:$TARGET_ROOT/Wan21/wan_models/Wan2.1-T2V-1.3B/"
"${SSH[@]}" "$DESTINATION" "bash '$TARGET_ROOT/deployment/bootstrap_target.sh'"
touch "$HERE/transfer.complete"
echo "TRANSFER AND PREPARATION COMPLETE: $TARGET_ROOT"
echo "On target: cd $TARGET_ROOT && nohup bash deployment/run_on_target.sh all > guider2_replay.console.log 2>&1 &"
