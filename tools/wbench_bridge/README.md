# minWM ↔ WBench Bridge

Standalone adapter layer. **Does not modify** existing minWM or WBench code.

## What it does

1. Read WBench navigation cases (`cases/` + `images/`)
2. Convert each case to minWM inputs:
   - `prompt` = environment + character + perspective
   - per-turn `trajectory` (`w/s/a/d/j/l/i/k`, 20 latent frames)
   - `image_path` = WBench `initial_image`
3. Run minWM DMD inference per turn (I2V), concatenate clips
4. Write WBench-ready outputs:
   - `work_dirs/minwm/videos/case_{id}_combined.mp4`
   - `work_dirs/minwm/turns.json`

## Quick start

```bash
export PYTHON_BIN=/root/data/huangrui/conda_envs/wanmodel/bin/python
export GPU_ID=0

# Smoke: one case
CASE_ID=11 LIMIT=0 bash /root/data/mcy/timeOPD/minWM/tools/wbench_bridge/run.sh --dry-run

# Convert only
/root/data/huangrui/conda_envs/wanmodel/bin/python \
  /root/data/mcy/timeOPD/minWM/tools/wbench_bridge/convert_cases.py \
  --case /root/data/wbench_handoff/WBench/data/cases/case_11.json

# Full pipeline (158 nav cases)
bash /root/data/mcy/timeOPD/minWM/tools/wbench_bridge/run.sh

# Scheme B: initial image + one merged 20-frame trajectory (no multi-turn stitch)
/root/data/huangrui/conda_envs/wanmodel/bin/python \
  tools/wbench_bridge/infer_wbench.py \
  --merged-navigation --limit 5 --overwrite
```

## WBench evaluation

```bash
cd /root/data/wbench_handoff/WBench
conda activate wbench-main
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python main.py --model minwm --phase precompute --gpus 0,1,2,3
python main.py --model minwm --phase gpu --gpus 0,1,2,3
python main.py --model minwm --phase vlm --vlm_workers 1
python main.py --model minwm --phase report
```

## Conversion rules

| WBench | minWM |
|--------|-------|
| `W/S/A/D` | `w/s/a/d` |
| `←/→/↑/↓` or left/right/up/down | `j/l/i/k` |
| multi-turn `interactions` | one trajectory per turn |
| `initial_image` | first-turn I2V conditioning |
| previous turn last frame | next-turn I2V conditioning |

Only cases with at least one navigation turn are included (WBench nav subset).
