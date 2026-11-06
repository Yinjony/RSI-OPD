"""Target entry wrapper: keep Hopper-only FA3 off Blackwell, then run a script."""
import importlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'Wan21'), str(ROOT), str(ROOT / 'shared')]
import torch

attention = importlib.import_module('wan.modules.attention')
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
    attention.FLASH_ATTN_3_AVAILABLE = False

if len(sys.argv) < 2:
    raise SystemExit('Usage: runtime.py TARGET_SCRIPT [arguments]')
script = sys.argv.pop(1)
sys.argv[0] = script
runpy.run_path(script, run_name='__main__')
