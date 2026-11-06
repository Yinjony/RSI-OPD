"""Small target CUDA/attention check. This does not load Wan model weights."""
import importlib
import json
from pathlib import Path
import sys
sys.path[:0] = [str(Path.cwd() / 'Wan21'), str(Path.cwd()), str(Path.cwd() / 'shared')]
import torch
import torchvision
import cv2
from torchvision.io import write_video

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError('Expose at least GPUs 0,1 for the planned two-GPU experiment')
result = dict(torch=torch.__version__, CUDA=torch.version.cuda,
              devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
for i in (0, 1):
    with torch.cuda.device(i):
        x = torch.randn(128, 128, device=f'cuda:{i}', dtype=torch.bfloat16, requires_grad=True)
        (x @ x).float().square().mean().backward()
        torch.cuda.synchronize(i)
attention = importlib.import_module('wan.modules.attention')
if torch.cuda.get_device_capability()[0] >= 10:
    attention.FLASH_ATTN_3_AVAILABLE = False
q = torch.randn(1, 64, 12, 128, device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
y = attention.attention(q, q, q)
y.float().square().mean().backward()
torch.cuda.synchronize()
if not torch.isfinite(y).all() or not torch.isfinite(q.grad).all():
    raise RuntimeError('Nonfinite attention probe')
result['attention'] = attention.attention_backend_info()
result['small_cuda_forward_backward'] = 'passed'
Path('deployment/runtime_check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
