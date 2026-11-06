"""Two-rank CUDA communication probe; no model or training data is loaded."""
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=60))
try:
    value = torch.tensor([rank + 1.0], device=f'cuda:{rank}')
    dist.all_reduce(value)
    torch.cuda.synchronize(rank)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    assert value.item() == expected
    if rank == 0:
        result = dict(backend='nccl', world_size=dist.get_world_size(),
                      all_reduce_sum=value.item(), result='passed')
        Path('deployment/ddp_check.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
finally:
    dist.destroy_process_group()
