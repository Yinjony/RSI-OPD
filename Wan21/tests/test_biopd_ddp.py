"""Two-process CPU/Gloo test of the real all-step pipeline's DDP accumulation."""

from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).parent))
from test_biopd import ROOT, TinyWrapper, tiny_config
from wan_utils.biopd_io import load_source


def _gradient_run(student, rank):
    module = load_source("ddp_biopd_pipeline", ROOT / "Wan21/pipeline/biopd_training.py")
    pipeline = module.BiOPDTrainingPipeline(student, TinyWrapper().requires_grad_(False), tiny_config())
    pipeline.noise_generator = torch.Generator().manual_seed(100 + rank)
    inputs = torch.Generator().manual_seed(200 + rank)
    conditions = [dict(prompt_embeds=torch.tensor(float(i))) for i in range(5)]
    for micro in range(2):
        noise = torch.randn(1, 8, 1, 2, 2, generator=inputs)
        pipeline.run(conditions, torch.zeros(1, 8, 4, 4), torch.zeros(1, 8, 3, 3), noise,
                     loss_scale=.5, sync_gradients=micro == 1)


def _worker(rank, rendezvous, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        raw = TinyWrapper()
        model = DistributedDataParallel(raw, broadcast_buffers=False)
        _gradient_run(model, rank)
        torch.save(raw.lora_weight.grad, Path(output_dir) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_accumulation_matches_global_reference(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    reference = []
    for rank in range(2):
        model = TinyWrapper()
        _gradient_run(model, rank)
        reference.append(model.lora_weight.grad)
    expected = torch.stack(reference).mean(0)
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True)
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
