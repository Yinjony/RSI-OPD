"""CPU validation checks: schedule, RNG isolation, frozen rollout and DDP means."""

import json
from pathlib import Path
import random
import sys

from omegaconf import OmegaConf
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).parent))
from test_biopd import ROOT, TinyWrapper, tiny_config
from wan_utils.biopd_dataset import collate_conditions
from wan_utils.biopd_io import load_source
from wan_utils.biopd_validation import (sample_noise_seed, validation_indices, validation_kind,
                                       validation_metric_names, validation_report)


class ValidationItems(list):
    def report(self):
        return dict(usable=len(self), split=dict(selected="val"))


def validation_items(count=5):
    return ValidationItems(dict(sample_id=f"held_out_{i}", texts=[str(i + branch) for branch in range(5)],
                                viewmats=torch.zeros(8, 4, 4), Ks=torch.zeros(8, 3, 3))
                           for i in range(count))


class DummyText:
    def encode(self, branches):
        # Simulate incidental global RNG use; evaluation must still preserve the
        # training RNG state, not just use a different rollout generator.
        torch.rand(1)
        random.random()
        return [dict(prompt_embeds=torch.tensor(float(branch[0]))) for branch in branches]


def make_validation_trainer(directory, *, rank=0, world_size=1, count=5, ddp=False):
    module = load_source("cpu_validation_trainer", ROOT / "Wan21/wan_trainer/camera_biopd.py")
    pipeline_module = load_source("cpu_validation_pipeline", ROOT / "Wan21/pipeline/biopd_training.py")
    trainer = module.Trainer.__new__(module.Trainer)
    trainer.config = OmegaConf.create(vars(tiny_config()))
    trainer.config.update(dict(num_training_frames=8, max_steps=475, val_every=100,
                               val_num_samples=2, val_seed=12345, val_at_start=True, val_full_at_end=True))
    trainer.raw_student = TinyWrapper().eval()
    trainer.guider = TinyWrapper().requires_grad_(False).eval()
    trainer.student = (DistributedDataParallel(trainer.raw_student, broadcast_buffers=False)
                       if ddp else trainer.raw_student)
    trainer.device = torch.device("cpu")
    trainer.rank, trainer.world_size, trainer.step = rank, world_size, 100
    trainer.noise_generator = torch.Generator().manual_seed(9)
    trainer.pipeline = pipeline_module.BiOPDTrainingPipeline(trainer.student, trainer.guider, trainer.config)
    trainer.pipeline.noise_generator = trainer.noise_generator
    trainer.text = DummyText()
    trainer.validation_dataset = validation_items(count)
    trainer.output_dir = Path(directory)
    return trainer


def test_schedule_and_stable_held_out_selection():
    cfg = OmegaConf.create(dict(max_steps=475, val_every=100, val_num_samples=20, val_seed=12345))
    dataset = validation_items(100)
    report = validation_report(dataset, cfg)
    assert [(r["step"], r["kind"], r["num_samples"]) for r in report["schedule"]] == [
        (0, "periodic", 20), (100, "periodic", 20), (200, "periodic", 20),
        (300, "periodic", 20), (400, "periodic", 20), (475, "full", 100)]
    selected = report["periodic_sample_ids"]
    reversed_data = ValidationItems(reversed(dataset))
    assert selected == [reversed_data[i]["sample_id"] for i in validation_indices(reversed_data, cfg)]
    assert all(sample_noise_seed(sample_id, 12345) == report["sample_noise_seeds"][sample_id]
               for sample_id in selected)
    cfg.max_steps = 500
    assert validation_kind(500, cfg) == "full"  # no second periodic event at the final step
    cfg.val_at_start = False
    assert validation_kind(0, cfg) is None
    cfg.val_every = 0
    assert all(validation_kind(step, cfg) is None for step in [0, 100, 500])
    assert validation_report(dataset, cfg)["schedule"] == []


def test_no_backward_rollout_bypasses_ddp_and_preserves_training_generator():
    module = load_source("eval_biopd_pipeline", ROOT / "Wan21/pipeline/biopd_training.py")

    class DDPGuard(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = TinyWrapper()

        def forward(self, *args, **kwargs):
            raise AssertionError("Evaluation must bypass DDP.forward")

        def no_sync(self):
            raise AssertionError("Evaluation must not touch DDP reducer state")

    student = DDPGuard()
    guider = TinyWrapper().requires_grad_(False)
    pipeline = module.BiOPDTrainingPipeline(student, guider, tiny_config())
    pipeline.noise_generator = torch.Generator().manual_seed(7)
    state = pipeline.noise_generator.get_state().clone()
    student.module.lora_weight.grad = torch.tensor(3.)
    conditions = [dict(prompt_embeds=torch.tensor(float(i))) for i in range(5)]
    noise = torch.ones(1, 8, 1, 2, 2)
    output, values = pipeline.run(conditions, torch.zeros(1, 8, 4, 4), torch.zeros(1, 8, 3, 3), noise,
                                  backward=False, noise_generator=torch.Generator().manual_seed(88))
    assert not output.requires_grad and all(not value.requires_grad for value in values.values())
    assert student.module.lora_weight.grad == 3 and guider.lora_weight.grad is None
    assert torch.equal(state, pipeline.noise_generator.get_state())
    assert values["prediction_steps"] == 8


def test_validation_repeatability_rng_gradients_and_full_subset(tmp_path):
    trainer = make_validation_trainer(tmp_path)
    trainer.raw_student.lora_weight.grad = torch.tensor(2.)
    training_rng = trainer.noise_generator.get_state().clone()
    global_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    first = trainer._validate()
    second = trainer._validate()
    assert first["num_samples"] == 2 and not first["full"]
    names = validation_metric_names(4)
    assert all(first[name] == second[name] for name in names)
    trainer.step = 475
    full = trainer._validate(full=True)
    assert full["num_samples"] == 5 and full["periodic_subset"]["num_samples"] == 2
    assert all(full["periodic_subset"][name] == first[name] for name in names)
    assert torch.equal(training_rng, trainer.noise_generator.get_state())
    assert torch.equal(global_rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert trainer.raw_student.lora_weight.grad == 2
    assert trainer.guider.lora_weight.grad is None
    rows = [json.loads(line) for line in (tmp_path / "validation_metrics.jsonl").read_text().splitlines()]
    assert [(r["step"], r["num_samples"]) for r in rows] == [(100, 2), (100, 2), (475, 5)]


@pytest.mark.parametrize("maximum", [3, 4])
def test_train_loop_triggers_start_periodic_final_validation_and_saves(tmp_path, monkeypatch, capsys, maximum):
    trainer = make_validation_trainer(tmp_path)
    trainer.config.update(dict(max_steps=maximum, val_every=2, save_every=2,
                               grad_accum_steps=1, log_every=1, max_grad_norm=1.))
    trainer.step, trainer.epoch = 0, 0
    trainer.trainable = list(trainer.raw_student.parameters())
    trainer.optimizer = torch.optim.SGD(trainer.trainable, lr=.01)
    batch = collate_conditions([trainer.validation_dataset[0]])
    trainer._batches = lambda: iter([batch] * maximum)
    validations, saves = [], []
    trainer._validate = lambda *, full=False: validations.append((trainer.step, full))
    trainer._save = lambda: saves.append(trainer.step)

    class TrainStep:
        def run(self, conditions, viewmats, ks, noise, **kwargs):
            loss = trainer.raw_student.lora_weight.square()
            loss.backward()
            return noise.detach(), dict(loss=loss.detach())

    trainer.pipeline = TrainStep()
    original_autocast = torch.autocast
    # Only redirect this tiny trainer test's CUDA autocast context to CPU.
    monkeypatch.setattr(torch, "autocast", lambda device_type, **kwargs: original_autocast("cpu", **kwargs))
    trainer.train()
    assert trainer.step == maximum
    assert validations == [(0, False), (2, False), (maximum, True)]
    assert saves == [2, maximum]
    assert trainer.raw_student.lora_weight.item() < .1
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert all("eta_seconds" in row and "estimated_total_seconds" in row for row in metrics)
    assert metrics[-1]["progress_percent"] == 100
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()
                if line.startswith('{"event": "progress"')]
    assert progress[-1]["phase"] == "完成" and progress[-1]["eta_seconds"] == 0
    assert progress[-1]["remaining_validation_runs"] == progress[-1]["remaining_checkpoints"] == 0


def _validation_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        trainer = make_validation_trainer(directory, rank=rank, world_size=2, count=3, ddp=True)
        row = trainer._validate(full=True)  # uneven work: rank 0 gets 2, rank 1 gets 1
        assert trainer.raw_student.lora_weight.grad is None
        torch.save(row, Path(directory) / f"val_rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_validation_matches_unpadded_single_rank_mean(tmp_path):
    mp.spawn(_validation_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    reference = make_validation_trainer(tmp_path, count=3)._validate(full=True)
    for rank in range(2):
        actual = torch.load(tmp_path / f"val_rank_{rank}.pt", weights_only=True)
        assert actual["num_samples"] == reference["num_samples"] == 3
        assert actual["periodic_subset"]["num_samples"] == 2
        for name in validation_metric_names(4):
            assert actual[name] == pytest.approx(reference[name], abs=1e-8)
            assert actual["periodic_subset"][name] == pytest.approx(reference["periodic_subset"][name], abs=1e-8)
