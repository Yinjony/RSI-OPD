"""CPU objective/rollout/resume and two-rank DDP regression tests; no GPU training."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]

from wan_utils.biopd_io import load_source
from wan_utils.biopd_lora import inject_lora
from wan_utils.guider_sft_anchor import FourStepAnchor, backward_sft_with_anchor, load_frozen_reference
from wan_utils.guider_sft_config import OBJECTIVE_DEFAULTS, load_config
from wan_utils.guider_sft_training import backward_microbatch
from wan_utils.scheduler import FlowMatchScheduler


class TinyCausalWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        block = nn.Module()
        for name in ("q", "k", "v", "o", "prope_o"):
            setattr(block, name, nn.Linear(4, 4))
        self.model.blocks = nn.ModuleList([block])
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.trace = []
        self.gradient_checkpointing = False

    def get_scheduler(self):
        return self.scheduler

    def forward(self, noisy_image_or_video, conditional_dict, timestep, *, clean_x=None,
                viewmats=None, Ks=None, kv_cache=None, prope_kv_cache=None,
                crossattn_cache=None, current_start=None):
        x = noisy_image_or_video
        block = self.model.blocks[0]
        def velocity(value):
            return sum(getattr(block, n)(value) for n in ("q", "k", "v", "o", "prope_o")) / 5
        if self.gradient_checkpointing and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            flow = checkpoint(velocity, x, use_reentrant=False)
        else:
            flow = velocity(x)
        past = flow.new_zeros(())
        write = kv_cache is not None and kv_cache[0]["write"]
        if kv_cache is not None and kv_cache[0]["k"] is not None:
            past = kv_cache[0]["k"].mean()
        context = conditional_dict["prompt_embeds"].mean() + viewmats.mean() + Ks.mean()
        flow = flow + .01 * context + .1 * (clean_x.mean() if clean_x is not None else past)
        if kv_cache is not None:
            assert kv_cache[0]["end"] == current_start
            self.trace.append(dict(x=x.detach().clone(), t=timestep.detach().clone(),
                                   text=conditional_dict["prompt_embeds"].detach().clone(),
                                   vm=viewmats.detach().clone(), ks=Ks.detach().clone(),
                                   start=current_start, write=write, grad=torch.is_grad_enabled(),
                                   input_requires_grad=x.requires_grad, clean_x=clean_x))
        if write:
            assert not torch.is_grad_enabled()
            for caches in (kv_cache, prope_kv_cache):
                caches[0]["k"] = flow.detach().clone()
                caches[0]["v"] = flow.detach().clone()
                caches[0]["end"] = current_start + x.shape[1] * (x.shape[-2] // 2) * (x.shape[-1] // 2)
        ids = (self.scheduler.timesteps[:, None] - timestep.flatten()[None, :]).abs().argmin(0)
        sigma = self.scheduler.sigmas[ids].reshape(*timestep.shape, 1, 1, 1)
        return flow, x - sigma * flow


def models(perturb=True):
    torch.manual_seed(7)
    student = TinyCausalWrapper()
    reference = copy.deepcopy(student).requires_grad_(False).eval()
    inject_lora(student.model, 2, 2, ["q", "k", "v", "o", "prope_o"])
    if perturb:
        with torch.no_grad():
            for name, p in student.named_parameters():
                if "lora_B" in name:
                    p.fill_(.05)
    return student, reference


def config(states=1):
    return SimpleNamespace(num_frame_per_block=4, anchor_denoising_step_list=[1000, 750, 500, 250],
                           anchor_warp_denoising_step=True, anchor_states_per_batch=states,
                           anchor_cache_device="cpu")


def batch(seed=0):
    rng = torch.Generator().manual_seed(seed)
    return dict(clean_latent=torch.randn(1, 8, 1, 2, 4, generator=rng),
                embeddings=dict(full=torch.randn(1, 3, 4, generator=rng)),
                viewmats=torch.eye(4).expand(1, 8, 4, 4), Ks=torch.eye(3).expand(1, 8, 3, 3))


def grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def test_zero_anchor_at_training_start_and_reference_stopgrad():
    student, reference = models(perturb=False)
    anchor = FourStepAnchor(student, reference, config(states=8))
    loss = anchor.loss(batch(), torch.Generator().manual_seed(1), backward=True,
                       state_indices=list(range(8)))
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=0, rtol=0)
    assert all(p.grad is None for p in reference.parameters())
    assert all(p.grad is None for n, p in student.named_parameters() if "lora_" not in n)
    assert student.training  # temporarily eval only during rollout


def test_four_step_states_renoise_shared_inputs_and_same_clean_history():
    student, reference = models()
    anchor = FourStepAnchor(student, reference, config(states=8))
    loss = anchor.loss(batch(), torch.Generator().manual_seed(2), backward=True,
                       state_indices=list(range(8)))
    assert loss.item() > 0
    steps = [x for x in student.trace if not x["write"]]
    ref_steps = [x for x in reference.trace if not x["write"]]
    assert len(steps) == len(ref_steps) == 8
    assert [s["t"][0, 0].item() for s in steps] == pytest.approx(anchor.times * 2)
    for s, r in zip(steps, ref_steps):
        for key in ("x", "t", "text", "vm", "ks"):
            torch.testing.assert_close(s[key], r[key], atol=0, rtol=0)
        assert s["grad"] and not r["grad"]
        assert not s["input_requires_grad"] and s["clean_x"] is None
        assert s["start"] == r["start"]
    commits = [x for x in student.trace if x["write"]]
    ref_commits = [x for x in reference.trace if x["write"]]
    assert len(commits) == len(ref_commits) == 1
    torch.testing.assert_close(commits[0]["x"], ref_commits[0]["x"], atol=0, rtol=0)
    assert not commits[0]["grad"] and not ref_commits[0]["grad"]
    # Verify the first denoise-and-renoise transition against the scheduler.
    raw, _ = models()
    s = steps[0]
    flow, clean = raw(s["x"], dict(prompt_embeds=s["text"]), s["t"],
                      viewmats=s["vm"], Ks=s["ks"])
    rng = torch.Generator().manual_seed(2)
    torch.randn(batch()["clean_latent"].shape, generator=rng)  # initial noise
    eps = torch.randn(clean.shape, generator=rng)
    expected = raw.scheduler.add_noise(clean.flatten(0, 1), eps.flatten(0, 1),
                                       torch.full((4,), anchor.times[1])).unflatten(0, clean.shape[:2])
    torch.testing.assert_close(steps[1]["x"], expected, atol=0, rtol=0)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    assert all(p.grad is None for p in reference.parameters())


def test_sampled_state_expectation_and_gradient_equal_full_state_mean():
    full, reference = models()
    anchor = FourStepAnchor(full, reference, config(8))
    full_loss = anchor.loss(batch(), torch.Generator().manual_seed(4), backward=True,
                            state_indices=list(range(8)))
    expected_grads = grads(full)
    sampled, reference = models()
    sampled_anchor = FourStepAnchor(sampled, reference, config())
    terms = [sampled_anchor.loss(batch(), torch.Generator().manual_seed(4), backward=True,
                                loss_scale=1/8, state_indices=[i]) for i in range(8)]
    torch.testing.assert_close(torch.stack(terms).mean(0), full_loss, atol=1e-7, rtol=1e-6)
    for n, g in grads(sampled).items():
        torch.testing.assert_close(g, expected_grads[n], atol=1e-7, rtol=1e-6)


def test_eta_zero_matches_original_supervised_loss_gradient_and_rng():
    student, _ = models()
    original = copy.deepcopy(student)
    rng1, rng2 = [torch.Generator().manual_seed(5) for _ in range(2)]
    old = backward_microbatch(original, batch(), original.scheduler, rng1,
                             accumulation=2, synchronize=True)
    new = backward_sft_with_anchor(student, batch(), student.scheduler, rng2, None, None,
                                   eta=0, accumulation=2, synchronize=True)
    torch.testing.assert_close(new["loss"], old, atol=0, rtol=0)
    assert torch.equal(rng1.get_state(), rng2.get_state())
    for n, g in grads(student).items():
        torch.testing.assert_close(g, grads(original)[n], atol=0, rtol=0)


def test_combined_loss_and_gradient_is_supervised_plus_eta_anchor():
    eta = .3
    student, reference = models()
    result = backward_sft_with_anchor(student, batch(), student.scheduler, torch.Generator().manual_seed(6),
                                      FourStepAnchor(student, reference, config()),
                                      torch.Generator().manual_seed(7), eta=eta,
                                      accumulation=1, synchronize=True)
    torch.testing.assert_close(result["loss"], result["supervised_loss"] + eta * result["anchor_loss"])
    expected, reference = models()
    backward_microbatch(expected, batch(), expected.scheduler, torch.Generator().manual_seed(6),
                        accumulation=1, synchronize=True)
    FourStepAnchor(expected, reference, config()).loss(batch(), torch.Generator().manual_seed(7),
                                                       backward=True, loss_scale=eta)
    for n, g in grads(student).items():
        torch.testing.assert_close(g, grads(expected)[n], atol=0, rtol=0)


def test_anchor_rng_state_roundtrip_keeps_state_and_loss():
    student, reference = models()
    anchor = FourStepAnchor(student, reference, config())
    rng = torch.Generator().manual_seed(8)
    anchor.loss(batch(), rng)
    saved = rng.get_state().clone()
    expected = anchor.loss(batch(), rng)
    other_rng = torch.Generator().set_state(saved)
    actual = anchor.loss(batch(), other_rng)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(rng.get_state(), other_rng.get_state())


def test_bf16_autocast_checkpoint_recompute_preserves_combined_gradient():
    results = []
    for checkpointing in (False, True):
        student, reference = models()
        student.gradient_checkpointing = checkpointing
        losses = backward_sft_with_anchor(student, batch(), student.scheduler,
                                          torch.Generator().manual_seed(16),
                                          FourStepAnchor(student, reference, config(states=8)),
                                          torch.Generator().manual_seed(17), eta=.1,
                                          accumulation=1, synchronize=True, amp=True)
        results.append((losses, grads(student)))
        assert all(p.grad is None for p in reference.parameters())
        assert all(torch.isfinite(g).all() for g in results[-1][1].values())
    for key, value in results[0][0].items():
        torch.testing.assert_close(value, results[1][0][key], atol=0, rtol=0)
    for n, value in results[0][1].items():
        torch.testing.assert_close(value, results[1][1][n], atol=0, rtol=0)


def test_anchor_switch_keeps_initialization_data_and_cache_settings():
    for name in ("guider1", "guider2"):
        path = ROOT / f"Wan21/configs/{name}.yaml"
        active = load_config(path, ["anchor=true"])
        original = load_config(path, ["anchor=false"])
        assert active.sft_loss_type == "supervised_anchor" and active.anchor_eta == .1
        assert original.sft_loss_type == "supervised"
        assert active.generator_ckpt == original.generator_ckpt
        assert active.get("initial_adapter") == original.get("initial_adapter")
        assert active.cache_dir == original.cache_dir
        assert active.output_dir == original.output_dir
        assert "anchor" not in active and "relora" not in active
    with pytest.raises(ValueError, match="anchor_eta"):
        load_config(ROOT / "Wan21/configs/guider2.yaml", ["anchor_eta=-1"])
    with pytest.raises(ValueError, match="strictly descending"):
        load_config(ROOT / "Wan21/configs/guider2.yaml", ["anchor_denoising_step_list=[500,1000]"])
    with pytest.raises(ValueError, match="anchor_states_per_batch"):
        load_config(ROOT / "Wan21/configs/guider2.yaml", ["anchor_states_per_batch=101"])


def test_reference_loader_uses_start_checkpoint_without_lora():
    template = TinyCausalWrapper()
    cfg = SimpleNamespace(model_path="model", generator_ckpt="training_start/model.pt",
                          timestep_shift=5, num_frame_per_block=4)
    fake_module = SimpleNamespace(WanDiffusionWrapper=lambda **kwargs: copy.deepcopy(template))
    with patch.dict(sys.modules, {"wan_utils.wan_wrapper": fake_module}), \
         patch("wan_utils.biopd_io.load_generator_state", return_value=template.state_dict()) as loader, \
         patch("wan_utils.guider_sft_anchor.install_biopd_attention"):
        reference = load_frozen_reference(cfg, "cpu")
    loader.assert_called_once_with("training_start/model.pt", "generator")
    assert not reference.training and not any(p.requires_grad for p in reference.parameters())
    assert not any("lora_" in n for n, _ in reference.named_parameters())


def test_original_resume_contract_compatible_and_anchor_contract_distinct():
    trainer = load_source("test_anchor_resume_contract", ROOT / "Wan21/wan_trainer/camera_guider_sft.py")
    cfg = load_config(ROOT / "Wan21/configs/guider2.yaml", ["anchor=false"])
    dataset = SimpleNamespace(sha256="cachehash")
    with patch.object(trainer, "file_identity", return_value={"path": "base"}):
        legacy = trainer.resume_contract(cfg, dataset, 4)
        assert not set(OBJECTIVE_DEFAULTS) & set(legacy["config"])
        cfg.sft_loss_type = "supervised_anchor"
        anchored = trainer.resume_contract(cfg, dataset, 4)
        assert anchored != legacy
        assert anchored["config"]["anchor_eta"] == .1


def _gradient_run(student, reference, rank):
    raw = getattr(student, "module", student)
    anchor = FourStepAnchor(student, reference, config(states=2))
    supervised_rng = torch.Generator().manual_seed(100 + rank)
    anchor_rng = torch.Generator().manual_seed(200 + rank)
    for micro in range(2):
        backward_sft_with_anchor(student, batch(rank * 10 + micro), raw.scheduler, supervised_rng,
                                 anchor, anchor_rng, eta=.3, accumulation=2, synchronize=micro == 1)


def _ddp_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        raw, reference = models()
        student = DDP(raw, broadcast_buffers=False)
        _gradient_run(student, reference, rank)
        torch.save(grads(raw), Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_combined_supervised_anchor_accumulation_matches_global_mean(tmp_path):
    mp.spawn(_ddp_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    references = []
    for rank in range(2):
        student, reference = models()
        _gradient_run(student, reference, rank)
        references.append(grads(student))
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        for n, g in actual.items():
            expected = (references[0][n] + references[1][n]) / 2
            torch.testing.assert_close(g, expected, atol=2e-6, rtol=2e-5)
