"""Packed guider parity: branch/sample isolation, histories, losses and gradients."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from test_biopd import ROOT, _sdpa, tiny_config
from wan_utils.biopd_causal_adapter import history_kv, install_biopd_attention
from wan_utils.biopd_guider import GuiderBranches
from wan_utils.biopd_io import load_source
from wan_utils.scheduler import FlowMatchScheduler


class HistoryWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = nn.Parameter(torch.tensor(.1))
        self.model = SimpleNamespace(blocks=[None])
        self.calls = []
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

    def get_scheduler(self):
        return self.scheduler

    def forward(self, x, condition, timestep, kv_cache, prope_kv_cache, current_start, viewmats, Ks, **kwargs):
        if self.lora_weight.requires_grad:
            assert all(cache is None for cache in kwargs["crossattn_cache"])
        self.calls.append((x.shape[0], torch.is_grad_enabled(), kv_cache[0]["write"]))
        text = condition["prompt_embeds"].mean(dim=(1, 2))
        keys = (x.mean(dim=(2, 3, 4)) + text[:, None]).unsqueeze(-1).unsqueeze(-1)
        hk, hv = history_kv(kv_cache[0], keys, keys * .5, current_start, 20)
        pk, pv = history_kv(prope_kv_cache[0], keys + .1, keys * .3, current_start, 20)
        history = (hk + hv + pk + pv).mean(dim=(1, 2, 3))
        camera = viewmats.mean(dim=(1, 2, 3)) + Ks.mean(dim=(1, 2, 3))
        flow = x * self.lora_weight + (text + .05 * history + .01 * camera)[:, None, None, None, None]
        return flow, x - (timestep / 1000)[:, :, None, None, None] * flow


@pytest.mark.parametrize("batch_size", [1, 2])
def test_all_step_rollout_loss_gradient_and_call_count(batch_size):
    module = load_source("packed_guider_pipeline", ROOT / "Wan21/pipeline/biopd_training.py")
    noise = torch.randn(batch_size, 8, 1, 2, 2, generator=torch.Generator().manual_seed(4))
    conditions = [dict(prompt_embeds=(torch.arange(batch_size).float()[:, None, None] * .02 + i * .1)
                       .expand(-1, 2, 3)) for i in range(5)]
    viewmats = torch.eye(4).expand(batch_size, 8, 4, 4).clone()
    viewmats[:, :, 0, 3] = torch.arange(batch_size).float()[:, None]
    ks = torch.eye(3).expand(batch_size, 8, 3, 3).clone()
    results = []
    for pack, text_cache, optimizations in ((pack, text, enabled) for pack in (1, 2, 4)
                                             for text in (False, True) for enabled in (False, True)):
        student, guider = HistoryWrapper(), HistoryWrapper().requires_grad_(False)
        cfg = tiny_config()
        cfg.guider_branch_batch_size = pack
        cfg.guider_text_kv_cache = text_cache
        cfg.prope_transform_cache = optimizations
        cfg.guider_video_kv_buffer = optimizations
        pipeline = module.BiOPDTrainingPipeline(student, guider, cfg)
        pipeline.noise_generator = torch.Generator().manual_seed(7)
        output, metrics = pipeline.run(conditions, viewmats, ks, noise)
        results.append((output, metrics, student.lora_weight.grad.clone()))
        assert len(guider.calls) == (8 + 1) * (4 // pack)
        assert all(size == batch_size * pack and not grad for size, grad, _ in guider.calls)
        assert len(student.calls) == 8 + 1
        assert guider.lora_weight.grad is None
    for output, metrics, gradient in results[1:]:
        assert torch.allclose(output, results[0][0], atol=1e-6)
        assert torch.allclose(gradient, results[0][2], atol=1e-6)
        assert all(torch.allclose(value, results[0][1][name], atol=1e-6) for name, value in metrics.items())


@pytest.fixture
def camera_guider(monkeypatch):
    with patch("torch.cuda.current_device", return_value=0):
        import wan.modules.causal_model as cm
        import wan.modules.attention as am
        import wan.modules.model as wm
        from wan.modules.prope import add_prope_parameters
    monkeypatch.setattr(cm, "attention", _sdpa)
    monkeypatch.setattr(am, "attention", _sdpa)
    monkeypatch.setattr(wm, "flash_attention", _sdpa)
    torch.manual_seed(8)
    model = cm.CausalWanModel(dim=128, ffn_dim=256, freq_dim=16, text_dim=64,
                             text_len=8, num_heads=1, num_layers=2,
                             in_dim=4, out_dim=4, local_attn_size=20).eval()
    add_prope_parameters(model)
    nn.init.normal_(model.head.head.weight, std=.02)
    nn.init.normal_(model.blocks[0].self_attn.prope_o.weight, std=.02)
    install_biopd_attention(model)

    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.requires_grad_(False)

        def forward(self, x, condition, timestep, **kwargs):
            flow = self.model(x.permute(0, 2, 1, 3, 4), t=timestep,
                              context=condition["prompt_embeds"], seq_len=16, **kwargs)
            return flow.permute(0, 2, 1, 3, 4), x

    return Wrapper(model)


def test_real_wan_camera_branches_match_serial_with_clean_history(camera_guider):
    wrapper = camera_guider
    conditions = [dict(prompt_embeds=torch.randn(2, 5, 64)) for _ in range(4)]
    serial = GuiderBranches(wrapper, conditions, branches_per_call=1)
    packed = GuiderBranches(wrapper, conditions, branches_per_call=4)
    vm = torch.eye(4).expand(2, 4, 4, 4).clone()
    vm[1, :, 0, 3] = .2
    ks = torch.eye(3).expand(2, 4, 3, 3).clone()
    for chunk in range(2):
        common = dict(current_start=16 * chunk, viewmats=vm, Ks=ks)
        serial.set_chunk(common)
        packed.set_chunk(common)
        for time in (1000., 625.):
            x = torch.randn(2, 4, 4, 4, 4)
            timestep = torch.full((2, 4), time)
            expected, actual = serial.predict(x, timestep), packed.predict(x, timestep)
            assert all(torch.allclose(a, b, atol=3e-6, rtol=3e-5) for a, b in zip(expected, actual))
            assert all(not result.requires_grad for result in actual)
        clean = torch.randn(2, 4, 4, 4, 4)
        serial.commit(clean, torch.zeros(2, 4))
        packed.commit(clean, torch.zeros(2, 4))
        for kind in ("kv_cache", "prope_kv_cache"):
            for layer in range(2):
                for name in ("k", "v"):
                    expected = torch.cat([group["history"][kind][layer][name] for group in serial.groups])
                    actual = packed.groups[0]["history"][kind][layer][name]
                    assert torch.allclose(expected, actual, atol=3e-6, rtol=3e-5)


def test_packing_is_resume_compatible(tmp_path):
    from omegaconf import OmegaConf
    source = tmp_path / "weights.pt"
    source.write_bytes(b"identity")
    module = load_source("packed_resume_contract", ROOT / "Wan21/wan_trainer/camera_biopd.py")
    cfg = OmegaConf.create(dict(generator_ckpt=str(source), guider_ckpt=str(source), guider_adapter_ckpt=None))
    dataset = SimpleNamespace(sha256="same", split_info=None)
    expected = module.resume_contract(cfg, dataset, 4)
    for packing in (1, 2, 4):
        changed = copy.deepcopy(cfg)
        changed.guider_branch_batch_size = packing
        changed.guider_text_kv_cache = True
        changed.prope_transform_cache = True
        changed.guider_video_kv_buffer = True
        assert module.resume_contract(changed, dataset, 4) == expected


@pytest.mark.parametrize("packing", [1, 2, 4])
def test_frozen_text_kv_reuse_reset_and_real_wan_prediction_parity(camera_guider, packing):
    model = camera_guider.model
    conditions = [dict(prompt_embeds=torch.randn(2, 5, 64)) for _ in range(4)]
    noisy = [torch.randn(2, 4, 4, 4, 4) for _ in range(4)]
    clean = torch.randn(2, 4, 4, 4, 4)
    vm = torch.eye(4).expand(2, 4, 4, 4).clone()
    ks = torch.eye(3).expand(2, 4, 3, 3).clone()
    stats = {name: 0 for name in ("q", "k", "v")}
    hooks = []

    def count(name):
        def callback(*args):
            stats[name] += 1
        return callback

    for block in model.blocks:
        for name in stats:
            hooks.append(getattr(block.cross_attn, name).register_forward_hook(count(name)))
    outputs = []
    try:
        for enabled in (False, True):
            stats.update(q=0, k=0, v=0)
            predictor = GuiderBranches(camera_guider, conditions, branches_per_call=packing,
                                       cache_text_kv=enabled)
            result, addresses = [], None
            for chunk in range(2):
                predictor.set_chunk(dict(current_start=chunk * 16, viewmats=vm, Ks=ks))
                for step in range(2):
                    result += predictor.predict(noisy[chunk * 2 + step], torch.full((2, 4), 625.))
                    if enabled:
                        caches = [cache for group in predictor.groups for cache in group["history"]["crossattn_cache"]]
                        assert all(cache["is_init"] and not cache["k"].requires_grad and not cache["v"].requires_grad
                                   for cache in caches)
                        current = [(cache["k"].data_ptr(), cache["v"].data_ptr()) for cache in caches]
                        if addresses is None:
                            addresses = current
                        assert current == addresses  # Reuse survives both chunk and timestep changes.
                if chunk == 0:
                    predictor.commit(clean, torch.zeros(2, 4))
            multiplier = len(model.blocks) * (4 // packing)
            assert stats == dict(q=5 * multiplier,
                                 k=(1 if enabled else 5) * multiplier,
                                 v=(1 if enabled else 5) * multiplier)
            outputs.append(result)
        assert all(torch.allclose(a, b, atol=3e-6, rtol=3e-5) for a, b in zip(*outputs))

        # A new sample/rollout gets empty caches and computes the NEW text K/V.
        fresh_conditions = [dict(prompt_embeds=branch["prompt_embeds"] + .7) for branch in conditions]
        cached = GuiderBranches(camera_guider, fresh_conditions, branches_per_call=packing)
        assert all(not entry["is_init"] for group in cached.groups for entry in group["history"]["crossattn_cache"])
        reference = GuiderBranches(camera_guider, fresh_conditions, branches_per_call=packing, cache_text_kv=False)
        common = dict(current_start=0, viewmats=vm, Ks=ks)
        cached.set_chunk(common)
        reference.set_chunk(common)
        a = cached.predict(noisy[0], torch.full((2, 4), 625.))
        b = reference.predict(noisy[0], torch.full((2, 4), 625.))
        assert all(torch.allclose(x, y, atol=3e-6, rtol=3e-5) for x, y in zip(a, b))
    finally:
        for hook in hooks:
            hook.remove()


def test_text_kv_caching_rejects_trainable_guider():
    conditions = [dict(prompt_embeds=torch.zeros(1, 2, 3)) for _ in range(4)]
    with pytest.raises(ValueError, match="fully frozen"):
        GuiderBranches(HistoryWrapper(), conditions)
