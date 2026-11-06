"""CPU checks: math, conditions, cache semantics, LoRA export, and full rollout.

Run with the project's timeopd Python environment; no GPU/checkpoint load needed.
"""
# ruff: noqa: E402

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]

from shared.algorithms.biopd import biopd_loss, guider_targets, resolve_schedule
from wan_utils.biopd_causal_adapter import history_kv, install_biopd_attention, new_history, set_write
from wan_utils.biopd_dataset import BiOPDDataset, collate_conditions, json_digest
from wan_utils.biopd_io import adapter_state, load_source, merged_generator_state, restore_adapter
from wan_utils.scheduler import FlowMatchScheduler


def test_runtime_log_preserves_failure_and_restores_streams(tmp_path):
    from wan_utils.biopd_io import runtime_log
    original = sys.stdout
    with pytest.raises(RuntimeError, match="test failure"):
        with runtime_log(tmp_path, 1):
            print("loading guider")
            raise RuntimeError("test failure")
    assert sys.stdout is original
    text = (tmp_path / "runtime_rank1.log").read_text()
    assert "loading guider" in text and "RuntimeError: test failure" in text


def test_loss_exact_gradient_and_teacher_stopgrad():
    torch.manual_seed(3)
    v, p, n = [torch.randn(2, 4, 3, requires_grad=True) for _ in range(3)]
    coefficient = 0.7 * 0.8**2 * (1 - 0.3)**2 / 2
    loss, metrics = biopd_loss(v, p, n, sigma=.8, next_sigma=.3, weight=.7, lam=.25)
    expected = coefficient * ((v - p.detach()).square().mean() - .25 * (v - n.detach()).square().mean())
    assert torch.allclose(torch.autograd.grad(loss, v, retain_graph=True)[0],
                          torch.autograd.grad(expected, v)[0], atol=1e-7)
    loss.backward()
    assert p.grad is None and n.grad is None
    assert torch.allclose(expected.detach(), metrics["signed_loss"])
    constant = -coefficient * .25 / .75 * (p - n).square().mean()
    assert torch.allclose(metrics["signed_loss"], loss.detach() + constant.detach(), atol=1e-7)


@pytest.mark.parametrize("lam", [-.1, 1., 1.1, float("nan")])
def test_invalid_lambda(lam):
    with pytest.raises(ValueError):
        biopd_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2),
                   sigma=1, next_sigma=0, weight=1, lam=lam)


def test_terminal_and_positive_only_limit():
    v = torch.randn(2, requires_grad=True)
    p, n = torch.randn(2), torch.randn(2)
    loss, _ = biopd_loss(v, p, n, sigma=.625, next_sigma=0, weight=1, lam=0)
    assert torch.allclose(loss, .625**2 / 2 * (v - p).square().mean())
    assert torch.isfinite(torch.autograd.grad(loss, v)[0]).all()


def test_exact_four_branch_formula():
    yp, n, yn, p = [torch.tensor(float(i)) for i in (1, 3, 5, 7)]
    positive, negative = guider_targets([yp, n, yn, p], 2, 3)
    assert positive == -3 and negative == -1


def test_schedule_matches_deployed_warp_and_renoise():
    scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    times, sigmas = resolve_schedule(scheduler, [1000, 750, 500, 250])
    assert sigmas == pytest.approx([1, .9375, 5/6, .625, 0])
    expected = torch.cat([scheduler.timesteps, torch.zeros(1)])[1000 - torch.tensor([1000,750,500,250])]
    assert torch.equal(torch.tensor(times), expected)
    x0, eps = torch.randn(3, 2, 4, 4), torch.randn(3, 2, 4, 4)
    for index in range(1, len(times)):
        actual = scheduler.add_noise(x0, eps, torch.full((3,), times[index]))
        assert torch.allclose(actual, (1 - sigmas[index]) * x0 + sigmas[index] * eps)


def test_dataset_empty_negative_and_branch_contents(tmp_path):
    record = dict(sample_id="a", prompt="scene", action="j*8,a*11",
                  positive_attributes=[dict(name="clear", attribute_id=1)],
                  negative_attributes=[dict(name="blur", attribute_id=2)])
    path = tmp_path / "data.json"
    path.write_text(json.dumps([record, dict(record, sample_id="b", negative_attributes=[])]))
    dataset = BiOPDDataset(path)
    assert len(dataset) == 2 and dataset.skipped_ids == []
    item = dataset[0]
    assert item["texts"] == ["scene", "scene\nAttributes: clear", "Attributes: blur",
                             "scene\nAttributes: blur", "Attributes: clear"]
    batch = collate_conditions([item, item])
    assert batch["viewmats"].shape == (2, 20, 4, 4)
    assert batch["Ks"].shape == (2, 20, 3, 3)
    assert dataset[1]["texts"] == ["scene", "scene\nAttributes: clear", "",
                                    "scene", "Attributes: clear"]
    assert dataset.report()["skipped"] == 0 and dataset.report()["empty_negative"] == 1
    with pytest.raises(ValueError, match="missing_attributes=keep"):
        BiOPDDataset(path, missing_attributes="skip")


def test_guider_sft_conditions_match_training_template(tmp_path):
    record = dict(sample_id="sft", prompt="scene", action="w*19",
                  positive_attributes=[dict(name="clear  edges"), dict(name="clear edges")],
                  negative_attributes=[dict(name="blur")])
    path = tmp_path / "sft.json"
    path.write_text(json.dumps([record, dict(record, sample_id="empty", negative_attributes=None),
                                dict(sample_id="neither", prompt="scene", action="w*19")]))
    dataset = BiOPDDataset(path, condition_format="guider_sft")
    pos = "Observed positive video attributes:\nclear edges."
    neg = "Observed negative video attributes:\nblur."
    assert dataset[0]["texts"] == ["scene", "scene\n\n" + pos, neg, "scene\n\n" + neg, pos]
    assert dataset[1]["texts"] == ["scene", "scene\n\n" + pos, "", "scene", pos]
    assert dataset[2]["texts"] == ["scene", "scene", "", "scene", ""]
    assert dataset.report()["usable"] == 3 and dataset.report()["skipped"] == 0
    with pytest.raises(ValueError, match="condition_format"):
        BiOPDDataset(path, condition_format="unknown")


def split_fixture(tmp_path):
    originals = [dict(sample_id=sample_id, prompt="scene", action="w*19", video_path=video,
                      positive_attributes=[dict(name="old")], negative_attributes=[], split=split)
                 for sample_id, video, split in [("a", "train.mp4", "train"),
                                                  ("b", "train.mp4", "train"),
                                                  ("c", "val.mp4", "val")]]
    contract = dict(val_fraction=.05, split_seed=42)
    manifest = dict(format="wan_guider_sft_cache_v1", contract=contract,
                    signature=json_digest(contract), records=originals)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(manifest))
    # Changed attributes and reordered source must NOT change split membership.
    source = [dict(record, positive_attributes=[dict(name="selected good")],
                   negative_attributes=[dict(name="selected bad")]) for record in reversed(originals)]
    for record in source:
        record.pop("split")
    path = tmp_path / "selector.json"
    path.write_text(json.dumps(source))
    return path, reference, source, manifest


def test_manifest_split_reuses_ids_and_selector_attributes_without_leakage(tmp_path):
    path, reference, source, _ = split_fixture(tmp_path)
    train = BiOPDDataset(path, split_manifest=reference, split="train", condition_format="guider_sft")
    val = BiOPDDataset(path, split_manifest=reference, split="val", condition_format="guider_sft")
    assert [r["sample_id"] for r in train.records] == ["b", "a"]
    assert [r["sample_id"] for r in val.records] == ["c"]
    assert {r["video_path"] for r in train.records}.isdisjoint(r["video_path"] for r in val.records)
    assert train.report()["total"] == 3 and train.report()["usable"] == 2
    assert train.report()["skipped"] == 0
    assert train.report()["split"]["train"] == 2 and train.report()["split"]["val"] == 1
    assert "selected good" in train[0]["texts"][1] and "selected bad" in train[0]["texts"][2]
    output = Path(train.write_splits(tmp_path / "export"))
    exported = json.loads((output / "manifest.json").read_text())
    assert exported["format"] == "wan_biopd_condition_split_v1"
    assert [r["sample_id"] for r in json.loads((output / "train.json").read_text())] == ["b", "a"]
    assert json.loads((output / "val.json").read_text())[0]["negative_attributes"] == source[0]["negative_attributes"]
    assert json.loads(path.read_text()) == source


@pytest.mark.parametrize("change,match", [("ids", "sample_ids"), ("prompt", "prompt differs"),
                                         ("signature", "signature"), ("video_leak", "same video")])
def test_manifest_split_rejects_mismatch_and_video_leakage(tmp_path, change, match):
    path, reference, source, manifest = split_fixture(tmp_path)
    if change == "ids":
        source[0]["sample_id"] = "unknown"
    elif change == "prompt":
        source[0]["prompt"] = "different scene"
    elif change == "signature":
        manifest["signature"] = "invalid"
    elif change == "video_leak":
        manifest["records"][1]["split"] = "val"
    path.write_text(json.dumps(source))
    reference.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        BiOPDDataset(path, split_manifest=reference, split="train")


def test_partition_identity_is_in_resume_contract_and_trainer_rejects_val(tmp_path):
    from omegaconf import OmegaConf
    module = load_source("test_partition_trainer", ROOT / "Wan21/wan_trainer/camera_biopd.py")
    path, reference, _, _ = split_fixture(tmp_path)
    train = BiOPDDataset(path, split_manifest=reference, split="train")
    val = BiOPDDataset(path, split_manifest=reference, split="val")
    cfg = OmegaConf.create(dict(generator_ckpt=str(path), guider_ckpt=str(reference), guider_adapter_ckpt=None))
    train_contract = module.resume_contract(cfg, train, 4)
    val_contract = module.resume_contract(cfg, val, 4)
    assert train_contract["dataset_partition"]["sha256"] == train.split_info["sha256"]
    assert train_contract != val_contract
    with pytest.raises(ValueError, match="only the train"):
        module.Trainer(cfg, val)


@pytest.mark.parametrize("positive,negative", [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize("empty_form", ["list", "null", "absent"])
def test_empty_attributes_keep_records_and_four_branch_guidance(tmp_path, positive, negative, empty_form):
    record = dict(sample_id="empty", prompt="scene", action="j*8,a*11")
    for key, present, name in [("positive_attributes", positive, "clear"),
                               ("negative_attributes", negative, "blur")]:
        if present:
            record[key] = [dict(name=name)]
        elif empty_form != "absent":
            record[key] = [] if empty_form == "list" else None
    path = tmp_path / "empty.json"
    path.write_text(json.dumps([record]))
    dataset = BiOPDDataset(path)
    assert len(dataset) == 1 and dataset.report()["skipped"] == 0
    expected = ["scene",
                "scene\nAttributes: clear" if positive else "scene",
                "Attributes: blur" if negative else "",
                "scene\nAttributes: blur" if negative else "scene",
                "Attributes: clear" if positive else ""]
    assert dataset[0]["texts"] == expected
    # All four guider branches still participate, including empty text branches.
    def g(text):
        return torch.tensor(float(len(text)))
    yp, n, yn, p = [g(text) for text in dataset[0]["texts"][1:]]
    v_pos, v_neg = guider_targets([yp, n, yn, p], 2., 3.)
    assert v_pos == g(expected[1]) + 2 * (g(expected[1]) - g(expected[2]))
    assert v_neg == g(expected[3]) + 3 * (g(expected[3]) - g(expected[4]))


def test_cache_checkpoint_replay_and_detached_history():
    history = new_history(1)
    cache = history["kv_cache"][0]
    past = torch.randn(1, 3, 1, 4, requires_grad=True)
    set_write(history, True)
    with torch.no_grad():
        history_kv(cache, past, past, 0, 6)
    set_write(history, False)
    frozen = cache["k"].clone()
    current = torch.randn(1, 2, 1, 4, requires_grad=True)

    def forward(x):
        k, v = history_kv(cache, x, x, 3, 6)
        return k.square().sum() + v.square().sum()
    loss = checkpoint(forward, current, use_reentrant=False)
    loss.backward()
    assert torch.allclose(current.grad, 4 * current)
    assert past.grad is None
    assert cache["end"] == 3 and torch.equal(cache["k"], frozen)
    set_write(history, True)
    with pytest.raises(RuntimeError, match="no_grad"):
        history_kv(cache, current, current, 3, 6)


def test_lora_export_and_strict_reload(tmp_path):
    from wan_utils.biopd_lora import inject_lora, save_adapter, load_and_merge_adapter

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4)
            self.v = nn.Linear(4, 4)
        def forward(self, x):
            return self.v(self.q(x))
    base = Tiny()
    wrapper = nn.Module()
    wrapper.model = copy.deepcopy(base)
    inject_lora(wrapper.model, 2, 3, ["q", "v"])
    for name, p in wrapper.named_parameters():
        if "lora_B" in name:
            p.data.normal_()
    x = torch.randn(2, 4)
    expected = wrapper.model(x)
    state = merged_generator_state(wrapper)
    bare = nn.Module()
    bare.model = base
    bare.load_state_dict(state, strict=True)
    assert torch.allclose(bare.model(x), expected, atol=2e-6)
    restored = Tiny()
    inject_lora(restored, 2, 3, ["q", "v"])
    restore_adapter(restored, adapter_state(wrapper.model))
    for name, value in adapter_state(restored).items():
        assert torch.equal(value, adapter_state(wrapper.model)[name])
    # A frozen guider can be replaced with an adapter directory as well.
    guider = copy.deepcopy(base)
    # base now contains merged weights; use the original frozen base layers.
    guider.q.load_state_dict(wrapper.model.q.base_layer.state_dict())
    guider.v.load_state_dict(wrapper.model.v.base_layer.state_dict())
    save_adapter(wrapper.model, tmp_path / "adapter", rank=2, alpha=3,
                 targets=["q", "v"], base_model_path="unused")
    load_and_merge_adapter(guider, tmp_path / "adapter")
    assert torch.allclose(guider(x), expected, atol=2e-6)
    assert all(not p.requires_grad for p in guider.parameters())


def test_trainer_resume_restores_optimizer_and_rollout_rng(tmp_path):
    from wan_utils.biopd_lora import inject_lora
    module = load_source("test_biopd_trainer", ROOT / "Wan21/wan_trainer/camera_biopd.py")

    def make_wrapper():
        wrapper = nn.Module()
        wrapper.model = nn.Sequential(nn.Linear(3, 3))
        inject_lora(wrapper.model, 2, 2, ["0"])
        return wrapper

    trained = make_wrapper()
    initial_base = copy.deepcopy(trained)
    optimizer = torch.optim.AdamW([p for p in trained.parameters() if p.requires_grad], lr=.01)
    x = torch.randn(2, 3)
    trained.model(x).square().mean().backward()
    optimizer.step()
    noise_rng = torch.Generator().manual_seed(91)
    torch.randn(8, generator=noise_rng)
    checkpoint_path = tmp_path / "training_state.pt"
    torch.save(dict(format="wan_biopd_lora_v1", contract={"world_size": 1},
                    adapter=adapter_state(trained.model), optimizer=optimizer.state_dict(),
                    step=7, epoch=3, batch_in_epoch=2,
                    rng_by_rank=[dict(noise=noise_rng.get_state())]), checkpoint_path)
    trainer = module.Trainer.__new__(module.Trainer)
    trainer.rank = 0
    trainer.contract = {"world_size": 1}
    trainer.raw_student = initial_base
    trainer.optimizer = torch.optim.AdamW([p for p in initial_base.parameters() if p.requires_grad], lr=.01)
    trainer.noise_generator = torch.Generator().manual_seed(0)
    trainer._restore(checkpoint_path)
    assert (trainer.step, trainer.epoch, trainer.batch_in_epoch) == (7,3,2)
    assert torch.equal(torch.randn(8, generator=noise_rng), torch.randn(8, generator=trainer.noise_generator))
    for model, opt in ((trained, optimizer), (trainer.raw_student, trainer.optimizer)):
        opt.zero_grad(set_to_none=True)
        model.model(x).square().mean().backward()
        opt.step()
    for name, value in adapter_state(trained.model).items():
        assert torch.equal(value, adapter_state(trainer.raw_student.model)[name])
    trainer.contract = {"world_size": 2}
    with pytest.raises(ValueError, match="contract"):
        trainer._restore(checkpoint_path)


def _sdpa(q, k, v, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def test_real_attention_adapter_matches_original_with_camera(monkeypatch):
    # Legacy modules evaluate cuda.current_device in a default argument at
    # import time. Patch ONLY that import-time default for this CPU unit test.
    with patch("torch.cuda.current_device", return_value=0):
        import wan.modules.causal_model as cm
        import wan.modules.attention as am
        from wan.modules.prope import add_prope_parameters
    monkeypatch.setattr(cm, "attention", _sdpa)
    monkeypatch.setattr(am, "attention", _sdpa)
    base = cm.CausalWanModel(dim=128, ffn_dim=256, freq_dim=16, text_dim=64,
                            num_heads=1, num_layers=1, local_attn_size=20).eval()
    add_prope_parameters(base)
    # Camera output is nonzero, so the comparison actually tests PRoPE.
    nn.init.normal_(base.blocks[0].self_attn.prope_o.weight, std=.02)
    adapted = copy.deepcopy(base)
    install_biopd_attention(adapted)
    orig, new = base.blocks[0].self_attn, adapted.blocks[0].self_attn
    history = new_history(1)
    def old_cache():
        return dict(k=torch.zeros(1, 64, 1, 128), v=torch.zeros(1, 64, 1, 128),
                    global_end_index=torch.zeros(1, dtype=torch.long),
                    local_end_index=torch.zeros(1, dtype=torch.long))
    old, old_p = old_cache(), old_cache()
    grid = torch.tensor([[4, 2, 2]])
    vm = torch.eye(4).expand(1, 16, 4, 4).clone()
    vm[:, :, 0, 3] = .2
    ks = torch.eye(3).expand(1, 16, 3, 3).clone()
    common = dict(seq_lens=torch.tensor([16]), grid_sizes=grid, freqs=base.freqs,
                  block_mask=None, viewmats=vm, Ks=ks)
    torch.manual_seed(6)
    for chunk in range(2):
        for _ in range(4):
            x = torch.randn(1, 16, 128)
            with torch.no_grad():
                expected = orig(x, kv_cache=old, prope_kv_cache=old_p,
                                current_start=chunk * 16, **common)
                actual = new(x, kv_cache=history["kv_cache"][0],
                             prope_kv_cache=history["prope_kv_cache"][0],
                             current_start=chunk * 16, **common)
            assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
        clean = torch.randn(1, 16, 128)
        set_write(history, True)
        with torch.no_grad():
            orig(clean, kv_cache=old, prope_kv_cache=old_p, current_start=chunk*16, **common)
            new(clean, kv_cache=history["kv_cache"][0], prope_kv_cache=history["prope_kv_cache"][0],
                current_start=chunk*16, **common)
        set_write(history, False)


def test_real_causal_model_lora_checkpointed_backward(monkeypatch):
    from wan_utils.biopd_lora import inject_lora
    with patch("torch.cuda.current_device", return_value=0):
        import wan.modules.causal_model as cm
        import wan.modules.attention as am
        import wan.modules.model as wm
        from wan.modules.prope import add_prope_parameters
    monkeypatch.setattr(cm, "attention", _sdpa)
    monkeypatch.setattr(am, "attention", _sdpa)
    monkeypatch.setattr(wm, "flash_attention", _sdpa)
    torch.manual_seed(12)
    plain = cm.CausalWanModel(dim=128, ffn_dim=256, freq_dim=16, text_dim=64,
                             text_len=8, num_heads=1, num_layers=2,
                             in_dim=4, out_dim=4, local_attn_size=20).eval()
    add_prope_parameters(plain)
    # Default Wan head is zero initialized; make this a nontrivial gradient test.
    nn.init.normal_(plain.head.head.weight, std=.02)
    install_biopd_attention(plain)
    inject_lora(plain, 2, 2, ["q", "k", "v", "o", "prope_o"])
    checked = copy.deepcopy(plain)
    checked.enable_gradient_checkpointing()
    assert checked.gradient_checkpointing
    clean, noisy = torch.randn(1,4,4,4,4), torch.randn(1,4,4,4,4)
    history = new_history(2)
    vm = torch.eye(4).expand(1,4,4,4).clone()
    ks = torch.eye(3).expand(1,4,3,3).clone()
    common = dict(context=torch.randn(1,5,64), seq_len=16, viewmats=vm, Ks=ks)
    set_write(history, True)
    with torch.no_grad():
        plain(clean, t=torch.zeros(1,4), current_start=0, **history, **common)
    set_write(history, False)
    outputs = []
    for model in (plain, checked):
        output = model(noisy, t=torch.full((1,4), 625.), current_start=16,
                       **copy.deepcopy(history), **common)
        output.square().mean().backward()
        outputs.append(output.detach())
    assert torch.allclose(*outputs, atol=2e-6, rtol=2e-5)
    checked_params = dict(checked.named_parameters())
    for name, param in plain.named_parameters():
        if param.requires_grad:
            assert "lora_" in name and param.grad is not None
            assert torch.allclose(param.grad, checked_params[name].grad, atol=2e-6, rtol=2e-5), name
        else:
            assert param.grad is None
    assert plain.blocks[0].cross_attn.k.lora_B.weight.grad.abs().sum() > 0
    assert plain.blocks[0].cross_attn.v.lora_B.weight.grad.abs().sum() > 0


class TinyWrapper(nn.Module):
    """Exercises pipeline semantics, not a replacement for a Wan GPU smoke test."""
    def __init__(self):
        super().__init__()
        self.lora_weight = nn.Parameter(torch.tensor(.1))
        self.model = SimpleNamespace(blocks=[None])
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.calls = []
    def get_scheduler(self):
        return self.scheduler
    def forward(self, x, condition, timestep, kv_cache, **kwargs):
        self.calls.append((float(timestep[0,0]), condition["prompt_embeds"].item(),
                           x.requires_grad, kv_cache[0]["write"]))
        flow = x * self.lora_weight + condition["prompt_embeds"]
        sigma = timestep[0,0] / 1000
        return flow, (x - sigma * flow)


def tiny_config():
    return SimpleNamespace(denoising_step_list=[1000,750,500,250], warp_denoising_step=True,
                           step_weights=[1.,1.,1.,1.], num_frame_per_block=4,
                           latent_height=2, latent_width=2, cache_device="cpu",
                           omega_positive=1., omega_negative=1., biopd_lambda=.25)


def test_autocast_lora_backward_survives_no_grad_chunk_commits():
    from wan_utils.biopd_lora import LoRALinear
    module = load_source("autocast_biopd_pipeline", ROOT / "Wan21/pipeline/biopd_training.py")

    class CastWrapper(TinyWrapper):
        def __init__(self):
            super().__init__()
            self.lora_weight.requires_grad_(False)
            self.projection = LoRALinear(nn.Linear(2, 2).to(torch.bfloat16), 2, 2)
            self.grad_outputs = []

        def forward(self, x, condition, timestep, kv_cache, **kwargs):
            flow = self.projection(x) + condition["prompt_embeds"]
            if torch.is_grad_enabled():
                self.grad_outputs.append(flow.requires_grad)
            return flow, x - timestep[0, 0] / 1000 * flow

    student, guider = CastWrapper(), CastWrapper().requires_grad_(False)
    pipeline = module.BiOPDTrainingPipeline(student, guider, tiny_config())
    pipeline.noise_generator = torch.Generator().manual_seed(7)
    conditions = [dict(prompt_embeds=torch.tensor(float(i))) for i in range(5)]
    noise = torch.randn(1, 8, 1, 2, 2, dtype=torch.bfloat16)
    # Cached no_grad casts must not detach the subsequent student forwards.
    # This reproduces the production BF16/FP32 LoRA interaction without a GPU.
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=True):
        with torch.no_grad():
            student.projection(noise[:, :4])
        output, values = pipeline.run(conditions, torch.zeros(1,8,4,4), torch.zeros(1,8,3,3), noise)
    assert values["prediction_steps"] == 8 and not output.requires_grad
    assert student.grad_outputs == [True] * 8
    for name, param in student.named_parameters():
        if param.requires_grad:
            assert "lora_" in name and param.grad is not None and torch.isfinite(param.grad).all()
        else:
            assert param.grad is None
    assert all(param.grad is None for param in guider.parameters())


def test_all_step_pipeline_and_frozen_guider():
    module = load_source("test_biopd_pipeline", ROOT / "Wan21/pipeline/biopd_training.py")
    student, guider = TinyWrapper(), TinyWrapper().requires_grad_(False)
    pipeline = module.BiOPDTrainingPipeline(student, guider, tiny_config())
    pipeline.noise_generator = torch.Generator().manual_seed(7)
    conditions = [dict(prompt_embeds=torch.tensor(float(i))) for i in range(5)]
    noise = torch.randn(1,20,1,2,2)
    output, values = pipeline.run(conditions, torch.zeros(1,20,4,4), torch.zeros(1,20,3,3), noise)
    assert output.shape == noise.shape and not output.requires_grad
    assert values["prediction_steps"] == 20
    assert len(student.calls) == 20 + 4  # Last clean commit intentionally unnecessary.
    assert len(guider.calls) == 4 * (20 + 4)
    assert all(call[1] == 0 and not call[2] for call in student.calls)
    assert student.lora_weight.grad is not None and torch.isfinite(student.lora_weight.grad)
    assert guider.lora_weight.grad is None
    # A separate, direct predict-x0/add_noise loop with the same noise stream.
    reference = torch.empty_like(noise)
    rng = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for start in range(0, 20, 4):
            x = noise[:, start:start+4]
            for j, t in enumerate(pipeline.times):
                clean = x - (t / 1000) * (.1 * x)
                if j < len(pipeline.times) - 1:
                    eps = torch.randn(clean.shape, generator=rng)
                    x = student.scheduler.add_noise(clean.flatten(0,1), eps.flatten(0,1),
                                                      torch.full((4,), pipeline.times[j+1])).unflatten(0,(1,4))
            reference[:, start:start+4] = clean
    assert torch.equal(output, reference)
