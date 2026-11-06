"""CPU regression: BiOPD preserves G1 A/B and rejects a merged student base."""

import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]

from wan_utils.biopd_io import file_identity, load_source, merged_generator_state
from wan_utils.biopd_lora import get_adapter_state, inject_lora, save_adapter


class TinyWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = nn.ModuleDict({name: nn.Linear(4, 4) for name in ("q", "v")})

    def forward(self, x):
        return self.model["v"](self.model["q"](x))


@pytest.fixture
def continuation(tmp_path, monkeypatch):
    torch.manual_seed(42)
    base = TinyWrapper()
    base_path = tmp_path / "original_base.pt"
    torch.save(dict(generator=base.state_dict()), base_path)
    old = copy.deepcopy(base).to(dtype=torch.bfloat16)
    inject_lora(old.model, 2, 2, ["q", "v"])
    with torch.no_grad():
        for name, parameter in old.named_parameters():
            if "lora_B" in name:
                parameter.normal_()
    step = tmp_path / "step_0000475"
    save_adapter(old.model, step / "adapter", rank=2, alpha=2, targets=["q", "v"],
                 base_model_path=str(tmp_path))
    (step / "metadata.json").write_text(json.dumps(dict(step=475, contract=dict(
        student_identity=file_identity(base_path), generator_ckpt=str(base_path),
        model_path=str(tmp_path)))))
    cfg = OmegaConf.create(dict(
        generator_ckpt=str(base_path), generator_checkpoint_key="generator",
        initial_adapter=str(step / "adapter"), model_path=str(tmp_path),
        guider_ckpt=str(base_path), guider_checkpoint_key="generator", guider_adapter_ckpt=None,
        timestep_shift=5., local_attn_size=20, num_frame_per_block=4,
        lora_rank=2, lora_alpha=2, lora_target_modules=["q", "v"], gradient_checkpointing=False))
    wrapper_module = ModuleType("wan_utils.wan_wrapper")
    wrapper_module.WanDiffusionWrapper = TinyWrapper
    monkeypatch.setitem(sys.modules, "wan_utils.wan_wrapper", wrapper_module)
    module = load_source("test_continued_biopd_model", ROOT / "Wan21/model/camera_biopd.py")
    monkeypatch.setattr(module, "install_biopd_attention", lambda model: None)
    return cfg, old, module


def test_biopd_loader_preserves_ab_predictions_and_updates_only_ab(continuation):
    cfg, old, module = continuation
    student, guider = module.load_models(cfg, torch.device("cpu"))
    initial = get_adapter_state(old.model)
    assert all(torch.equal(value, initial[key]) for key, value in get_adapter_state(student.model).items())
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    assert torch.equal(student(x), old(x))
    frozen = {n: p.detach().clone() for n, p in student.named_parameters() if not p.requires_grad}
    assert all("lora_" in n for n, p in student.named_parameters() if p.requires_grad)
    assert not any(p.requires_grad for p in guider.parameters())
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=.01)
    student(x).float().square().mean().backward()
    optimizer.step()
    updated = get_adapter_state(student.model)
    assert any(not torch.equal(value, initial[key]) for key, value in updated.items())
    assert all(torch.equal(p, frozen[n]) for n, p in student.named_parameters() if n in frozen)
    assert all(p.grad is None for p in guider.parameters())
    # Exporting a merged copy must not reset the active A/B or its Adam state.
    merged_generator_state(student)
    assert all(torch.equal(value, updated[key]) for key, value in get_adapter_state(student.model).items())
    assert all(state["step"].item() == 1 for state in optimizer.state.values())


def test_biopd_loader_rejects_adding_old_adapter_to_merged_base(continuation, tmp_path):
    cfg, old, module = continuation
    merged = tmp_path / "g1_merged.pt"
    torch.save(dict(generator=merged_generator_state(old)), merged)
    cfg.generator_ckpt = str(merged)
    with pytest.raises(ValueError, match="base_identity"):
        module.load_models(cfg, torch.device("cpu"))


def test_existing_fresh_lora_configs_keep_their_initialization(continuation):
    cfg, _, module = continuation
    del cfg.initial_adapter
    student, _ = module.load_models(cfg, torch.device("cpu"))
    assert all(torch.count_nonzero(value) == 0 for key, value in get_adapter_state(student.model).items()
               if ".lora_B." in key)


def test_resume_contract_tracks_initial_adapter_source_changes(continuation):
    cfg, _, _ = continuation
    module = load_source("test_continued_biopd_contract", ROOT / "Wan21/wan_trainer/camera_biopd.py")
    dataset = SimpleNamespace(sha256="same-data", split_info=None)
    before = module.resume_contract(cfg, dataset, 4)
    metadata = Path(cfg.initial_adapter).parent / "metadata.json"
    payload = json.loads(metadata.read_text())
    payload["note"] = "changed source metadata"
    metadata.write_text(json.dumps(payload))
    after = module.resume_contract(cfg, dataset, 4)
    assert before["initial_adapter_identity"] != after["initial_adapter_identity"]
