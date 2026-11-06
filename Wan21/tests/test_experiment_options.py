"""CPU checks for continuation controls, phase resets, and shared launchers."""
import copy
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]

from wan_utils.experiment_options import accumulation, gpu_selection, prepare_relora, training_config
from wan_utils.guider_sft_dataset import file_identity
from wan_utils.biopd_io import load_generator_state
from wan_utils.guider_sft_lora_init import validate_initial_adapter
from run_experiment import saved_config


def test_gpu_count_preserves_effective_batch_and_rejects_bad_selections():
    for count, expected in ((1, 4), (2, 2), (4, 1)):
        assert accumulation(4, 1, count) == expected
    with pytest.raises(ValueError, match="divisible"):
        accumulation(4, 1, 3)
    for value in ("0,0", "", "a,1", [-1], [True]):
        with pytest.raises(ValueError):
            gpu_selection(value)


def test_controls_do_not_pollute_old_resume_contract_and_false_keeps_adapter(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    recipe = OmegaConf.create(dict(anchor=False, relora=False, initial_adapter="old/adapter.pt",
                                  gpus=[0, 1], effective_batch_size=4, batch_size=1,
                                  inference=dict(checkpoint="unused")))
    before = OmegaConf.to_container(recipe)
    cfg = training_config(recipe)
    assert cfg.initial_adapter == "old/adapter.pt" and cfg.sft_loss_type == "supervised"
    assert cfg.grad_accum_steps == 2
    assert not set(("gpus", "effective_batch_size", "anchor", "relora", "inference")) & set(cfg)
    assert OmegaConf.to_container(recipe) == before
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    assert training_config(recipe, selected_gpus=[0, 1, 2, 3]).grad_accum_steps == 1
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    recipe.anchor = True
    assert training_config(recipe).sft_loss_type == "supervised_anchor"
    for flag in ("anchor", "relora"):
        invalid = copy.deepcopy(recipe)
        invalid[flag] = "false"
        with pytest.raises(ValueError, match=flag):
            training_config(invalid)
    recipe.relora = True
    with pytest.raises(ValueError, match="shared"):
        training_config(recipe)


def test_relora_merges_once_rebases_metadata_and_never_reapplies_old_adapter(tmp_path):
    base = tmp_path / "base.pt"
    weights = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save(dict(generator={"model.q.weight": weights}), base)
    a, b = torch.ones(2, 4), torch.ones(3, 2)
    payload = dict(format="wan_guider_sft_lora_v1", rank=2, alpha=2, dropout=0., targets=["q"],
                   base_identity=file_identity(base), model_path=str(tmp_path),
                   adapter={"base_model.model.q.lora_A.weight": a, "base_model.model.q.lora_B.weight": b})
    source = tmp_path / "adapter.pt"
    torch.save(payload, source)
    cfg = OmegaConf.create(dict(generator_ckpt=str(base), initial_adapter=str(source),
                               lora_rank=2, lora_alpha=2, lora_target_modules=["q"], model_path=str(tmp_path)))
    initial_base, initial_source = base.read_bytes(), source.read_bytes()
    prepared = prepare_relora(cfg, tmp_path / "prepared")
    assert "initial_adapter" not in prepared
    torch.testing.assert_close(load_generator_state(prepared.generator_ckpt)["model.q.weight"], weights + b @ a)
    assert base.read_bytes() == initial_base and source.read_bytes() == initial_source
    provenance = json.loads(Path(prepared.generator_ckpt + ".merge.json").read_text())
    assert provenance["merged"] == file_identity(prepared.generator_ckpt)
    with pytest.raises(ValueError, match="base_identity"):
        validate_initial_adapter(payload, prepared)
    # The new phase loads the merged base and initializes B=0; the old adapter
    # cannot silently be applied for a second time against the changed base.
    original = copy.deepcopy(prepared)
    original.generator_ckpt = str(base)
    original.initial_adapter = str(source)
    with pytest.raises(FileExistsError):
        prepare_relora(original, tmp_path / "prepared")


def test_resume_uses_saved_run_not_changed_recipe(tmp_path):
    run = tmp_path / "run"
    step = run / "checkpoint_000001"
    step.mkdir(parents=True)
    state = step / "training_state.pt"
    state.touch()
    OmegaConf.save(OmegaConf.create(dict(generator_ckpt="prepared/relora_base.pt", grad_accum_steps=2)), run / "config.yaml")
    cfg = saved_config(state)
    assert cfg.generator_ckpt == "prepared/relora_base.pt" and cfg.grad_accum_steps == 2
    assert cfg.resume == str(state)


def test_recipes_keep_dmd_base_and_replay_guider_starts_fresh(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    recipes = {name: OmegaConf.load(ROOT / f"Wan21/configs/{name}.yaml")
               for name in ("guider1", "guider2", "generator1", "generator2")}
    assert all(c.lora_rank == 64 and c.lora_alpha == 64 and c.relora is False for c in recipes.values())
    assert recipes["guider1"].initial_adapter is None and recipes["generator1"].initial_adapter is None
    assert recipes["guider2"].initial_adapter is None
    assert recipes["guider1"].attribute_dropout is True and recipes["guider2"].attribute_dropout is True
    assert recipes["guider2"].annotation_path.endswith("iteration001_replay_new/Guider_2_new.json")
    assert "step_0000475/adapter" in recipes["generator2"].initial_adapter
    assert recipes["generator2"].guider_ckpt == recipes["guider2"].inference.checkpoint
    # Generator1 retains its already selected first-round guider; the current
    # Guider1 recipe may describe a new run rather than that historical model.
    assert recipes["generator2"].annotation_path.endswith("selector_iter2.json")
    assert len({c.generator_ckpt for c in recipes.values()}) == 1
