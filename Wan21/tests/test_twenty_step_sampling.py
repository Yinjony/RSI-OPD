"""CPU rollout/gradient/entrypoint checks; no real Wan weights or CUDA execution."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]

from shared.algorithms.biopd import resolve_schedule
from wan_utils.biopd_io import load_source
from wan_utils.experiment_options import inference_sampler, training_config
from wan_utils.guider_sft_anchor import OnPolicyAnchor
from wan_utils.guider_sft_config import load_config
from wan_utils.scheduler import FlowMatchScheduler
from run_experiment import inference_command, inference_config_file

STEPS = list(range(1000, 0, -50))
biopd_helpers = load_source("twenty_biopd_helpers", ROOT / "Wan21/tests/test_biopd.py")
anchor_helpers = load_source("twenty_anchor_helpers", ROOT / "Wan21/tests/test_guider_sft_anchor.py")
inference_helpers = load_source("twenty_inference_helpers", ROOT / "Wan21/tests/test_guider_sft_cfg_inference.py")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def template():
    return OmegaConf.load(ROOT / "Wan21/configs/_base/inference.yaml")


def test_all_recipes_align_training_anchor_and_video_schedule(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for name in ("guider1", "guider2", "generator1", "generator2"):
        recipe = OmegaConf.load(ROOT / f"Wan21/configs/{name}.yaml")
        cfg = training_config(recipe)
        sampler = inference_sampler(recipe, template())
        assert list(cfg.denoising_step_list) == list(sampler.denoising_step_list) == STEPS
        assert cfg.timestep_shift == sampler.timestep_shift == sampler.model_kwargs.timestep_shift == 5
        assert cfg.warp_denoising_step is sampler.warp_denoising_step is True
        assert cfg.num_frame_per_block == sampler.num_frame_per_block == 4
        assert cfg.lora_rank == cfg.lora_alpha == 64 and recipe.relora is False
        if name.startswith("guider"):
            sft = load_config(ROOT / f"Wan21/configs/{name}.yaml")
            assert list(sft.anchor_denoising_step_list) == STEPS
        else:
            assert list(cfg.step_weights) == [1.] * 20


def test_sampler_overrides_and_anchor_drift_are_not_silent():
    generator = OmegaConf.load(ROOT / "Wan21/configs/generator1.yaml")
    generator.denoising_step_list = [1000, 750, 500, 250]
    generator.timestep_shift = 3
    sampler = inference_sampler(generator, template())
    assert list(sampler.denoising_step_list) == [1000, 750, 500, 250]
    assert sampler.model_kwargs.timestep_shift == sampler.timestep_shift == 3
    guider = OmegaConf.load(ROOT / "Wan21/configs/guider2.yaml")
    guider.denoising_step_list = [1000, 500]
    assert list(inference_sampler(guider, template()).denoising_step_list) == [1000, 500]
    guider.anchor_denoising_step_list = STEPS
    with pytest.raises(ValueError, match="must match"):
        inference_sampler(guider, template())


@pytest.mark.parametrize("steps", [[1000, 500, 0], [1000, 750, 500, 250], STEPS])
def test_schedule_shared_by_inference_and_training(steps):
    scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    times, sigmas = resolve_schedule(scheduler, steps, True)
    assert len(times) == len([t for t in steps if t])
    assert sigmas[-1] == 0 and all(b < a for a, b in zip(sigmas, sigmas[1:]))


@pytest.mark.parametrize("steps", [[1000, 1000], [500, 1000], [True], [float("nan")], [0]])
def test_bad_schedules_rejected(steps):
    scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    with pytest.raises(ValueError):
        resolve_schedule(scheduler, steps)


def test_biopd_twenty_step_rollout_gradients_and_independent_renoise_reference():
    torch.manual_seed(7)
    student, guider = biopd_helpers.TinyWrapper(), biopd_helpers.TinyWrapper().requires_grad_(False)
    cfg = biopd_helpers.tiny_config()
    cfg.denoising_step_list, cfg.step_weights = STEPS, [1.] * 20
    module = load_source("twenty_training_pipeline", ROOT / "Wan21/pipeline/biopd_training.py")
    pipeline = module.BiOPDTrainingPipeline(student, guider, cfg)
    pipeline.noise_generator = torch.Generator().manual_seed(11)
    conditions = [dict(prompt_embeds=torch.tensor(float(i))) for i in range(5)]
    noise = torch.randn(1, 20, 1, 2, 2)
    with patch.object(pipeline.scheduler, "add_noise", wraps=pipeline.scheduler.add_noise) as add_noise:
        output, values = pipeline.run(conditions, torch.zeros(1,20,4,4), torch.zeros(1,20,3,3), noise)
    assert add_noise.call_count == 5 * 19
    assert values["prediction_steps"] == 100
    assert len([k for k in values if k.startswith("step_")]) == 20
    assert len(student.calls) == 104 and len(guider.calls) == 416
    assert student.lora_weight.grad is not None and torch.isfinite(student.lora_weight.grad)
    assert student.lora_weight.grad.abs() > 0 and guider.lora_weight.grad is None
    assert not output.requires_grad
    expected = torch.empty_like(noise)
    rng = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for start in range(0, 20, 4):
            x = noise[:, start:start+4]
            for index, time in enumerate(pipeline.times):
                clean = x - time / 1000 * (x * student.lora_weight)
                if index < 19:
                    eps = torch.randn(clean.shape, generator=rng)
                    x = pipeline.scheduler.add_noise(clean.flatten(0,1), eps.flatten(0,1),
                        torch.full((4,), pipeline.times[index+1])).unflatten(0,(1,4))
            expected[:, start:start+4] = clean
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_anchor_twenty_steps_frozen_reference_and_sample_mean():
    student, reference = anchor_helpers.models()
    cfg = anchor_helpers.config()
    cfg.anchor_denoising_step_list = STEPS
    anchor = OnPolicyAnchor(student, reference, cfg)
    batch = anchor_helpers.batch()
    batch["clean_latent"] = batch["clean_latent"].repeat(1, 3, 1, 1, 1)[:, :20]
    batch["viewmats"] = torch.eye(4).expand(1,20,4,4)
    batch["Ks"] = torch.eye(3).expand(1,20,3,3)
    def evaluate(indices, backward=False):
        return anchor.loss(batch, torch.Generator().manual_seed(13),
                           state_indices=indices, backward=backward)
    loss = evaluate([0, 99], backward=True)
    assert len([call for call in student.trace if not call["write"]]) == 100
    assert len([call for call in student.trace if call["write"]]) == 4
    assert all(not call["input_requires_grad"] for call in student.trace)
    torch.testing.assert_close(loss, (evaluate([0]) + evaluate([99])) / 2)
    assert all(p.grad is None for p in reference.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in student.parameters() if p.requires_grad)
    assert all(p.grad is None for p in student.parameters() if not p.requires_grad)
    with pytest.raises(ValueError, match="anchor_states_per_batch"):
        cfg.anchor_states_per_batch = 101
        anchor.loss(batch, torch.Generator().manual_seed(13))
    # The config guard scales with actual schedule length, while old saved YAMLs still work.
    loaded = load_config(ROOT / "Wan21/configs/guider2.yaml", ["anchor_states_per_batch=100"])
    assert loaded.anchor_states_per_batch == 100


@pytest.mark.parametrize("steps", [[1000,750,500,250], STEPS, STEPS + [0]])
@pytest.mark.parametrize("cfg_enabled", [False, True])
def test_inference_prediction_renoise_counts_and_exact_schedule(steps, cfg_enabled):
    args = SimpleNamespace(denoising_step_list=steps, warp_denoising_step=True,
        num_frame_per_block=4, independent_first_frame=False, context_noise=0)
    generator = inference_helpers.FakeGenerator()
    scheduler = generator.get_scheduler()
    # Exercise real re-noising, rather than the identity operation in the old CFG unit test.
    scheduler.add_noise = FlowMatchScheduler.add_noise.__get__(scheduler)
    generator.get_scheduler = lambda: scheduler
    pipeline = inference_helpers.CausalInferencePipeline(args, torch.device("cpu"), generator=generator,
        text_encoder=inference_helpers.FakeTextEncoder(), vae=inference_helpers.FakeVAE())
    pipeline.num_transformer_blocks = pipeline.frame_seq_length = 1
    pipeline._initialize_kv_cache = lambda *a, **kw: setattr(pipeline, "kv_cache1",
        [dict(global_end_index=torch.tensor([0]), local_end_index=torch.tensor([0]))])
    pipeline._initialize_prope_kv_cache = lambda *a, **kw: setattr(pipeline, "prope_kv_cache1",
        [dict(global_end_index=torch.tensor([0]), local_end_index=torch.tensor([0]))])
    pipeline._initialize_crossattn_cache = lambda *a, **kw: setattr(pipeline, "crossattn_cache", [dict(is_init=False)])
    times, _ = resolve_schedule(scheduler, steps, True)
    torch.testing.assert_close(pipeline.denoising_step_list, torch.tensor(times), rtol=0, atol=0)
    options = dict(negative_text_prompts=["negative"], guidance_scale=2) if cfg_enabled else {}
    with patch("torch.cuda.synchronize"), patch.object(scheduler, "add_noise", wraps=scheduler.add_noise) as add_noise:
        pipeline.inference(torch.randn(1,20,1,2,2), ["positive"], **options)
    assert add_noise.call_count == 5 * (len(times) - 1)
    assert len(generator.calls) == 5 * (len(times) + 1) * (2 if cfg_enabled else 1)


@pytest.mark.parametrize("name", ["guider1", "guider2", "generator1", "generator2"])
def test_entrypoint_passes_exported_schedule_and_preserves_resume_identity(tmp_path, name):
    recipe = OmegaConf.load(ROOT / f"Wan21/configs/{name}.yaml")
    sampler = inference_sampler(recipe, template())
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    output = tmp_path / "videos"
    args = SimpleNamespace(model=name, checkpoint=str(checkpoint), output_dir=str(output),
                           num_samples=1, dry_run=True)
    flag = "--inference-config" if name.startswith("guider") else "--config"
    with inference_config_file(sampler, output, dry_run=True) as path:
        command, actual_output, gpus = inference_command(args, recipe, ["2", "3"], config_path=path)
        assert command[command.index(flag) + 1] == str(path)
        assert list(OmegaConf.load(path).denoising_step_list) == STEPS
        assert actual_output == output and gpus == ["2"] and "--dry-run" in command
    assert not output.exists()
    with inference_config_file(sampler, output, dry_run=False) as path:
        identity = path.stat().st_mtime_ns
    with inference_config_file(sampler, output, dry_run=False) as path:
        assert path.stat().st_mtime_ns == identity
    changed = copy.deepcopy(sampler)
    changed.denoising_step_list = [1000, 500]
    with pytest.raises(ValueError, match="different inference"):
        with inference_config_file(changed, output, dry_run=False):
            pass
