"""Shared CLI/config and cache provenance, independent of Wan imports."""
import math
from pathlib import Path

from omegaconf import OmegaConf

from wan_utils.guider_sft_dataset import TEMPLATE_VERSION, file_identity, file_sha256

ROOT = Path(__file__).resolve().parents[2]

# Old saved YAMLs without these fields keep their original objective.
# Current recipes use the anchor boolean, resolved before validation.
OBJECTIVE_DEFAULTS = dict(sft_loss_type="supervised", anchor_eta=0.1,
                          anchor_denoising_step_list=[1000, 750, 500, 250],
                          anchor_warp_denoising_step=True, anchor_states_per_batch=1,
                          anchor_cache_device="cuda")


def load_config(path, overrides=(), *, data_only=False):
    from wan_utils.experiment_options import training_config
    cfg = training_config(OmegaConf.merge(OBJECTIVE_DEFAULTS, OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides))),
                          allow_relora=data_only)
    for name in ("annotation_path", "model_path", "generator_ckpt", "cache_dir", "output_dir", "resume"):
        if cfg.get(name):
            p = Path(cfg[name]).expanduser()
            cfg[name] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    # Absent in old saved runs: preserve their cache and exact-resume contracts.
    if "attribute_dropout" not in cfg:
        cfg.attribute_dropout = False
    if not isinstance(cfg.attribute_dropout, bool):
        raise ValueError("attribute_dropout must be boolean")
    if cfg.sft_loss_type not in ("supervised", "supervised_anchor"):
        raise ValueError("sft_loss_type must be supervised or supervised_anchor")
    if isinstance(cfg.anchor_eta, bool) or not math.isfinite(float(cfg.anchor_eta)) or cfg.anchor_eta < 0:
        raise ValueError("anchor_eta must be finite and nonnegative")
    steps = list(cfg.anchor_denoising_step_list)
    if (not steps or any(isinstance(t, bool) or not isinstance(t, (int, float))
                        or not math.isfinite(t) or int(t) != t or not 0 < t <= 1000 for t in steps)
            or any(b >= a for a, b in zip(steps, steps[1:]))):
        raise ValueError("anchor_denoising_step_list must contain strictly descending integer timesteps in [1, 1000]")
    if not isinstance(cfg.anchor_warp_denoising_step, bool):
        raise ValueError("anchor_warp_denoising_step must be boolean")
    if cfg.anchor_cache_device not in ("cpu", "cuda"):
        raise ValueError("anchor_cache_device must be cpu or cuda")
    # Recipe-level launchers derive accumulation from GPU count. Direct CPU
    # consumers (preencoding/inference) use one process unless given an override.
    if "grad_accum_steps" not in cfg:
        cfg.grad_accum_steps = 1
    for name in ("batch_size", "grad_accum_steps", "max_steps", "save_every", "log_every",
                 "lora_rank", "num_frames", "num_frame_per_block", "height", "width"):
        v = cfg[name]
        if isinstance(v, bool) or int(v) != v or v <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("lr", "max_grad_norm", "timestep_shift", "lora_alpha"):
        if not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("num_workers", "val_every", "warmup_steps"):
        if int(cfg[name]) != cfg[name] or cfg[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not (0 <= cfg.beta1 < 1 and 0 <= cfg.beta2 < 1):
        raise ValueError("Adam betas must be in [0, 1)")
    if not 0 <= cfg.val_fraction < 1:
        raise ValueError("val_fraction must be in [0,1)")
    if cfg.loss_weighting not in ("uniform", "scheduler"):
        raise ValueError("loss_weighting must be uniform or scheduler")
    if (cfg.num_frames, cfg.height, cfg.width, cfg.num_frame_per_block) != (77, 480, 832, 4):
        raise ValueError("This entry preserves 77 RGB / 20 latent frames, 480x832, 4-frame blocks")
    from shared.algorithms.biopd import resolve_schedule
    from wan_utils.scheduler import FlowMatchScheduler
    scheduler = FlowMatchScheduler(shift=cfg.timestep_shift, sigma_min=0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    times, _ = resolve_schedule(scheduler, steps, cfg.anchor_warp_denoising_step)
    if cfg.get("denoising_step_list") is not None:
        inference_times, _ = resolve_schedule(scheduler, cfg.denoising_step_list, cfg.warp_denoising_step)
        if times != inference_times:
            raise ValueError("Anchor and inference denoising schedules must match")
    latent_frames = (cfg.num_frames - 1) // 4 + 1
    num_states = latent_frames // cfg.num_frame_per_block * len(times)
    n = cfg.anchor_states_per_batch
    if isinstance(n, bool) or int(n) != n or not 1 <= n <= num_states:
        raise ValueError(f"anchor_states_per_batch must be an integer in [1, {num_states}]")
    if cfg.lora_dropout != 0:
        raise ValueError("Current standard LoRA implementation requires dropout=0")
    if set(cfg.lora_target_modules) != {"q", "k", "v", "o", "prope_o"}:
        raise ValueError("Expected LoRA targets q,k,v,o,prope_o")
    if len(cfg.lora_target_modules) != 5:
        raise ValueError("LoRA targets must not contain duplicates")
    for name in ("fx", "fy", "cx", "cy"):
        if not math.isfinite(cfg.camera[name]):
            raise ValueError(f"Invalid camera.{name}")
    if cfg.camera.fx <= 0 or cfg.camera.fy <= 0:
        raise ValueError("Camera focal lengths must be positive")


def cache_contract(cfg, records):
    model = Path(cfg.model_path)
    # Encodings do not depend on the trainable generator checkpoint.
    model_files = [model / "Wan2.1_VAE.pth", model / "models_t5_umt5-xxl-enc-bf16.pth"]
    model_files += sorted(p for p in (model / "google/umt5-xxl").rglob("*") if p.is_file())
    if len(model_files) == 2:
        raise FileNotFoundError("Missing local UMT5 tokenizer files")
    contract = dict(template_version=TEMPLATE_VERSION, annotation_sha256=file_sha256(cfg.annotation_path),
                records=records, videos=[file_identity(r["video_path"]) for r in records],
                model_files=[file_identity(p) for p in model_files], camera=OmegaConf.to_container(cfg.camera),
                height=cfg.height, width=cfg.width, num_frames=cfg.num_frames,
                val_fraction=cfg.val_fraction, split_seed=cfg.split_seed,
                validation_controls=cfg.validation_controls, latent_dtype="float16", text_dtype="bfloat16")
    if cfg.get("attribute_dropout", False):
        contract["attribute_dropout"] = True
    return contract
