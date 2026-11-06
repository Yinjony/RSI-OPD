"""Recipe controls. Trainers receive only their original, resolved parameters."""
from omegaconf import OmegaConf


def inference_sampler(recipe, template):
    """Use the recipe's training rollout schedule for video sampling as well."""
    from shared.algorithms.biopd import resolve_schedule
    from wan_utils.scheduler import FlowMatchScheduler

    cfg = OmegaConf.create(OmegaConf.to_container(template, resolve=True))
    for key in ("denoising_step_list", "warp_denoising_step", "timestep_shift", "num_frame_per_block"):
        if key in recipe:
            cfg[key] = OmegaConf.to_container(recipe[key]) if OmegaConf.is_config(recipe[key]) else recipe[key]
    if "denoising_step_list" not in recipe and "anchor_denoising_step_list" in recipe:
        cfg.denoising_step_list = list(recipe.anchor_denoising_step_list)
        cfg.warp_denoising_step = recipe.anchor_warp_denoising_step
    cfg.model_kwargs.timestep_shift = cfg.timestep_shift
    if recipe.get("model_path"):
        cfg.model_kwargs.model_path = recipe.model_path
    scheduler = FlowMatchScheduler(shift=cfg.timestep_shift, sigma_min=0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    times, _ = resolve_schedule(scheduler, cfg.denoising_step_list, cfg.warp_denoising_step)
    if "anchor_denoising_step_list" in recipe:
        anchor_times, _ = resolve_schedule(scheduler, recipe.anchor_denoising_step_list,
                                           recipe.anchor_warp_denoising_step)
        if times != anchor_times:
            raise ValueError("Anchor and inference denoising schedules must match")
    return cfg


def training_config(recipe, *, allow_relora=False, selected_gpus=None):
    cfg = OmegaConf.create(OmegaConf.to_container(recipe, resolve=True))
    for name in ("anchor", "relora"):
        if name in cfg and not isinstance(cfg[name], bool):
            raise ValueError(f"{name} must be true or false")
    if "anchor" in cfg:
        cfg.sft_loss_type = "supervised_anchor" if cfg.anchor else "supervised"
    if cfg.get("relora", False):
        if not cfg.get("initial_adapter"):
            raise ValueError("relora=true requires initial_adapter to merge/reset")
        if not allow_relora:
            raise ValueError("relora=true requires the shared Wan21/scripts/train.sh entry to prepare the merged base")
    if "effective_batch_size" in cfg and "grad_accum_steps" not in cfg:
        import os
        devices = gpu_selection(selected_gpus if selected_gpus is not None else
                                os.environ.get("CUDA_VISIBLE_DEVICES", cfg.get("gpus")))
        cfg.grad_accum_steps = accumulation(cfg.effective_batch_size, cfg.batch_size, len(devices))
    for key in ("anchor", "relora", "gpus", "effective_batch_size", "inference"):
        cfg.pop(key, None)
    return cfg


def gpu_selection(value):
    if isinstance(value, str):
        value = value.split(",")
    if not value:
        raise ValueError("Specify at least one GPU in gpus or --gpus")
    ids = [str(v).strip() for v in value]
    if any(not v.isdecimal() for v in ids) or len(ids) != len(set(ids)):
        raise ValueError("GPUs must be unique nonnegative integer IDs, e.g. 0,1")
    return ids


def accumulation(effective_batch, batch_size, world_size):
    values = (effective_batch, batch_size, world_size)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("effective_batch_size, batch_size and world_size must be positive integers")
    divisor = batch_size * world_size
    if effective_batch % divisor:
        raise ValueError("effective_batch_size must be divisible by batch_size * number of GPUs")
    return effective_batch // divisor


def prepare_relora(cfg, directory):
    """One phase-boundary merge/reset; never runs periodically during training.

    The merged file becomes the actual frozen base, so new adapters record its
    identity. The source adapter and its original base are never modified.
    """
    from pathlib import Path
    from wan_utils.biopd_io import load_generator_state
    from wan_utils.guider_sft_dataset import atomic_json, atomic_save, file_identity
    from wan_utils.guider_sft_lora_init import read_initial_adapter, validate_initial_adapter
    from wan_utils.guider_sft_training import merge_adapter_state

    payload, identities = read_initial_adapter(cfg.initial_adapter)
    validate_initial_adapter(payload, cfg)
    destination = Path(directory) / "relora_base.pt"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite ReLoRA base: {destination}")
    merged = merge_adapter_state(load_generator_state(cfg.generator_ckpt), payload)
    atomic_save(dict(generator=merged), destination)
    atomic_json(dict(operation="phase_boundary_merge_and_reset", base=file_identity(cfg.generator_ckpt),
                     adapter=identities, merged=file_identity(destination)),
                destination.with_suffix(".pt.merge.json"))
    cfg.generator_ckpt = str(destination.resolve())
    cfg.pop("initial_adapter", None)
    cfg.pop("initial_adapter_identity", None)
    return cfg
