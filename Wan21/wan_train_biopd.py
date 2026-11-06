#!/usr/bin/env python3
"""Standalone BiOPD CLI; --check-data requires no CUDA or model imports."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]


def validate_config(cfg):
    if not 0 <= cfg.biopd_lambda < 1:
        raise ValueError("biopd_lambda must satisfy 0 <= lambda < 1")
    for name in ("omega_positive", "omega_negative", "lr", "max_grad_norm", "timestep_shift",
                 "weight_decay", "lora_alpha"):
        if not math.isfinite(float(cfg[name])) or cfg[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("batch_size", "grad_accum_steps", "max_steps", "save_every", "log_every",
                 "lora_rank", "num_training_frames", "num_frame_per_block"):
        if int(cfg[name]) != cfg[name] or cfg[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if cfg.lr == 0 or cfg.max_grad_norm == 0 or cfg.timestep_shift == 0 or cfg.lora_alpha == 0:
        raise ValueError("lr, max_grad_norm, timestep_shift and lora_alpha must be positive")
    if not (0 <= cfg.beta1 < 1 and 0 <= cfg.beta2 < 1):
        raise ValueError("Adam betas must be in [0,1)")
    if cfg.num_training_frames % cfg.num_frame_per_block:
        raise ValueError("num_training_frames must be divisible by num_frame_per_block")
    # Preserve this checkpoint's supported camera/attention geometry. Tests use
    # small stand-ins, but production CLI must not silently change the sampler.
    if (cfg.latent_height, cfg.latent_width) != (60, 104):
        raise ValueError("This implementation preserves the checkpoint's 60x104 latent geometry")
    if cfg.num_frame_per_block != 4 or cfg.num_training_frames > 20 or cfg.local_attn_size != 20:
        raise ValueError("Use 4-frame chunks, <=20 latent frames, local_attn_size=20")
    if cfg.cache_device not in ("cpu", "cuda"):
        raise ValueError("cache_device must be cpu or cuda")
    if type(cfg.get("guider_branch_batch_size", 1)) is not int or cfg.get("guider_branch_batch_size", 1) not in (1, 2, 4):
        raise ValueError("guider_branch_batch_size must be 1, 2 or 4")
    for flag in ("guider_text_kv_cache", "prope_transform_cache", "guider_video_kv_buffer"):
        if not isinstance(cfg.get(flag, True), bool):
            raise ValueError(f"{flag} must be a boolean")
    if cfg.text_cache_size < 0 or int(cfg.text_cache_size) != cfg.text_cache_size:
        raise ValueError("text_cache_size must be a nonnegative integer")
    every = cfg.get("val_every", 0)
    if not math.isfinite(float(every)) or int(every) != every or every < 0:
        raise ValueError("val_every must be a nonnegative integer (0 disables validation)")
    if every:
        if not cfg.get("split_manifest"):
            raise ValueError("Automatic validation requires split_manifest")
        count, seed = cfg.get("val_num_samples", 20), cfg.get("val_seed", 12345)
        if not math.isfinite(float(count)) or int(count) != count or count <= 0:
            raise ValueError("val_num_samples must be a positive integer")
        if not math.isfinite(float(seed)) or int(seed) != seed or not 0 <= seed < 2**63:
            raise ValueError("val_seed must be an integer in [0, 2**63)")
        for name in ("val_at_start", "val_full_at_end"):
            if not isinstance(cfg.get(name, True), bool):
                raise ValueError(f"{name} must be a boolean")
    if not cfg.lora_target_modules or not set(cfg.lora_target_modules) <= {"q", "k", "v", "o", "prope_o"}:
        raise ValueError("LoRA targets must be a nonempty subset of q,k,v,o,prope_o")
    if any(not math.isfinite(float(w)) or w < 0 for w in cfg.step_weights) or not any(cfg.step_weights):
        raise ValueError("step_weights must be finite, nonnegative, with at least one positive entry")
    from shared.algorithms.biopd import resolve_schedule
    from wan_utils.scheduler import FlowMatchScheduler
    scheduler = FlowMatchScheduler(shift=cfg.timestep_shift, sigma_min=0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    times, sigmas = resolve_schedule(scheduler, cfg.denoising_step_list, cfg.warp_denoising_step)
    if len(times) != len(cfg.step_weights):
        raise ValueError("step_weights must match denoising_step_list (excluding terminal zero)")
    for name in ("generator_ckpt", "guider_ckpt", "annotation_path", "model_path"):
        if not Path(cfg[name]).exists():
            raise FileNotFoundError(f"{name}: {cfg[name]}")
    if cfg.guider_adapter_ckpt and not (Path(cfg.guider_adapter_ckpt) / "adapter_config.json").exists():
        raise ValueError("guider_adapter_ckpt must be a PEFT adapter directory")
    if cfg.get("initial_adapter"):
        from wan_utils.guider_sft_lora_init import read_initial_adapter, validate_initial_adapter
        payload, _ = read_initial_adapter(cfg.initial_adapter)
        validate_initial_adapter(payload, cfg)
        if cfg.generator_checkpoint_key != "generator":
            raise ValueError("Continued student adapters require generator_checkpoint_key=generator")
    return times, sigmas


def main():
    from omegaconf import OmegaConf
    from wan_utils.biopd_dataset import BiOPDDataset
    from wan_utils.biopd_io import load_source, runtime_log
    from wan_utils.biopd_validation import validation_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--config_path", default=str(ROOT / "Wan21/configs/generator1.yaml"))
    parser.add_argument("--override", action="append", default=[], help="OmegaConf key=value; repeatable")
    parser.add_argument("--resume", help="Path to training_state.pt (same world size/objective required)")
    parser.add_argument("--check-data", action="store_true", help="Validate config/data/schedule without model loading")
    parser.add_argument("--write-splits", metavar="DIRECTORY",
                        help="With --check-data, export Selector train/val JSONs and split manifest")
    args = parser.parse_args()
    if args.write_splits and not args.check_data:
        parser.error("--write-splits requires --check-data; it never starts training")
    config_path = Path(args.config).resolve()
    os.chdir(ROOT)
    from wan_utils.experiment_options import training_config
    cfg = training_config(OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_dotlist(args.override)))
    if "grad_accum_steps" not in cfg:
        cfg.grad_accum_steps = 1
    if args.resume:
        cfg.resume = args.resume
    # Every worker must reject an occupied output directory before CUDA/NCCL.
    # A rank-0-only check after DDP can strand other ranks in their barrier,
    # while rank 0 waits in destroy_process_group before printing its exception.
    if not args.check_data and not cfg.get("resume"):
        metrics = Path(cfg.output_dir) / "metrics.jsonl"
        if metrics.exists():
            raise FileExistsError(
                f"Output already contains an experiment: {metrics.parent.resolve()}. "
                "Choose a new output_dir (check BIOPD_OUTPUT_DIR), or explicitly "
                "resume the same experiment with --resume training_state.pt. "
                "Stopped before CUDA/DDP initialization."
            )
    times, sigmas = validate_config(cfg)
    split_manifest = cfg.get("split_manifest")
    split = cfg.get("dataset_split", "train" if split_manifest else "all")
    if split_manifest and not args.check_data and split != "train":
        raise ValueError("BiOPD training requires dataset_split=train when split_manifest is configured")
    dataset_options = dict(num_frames=cfg.num_training_frames,
                           missing_attributes=cfg.get("missing_attributes", "keep"),
                           camera=OmegaConf.to_container(cfg.camera),
                           prompt_template=cfg.prompt_attribute_template,
                           attribute_template=cfg.attribute_only_template,
                           separator=cfg.attribute_separator,
                           condition_format=cfg.get("condition_format", "generic"),
                           split_manifest=split_manifest)
    dataset = BiOPDDataset(cfg.annotation_path, split=split, **dataset_options)
    validation_dataset = (BiOPDDataset(cfg.annotation_path, split="val", **dataset_options)
                          if cfg.get("val_every", 0) else None)
    if args.check_data:
        report = dict(data=dataset.report(), times=times, sigmas=sigmas,
                      guider_branch_batch_size=cfg.get("guider_branch_batch_size", 1),
                      guider_text_kv_cache=cfg.get("guider_text_kv_cache", True),
                      prope_transform_cache=cfg.get("prope_transform_cache", True),
                      guider_video_kv_buffer=cfg.get("guider_video_kv_buffer", True),
                      example_conditions=dataset[0]["texts"])
        if cfg.get("initial_adapter"):
            from wan_utils.guider_sft_lora_init import read_initial_adapter
            payload, sources = read_initial_adapter(cfg.initial_adapter)
            report["student_initialization"] = dict(
                mode="continue_existing_lora", base_identity=payload["base_identity"],
                initial_adapter=str(Path(cfg.initial_adapter).resolve()), sources=sources,
                rank=payload["rank"], alpha=payload["alpha"],
                adapter_tensors=len(payload["adapter"]),
                adapter_parameters=sum(t.numel() for t in payload["adapter"].values()),
                optimizer="new for iteration 2; A/B preserved")
        if cfg.get("text_cache_dir"):
            from wan_utils.biopd_text_cache import PreencodedTextConditions
            cached_text = PreencodedTextConditions(cfg, dataset, "cpu")
            if validation_dataset is not None:
                PreencodedTextConditions(cfg, validation_dataset, "cpu")
            report["text_cache"] = cached_text.report
        if validation_dataset is not None:
            report["validation"] = validation_report(validation_dataset, cfg)
        if args.write_splits:
            report["split_output_dir"] = dataset.write_splits(args.write_splits)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    with runtime_log(cfg.output_dir, int(os.environ.get("RANK", "0"))):
        module = load_source("biopd_camera_trainer", ROOT / "Wan21/wan_trainer/camera_biopd.py")
        try:
            module.Trainer(cfg, dataset, validation_dataset=validation_dataset).train()
        finally:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
