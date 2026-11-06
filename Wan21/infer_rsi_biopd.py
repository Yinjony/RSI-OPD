#!/usr/bin/env python3
"""Iteration-aware inference for the RSI BiOPD prompt/trajectory pairs.

This is a standalone wrapper around minWM's existing causal Wan DMD inference
pipeline.  It adds deterministic iteration slicing, index-based filenames,
per-sample metadata, and safe resume behavior without changing the original
``wan_inference.py`` or causal inference implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "Wan21"), str(PROJECT_ROOT), str(PROJECT_ROOT / "shared")]
DEFAULT_CONFIG = PROJECT_ROOT / "Wan21/configs/_base/inference.yaml"
SHARED_ASSET_ROOT = Path("/root/data/mcy/timeOPD/minWM")
DEFAULT_CHECKPOINT = SHARED_ASSET_ROOT / "ckpts/Wan21/Action2V/dmd/model.pt"
DEFAULT_PROMPTS = SHARED_ASSET_ROOT / "dataset/RSI_BiOPD_4000_prompts.txt"
DEFAULT_TRAJECTORIES = SHARED_ASSET_ROOT / "dataset/RSI_BiOPD_4000_trajectories.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one iteration of RSI BiOPD inference with Wan Action2V DMD."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--trajectories", default=str(DEFAULT_TRAJECTORIES))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--samples-per-iteration", type=int, default=2000)
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional inclusive global start index for one shard of the iteration.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Optional exclusive global end index for one shard of the iteration.",
    )
    parser.add_argument(
        "--shard-name",
        default="",
        help="Optional safe name for separate run state under iteration_NNN/shards/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional smaller count for a smoke test; 0 uses the full iteration.",
    )
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--text-encoder-offload",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument("--fx", type=float, default=0.5050505)
    parser.add_argument("--fy", type=float, default=0.89786756)
    parser.add_argument("--cx", type=float, default=0.5)
    parser.add_argument("--cy", type=float, default=0.5)
    parser.add_argument("--empty-cache-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths, iteration bounds, config, and trajectories without CUDA.",
    )
    return parser.parse_args()


def resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def read_nonempty_aligned_lines(path: Path, label: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    empty_indices = [index for index, line in enumerate(lines) if not line.strip()]
    if empty_indices:
        preview = ", ".join(str(index) for index in empty_indices[:8])
        raise ValueError(
            f"{label} contains empty aligned rows at zero-based indices: {preview}"
        )
    return [line.strip() for line in lines]


def select_iteration_indices(
    dataset_size: int,
    iteration: int,
    samples_per_iteration: int,
    limit: int,
    start_index: int | None = None,
    end_index: int | None = None,
) -> list[int]:
    if dataset_size <= 0:
        raise ValueError("Dataset is empty")
    if iteration < 0:
        raise ValueError(f"iteration must be non-negative, got {iteration}")
    if samples_per_iteration <= 0:
        raise ValueError(
            f"samples-per-iteration must be positive, got {samples_per_iteration}"
        )
    if limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")
    iteration_start = iteration * samples_per_iteration
    if iteration_start >= dataset_size:
        raise ValueError(
            f"Iteration {iteration} starts at {iteration_start}, "
            f"outside dataset size {dataset_size}"
        )
    iteration_end = min(dataset_size, iteration_start + samples_per_iteration)
    if (start_index is None) != (end_index is None):
        raise ValueError("start-index and end-index must be specified together")
    if start_index is None:
        start = iteration_start
        end = iteration_end
    else:
        start = start_index
        end = end_index
        if start < iteration_start or end > iteration_end or start >= end:
            raise ValueError(
                f"Shard range [{start}, {end}) must be a non-empty subset of "
                f"iteration {iteration} range [{iteration_start}, {iteration_end})"
            )
    if limit > 0:
        end = min(end, start + limit)
    return list(range(start, end))


def resolve_run_state_dir(iteration_dir: Path, shard_name: str) -> Path:
    if not shard_name:
        return iteration_dir
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", shard_name) is None:
        raise ValueError(
            "shard-name may contain only letters, digits, dot, underscore, and hyphen"
        )
    return iteration_dir / "shards" / shard_name


def file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def load_config(config_path: Path, project_root: Path):
    if not config_path.is_file():
        raise FileNotFoundError(f"Inference config does not exist: {config_path}")
    default_path = project_root / "Wan21/configs/_base/default.yaml"
    config = OmegaConf.merge(
        OmegaConf.load(default_path), OmegaConf.load(config_path)
    )
    if not bool(config.get("causal", True)):
        raise ValueError("RSI BiOPD inference requires a causal Wan config")
    if "denoising_step_list" not in config:
        raise ValueError("DMD inference config must contain denoising_step_list")
    if not bool(config.get("use_camera", False)):
        raise ValueError("RSI BiOPD trajectories require use_camera=true")
    model_kwargs = dict(config.get("model_kwargs", {}) or {})
    if not bool(model_kwargs.get("use_camera", False)):
        raise ValueError("RSI BiOPD trajectories require model_kwargs.use_camera=true")
    if float(model_kwargs.get("timestep_shift", 8.0)) != float(config.timestep_shift):
        raise ValueError("model_kwargs.timestep_shift must match timestep_shift")
    from shared.algorithms.biopd import resolve_schedule
    from wan_utils.scheduler import FlowMatchScheduler
    scheduler = FlowMatchScheduler(shift=config.timestep_shift, sigma_min=0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    resolve_schedule(scheduler, config.denoising_step_list, config.warp_denoising_step)
    return config


def validate_selected_trajectories(
    trajectories: list[str], indices: list[int], expected_frames: int
) -> None:
    from wan_utils.camera_trajectory import parse_trajectory

    for index in indices:
        frame_count = len(parse_trajectory(trajectories[index]))
        if frame_count != expected_frames:
            raise ValueError(
                f"Trajectory row {index} creates {frame_count} latent frames; "
                f"the DMD config requires {expected_frames}: {trajectories[index]!r}"
            )


def extract_generator_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must contain a state-dict mapping: {checkpoint_path}"
        )
    for key in ("generator_ema", "generator", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break
    if not checkpoint or not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise ValueError(f"No tensor generator state dict found in {checkpoint_path}")

    fixed = {}
    for name, value in checkpoint.items():
        name = (
            name.replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        if name.startswith("model._fsdp_wrapped_module."):
            name = name.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[name] = value
    return fixed


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def save_video_atomic(path: Path, video: torch.Tensor, fps: int) -> None:
    frames = (
        video[0]
        .permute(0, 2, 3, 1)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.mp4")
    write_video(
        str(temporary),
        frames,
        fps=fps,
        video_codec="libx264",
        options={"crf": "18"},
    )
    os.replace(temporary, path)


def completed_sample_matches(
    metadata_path: Path, video_path: Path, run_signature: str, global_index: int
) -> bool:
    metadata = load_json(metadata_path)
    return bool(
        metadata
        and metadata.get("status") == "complete"
        and metadata.get("run_signature") == run_signature
        and metadata.get("global_index") == global_index
        and video_path.is_file()
        and video_path.stat().st_size > 0
    )


def write_manifest(
    path: Path, metadata_dir: Path, indices: list[int]
) -> tuple[int, int]:
    records = []
    completed = 0
    failed = 0
    for index in indices:
        record = load_json(metadata_dir / f"{index:06d}.json")
        if record is None:
            continue
        records.append(record)
        if record.get("status") == "complete":
            completed += 1
        elif record.get("status") == "error":
            failed += 1
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    atomic_write_text(path, text)
    return completed, failed


def initialise_pipeline(config, checkpoint_path: Path, device, dtype, offload_mode):
    from demo_utils import memory as memory_utils

    memory_utils.gpu = device
    from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
    from pipeline.causal_inference import CausalInferencePipeline

    pipeline = CausalInferencePipeline(config, device=device)
    generator_state = extract_generator_state(checkpoint_path)
    incompatible = pipeline.generator.load_state_dict(generator_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict generator load unexpectedly reported incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    del generator_state
    gc.collect()

    pipeline.eval().requires_grad_(False)
    pipeline.to(dtype=dtype)
    free_gib = get_cuda_free_memory_gb(device)
    offload_text = offload_mode == "on" or (
        offload_mode == "auto" and free_gib < 40.0
    )
    if offload_text:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    return pipeline, offload_text, free_gib


def build_contract(
    args: argparse.Namespace,
    config,
    paths: dict[str, Path],
    indices: list[int],
    latent_shape: list[int],
) -> dict[str, Any]:
    contract = {
        "schema_version": 1,
        "model": "Wan21 Action2V causal DMD",
        "checkpoint": file_identity(paths["checkpoint"]),
        "config": file_identity(paths["config"]),
        "prompts": file_identity(paths["prompts"]),
        "trajectories": file_identity(paths["trajectories"]),
        "iteration": args.iteration,
        "samples_per_iteration": args.samples_per_iteration,
        "selected_start": indices[0],
        "selected_end_exclusive": indices[-1] + 1,
        "selected_count": len(indices),
        "base_seed": args.base_seed,
        "fps": args.fps,
        "dtype": args.dtype,
        "latent_shape": latent_shape,
        "denoising_step_list": list(config.denoising_step_list),
        "warp_denoising_step": bool(config.warp_denoising_step),
        "timestep_shift": float(config.timestep_shift),
        "camera_intrinsics": {
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
        },
    }
    if args.shard_name:
        contract["shard_name"] = args.shard_name
    return contract


def main() -> None:
    args = parse_args()
    project_root = resolve_path(args.project_root, PROJECT_ROOT)
    paths = {
        "config": resolve_path(args.config, project_root),
        "checkpoint": resolve_path(args.checkpoint, project_root),
        "prompts": resolve_path(args.prompts, project_root),
        "trajectories": resolve_path(args.trajectories, project_root),
        "output_root": resolve_path(args.output_root, project_root),
    }
    if not paths["checkpoint"].is_file():
        raise FileNotFoundError(f"DMD checkpoint does not exist: {paths['checkpoint']}")
    config = load_config(paths["config"], project_root)
    prompts = read_nonempty_aligned_lines(paths["prompts"], "Prompt")
    trajectories = read_nonempty_aligned_lines(paths["trajectories"], "Trajectory")
    if len(prompts) != len(trajectories):
        raise ValueError(
            f"Prompt/trajectory row mismatch: {len(prompts)} vs {len(trajectories)}"
        )
    indices = select_iteration_indices(
        len(prompts),
        args.iteration,
        args.samples_per_iteration,
        args.limit,
        args.start_index,
        args.end_index,
    )
    latent_shape = [int(value) for value in config.image_or_video_shape]
    if len(latent_shape) != 5 or latent_shape[0] != 1:
        raise ValueError(f"Expected image_or_video_shape [1,F,C,H,W], got {latent_shape}")
    expected_frames = latent_shape[1]
    validate_selected_trajectories(trajectories, indices, expected_frames)
    contract = build_contract(args, config, paths, indices, latent_shape)
    run_signature = signature(contract)
    iteration_dir = paths["output_root"] / f"iteration_{args.iteration:03d}"
    run_state_dir = resolve_run_state_dir(iteration_dir, args.shard_name)

    print(
        f"Validated {len(prompts)} aligned pairs; iteration {args.iteration} selects "
        f"[{indices[0]}, {indices[-1] + 1}) ({len(indices)} samples)."
    )
    print(f"Checkpoint: {paths['checkpoint']}")
    print(f"Output: {iteration_dir}")
    if args.shard_name:
        print(f"Shard state: {run_state_dir}")
    print(f"Run signature: {run_signature}")
    if args.dry_run:
        print("Dry run complete; CUDA/model loading and output writes were skipped.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan Action2V DMD inference")

    os.chdir(project_root)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = dtype_from_name(args.dtype)

    videos_dir = iteration_dir / "videos"
    metadata_dir = iteration_dir / "metadata"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    run_state_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_state_dir / "run_config.json"
    previous_run = load_json(run_config_path)
    if (
        previous_run is not None
        and previous_run.get("run_signature") != run_signature
        and not args.overwrite
    ):
        raise RuntimeError(
            f"Output directory contains a different run contract: {run_config_path}. "
            "Use a different output root or pass --overwrite explicitly."
        )
    atomic_write_json(
        run_config_path,
        {
            **contract,
            "run_signature": run_signature,
            "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )

    print(
        f"CUDA logical device={device}, physical visibility="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')}, name={torch.cuda.get_device_name(device)}"
    )
    pipeline, text_offloaded, initial_free_gib = initialise_pipeline(
        config, paths["checkpoint"], device, dtype, args.text_encoder_offload
    )
    print(
        f"Model ready: dtype={args.dtype}, initial_free_vram={initial_free_gib:.2f} GiB, "
        f"text_encoder_offload={text_offloaded}"
    )

    from wan_utils.camera_trajectory import make_camera_tensors

    completed_now = 0
    skipped = 0
    errors = 0
    for position, index in enumerate(
        tqdm(indices, desc=f"RSI BiOPD iteration {args.iteration}", dynamic_ncols=True)
    ):
        video_path = videos_dir / f"{index:06d}.mp4"
        metadata_path = metadata_dir / f"{index:06d}.json"
        if not args.overwrite and completed_sample_matches(
            metadata_path, video_path, run_signature, index
        ):
            skipped += 1
            continue

        prompt = prompts[index]
        trajectory = trajectories[index]
        sample_seed = args.base_seed + index
        started_at = time.time()
        try:
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            viewmats, Ks = make_camera_tensors(
                trajectory,
                fx=args.fx,
                fy=args.fy,
                cx=args.cx,
                cy=args.cy,
                device=device,
                dtype=dtype,
            )
            noise = torch.randn(latent_shape, device=device, dtype=dtype)
            torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            with torch.inference_mode():
                video, latents = pipeline.inference(
                    noise=noise,
                    text_prompts=[prompt],
                    return_latents=True,
                    initial_latent=None,
                    viewmats=viewmats,
                    Ks=Ks,
                )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - inference_started
            save_video_atomic(video_path, video, args.fps)
            metadata = {
                "status": "complete",
                "run_signature": run_signature,
                "iteration": args.iteration,
                "iteration_offset": index - args.iteration * args.samples_per_iteration,
                "global_index": index,
                "prompt": prompt,
                "trajectory": trajectory,
                "seed": sample_seed,
                "video_path": str(video_path),
                "video_frames": int(video.shape[1]),
                "latent_shape": list(latents.shape),
                "elapsed_seconds": elapsed,
                "chunk0_latency_seconds": getattr(
                    pipeline, "last_chunk0_latency", None
                ),
                "completed_unix_time": time.time(),
            }
            atomic_write_json(metadata_path, metadata)
            completed_now += 1
            del video, latents, noise, viewmats, Ks
            pipeline.vae.model.clear_cache()
        except Exception as exc:
            errors += 1
            error_record = {
                "status": "error",
                "run_signature": run_signature,
                "iteration": args.iteration,
                "global_index": index,
                "prompt": prompt,
                "trajectory": trajectory,
                "seed": sample_seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_unix_time": time.time(),
                "elapsed_since_sample_start": time.time() - started_at,
            }
            atomic_write_json(metadata_path, error_record)
            try:
                pipeline.vae.model.clear_cache()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            if not args.continue_on_error:
                write_manifest(run_state_dir / "manifest.jsonl", metadata_dir, indices)
                raise

        if args.empty_cache_every > 0 and (position + 1) % args.empty_cache_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    completed_total, failed_total = write_manifest(
        run_state_dir / "manifest.jsonl", metadata_dir, indices
    )
    summary = {
        "run_signature": run_signature,
        "iteration": args.iteration,
        "selected_count": len(indices),
        "completed_now": completed_now,
        "skipped_existing": skipped,
        "errors_now": errors,
        "completed_total": completed_total,
        "failed_total": failed_total,
        "complete": completed_total == len(indices) and failed_total == 0,
        "updated_unix_time": time.time(),
    }
    atomic_write_json(run_state_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["complete"]:
        raise RuntimeError(
            f"Iteration finished with incomplete samples: {completed_total}/{len(indices)}"
        )


if __name__ == "__main__":
    main()
