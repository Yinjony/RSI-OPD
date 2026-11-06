#!/usr/bin/env python3
"""Generate the first N cached validation samples with single-GPU DMD CFG."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]

from infer_rsi_biopd import (  # reuse the existing DMD loader and atomic output helpers
    atomic_write_json, completed_sample_matches, dtype_from_name,
    file_identity, initialise_pipeline, load_json, save_video_atomic, signature,
)
from wan_utils.camera_trajectory import make_camera_tensors
from wan_utils.guider_sft_config import load_config as load_sft_config
from wan_utils.guider_sft_dataset import (
    FORMAT, attribute_names, digest, file_sha256, format_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", choices=("sft", "base"), default="sft")
    parser.add_argument("--sft-config", default=str(ROOT / "Wan21/configs/guider1.yaml"))
    parser.add_argument("--inference-config", default=str(ROOT / "Wan21/configs/_base/inference.yaml"))
    parser.add_argument("--adapter", help="Required for --model-source=sft")
    parser.add_argument("--base", default="/root/data/mcy/timeOPD/minWM/ckpts/Wan21/Action2V/dmd/model.pt")
    parser.add_argument("--merged-checkpoint", help="Required for --model-source=sft")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--no-cfg", action="store_true",
                        help="Use only the positive branch; ignore guidance-scale and record it as 1.0")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--text-encoder-offload", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def conditions(record):
    positive = format_prompt(record["prompt"], record["positive_attributes"], [])
    negatives = attribute_names(record["negative_attributes"])
    negative = "Observed negative video attributes:\n" + "; ".join(negatives) + "." if negatives else ""
    return positive, negative


def select_validation(sft_config, count):
    cache_dir = Path(sft_config.cache_dir)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT or manifest.get("signature") != digest(manifest["contract"]):
        raise ValueError("Invalid SFT cache manifest")
    contract = manifest["contract"]
    if contract["annotation_sha256"] != file_sha256(sft_config.annotation_path):
        raise ValueError("SFT JSON differs from the validation cache source")
    for key in ("val_fraction", "split_seed", "num_frames", "height", "width", "camera"):
        current = OmegaConf.to_container(sft_config.camera) if key == "camera" else sft_config[key]
        if contract[key] != current:
            raise ValueError(f"SFT config {key} differs from the validation cache")
    selected = [(i, record) for i, record in enumerate(manifest["records"]) if record["split"] == "val"]
    if not 1 <= count <= len(selected):
        raise ValueError(f"num-samples must be in [1, {len(selected)}]")
    return manifest_path, selected[:count], len(selected)


def check_adapter(adapter_path, base_path, sft_config):
    payload = torch.load(adapter_path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("format") != "wan_guider_sft_lora_v1":
        raise ValueError("Expected a guider SFT LoRA adapter")
    base = file_identity(base_path)
    if any(payload["base_identity"][key] != base[key] for key in ("path", "size", "mtime_ns")):
        raise ValueError("Base checkpoint is not the one recorded in the adapter")
    if payload.get("rank") != sft_config.lora_rank or payload.get("alpha") != sft_config.lora_alpha:
        raise ValueError("Adapter LoRA rank/alpha differs from the SFT config")
    if payload.get("model_path") != sft_config.model_path or list(payload.get("targets", [])) != list(sft_config.lora_target_modules):
        raise ValueError("Adapter model path or target layers differ from the SFT config")
    del payload


def ensure_merged_checkpoint(adapter_path, base_path, merged_path):
    """Merge once in a child process so the 6 GB CPU state is released before inference."""
    provenance_path = merged_path.with_name(merged_path.name + ".merge.json")
    sources = dict(adapter=file_identity(adapter_path), base=file_identity(base_path))
    if merged_path.exists():
        provenance = load_json(provenance_path)
        if provenance is None or provenance.get("sources") != sources or provenance.get("merged") != file_identity(merged_path):
            raise ValueError(f"Cannot verify existing merged checkpoint: {merged_path}; use a new path")
        return
    if provenance_path.exists():
        raise ValueError(f"Merge provenance exists without checkpoint: {provenance_path}")
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "Wan21/merge_guider_sft_lora.py"),
                    "--adapter", str(adapter_path), "--base", str(base_path),
                    "--output", str(merged_path)], cwd=ROOT, check=True)
    atomic_write_json(provenance_path, dict(sources=sources, merged=file_identity(merged_path)))


def main():
    args = parse_args()
    if args.no_cfg:
        args.guidance_scale = 1.0
    elif not math.isfinite(args.guidance_scale) or args.guidance_scale <= 1:
        raise ValueError("CFG guidance-scale must be finite and > 1; use --no-cfg to disable CFG")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.model_source == "sft" and (not args.adapter or not args.merged_checkpoint):
        raise ValueError("SFT inference requires --adapter and --merged-checkpoint")
    if args.model_source == "base" and (args.adapter or args.merged_checkpoint):
        raise ValueError("Base inference must not receive an adapter or merged checkpoint")
    paths = {key: Path(getattr(args, key)).expanduser().resolve() for key in
             ("sft_config", "inference_config", "base", "output_dir")}
    if args.model_source == "sft":
        paths["adapter"] = Path(args.adapter).expanduser().resolve()
        paths["merged_checkpoint"] = Path(args.merged_checkpoint).expanduser().resolve()
    sft_config = load_sft_config(paths["sft_config"], data_only=True)
    manifest_path, selected, total_val = select_validation(sft_config, args.num_samples)
    if args.model_source == "sft":
        check_adapter(paths["adapter"], paths["base"], sft_config)

    from infer_rsi_biopd import load_config
    inference_config = load_config(paths["inference_config"], ROOT)
    latent_shape = [int(v) for v in inference_config.image_or_video_shape]
    if latent_shape != [1, 20, 16, 60, 104] or int(inference_config.num_frame_per_block) != sft_config.num_frame_per_block:
        raise ValueError("Inference latent shape/block size differs from the SFT model")
    if float(inference_config.timestep_shift) != float(sft_config.timestep_shift):
        raise ValueError("Inference timestep_shift differs from the SFT model")
    if bool(inference_config.independent_first_frame):
        raise ValueError("SFT inference requires independent_first_frame=false")
    for _, record in selected:
        if len(make_camera_tensors(record["action"])[0][0]) != latent_shape[1]:
            raise ValueError(f"Invalid action length: {record['sample_id']}")
        conditions(record)

    print(f"Validation: first {len(selected)} of {total_val}; CFG scale={args.guidance_scale}; model source={args.model_source}")
    if args.no_cfg:
        print("CFG disabled: positive branch only; no negative text branch; effective scale=1.0")
    print("Model input:", paths["base"] if args.model_source == "base" else paths["adapter"])
    print("Output directory:", paths["output_dir"])
    print("First selected sample:", selected[0][1]["sample_id"])
    if args.dry_run:
        print("Dry run complete: no model loaded and no videos written")
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU, for example CUDA_VISIBLE_DEVICES=0")
    if args.model_source == "sft":
        ensure_merged_checkpoint(paths["adapter"], paths["base"], paths["merged_checkpoint"])
        checkpoint_path = paths["merged_checkpoint"]
        # Preserve the existing SFT run signature so interrupted jobs remain resumable.
        run_contract = dict(schema_version=1, mode="validation_cfg", selected_sample_ids=[r["sample_id"] for _, r in selected],
                            sft_config=file_identity(paths["sft_config"]), manifest=file_identity(manifest_path),
                            inference_config=file_identity(paths["inference_config"]),
                            adapter=file_identity(paths["adapter"]), base=file_identity(paths["base"]),
                            merged_checkpoint=file_identity(paths["merged_checkpoint"]),
                            guidance_scale=args.guidance_scale, base_seed=args.base_seed,
                            fps=args.fps, dtype=args.dtype)
    else:
        checkpoint_path = paths["base"]
        run_contract = dict(schema_version=1, mode="validation_cfg_base", selected_sample_ids=[r["sample_id"] for _, r in selected],
                            sft_config=file_identity(paths["sft_config"]), manifest=file_identity(manifest_path),
                            inference_config=file_identity(paths["inference_config"]),
                            base=file_identity(paths["base"]), guidance_scale=args.guidance_scale,
                            base_seed=args.base_seed, fps=args.fps, dtype=args.dtype)
    if args.no_cfg:
        run_contract["mode"] = "validation_no_cfg" if args.model_source == "sft" else "validation_no_cfg_base"
        run_contract["cfg_enabled"] = False
    run_signature = signature(run_contract)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = load_json(output_dir / "run_config.json")
    if previous is not None and previous.get("run_signature") != run_signature:
        raise RuntimeError("Output directory belongs to a different run; use a new --output-dir")
    atomic_write_json(output_dir / "run_config.json", dict(**run_contract, run_signature=run_signature))
    (output_dir / "videos").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    os.chdir(ROOT)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    dtype = dtype_from_name(args.dtype)
    pipeline, offloaded, free_gib = initialise_pipeline(
        inference_config, checkpoint_path, torch.device("cuda:0"), dtype, args.text_encoder_offload)
    print(f"Model ready: free VRAM={free_gib:.2f} GiB, text encoder offloaded={offloaded}")
    complete = skipped = errors = 0
    for val_index, (source_index, record) in enumerate(tqdm(selected, desc="Validation (no CFG)" if args.no_cfg else "Validation CFG")):
        video_path = output_dir / "videos" / f"{val_index:06d}.mp4"
        metadata_path = output_dir / "metadata" / f"{val_index:06d}.json"
        if completed_sample_matches(metadata_path, video_path, run_signature, val_index):
            skipped += 1
            continue
        positive, negative = conditions(record)
        if args.no_cfg:
            negative = ""
        seed = args.base_seed + source_index
        started = time.time()
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # PRoPE einsum runs without autocast here, so camera matrices must match Q/K/V dtype.
            viewmats, Ks = make_camera_tensors(record["action"], **OmegaConf.to_container(sft_config.camera),
                                                device="cuda:0", dtype=dtype)
            noise = torch.randn(latent_shape, device="cuda:0", dtype=dtype)
            with torch.inference_mode():
                video, latents = pipeline.inference(
                    noise=noise, text_prompts=[positive], negative_text_prompts=None if args.no_cfg else [negative],
                    guidance_scale=args.guidance_scale, return_latents=True,
                    viewmats=viewmats, Ks=Ks)
            save_video_atomic(video_path, video, args.fps)
            atomic_write_json(metadata_path, dict(
                status="complete", run_signature=run_signature, global_index=val_index,
                source_index=source_index, sample_id=record["sample_id"],
                reference_video_path=record["video_path"],
                prompt=record["prompt"], positive_attributes=record["positive_attributes"],
                negative_attributes=record["negative_attributes"],
                positive_text=positive, negative_text=negative, action=record["action"],
                guidance_scale=args.guidance_scale, seed=seed, video_path=str(video_path),
                video_frames=int(video.shape[1]), elapsed_seconds=time.time() - started,
                **({"cfg_enabled": False} if args.no_cfg else {})))
            complete += 1
            pipeline.vae.model.clear_cache()
            del video, latents, noise, viewmats, Ks
        except Exception as exc:
            errors += 1
            atomic_write_json(metadata_path, dict(status="error", run_signature=run_signature,
                                                  global_index=val_index, sample_id=record["sample_id"],
                                                  error=str(exc), traceback=traceback.format_exc()))
            pipeline.vae.model.clear_cache()
            gc.collect()
            torch.cuda.empty_cache()
            if not args.continue_on_error:
                raise
    summary = dict(selected=len(selected), completed_now=complete, skipped=skipped, errors=errors,
                   total_complete=complete + skipped, run_signature=run_signature)
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError(f"{errors} validation samples failed")


if __name__ == "__main__":
    main()
