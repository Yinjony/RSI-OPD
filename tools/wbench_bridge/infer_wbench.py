#!/usr/bin/env python3
"""Generate WBench-compatible videos for minWM from a bridge manifest."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torchvision.io import write_video

BRIDGE_DIR = Path(__file__).resolve().parent
MINWM_ROOT = BRIDGE_DIR.parents[1]
WAN21_ROOT = MINWM_ROOT / "Wan21"
WBENCH_WORKDIR_DEFAULT = Path("/root/data/wbench_handoff/WBench/work_dirs/minwm")
WBENCH_DATA_ROOT = Path("/root/data/wbench_handoff")
WBENCH_DATA_ROOT_LEGACY = Path("/root/wbench_handoff")
sys.path.insert(0, str(BRIDGE_DIR))
sys.path.insert(0, str(WAN21_ROOT))
sys.path.insert(0, str(MINWM_ROOT / "shared"))

from case_convert import (  # noqa: E402
    convert_case,
    load_wbench_case,
    merge_turns_to_single_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minWM on WBench bridge manifest")
    parser.add_argument(
        "--manifest",
        default=str(BRIDGE_DIR / "manifests" / "wbench_nav_minwm.jsonl"),
        help="JSONL manifest produced by convert_cases.py",
    )
    parser.add_argument(
        "--case-id",
        default="",
        help="Generate only one case id from the manifest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of cases",
    )
    parser.add_argument(
        "--project-root",
        default=str(MINWM_ROOT),
        help="minWM project root",
    )
    parser.add_argument(
        "--config",
        default=str(WAN21_ROOT / "configs" / "causal_forcing_dmd_camera.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(MINWM_ROOT / "ckpts/Wan21/Action2V/dmd/model.pt"),
    )
    parser.add_argument(
        "--output-root",
        default=str(WBENCH_WORKDIR_DEFAULT),
        help="WBench model work dir; videos go to <output-root>/videos/",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--text-encoder-offload",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=0.5050505,
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=0.89786756,
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--no-i2v",
        action="store_true",
        help="Disable image conditioning (not recommended for WBench)",
    )
    parser.add_argument(
        "--merged-navigation",
        action="store_true",
        help="Merge all navigation turns into one 20-frame clip (I2V once per case)",
    )
    parser.add_argument(
        "--camera-continuation",
        action="store_true",
        help=(
            "WBench camera-controlled protocol: I2V with initial image on turn 1, "
            "then each next turn conditions on the previous turn's last RGB frame "
            "plus that turn's camera trajectory"
        ),
    )
    parser.add_argument(
        "--t2v-after-first",
        action="store_true",
        help=(
            "With --camera-continuation, use native T2V (no image conditioning) "
            "from turn 2 onward; only turn 1 stays I2V"
        ),
    )
    parser.add_argument(
        "--freeze-initial-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replace the static-camera prefix with the exact input image to avoid "
            "VAE round-trip flicker at the start of I2V clips"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load manifest and print planned outputs without CUDA inference",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def resolve_data_path(path: str | Path) -> Path:
    """Prefer data-disk paths when manifests still reference the legacy root."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    legacy_root = WBENCH_DATA_ROOT_LEGACY
    modern_root = WBENCH_DATA_ROOT
    try:
        relative = candidate.relative_to(legacy_root)
    except ValueError:
        return candidate
    remapped = modern_root / relative
    return remapped if remapped.is_file() else candidate


def align_i2v_for_block(
    initial_latent: torch.Tensor,
    expected_latent_frames: int,
    num_frame_per_block: int,
) -> tuple[torch.Tensor, int]:
    """Pad I2V conditioning so noise length is divisible by num_frame_per_block."""
    block = max(int(num_frame_per_block), 1)
    input_frames = int(initial_latent.shape[1])
    if block == 1:
        noise_frames = expected_latent_frames - input_frames
        if noise_frames < 0:
            raise ValueError(
                f"I2V frame count mismatch: expected={expected_latent_frames}, "
                f"input={input_frames}, block={block}"
            )
        return initial_latent, noise_frames

    latent = initial_latent
    while input_frames < block:
        repeat = min(block - input_frames, 1)
        latent = torch.cat(
            [latent, latent[:, -1:].expand(-1, repeat, -1, -1, -1)],
            dim=1,
        )
        input_frames += repeat

    remainder = input_frames % block
    if remainder:
        pad = block - remainder
        latent = torch.cat(
            [latent, latent[:, -1:].expand(-1, pad, -1, -1, -1)],
            dim=1,
        )
        input_frames += pad

    noise_frames = expected_latent_frames - input_frames
    if noise_frames <= 0 or noise_frames % block != 0:
        raise ValueError(
            f"Cannot align I2V frames for block={block}: "
            f"expected={expected_latent_frames}, input={input_frames}, noise={noise_frames}"
        )
    return latent, noise_frames


def align_viewmats_for_i2v_padding(
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    num_input_frames: int,
    expected_latent_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the start pose fixed while I2V conditioning latents are block-padded."""
    if num_input_frames <= 1:
        return viewmats, ks

    tail_len = expected_latent_frames - num_input_frames
    if tail_len <= 0:
        raise ValueError(
            f"Invalid I2V viewmat split: expected={expected_latent_frames}, "
            f"input={num_input_frames}"
        )

    vm0 = viewmats[:, :1]
    ks0 = ks[:, :1]
    viewmats = torch.cat(
        [
            vm0.expand(-1, num_input_frames, -1, -1),
            viewmats[:, 1 : 1 + tail_len],
        ],
        dim=1,
    )
    ks = torch.cat(
        [
            ks0.expand(-1, num_input_frames, -1, -1),
            ks[:, 1 : 1 + tail_len],
        ],
        dim=1,
    )
    if int(viewmats.shape[1]) != expected_latent_frames:
        raise ValueError(
            f"Aligned viewmats length {viewmats.shape[1]} != {expected_latent_frames}"
        )
    return viewmats, ks


def load_image_pixel_tensor(image_path: Path) -> torch.Tensor:
    """Load one RGB frame as (C, H, W) in [0, 1] for video compositing."""
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def freeze_initial_image_hold(
    video: torch.Tensor,
    image_path: Path,
    conditioning_latent_frames: int,
    total_latent_frames: int,
    crossfade_frames: int = 3,
) -> tuple[torch.Tensor, int]:
    """Hold the input image over the static-camera I2V prefix, then crossfade."""
    if conditioning_latent_frames <= 0 or total_latent_frames <= 0:
        return video, 0

    pixel_frames = int(video.shape[1])
    hold_pixels = max(
        1,
        int(round(pixel_frames * conditioning_latent_frames / total_latent_frames)),
    )
    hold_pixels = min(hold_pixels, pixel_frames)

    image = load_image_pixel_tensor(image_path).to(device=video.device, dtype=video.dtype)
    frozen = video.clone()
    frozen[0, :hold_pixels] = image.unsqueeze(0).expand(hold_pixels, -1, -1, -1)

    fade_len = min(crossfade_frames, pixel_frames - hold_pixels)
    if fade_len > 0:
        for offset in range(fade_len):
            alpha = (offset + 1) / (fade_len + 1)
            index = hold_pixels + offset
            frozen[0, index] = (1.0 - alpha) * image + alpha * video[0, index]

    return frozen, hold_pixels


def load_image_tensor(image_path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    from PIL import Image
    from torchvision import transforms

    # Match Wan21 I2V preprocessing (480x832, normalized to [-1, 1]).
    transform = transforms.Compose(
        [
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).unsqueeze(2)  # B,C,1,H,W
    return tensor.to(device=device, dtype=dtype)


def save_last_frame_image(video: torch.Tensor, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = (
        video[0, -1]
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    Image.fromarray(frame).save(path)


def save_video(path: Path, video: torch.Tensor, fps: int) -> None:
    frames = (
        video[0]
        .permute(0, 2, 3, 1)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.mp4")
    write_video(
        str(temporary),
        frames,
        fps=fps,
        video_codec="libx264",
        options={"crf": "18"},
    )
    os.replace(temporary, path)


def generate_case_video(
    sample: dict[str, Any],
    pipeline,
    device: torch.device,
    dtype: torch.dtype,
    latent_shape: list[int],
    fps: int,
    base_seed: int,
    use_i2v: bool,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    merged_navigation: bool = False,
    camera_continuation: bool = False,
    t2v_after_first: bool = False,
    freeze_initial_hold: bool = True,
) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
    from wan_utils.camera_trajectory import make_camera_tensors

    prompt = sample["prompt"]
    expected_latent_frames = int(sample.get("expected_latent_frames", latent_shape[1]))
    turn_frames: list[int] = [0]
    all_chunks: list[torch.Tensor] = []

    num_frame_per_block = int(getattr(pipeline, "num_frame_per_block", 1))
    current_image_path = resolve_data_path(sample["image_path"])
    current_initial_latent: torch.Tensor | None = None
    turn_specs = sample["turns"]
    if merged_navigation:
        merged_trajectory = sample.get("merged_trajectory") or merge_turns_to_single_trajectory(
            turn_specs,
            expected_latent_frames,
        )
        turn_specs = [
            {
                "turn": 1,
                "action": "merged",
                "trajectory": merged_trajectory,
            }
        ]

    temp_dir = Path(sample.get("_temp_dir", "/tmp"))
    hold_info: dict[str, Any] = {"freeze_initial_hold": freeze_initial_hold}
    for turn_index, turn in enumerate(turn_specs):
        trajectory = turn["trajectory"]
        viewmats, ks = make_camera_tensors(
            trajectory,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            device=device,
            dtype=dtype,
        )

        seed = base_seed + int(sample["case_id"]) * 100 + int(turn["turn"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        initial_latent = None
        noise_frames = expected_latent_frames
        use_i2v_this_turn = use_i2v and not (
            camera_continuation and t2v_after_first and turn_index > 0
        )
        if use_i2v_this_turn:
            if camera_continuation or current_initial_latent is None:
                image_tensor = load_image_tensor(current_image_path, device, dtype)
                conditioning_latent = pipeline.vae.encode_to_latent(image_tensor).to(
                    device=device, dtype=dtype
                )
            else:
                conditioning_latent = current_initial_latent
            initial_latent, noise_frames = align_i2v_for_block(
                conditioning_latent,
                expected_latent_frames,
                num_frame_per_block,
            )
            viewmats, ks = align_viewmats_for_i2v_padding(
                viewmats,
                ks,
                int(initial_latent.shape[1]),
                expected_latent_frames,
            )

        noise = torch.randn(
            [1, noise_frames, latent_shape[2], latent_shape[3], latent_shape[4]],
            device=device,
            dtype=dtype,
        )

        with torch.inference_mode():
            video, latents = pipeline.inference(
                noise=noise,
                text_prompts=[prompt],
                return_latents=True,
                initial_latent=initial_latent,
                viewmats=viewmats,
                Ks=ks,
            )

        chunk = video.detach()
        if (
            freeze_initial_hold
            and use_i2v_this_turn
            and initial_latent is not None
            and turn_index == 0
        ):
            chunk, hold_pixels = freeze_initial_image_hold(
                chunk,
                current_image_path,
                int(initial_latent.shape[1]),
                expected_latent_frames,
            )
            hold_info["hold_pixels"] = hold_pixels
            hold_info["conditioning_latent_frames"] = int(initial_latent.shape[1])

        all_chunks.append(chunk)
        turn_frames.append(turn_frames[-1] + int(video.shape[1]))

        if use_i2v and not merged_navigation:
            current_initial_latent = latents[:, -1:].detach().to(device=device, dtype=dtype)

        pipeline.vae.model.clear_cache()

    combined = torch.cat(all_chunks, dim=1)
    return combined, turn_frames, hold_info


def write_turns_json(output_root: Path, case_id: str, fps: int, turn_frames: list[int]) -> None:
    turns_path = output_root / "turns.json"
    lock_path = output_root / ".turns.json.lock"
    output_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        payload: dict[str, Any]
        if turns_path.is_file():
            with turns_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = {"fps": fps, "cases": {}}

        payload.setdefault("fps", fps)
        payload.setdefault("cases", {})
        payload["cases"][str(case_id)] = {"turn_frames": turn_frames}
        with turns_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    project_root = Path(args.project_root)
    output_root = Path(args.output_root)
    videos_dir = output_root / "videos"
    metadata_dir = output_root / "wbench_bridge_metadata"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    samples = read_manifest(manifest_path)
    if args.case_id:
        samples = [sample for sample in samples if str(sample["case_id"]) == str(args.case_id)]
    if args.limit > 0:
        samples = samples[: args.limit]

    if not samples:
        raise SystemExit("No manifest samples selected.")

    os.chdir(project_root)

    from infer_rsi_biopd import dtype_from_name, initialise_pipeline, load_config

    config = load_config(Path(args.config), project_root)
    latent_shape = [int(value) for value in config.image_or_video_shape]

    if args.dry_run:
        for sample in samples:
            out_video = videos_dir / f"case_{sample['case_id']}_combined.mp4"
            print(
                f"[dry-run] case_{sample['case_id']}: "
                f"{len(sample['turns'])} turns -> {out_video}"
            )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for minWM WBench inference")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = dtype_from_name(args.dtype)
    pipeline, _, _ = initialise_pipeline(
        config,
        Path(args.checkpoint),
        device,
        dtype,
        args.text_encoder_offload,
    )

    completed = 0
    skipped = 0
    for sample in samples:
        case_id = str(sample["case_id"])
        out_video = videos_dir / f"case_{case_id}_combined.mp4"
        meta_path = metadata_dir / f"case_{case_id}.json"
        if out_video.is_file() and not args.overwrite:
            skipped += 1
            continue

        sample = dict(sample)
        sample["_temp_dir"] = str(metadata_dir / "temp_frames")
        started = time.time()
        try:
            video, turn_frames, hold_info = generate_case_video(
                sample,
                pipeline,
                device,
                dtype,
                latent_shape,
                args.fps,
                args.base_seed,
                not args.no_i2v,
                args.fx,
                args.fy,
                args.cx,
                args.cy,
                args.merged_navigation,
                freeze_initial_hold=args.freeze_initial_hold,
            )
            save_video(out_video, video, args.fps)
            write_turns_json(output_root, case_id, args.fps, turn_frames)
            meta = {
                "status": "complete",
                "case_id": case_id,
                "video_path": str(out_video),
                "turn_frames": turn_frames,
                "num_frames": int(video.shape[1]),
                "generation_mode": (
                    "merged_single_clip" if args.merged_navigation else "multi_turn_i2v"
                ),
                "elapsed_seconds": time.time() - started,
                **hold_info,
            }
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            completed += 1
            print(f"[ok] case_{case_id} -> {out_video} ({video.shape[1]} frames)")
        except Exception as exc:
            error = {
                "status": "error",
                "case_id": case_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(error, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(f"[error] case_{case_id}: {exc}")

    print(f"Done. completed={completed}, skipped={skipped}, total={len(samples)}")


if __name__ == "__main__":
    main()
