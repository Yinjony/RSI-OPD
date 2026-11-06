#!/usr/bin/env python3
"""Validate/encode guider_sft.json to resumable local per-sample caches."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]

import torch
from omegaconf import OmegaConf
from wan_utils.guider_sft_config import cache_contract, load_config
from wan_utils.guider_sft_dataset import (
    FORMAT, atomic_json, atomic_save, condition_texts, digest, file_sha256, load_records,
    read_video, source_report, split_records, validate_cached_sample, validate_token_lengths,
)
from wan_utils.camera_trajectory import make_camera_tensors


def preencode(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("Preencoding requires CUDA plus the Wan dependencies; --check-data runs on CPU")
    torch.cuda.set_device(0)  # select physical GPU with CUDA_VISIBLE_DEVICES
    device = torch.device("cuda", 0)
    records = load_records(cfg.annotation_path, OmegaConf.to_container(cfg.camera))
    contract = cache_contract(cfg, records)
    signature = digest(contract)
    root = Path(cfg.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Cache configuration/source changed. Select a new cache_dir.")
    atomic_json(contract, contract_path)
    names = [digest(r["sample_id"])[:24] for r in records]
    shape = (20, 16, 60, 104)
    pending = []
    for i, name in enumerate(names):
        p = root / f"{name}.pt"
        if p.exists():
            sample = torch.load(p, map_location="cpu", weights_only=True)
            validate_cached_sample(sample, signature, shape, cfg.validation_controls, cfg.attribute_dropout)
            if sample["sample_id"] != records[i]["sample_id"]:
                raise ValueError("Cached sample_id mismatch")
        else:
            pending.append(i)

    # Check every text BEFORE spending GPU time on VAE or T5.
    from wan.modules.tokenizers import HuggingfaceTokenizer
    tokenizer = HuggingfaceTokenizer(str(Path(cfg.model_path) / "google/umt5-xxl"), seq_len=512, clean="whitespace")
    counts = []
    for i, record in enumerate(records):
        try:
            counts.append(validate_token_lengths(tokenizer, condition_texts(
                records, i, cfg.validation_controls, cfg.attribute_dropout)))
        except ValueError as exc:
            raise ValueError(f"{record['sample_id']}: {exc}") from exc
    del tokenizer

    from wan_utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
    need_latents = [i for i in pending if not (root / f"{names[i]}.latent.pt").exists()]
    if need_latents:
        vae = WanVAEWrapper(model_path=cfg.model_path).to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        for i in need_latents:
            r = records[i]
            video = read_video(r["video_path"], cfg.height, cfg.width, cfg.num_frames)
            with torch.no_grad():
                latent = vae.encode_to_latent(video[None].to(device, dtype=torch.bfloat16))[0].cpu().half()
            if tuple(latent.shape) != shape or not torch.isfinite(latent).all():
                raise ValueError(f"Invalid VAE latent: {r['sample_id']} {latent.shape}")
            atomic_save(dict(signature=signature, clean_latent=latent), root / f"{names[i]}.latent.pt")
            print(f"[VAE] {i + 1}/{len(records)} {r['sample_id']}", flush=True)
        del vae, video, latent
        gc.collect()
        torch.cuda.empty_cache()

    if pending:
        encoder = WanTextEncoder(model_path=cfg.model_path).to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        for i in pending:
            r = records[i]
            texts = condition_texts(records, i, cfg.validation_controls, cfg.attribute_dropout)
            latent = torch.load(root / f"{names[i]}.latent.pt", map_location="cpu", weights_only=True)
            if latent["signature"] != signature:
                raise ValueError("Stale partial latent cache")
            embeddings = {}
            with torch.no_grad():
                for name, text in texts.items():
                    embeddings[name] = encoder([text])["prompt_embeds"][0].cpu().to(torch.bfloat16)
            vm, ks = make_camera_tensors(r["action"], **OmegaConf.to_container(cfg.camera))
            sample = dict(signature=signature, sample_id=r["sample_id"], clean_latent=latent["clean_latent"],
                          embeddings=embeddings, viewmats=vm[0], Ks=ks[0], texts=texts, token_counts=counts[i])
            validate_cached_sample(sample, signature, shape, cfg.validation_controls, cfg.attribute_dropout)
            atomic_save(sample, root / f"{names[i]}.pt")
            (root / f"{names[i]}.latent.pt").unlink()
            print(f"[T5] {i + 1}/{len(records)} {r['sample_id']}", flush=True)
        del encoder

    splits = split_records(records, cfg.val_fraction, cfg.split_seed)
    entries = [dict(**r, split=split, cache_file=f"{name}.pt", sha256=file_sha256(root / f"{name}.pt"),
                    token_counts=count) for r, split, name, count in zip(records, splits, names, counts)]
    report = source_report(records)
    report.update(train=splits.count("train"), val=splits.count("val"), errors=0)
    atomic_json(dict(format=FORMAT, signature=signature, contract=contract, records=entries, report=report), root / "manifest.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "Wan21/configs/guider1.yaml"))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--check-data", action="store_true", help="Validate source JSON/actions/paths without models")
    parser.add_argument("--decode-videos", action="store_true", help="With --check-data, decode every video on CPU")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override, data_only=True)
    if args.check_data:
        records = load_records(cfg.annotation_path, OmegaConf.to_container(cfg.camera))
        report = source_report(records)
        report["example_texts"] = condition_texts(records, 0, False, cfg.attribute_dropout)
        report["splits"] = {s: split_records(records, cfg.val_fraction, cfg.split_seed).count(s) for s in ("train", "val")}
        if args.decode_videos:
            for r in records:
                read_video(r["video_path"], cfg.height, cfg.width, cfg.num_frames)
            report["decoded_videos"] = len(records)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        preencode(cfg)


if __name__ == "__main__":
    main()
