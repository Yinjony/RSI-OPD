#!/usr/bin/env python3
"""Standalone attribute-conditioned LoRA SFT: python or torchrun."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]


def main():
    from omegaconf import OmegaConf
    from wan_utils.guider_sft_config import cache_contract, load_config
    from wan_utils.guider_sft_dataset import GuiderSFTDataset, digest, format_prompt, load_records, source_report, atomic_json
    from wan_utils.biopd_io import load_source, runtime_log
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--config_path", default=str(ROOT / "Wan21/configs/guider1.yaml"))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", help="Path to training_state.pt; same world size and training contract required")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--export-merged", action="store_true", help="Export the final adapter plus its actual base on CPU")
    parser.add_argument("--export-prompts", help="Export matching one-line prompts.txt and trajectories.txt to a new directory")
    args, extra = parser.parse_known_args()
    overrides = list(args.override)
    if args.resume:
        overrides.append("resume=" + str(Path(args.resume).resolve()))
    cfg = load_config(args.config, overrides)
    if cfg.get("initial_adapter") and not args.export_prompts:
        # Keep the standard launcher usable for warm-starting the original A/B.
        # The continuation entry parses its own flags, including --check-init and
        # --export-merged, before constructing any GPU model or optimizer.
        from wan_train_guider_sft_continue_lora import main as continue_main
        return continue_main()
    if extra:
        parser.error("unrecognized arguments: " + " ".join(extra))
    records = load_records(cfg.annotation_path, OmegaConf.to_container(cfg.camera))
    if args.export_prompts:
        dest = Path(args.export_prompts).resolve()
        dest.mkdir(parents=True, exist_ok=False)
        texts = [format_prompt(r["prompt"], r["positive_attributes"], r["negative_attributes"]) for r in records]
        # Wan's whitespace-clean tokenizer treats these identically to the multiline template.
        (dest / "prompts.txt").write_text("".join(" ".join(t.split()) + "\n" for t in texts))
        (dest / "trajectories.txt").write_text("".join(r["action"] + "\n" for r in records))
        atomic_json(dict(sample_ids=[r["sample_id"] for r in records], camera=OmegaConf.to_container(cfg.camera)), dest / "metadata.json")
        print(f"Exported {len(records)} aligned prompt/action pairs to {dest}")
        return
    if args.check_data:
        print(json.dumps(source_report(records), ensure_ascii=False, indent=2))
        return
    train = GuiderSFTDataset(cfg.cache_dir, "train")
    val = GuiderSFTDataset(cfg.cache_dir, "val")
    if train.manifest["signature"] != digest(cache_contract(cfg, records)):
        raise ValueError("Cache does not match current annotations/video files/encoders/config; preencode to a new cache_dir")
    if args.check_cache:
        print(json.dumps(dict(train=len(train), val=len(val), report=train.manifest["report"],
                              cache_sha256=train.sha256), ensure_ascii=False, indent=2))
        return
    os.chdir(ROOT)
    # Import the dedicated trainer without triggering wan_trainer/__init__.py's optional algorithms.
    module = load_source("guider_sft_trainer", ROOT / "Wan21/wan_trainer/camera_guider_sft.py")
    with runtime_log(cfg.output_dir, int(os.environ.get("RANK", 0))):
        try:
            module.Trainer(cfg, train, val).train()
        finally:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        if args.export_merged and int(os.environ.get("RANK", 0)) == 0:
            from wan_train_guider_sft_continue_lora import export_final_checkpoint
            export_final_checkpoint(cfg, [])


if __name__ == "__main__":
    main()
