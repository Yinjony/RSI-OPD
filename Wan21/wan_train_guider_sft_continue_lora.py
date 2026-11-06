#!/usr/bin/env python3
"""Start a new SFT phase from existing LoRA A/B weights, with an optional exact resume."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]


def export_final_checkpoint(cfg, initial_identities):
    """Export the final trained adapter, retaining the original base and source adapters."""
    import torch
    from wan_utils.biopd_io import load_generator_state
    from wan_utils.guider_sft_dataset import atomic_json, atomic_save, file_identity
    from wan_utils.guider_sft_lora_init import validate_initial_adapter
    from wan_utils.guider_sft_training import merge_adapter_state
    latest = json.loads((Path(cfg.output_dir) / "latest.json").read_text())
    adapter_path = Path(latest["adapter"])
    destination = adapter_path.with_name("model.pt")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite merged model: {destination}")
    final_payload = torch.load(adapter_path, map_location="cpu", weights_only=True, mmap=True)
    validate_initial_adapter(final_payload, cfg)
    print(f"Exporting final guider on CPU: {destination}", flush=True)
    merged = merge_adapter_state(load_generator_state(cfg.generator_ckpt), final_payload)
    atomic_save(dict(generator=merged), destination)
    atomic_json(dict(sources=dict(adapter=file_identity(adapter_path),
                                  base=file_identity(cfg.generator_ckpt),
                                  initial_adapter=initial_identities),
                     merged=file_identity(destination)), destination.with_suffix(".pt.merge.json"))
    print(f"Exported {len(merged)} tensors to {destination}", flush=True)
    return destination


def main():
    from omegaconf import OmegaConf
    from wan_utils.biopd_io import load_source, runtime_log
    from wan_utils.guider_sft_config import cache_contract, load_config
    from wan_utils.guider_sft_dataset import GuiderSFTDataset, digest, file_sha256, load_records, source_report
    from wan_utils.guider_sft_lora_init import (
        read_initial_adapter, restore_initial_adapter, restore_initial_reference, validate_initial_adapter,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--config_path", required=True)
    parser.add_argument("--init-adapter", help="Existing SFT adapter.pt or BiOPD adapter directory; overrides config.initial_adapter")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", help="Only a training_state.pt from this continuation run")
    parser.add_argument("--check-init", action="store_true", help="CPU metadata/adapter/data-split checks; no model or GPU initialization")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--export-merged", action="store_true",
                        help="After training, export final adapter + its original base to checkpoint_*/model.pt on CPU")
    args = parser.parse_args()
    overrides = list(args.override)
    if args.resume:
        overrides.append("resume=" + str(Path(args.resume).resolve()))
    cfg = load_config(args.config, overrides)
    source_value = args.init_adapter or cfg.get("initial_adapter")
    if not source_value:
        raise ValueError("Set --init-adapter or config.initial_adapter")
    source = Path(source_value).expanduser()
    source = (ROOT / source).resolve() if not source.is_absolute() else source.resolve()
    payload, identities = read_initial_adapter(source)
    validate_initial_adapter(payload, cfg)
    cfg.initial_adapter = str(source)
    cfg.initial_adapter_identity = identities
    if args.check_init:
        manifest = json.loads((Path(cfg.cache_dir) / "manifest.json").read_text())
        if manifest["contract"]["annotation_sha256"] != file_sha256(cfg.annotation_path):
            raise ValueError("Annotation file differs from the selected cache")
        train_samples = sum(r["split"] == "train" for r in manifest["records"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        effective_batch = world_size * cfg.batch_size * cfg.grad_accum_steps
        print(json.dumps(dict(initialization="continue_existing_lora_weights", initial_adapter=str(source),
                              base_identity=payload["base_identity"], rank=payload["rank"], alpha=payload["alpha"],
                              adapter_tensors=len(payload["adapter"]), new_optimizer=not bool(cfg.resume),
                              phase_start_step="restored" if cfg.resume else 0, sft_loss_type=cfg.sft_loss_type,
                              anchor_reference="base plus initial adapter" if cfg.sft_loss_type == "supervised_anchor" else None,
                              train=train_samples,
                              val=sum(r["split"] == "val" for r in manifest["records"]),
                              world_size=world_size, effective_batch_size=effective_batch,
                              unpadded_train_passes=cfg.max_steps * effective_batch / train_samples,
                              grad_accum_steps=cfg.grad_accum_steps, max_steps=cfg.max_steps, output_dir=cfg.output_dir,
                              export_merged=args.export_merged),
                         ensure_ascii=False, indent=2))
        return
    records = load_records(cfg.annotation_path, OmegaConf.to_container(cfg.camera))
    if args.check_data:
        print(json.dumps(source_report(records), ensure_ascii=False, indent=2))
        return
    train, val = GuiderSFTDataset(cfg.cache_dir, "train"), GuiderSFTDataset(cfg.cache_dir, "val")
    if train.manifest["signature"] != digest(cache_contract(cfg, records)):
        raise ValueError("Cache does not match source data/encoder/config")
    if args.check_cache:
        print(json.dumps(dict(train=len(train), val=len(val), cache_sha256=train.sha256), indent=2))
        return
    os.chdir(ROOT)
    trainer_module = load_source("guider_sft_continue_trainer", ROOT / "Wan21/wan_trainer/camera_guider_sft.py")
    original_generator_loader = trainer_module.load_lora_generator
    original_reference_loader = trainer_module.load_frozen_reference

    def load_continued_generator(config, device):
        raw = restore_initial_adapter(original_generator_loader(config, device), payload, config)
        print(f"Restored existing LoRA A/B from {source}; only these LoRA weights are trainable", flush=True)
        return raw

    def load_continued_reference(config, device):
        reference = restore_initial_reference(original_reference_loader(config, device), payload, config)
        print(f"Frozen anchor reference is the phase-start model: original base + {source}; "
              "anchor gradients update only the current Guider2 LoRA", flush=True)
        return reference

    # Bind this dedicated trainer instance; the original training files stay unchanged.
    trainer_module.load_lora_generator = load_continued_generator
    trainer_module.load_frozen_reference = load_continued_reference
    with runtime_log(cfg.output_dir, int(os.environ.get("RANK", 0))):
        try:
            trainer_module.Trainer(cfg, train, val).train()
        finally:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        if args.export_merged and int(os.environ.get("RANK", 0)) == 0:
            export_final_checkpoint(cfg, identities)


if __name__ == "__main__":
    main()
