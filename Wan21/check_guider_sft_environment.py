#!/usr/bin/env python3
"""Read-only dependency and local checkpoint preflight for the entire SFT path."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "Wan21/configs/guider1.yaml"))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", help="Optional JSON audit artifact")
    args = parser.parse_args()
    report = dict(python=sys.executable, prefix=sys.prefix, imports={}, errors=[])
    names = ["torch", "torchvision", "numpy", "cv2", "omegaconf", "transformers", "diffusers", "ftfy",
             "safetensors", "flash_attn", "triton", "einops", "easydict", "wan_utils.wan_wrapper",
             "wan_utils.guider_sft_dataset", "wan_utils.guider_sft_training"]
    for name in names:
        try:
            mod = importlib.import_module(name)
            report["imports"][name] = dict(ok=True, version=getattr(mod, "__version__", None))
        except Exception as exc:
            # FlashAttention is optional: the existing attention backend has an SDPA fallback.
            report["imports"][name] = dict(ok=False, error=repr(exc), optional=name == "flash_attn")
            if name != "flash_attn":
                report["errors"].append(f"{name}: {exc}")
    try:
        import torch
        from omegaconf import OmegaConf
        from wan_utils.biopd_io import load_generator_state
        from wan_utils.guider_sft_config import load_config, cache_contract
        from wan_utils.guider_sft_dataset import load_records, source_report, condition_texts, validate_token_lengths
        from wan.modules.tokenizers import HuggingfaceTokenizer
        cfg = load_config(args.config, args.override)
        records = load_records(cfg.annotation_path, OmegaConf.to_container(cfg.camera))
        report["data"] = source_report(records)
        report["encoder_files"] = cache_contract(cfg, records)["model_files"]
        tokenizer = HuggingfaceTokenizer(str(Path(cfg.model_path) / "google/umt5-xxl"),
                                         seq_len=512, clean="whitespace", local_files_only=True)
        counts = [validate_token_lengths(tokenizer, condition_texts(
                      records, i, cfg.validation_controls, cfg.attribute_dropout))
                  for i in range(len(records))]
        report["text"] = dict(conditions=sum(len(c) for c in counts), max_tokens=max(max(c.values()) for c in counts))
        state = load_generator_state(cfg.generator_ckpt, "generator")
        targets = [(k, v) for k, v in state.items() if k.endswith(".weight") and v.ndim == 2
                   and k.split(".")[-2] in cfg.lora_target_modules]
        report["checkpoint"] = dict(tensors=len(state), lora_layers=len(targets),
                                     lora_parameters=sum(cfg.lora_rank * sum(v.shape) for _, v in targets))
        report["cuda"] = dict(available=torch.cuda.is_available(), version=torch.version.cuda,
                              nccl=torch.distributed.is_nccl_available(), gloo=torch.distributed.is_gloo_available())
        if torch.cuda.is_available():
            report["cuda"]["devices"] = [dict(index=i, name=torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
            report["cuda"]["bf16"] = torch.cuda.is_bf16_supported()
            if not report["cuda"]["bf16"]:
                report["errors"].append("BF16 not supported")
        else:
            report["errors"].append("CUDA unavailable for preencoding/training")
        if not report["cuda"]["nccl"]:
            report["errors"].append("NCCL unavailable for multi-GPU training")
    except Exception as exc:
        report["errors"].append(str(exc))
        report["traceback"] = traceback.format_exc()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered)
    return bool(report["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
