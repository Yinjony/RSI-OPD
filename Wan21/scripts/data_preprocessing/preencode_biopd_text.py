#!/usr/bin/env python3
"""Encode all five Selector BiOPD text conditions, without VAE/student/guider."""
# ruff: noqa: E402

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]

import torch
from omegaconf import OmegaConf

from wan_utils.biopd_text_cache import all_conditions, finalize_cache, prepare_cache, encode_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "Wan21/configs/generator1.yaml"))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--cache-dir", help="Defaults to text_cache_dir in config")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--check-data", action="store_true", help="CPU-only source/contract inventory")
    parser.add_argument("--finalize-only", action="store_true", help="CPU-only verify all shards and publish manifest")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.override))
    root = args.cache_dir or cfg.get("text_cache_dir")
    if not root:
        parser.error("Specify --cache-dir or text_cache_dir")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard-index < num-shards")
    dataset = all_conditions(cfg)
    _, signature, _, texts = prepare_cache(cfg, dataset, root)
    print(json.dumps(dict(event="text_cache_inventory", signature=signature, samples=len(dataset),
                          unique_texts=len(texts), branch_references=5 * len(dataset)), ensure_ascii=False), flush=True)
    if args.check_data:
        return
    if args.finalize_only:
        print(json.dumps(finalize_cache(cfg, dataset, root), ensure_ascii=False), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Text preencoding requires CUDA; --check-data/--finalize-only use CPU")
    torch.cuda.set_device(0)  # physical GPU selected by CUDA_VISIBLE_DEVICES
    # Exact online encoder: BF16 eval, no autocast, one text at a time.
    from wan_utils.wan_wrapper import WanTextEncoder
    encoder = WanTextEncoder(model_path=cfg.model_path).to(device="cuda:0", dtype=torch.bfloat16).eval()
    encode_shard(cfg, dataset, root, encoder, encoder.tokenizer, shard_index=args.shard_index,
                 num_shards=args.num_shards, log_every=args.log_every)
    del encoder
    torch.cuda.empty_cache()
    if args.num_shards == 1:
        print(json.dumps(finalize_cache(cfg, dataset, root), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
