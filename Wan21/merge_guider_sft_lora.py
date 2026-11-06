#!/usr/bin/env python3
"""Export an adapter + its base checkpoint into an ordinary generator model.pt, on CPU."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Wan21"))


def main():
    import torch
    from wan_utils.biopd_io import load_generator_state
    from wan_utils.guider_sft_dataset import atomic_save, file_identity
    from wan_utils.guider_sft_training import merge_adapter_state
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base", help="Default: recorded base checkpoint; replacement must match size and mtime")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
    base = args.base or payload["base_identity"]["path"]
    identity = file_identity(base)
    for name in ("size", "mtime_ns"):
        if identity[name] != payload["base_identity"][name]:
            raise ValueError("Base file differs from the checkpoint used for training")
    merged = merge_adapter_state(load_generator_state(base, "generator"), payload)
    atomic_save(dict(generator=merged), out)
    print(f"Exported {len(merged)} tensors to {out}")


if __name__ == "__main__":
    main()
