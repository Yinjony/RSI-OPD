#!/usr/bin/env python3
"""Export WBench navigation cases to a minWM manifest JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
MINWM_ROOT = BRIDGE_DIR.parents[1]
WAN21_ROOT = MINWM_ROOT / "Wan21"
WBENCH_DATA_DEFAULT = Path("/root/data/wbench_handoff/WBench/data")
sys.path.insert(0, str(BRIDGE_DIR))
sys.path.insert(0, str(WAN21_ROOT))

from case_convert import convert_case, discover_navigation_cases, load_wbench_case  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert WBench nav cases to minWM manifest")
    parser.add_argument(
        "--wbench-data-dir",
        default=str(WBENCH_DATA_DEFAULT),
        help="WBench data directory containing cases/ and images/",
    )
    parser.add_argument(
        "--out",
        default=str(BRIDGE_DIR / "manifests" / "wbench_nav_minwm.jsonl"),
        help="Output JSONL manifest path",
    )
    parser.add_argument(
        "--expected-latent-frames",
        type=int,
        default=20,
        help="Latent frame count required by the DMD config",
    )
    parser.add_argument(
        "--case",
        default="",
        help="Optional single case JSON path for a smoke conversion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.case:
        case_path = Path(args.case)
        case = load_wbench_case(case_path)
        wbench_data_dir = case_path.parent.parent
        sample = convert_case(
            case,
            wbench_data_dir / "images",
            expected_latent_frames=args.expected_latent_frames,
        )
        if sample is None:
            raise SystemExit(f"Failed to convert case: {case_path}")
        sample["wbench_case_path"] = str(case_path)
        samples = [sample]
        skipped = []
    else:
        from case_convert import convert_cases_dir

        samples, skipped = convert_cases_dir(
            Path(args.wbench_data_dir),
            expected_latent_frames=args.expected_latent_frames,
        )

    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples -> {out_path}")
    if skipped:
        print(f"Skipped {len(skipped)} cases without images or navigation: {skipped[:20]}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")


if __name__ == "__main__":
    main()
