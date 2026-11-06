"""Attribute text, source validation and immutable per-video SFT caches.

No Wan/model imports: data auditing and cache tests also run on CPU.
"""
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from wan_utils.camera_trajectory import make_camera_tensors

FORMAT = "wan_guider_sft_cache_v1"
TEMPLATE_VERSION = "observed_positive_negative_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path):
    p = Path(path).resolve()
    s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def attribute_names(attributes):
    if not isinstance(attributes, list):
        raise ValueError("Attributes must be lists of {attribute_id, name}")
    names = []
    for item in attributes:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            raise ValueError(f"Invalid attribute: {item!r}")
        if "attribute_id" not in item:
            raise ValueError(f"Missing attribute_id: {item!r}")
        name = " ".join(item["name"].split())
        if name not in names:
            names.append(name)
    return names


def format_prompt(prompt, positive_attributes, negative_attributes):
    """Shared training/inference template; negative names describe observed defects."""
    parts = [prompt.strip()]
    for title, attrs in (("positive", positive_attributes), ("negative", negative_attributes)):
        names = attribute_names(attrs)
        if names:
            parts.append(f"Observed {title} video attributes:\n" + "; ".join(names) + ".")
    return "\n\n".join(parts)


def load_records(path, camera, num_frames=20):
    path = Path(path).resolve()
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Input JSON must be a nonempty list")
    ids = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Each record must be an object")
        for key in ("sample_id", "video_path", "prompt", "action"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Missing/non-string {key} in {item!r}")
        if item["sample_id"] in ids:
            raise ValueError(f"Duplicate sample_id: {item['sample_id']}")
        ids.add(item["sample_id"])
        for key in ("positive_attributes", "negative_attributes"):
            if key not in item:
                raise ValueError(f"Missing {key} in {item['sample_id']}")
            attribute_names(item[key])
        video = Path(item["video_path"])
        video = video if video.is_absolute() else path.parent / video
        if not video.is_file():
            raise FileNotFoundError(video)
        item["video_path"] = str(video.resolve())
        vm, ks = make_camera_tensors(item["action"], **camera)
        if vm.shape[1] != num_frames:
            raise ValueError(f"{item['sample_id']}: action produces {vm.shape[1]} poses, expected {num_frames}")
        if not torch.isfinite(vm).all() or not torch.isfinite(ks).all():
            raise ValueError(f"Nonfinite camera matrices: {item['sample_id']}")
    return records


def source_report(records):
    return dict(records=len(records), unique_videos=len({r["video_path"] for r in records}),
                with_positive=sum(bool(r["positive_attributes"]) for r in records),
                with_negative=sum(bool(r["negative_attributes"]) for r in records),
                positive_occurrences=sum(len(r["positive_attributes"]) for r in records),
                negative_occurrences=sum(len(r["negative_attributes"]) for r in records),
                unique_positive=len({a["name"] for r in records for a in r["positive_attributes"]}),
                unique_negative=len({a["name"] for r in records for a in r["negative_attributes"]}))


def split_records(records, fraction, seed):
    """Group by video; stratify by presence of negative annotations when possible."""
    if not 0 <= fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    groups = {}
    for r in records:
        groups.setdefault(r["video_path"], []).append(r)
    rng, validation = random.Random(seed), set()
    for negative in (False, True):
        keys = sorted(k for k, rs in groups.items() if any(r["negative_attributes"] for r in rs) == negative)
        rng.shuffle(keys)
        n = min(len(keys) - 1, max(1, round(len(keys) * fraction))) if fraction and len(keys) > 1 else 0
        validation.update(keys[:n])
    return ["val" if r["video_path"] in validation else "train" for r in records]


def condition_texts(records, index, controls=True, attribute_dropout=False):
    r = records[index]
    texts = dict(full=format_prompt(r["prompt"], r["positive_attributes"], r["negative_attributes"]))
    if attribute_dropout:
        # Encode independently: dropping tokens from a combined T5 embedding
        # would still leak the other attribute set through self-attention.
        texts["positive"] = format_prompt(r["prompt"], r["positive_attributes"], [])
        texts["negative"] = format_prompt(r["prompt"], [], r["negative_attributes"])
    if controls:
        donor = records[(index + 1) % len(records)]
        texts["no_attributes"] = format_prompt(r["prompt"], [], [])
        texts["shuffled_attributes"] = format_prompt(r["prompt"], donor["positive_attributes"], donor["negative_attributes"])
    return texts


def validate_token_lengths(tokenizer, texts, limit=512):
    counts = {}
    for name, text in texts.items():
        ids = tokenizer(text, add_special_tokens=True, truncation=False, padding=False)
        n = ids.shape[-1]
        if n > limit:
            raise ValueError(f"{name}: {n} tokens exceeds {limit}; shorten prompt/attributes explicitly")
        counts[name] = n
    return counts


def read_video(path, height=480, width=832, num_frames=77):
    import cv2
    import numpy as np
    import torch.nn.functional as F
    cap, frames = cv2.VideoCapture(str(path)), []
    try:
        # Read one extra frame to detect mismatched action/video durations.
        for _ in range(num_frames + 1):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if len(frames) != num_frames:
        raise ValueError(f"{path}: decoded {len(frames)} frames (capped at {num_frames + 1}), expected exactly {num_frames}")
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    if x.shape[-2:] != (height, width):
        # Spatial resize only; normalized camera intrinsics remain unchanged.
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.permute(1, 0, 2, 3).div_(127.5).sub_(1).contiguous()


def validate_cached_sample(sample, signature, shape, controls, attribute_dropout=False):
    if sample.get("signature") != signature:
        raise ValueError("Stale/incompatible cache signature")
    x = sample["clean_latent"]
    if tuple(x.shape) != tuple(shape) or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("Invalid cached video latent")
    expected = {"full", "no_attributes", "shuffled_attributes"} if controls else {"full"}
    if attribute_dropout:
        expected.update(("positive", "negative"))
    if set(sample["embeddings"]) != expected:
        raise ValueError("Cached text conditions do not match configuration")
    for emb in sample["embeddings"].values():
        if tuple(emb.shape) != (512, 4096) or not torch.isfinite(emb).all():
            raise ValueError("Invalid cached text embedding")
    for name, dims in (("viewmats", (shape[0], 4, 4)), ("Ks", (shape[0], 3, 3))):
        if tuple(sample[name].shape) != dims or not torch.isfinite(sample[name]).all():
            raise ValueError(f"Invalid cached {name}")


class GuiderSFTDataset(Dataset):
    def __init__(self, cache_dir, split, verify_hashes=True):
        self.root = Path(cache_dir).resolve()
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError("Unsupported SFT cache format")
        if self.manifest.get("signature") != digest(self.manifest["contract"]):
            raise ValueError("Cache manifest contract/signature mismatch")
        self.sha256 = file_sha256(manifest_path)
        self.records = [r for r in self.manifest["records"] if r["split"] == split]
        for r in self.records:
            p = self.root / r["cache_file"]
            if not p.is_file() or (verify_hashes and file_sha256(p) != r["sha256"]):
                raise ValueError(f"Missing/changed cache file: {p}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        branch = None
        if isinstance(index, tuple):
            index, branch = index
            if branch not in ("positive", "negative"):
                raise ValueError(f"Unknown SFT attribute branch: {branch}")
        r = self.records[index]
        if branch is not None and not r.get(branch + "_attributes"):
            raise ValueError(f"{r['sample_id']}: selected {branch} attributes are empty")
        sample = torch.load(self.root / r["cache_file"], map_location="cpu", weights_only=True)
        if sample["signature"] != self.manifest["signature"] or sample["sample_id"] != r["sample_id"]:
            raise ValueError("Cache record mismatch")
        embeddings = sample["embeddings"]
        if branch is not None:
            if branch not in embeddings:
                raise ValueError("Attribute dropout requires a new cache with positive/negative text embeddings")
            # Only this independently encoded branch reaches training/anchor.
            embeddings = {"selected": embeddings[branch]}
        result = dict(clean_latent=sample["clean_latent"], embeddings=embeddings,
                    viewmats=sample["viewmats"], Ks=sample["Ks"], sample_id=r["sample_id"],
                    has_positive=bool(r.get("positive_attributes")),
                    has_negative=bool(r["negative_attributes"]))
        if branch is not None:
            result["branch"] = branch
        return result
