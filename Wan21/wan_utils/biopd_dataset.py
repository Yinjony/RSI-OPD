"""Condition-only data for on-policy BiOPD; source videos are not targets."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from wan_utils.camera_trajectory import make_camera_tensors


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def manifest_splits(records, path):
    """Reuse SFT video assignments, while keeping the NEW annotation attributes."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or manifest.get("format") != "wan_guider_sft_cache_v1"
            or not isinstance(manifest.get("contract"), dict)
            or not isinstance(manifest.get("records"), list)
            or manifest.get("signature") != json_digest(manifest["contract"])):
        raise ValueError("Invalid guider SFT split manifest or contract signature")
    reference = {}
    video_splits = {}
    for record in manifest["records"]:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in reference:
            raise ValueError("Split manifest must contain unique nonempty sample_ids")
        if record.get("split") not in ("train", "val"):
            raise ValueError(f"{sample_id}: split must be train or val")
        video = record.get("video_path")
        if not isinstance(video, str) or not video:
            raise ValueError(f"{sample_id}: split manifest requires video_path")
        if video in video_splits and video_splits[video] != record["split"]:
            raise ValueError(f"Split manifest leaks the same video into train and val: {video}")
        video_splits[video] = record["split"]
        reference[sample_id] = record
    source_ids = [record.get("sample_id") for record in records]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Annotation sample_ids must be unique for manifest splitting")
    if set(source_ids) != set(reference):
        raise ValueError("Annotations and split manifest must contain exactly the same sample_ids")
    splits = []
    for record in records:
        sample_id = record["sample_id"]
        original = reference[sample_id]
        for key in ("video_path", "prompt", "action"):
            if record.get(key) != original.get(key):
                raise ValueError(f"{sample_id}: {key} differs from the split manifest")
        splits.append(original["split"])
    info = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                method="reuse_guider_sft_manifest", val_fraction=manifest["contract"].get("val_fraction"),
                split_seed=manifest["contract"].get("split_seed"),
                train=splits.count("train"), val=splits.count("val"))
    return splits, info


def attribute_names(attributes):
    if attributes is None:
        return []
    if not isinstance(attributes, list):
        raise ValueError("Attribute fields must be lists")
    names = []
    for item in attributes:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            raise ValueError("Each attribute must have a nonempty string name")
        names.append(item["name"].strip())
    return names


def condition_texts(record, *, separator="; ", prompt_template="{prompt}\nAttributes: {attributes}",
                    attribute_template="Attributes: {attributes}", condition_format="generic"):
    if condition_format == "guider_sft":
        # Match guider_sft_dataset.format_prompt: preserve polarity, normalize
        # whitespace, deduplicate names, and omit sections with no attributes.
        def section(attributes, polarity):
            names = list(dict.fromkeys(" ".join(name.split()) for name in attribute_names(attributes)))
            return f"Observed {polarity} video attributes:\n" + "; ".join(names) + "." if names else ""

        pos = section(record.get("positive_attributes"), "positive")
        neg = section(record.get("negative_attributes"), "negative")
        prompt = record["prompt"].strip()
        return [record["prompt"], "\n\n".join([prompt, pos]) if pos else prompt,
                neg, "\n\n".join([prompt, neg]) if neg else prompt, pos]
    if condition_format != "generic":
        raise ValueError("condition_format must be generic or guider_sft")
    pos = separator.join(attribute_names(record.get("positive_attributes")))
    neg = separator.join(attribute_names(record.get("negative_attributes")))
    prompt = record["prompt"]
    # Only the first string is a student condition.
    return [prompt,
            prompt_template.format(prompt=prompt, attributes=pos) if pos else prompt,
            attribute_template.format(attributes=neg) if neg else "",
            prompt_template.format(prompt=prompt, attributes=neg) if neg else prompt,
            attribute_template.format(attributes=pos) if pos else ""]


class BiOPDDataset(Dataset):
    def __init__(self, path, *, num_frames=20, missing_attributes="keep", camera=None,
                 prompt_template="{prompt}\nAttributes: {attributes}",
                 attribute_template="Attributes: {attributes}", separator="; ",
                 condition_format="generic", split_manifest=None, split="all"):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        records = json.loads(raw)
        if not isinstance(records, list):
            raise ValueError("BiOPD annotations must be a JSON list")
        if any(not isinstance(record, dict) for record in records):
            raise ValueError("Each BiOPD annotation must be an object")
        if split not in ("all", "train", "val"):
            raise ValueError("split must be all, train or val")
        if split != "all" and not split_manifest:
            raise ValueError("Selecting train/val requires split_manifest")
        self.source_records = records
        self.split_info = None
        self.source_splits = ["all"] * len(records)
        if split_manifest:
            self.source_splits, self.split_info = manifest_splits(records, split_manifest)
            self.split_info["selected"] = split
            self.split_info["selected_ids_sha256"] = json_digest(
                [record["sample_id"] for record, label in zip(records, self.source_splits)
                 if split == "all" or label == split])
        if missing_attributes != "keep":
            raise ValueError("Missing attributes no longer filter records; set missing_attributes=keep")
        if condition_format not in ("generic", "guider_sft"):
            raise ValueError("condition_format must be generic or guider_sft")
        if "{prompt}" not in prompt_template or "{attributes}" not in prompt_template:
            raise ValueError("prompt_template must contain {prompt} and {attributes}")
        if "{attributes}" not in attribute_template or "{prompt}" in attribute_template:
            raise ValueError("attribute_template must contain {attributes}, never {prompt}")
        self.records, self.skipped_ids = [], []
        self.empty_positive = self.empty_negative = self.empty_both = 0
        self.items = []
        seen = set()
        for index, (record, label) in enumerate(zip(records, self.source_splits)):
            sample_id = record.get("sample_id", str(index))
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            seen.add(sample_id)
            for key in ("prompt", "action"):
                if not isinstance(record.get(key), str) or not record[key].strip():
                    raise ValueError(f"{sample_id}: missing/non-string {key}")
            pos = attribute_names(record.get("positive_attributes"))
            neg = attribute_names(record.get("negative_attributes"))
            viewmats, ks = make_camera_tensors(record["action"], **(camera or {}))
            if viewmats.shape[1] != num_frames:
                raise ValueError(f"{sample_id}: action yields {viewmats.shape[1]} frames, expected {num_frames}")
            texts = condition_texts(record, separator=separator, prompt_template=prompt_template,
                                    attribute_template=attribute_template, condition_format=condition_format)
            # Validate ALL annotations, but only expose the requested split to
            # DataLoader. This is a held-out partition, never attribute filtering.
            if split != "all" and label != split:
                continue
            self.empty_positive += int(not pos)
            self.empty_negative += int(not neg)
            self.empty_both += int(not pos and not neg)
            self.records.append(record)
            self.items.append(dict(sample_id=sample_id, texts=texts,
                                   viewmats=viewmats[0], Ks=ks[0]))
        if not self.items:
            raise ValueError("Selected BiOPD split contains no records")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def report(self):
        report = dict(path=str(self.path), sha256=self.sha256, usable=len(self),
                      skipped=0, skipped_ids=self.skipped_ids,
                      empty_positive=self.empty_positive, empty_negative=self.empty_negative,
                      empty_both=self.empty_both)
        if self.split_info:
            report.update(total=len(self.source_records), split=self.split_info)
        return report

    def write_splits(self, directory):
        """Export condition JSONs, not SFT cache tensors; never modify the source."""
        if not self.split_info:
            raise ValueError("Writing train/val partitions requires split_manifest")
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        records = [dict(record, split=label) for record, label in zip(self.source_records, self.source_splits)]
        payloads = {
            "manifest.json": dict(format="wan_biopd_condition_split_v1", source=self.report(), records=records),
            "train.json": [record for record, label in zip(self.source_records, self.source_splits) if label == "train"],
            "val.json": [record for record, label in zip(self.source_records, self.source_splits) if label == "val"],
        }
        for name in payloads:
            path = directory / name
            # Guard against using the annotation directory to replace an input
            # file, including the authoritative SFT manifest.
            if path == self.path or path == Path(self.split_info["path"]):
                raise ValueError("Refusing to overwrite source annotations or reference manifest")
        for name, payload in payloads.items():
            path = directory / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(path)
        return str(directory)


def collate_conditions(items):
    return dict(sample_ids=[x["sample_id"] for x in items],
                texts=[[x["texts"][i] for x in items] for i in range(5)],
                viewmats=torch.stack([x["viewmats"] for x in items]),
                Ks=torch.stack([x["Ks"] for x in items]))
