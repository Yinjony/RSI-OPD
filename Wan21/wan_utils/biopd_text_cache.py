"""Resumable, deduplicated UMT5 cache for all five BiOPD text branches."""

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import torch

from wan_utils.biopd_dataset import BiOPDDataset, json_digest
from wan_utils.biopd_io import file_identity

FORMAT = "wan_biopd_text_cache_v1"
BRANCHES = ["student_prompt", "prompt_positive", "negative_only", "prompt_negative", "positive_only"]
EMBEDDING_SHAPE = (512, 4096)


def text_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, payload, *, tensor=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        if tensor:
            torch.save(payload, temporary)
        else:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def all_conditions(config):
    """Use exactly the production formatter and original held-out membership."""
    return BiOPDDataset(config.annotation_path, split="all", split_manifest=config.get("split_manifest"),
                        num_frames=config.num_training_frames, camera=dict(config.camera),
                        missing_attributes=config.get("missing_attributes", "keep"),
                        condition_format=config.get("condition_format", "generic"),
                        prompt_template=config.prompt_attribute_template,
                        attribute_template=config.attribute_only_template,
                        separator=config.attribute_separator)


def cache_contract(config, dataset):
    model_path = Path(config.model_path).resolve()
    tokenizer = model_path / "google/umt5-xxl"
    files = [model_path / "models_t5_umt5-xxl-enc-bf16.pth"]
    files += sorted(path for path in tokenizer.rglob("*") if path.is_file())
    if not tokenizer.is_dir() or len(files) == 1:
        raise ValueError(f"Missing local UMT5 tokenizer: {tokenizer}")
    return dict(format=FORMAT, annotation_sha256=dataset.sha256,
                split_manifest_sha256=dataset.split_info["sha256"] if dataset.split_info else None,
                branches=BRANCHES, condition_format=config.get("condition_format", "generic"),
                prompt_attribute_template=config.prompt_attribute_template,
                attribute_only_template=config.attribute_only_template,
                attribute_separator=config.attribute_separator,
                encoder_files=[file_identity(path) for path in files],
                embedding_shape=list(EMBEDDING_SHAPE), dtype="bfloat16",
                encoding="WanTextEncoder([text]); eval; no_grad; BF16; no autocast; zero padded tokens")


def inventory(dataset):
    records, texts = [], {}
    for item, split in zip(dataset.items, dataset.source_splits):
        keys = [text_key(text) for text in item["texts"]]
        records.append(dict(sample_id=item["sample_id"], split=split, texts=item["texts"], text_keys=keys))
        texts.update(zip(keys, item["texts"]))
    return records, texts


def validate_embedding(payload, *, signature, text):
    embedding = payload.get("embedding")
    if (payload.get("format") != FORMAT or payload.get("signature") != signature
            or payload.get("text") != text or payload.get("text_key") != text_key(text)):
        raise ValueError("Stale or mismatched BiOPD text embedding")
    if (not isinstance(embedding, torch.Tensor) or tuple(embedding.shape) != EMBEDDING_SHAPE
            or embedding.dtype != torch.bfloat16 or not torch.isfinite(embedding).all()):
        raise ValueError("Invalid BiOPD text embedding shape/dtype/values")
    count = payload.get("token_count")
    if not isinstance(count, int) or not 0 < count <= EMBEDDING_SHAPE[0]:
        raise ValueError("Invalid text token count")
    if torch.count_nonzero(embedding[count:]):
        raise ValueError("Text embedding padding must be zero")
    return embedding


def prepare_cache(config, dataset, root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    contract = cache_contract(config, dataset)
    path = root / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Text cache source/template/encoder changed; select a new text_cache_dir")
    atomic_write(path, contract)
    records, texts = inventory(dataset)
    return contract, json_digest(contract), records, texts


def encode_shard(config, dataset, root, encoder, tokenizer, *, shard_index=0, num_shards=1, log_every=100):
    if not 0 <= shard_index < num_shards or log_every < 1:
        raise ValueError("Invalid shard index/count or log interval")
    root = Path(root)
    _, signature, _, texts = prepare_cache(config, dataset, root)
    assigned = [(key, text) for index, (key, text) in enumerate(texts.items())
                if index % num_shards == shard_index]
    # Validate the whole shard before loading/using the encoder.
    counts = []
    for _, text in assigned:
        ids, _ = tokenizer(text, return_mask=True, add_special_tokens=True, truncation=False, padding=False)
        count = ids.shape[-1]
        if not 0 < count <= EMBEDDING_SHAPE[0]:
            raise ValueError(f"Text exceeds {EMBEDDING_SHAPE[0]} tokens: {text[:100]}")
        counts.append(count)
    started, encoded, reused = time.monotonic(), 0, 0
    for position, ((key, text), count) in enumerate(zip(assigned, counts), 1):
        path = root / "text_embeddings" / f"{key}.pt"
        if path.exists():
            validate_embedding(torch.load(path, map_location="cpu", weights_only=True),
                               signature=signature, text=text)
            reused += 1
        else:
            with torch.no_grad():
                embedding = encoder([text])["prompt_embeds"][0].detach().cpu()
            payload = dict(format=FORMAT, signature=signature, text=text, text_key=key,
                           token_count=count, embedding=embedding)
            validate_embedding(payload, signature=signature, text=text)
            atomic_write(path, payload, tensor=True)
            encoded += 1
        if position == 1 or position % log_every == 0 or position == len(assigned):
            elapsed = time.monotonic() - started
            print(json.dumps(dict(event="text_preencode", shard=shard_index, done=position,
                                  total=len(assigned), encoded=encoded, reused=reused,
                                  seconds=elapsed, eta_seconds=elapsed / position * (len(assigned) - position))),
                  flush=True)
    return dict(encoded=encoded, reused=reused)


def finalize_cache(config, dataset, root):
    """Publish a complete manifest only after ALL shards/tensors are verified."""
    root = Path(root)
    contract, signature, records, texts = prepare_cache(config, dataset, root)
    entries = {}
    for index, (key, text) in enumerate(texts.items(), 1):
        path = root / "text_embeddings" / f"{key}.pt"
        if not path.is_file():
            raise ValueError(f"Incomplete text cache: missing {path}; finish all preencoding shards")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_embedding(payload, signature=signature, text=text)
        entries[key] = dict(text=text, cache_file=f"text_embeddings/{key}.pt",
                            token_count=payload["token_count"], sha256=file_sha256(path))
        if index % 1000 == 0:
            print(json.dumps(dict(event="text_cache_verify", done=index, total=len(texts))), flush=True)
    report = dict(records=len(records), train=sum(r["split"] == "train" for r in records),
                  val=sum(r["split"] == "val" for r in records), unique_texts=len(entries),
                  branch_references=5 * len(records))
    manifest = dict(format=FORMAT, signature=signature, contract=contract, branches=BRANCHES,
                    records=records, texts=entries, report=report)
    # Keep the existing condition-only manifest.json/train.json/val.json intact.
    atomic_write(root / "text_cache_manifest.json", manifest)
    return report


class PreencodedTextConditions:
    """Same .encode(branches) interface as online T5, without importing/loading T5."""
    def __init__(self, config, dataset, device):
        self.root = Path(config.text_cache_dir).resolve()
        path = self.root / "text_cache_manifest.json"
        if not path.is_file():
            raise ValueError(f"Missing completed text cache: {path}; run preencode_biopd_text.py first")
        manifest = json.loads(path.read_text())
        contract = cache_contract(config, dataset)
        self.signature = json_digest(contract)
        if (manifest.get("format") != FORMAT or manifest.get("contract") != contract
                or manifest.get("signature") != self.signature or manifest.get("branches") != BRANCHES
                or json.loads((self.root / "contract.json").read_text()) != contract):
            raise ValueError("Text cache source/template/encoder contract mismatch")
        records = manifest.get("records", [])
        by_id = {record["sample_id"]: record for record in records}
        source_ids = {record.get("sample_id", str(index)) for index, record in enumerate(dataset.source_records)}
        if len(by_id) != len(records) or set(by_id) != source_ids:
            raise ValueError("Text cache sample_ids do not cover the annotation dataset")
        expected_splits = {r.get("sample_id", str(i)): split for i, (r, split)
                           in enumerate(zip(dataset.source_records, dataset.source_splits))}
        if any(record["split"] != expected_splits[record["sample_id"]] for record in records):
            raise ValueError("Text cache train/val assignments differ")
        self.entries = manifest["texts"]
        for item in dataset.items:
            record = by_id[item["sample_id"]]
            keys = [text_key(text) for text in item["texts"]]
            if record["texts"] != item["texts"] or record["text_keys"] != keys:
                raise ValueError(f"Text cache conditions differ for {item['sample_id']}")
        for key, entry in self.entries.items():
            if key != text_key(entry["text"]) or entry["cache_file"] != f"text_embeddings/{key}.pt":
                raise ValueError("Invalid text cache key/path")
            if not (self.root / entry["cache_file"]).is_file():
                raise ValueError("Incomplete text cache: embedding file missing")
        if {key for record in records for key in record["text_keys"]} != set(self.entries):
            raise ValueError("Incomplete text cache branch coverage")
        self.device = device
        self.limit = config.text_cache_size
        self.cache = OrderedDict()
        self.report = dict(path=str(self.root), signature=self.signature, **manifest["report"])

    @torch.no_grad()
    def encode(self, text_branches):
        batch_cache = {}
        for text in dict.fromkeys(text for branch in text_branches for text in branch):
            key = text_key(text)
            if key in self.cache:
                embedding = self.cache[key]
                self.cache.move_to_end(key)
            else:
                entry = self.entries.get(key)
                if entry is None or entry["text"] != text:
                    raise ValueError("Requested text is absent from the preencoded cache")
                path = self.root / entry["cache_file"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError(f"Text cache checksum mismatch: {path}")
                embedding = validate_embedding(torch.load(path, map_location="cpu", weights_only=True),
                                               signature=self.signature, text=text)
                if self.limit:
                    self.cache[key] = embedding
                    while len(self.cache) > self.limit:
                        self.cache.popitem(last=False)
            batch_cache[text] = embedding
        return [dict(prompt_embeds=torch.stack([batch_cache[text] for text in branch]).to(self.device))
                for branch in text_branches]
