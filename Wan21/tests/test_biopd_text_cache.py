"""Cache parity, branch coverage, resume and stale/corrupt data rejection."""

import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_biopd import ROOT, split_fixture
from wan_utils.biopd_dataset import BiOPDDataset, collate_conditions
from wan_utils.biopd_io import load_source
import wan_utils.biopd_text_cache as cache


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.extend(texts)
        result = torch.zeros(len(texts), *cache.EMBEDDING_SHAPE, dtype=torch.bfloat16)
        for i, text in enumerate(texts):
            result[i, :2] = sum(text.encode()) % 31
        return dict(prompt_embeds=result)


def tokenizer(text, **kwargs):
    return torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long)


@pytest.fixture
def setup_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "EMBEDDING_SHAPE", (4, 3))
    path, reference, _, _ = split_fixture(tmp_path)
    records = json.loads(path.read_text())
    records[0]["negative_attributes"] = []
    path.write_text(json.dumps(records))
    model = tmp_path / "model"
    tokens = model / "google/umt5-xxl"
    tokens.mkdir(parents=True)
    (tokens / "tokenizer.json").write_text("{}")
    (model / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"weights")
    cfg = OmegaConf.create(dict(annotation_path=str(path), split_manifest=str(reference), model_path=str(model),
                               num_training_frames=20, camera={}, missing_attributes="keep",
                               condition_format="guider_sft", prompt_attribute_template="{prompt} {attributes}",
                               attribute_only_template="{attributes}", attribute_separator="; ",
                               text_cache_dir=str(tmp_path / "cache"), text_cache_size=1,
                               generator_ckpt=str(path), guider_ckpt=str(reference), guider_adapter_ckpt=None))
    dataset = cache.all_conditions(cfg)
    return cfg, dataset, Path(cfg.text_cache_dir)


def test_shards_resume_parity_and_original_split_exports(setup_cache):
    cfg, dataset, root = setup_cache
    dataset.write_splits(root)
    original = (root / "manifest.json").read_bytes()
    encoder = FakeEncoder()
    cache.encode_shard(cfg, dataset, root, encoder, tokenizer, shard_index=0, num_shards=2)
    calls = len(encoder.calls)
    with pytest.raises(ValueError, match="Incomplete"):
        cache.finalize_cache(cfg, dataset, root)
    assert not (root / "text_cache_manifest.json").exists()
    resumed = cache.encode_shard(cfg, dataset, root, encoder, tokenizer, shard_index=0, num_shards=2)
    assert resumed["encoded"] == 0 and resumed["reused"] == calls
    assert len(encoder.calls) == calls
    cache.encode_shard(cfg, dataset, root, encoder, tokenizer, shard_index=1, num_shards=2)
    report = cache.finalize_cache(cfg, dataset, root)
    assert report["records"] == 3 and report["train"] == 2 and report["val"] == 1
    assert report["unique_texts"] == len(encoder.calls) < 15
    assert (root / "manifest.json").read_bytes() == original
    provider = cache.PreencodedTextConditions(cfg, dataset, "cpu")
    batch = collate_conditions(dataset.items)
    global_rng = torch.get_rng_state().clone()
    for output, texts in zip(provider.encode(batch["texts"]), batch["texts"]):
        assert torch.equal(output["prompt_embeds"], FakeEncoder()(texts)["prompt_embeds"])
        assert not output["prompt_embeds"].requires_grad
    assert torch.equal(global_rng, torch.get_rng_state())
    for split in ("train", "val"):
        subset = BiOPDDataset(cfg.annotation_path, split_manifest=cfg.split_manifest, split=split,
                             condition_format=cfg.condition_format)
        cache.PreencodedTextConditions(cfg, subset, "cpu")


def test_reject_stale_corrupt_missing_and_unknown_text(setup_cache):
    cfg, dataset, root = setup_cache
    with pytest.raises(ValueError, match="Missing completed"):
        cache.PreencodedTextConditions(cfg, dataset, "cpu")
    cache.encode_shard(cfg, dataset, root, FakeEncoder(), tokenizer)
    cache.finalize_cache(cfg, dataset, root)
    cfg.text_cache_size = 0
    provider = cache.PreencodedTextConditions(cfg, dataset, "cpu")
    with pytest.raises(ValueError, match="absent"):
        provider.encode([["unknown"]] * 5)
    changed = OmegaConf.create(OmegaConf.to_container(cfg))
    changed.condition_format = "generic"
    with pytest.raises(ValueError, match="contract mismatch"):
        cache.PreencodedTextConditions(changed, dataset, "cpu")
    key = cache.text_key(dataset[0]["texts"][0])
    path = root / "text_embeddings" / f"{key}.pt"
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        provider.encode([[dataset[0]["texts"][0]]] * 5)
    path.unlink()
    with pytest.raises(ValueError, match="embedding file missing"):
        cache.PreencodedTextConditions(cfg, dataset, "cpu")


def test_token_overflow_rejected_before_encoding(setup_cache):
    cfg, dataset, root = setup_cache
    encoder = FakeEncoder()

    def too_long(text, **kwargs):
        return torch.ones(1, 5), torch.ones(1, 5)

    with pytest.raises(ValueError, match="exceeds"):
        cache.encode_shard(cfg, dataset, root, encoder, too_long)
    assert encoder.calls == []


def test_enable_cache_preserves_old_checkpoint_contract(setup_cache):
    cfg, dataset, _ = setup_cache
    module = load_source("cache_resume_contract", ROOT / "Wan21/wan_trainer/camera_biopd.py")
    online = OmegaConf.create(OmegaConf.to_container(cfg))
    del online.text_cache_dir
    assert module.resume_contract(cfg, dataset, 4) == module.resume_contract(online, dataset, 4)
