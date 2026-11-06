"""CPU checks of the production branch sampler, cached inputs and resume stream."""
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]

from wan_utils.biopd_io import load_source
from wan_utils.guider_sft_anchor import backward_sft_with_anchor
from wan_utils.guider_sft_config import cache_contract, load_config
from wan_utils.guider_sft_dataset import (
    FORMAT, GuiderSFTDataset, atomic_json, atomic_save, condition_texts, digest,
    file_sha256, validate_cached_sample,
)
from wan_utils.guider_sft_sampling import BalancedAttributeBatchSampler


def records():
    pos = [{"attribute_id": 1, "name": "Sharp foreground"}]
    neg = [{"attribute_id": 2, "name": "Missing walking motion"}]
    return [dict(sample_id=str(i), prompt="Robot walking", positive_attributes=p, negative_attributes=n)
            for i, (p, n) in enumerate([(pos, neg)] + [(pos, [])] * 8 + [([], neg), ([], [])])]


def test_text_conditions_keep_prompt_and_drop_the_whole_other_side():
    texts = condition_texts(records(), 0, controls=False, attribute_dropout=True)
    assert "Sharp foreground" in texts["positive"] and "Missing walking motion" not in texts["positive"]
    assert "Missing walking motion" in texts["negative"] and "Sharp foreground" not in texts["negative"]
    assert all(texts[k].startswith("Robot walking") for k in ("positive", "negative"))
    assert set(condition_texts(records(), 0, False)) == {"full"}
    assert condition_texts(records(), 1, False, True)["negative"] == "Robot walking"


@pytest.mark.parametrize("world,batch,accumulation", [(1, 1, 4), (2, 1, 2), (4, 1, 1),
                                                    (2, 2, 2), (1, 3, 1), (1, 1, 1)])
def test_global_balance_nonempty_and_dual_membership(world, batch, accumulation):
    rs = records()
    samplers = [BalancedAttributeBatchSampler(rs, batch_size=batch, accumulation=accumulation,
                                             world_size=world, rank=rank) for rank in range(world)]
    totals, dual_branches = Counter(), set()
    draw_order = dict(positive=[], negative=[])
    for epoch in range(6):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        streams = [list(sampler) for sampler in samplers]
        assert all(len(s) == len(samplers[0]) for s in streams)
        for offset in range(0, len(streams[0]), accumulation):
            update = Counter()
            for micro in range(offset, offset + accumulation):
                for stream in streams:
                    assert len(stream[micro]) == batch
                    for index, branch in stream[micro]:
                        assert rs[index][branch + "_attributes"]
                        update[branch] += 1
                        draw_order[branch].append(index)
                        if index == 0:
                            dual_branches.add(branch)
            assert abs(update["positive"] - update["negative"]) <= 1
            totals.update(update)
            assert abs(totals["positive"] - totals["negative"]) <= 1
    assert dual_branches == {"positive", "negative"}
    for branch, order in draw_order.items():
        size = len(samplers[0].pools[branch])
        for start in range(0, len(order), size):
            # Each shuffled pool is exhausted before repeating, across epochs.
            assert len(set(order[start:start + size])) == len(order[start:start + size])


def test_missing_pool_fails_instead_of_falling_back_to_another_branch():
    with pytest.raises(ValueError, match="positive AND negative"):
        BalancedAttributeBatchSampler(records()[1:5], batch_size=1, accumulation=4)


def cached_dataset(tmp_path):
    rs = records()
    contract = {"attribute_dropout": True}
    signature = digest(contract)
    entries = []
    for i, record in enumerate(rs):
        sample = dict(signature=signature, sample_id=record["sample_id"],
                      clean_latent=torch.zeros(4, 1, 2, 4),
                      embeddings={"full": torch.full((3, 4), 99.),
                                  "positive": torch.full((3, 4), 2.),
                                  "negative": torch.full((3, 4), -3.)},
                      viewmats=torch.eye(4).expand(4, 4, 4), Ks=torch.eye(3).expand(4, 3, 3))
        path = tmp_path / f"{i}.pt"
        atomic_save(sample, path)
        entries.append(dict(**record, split="train", cache_file=path.name, sha256=file_sha256(path)))
    atomic_json(dict(format=FORMAT, signature=signature, contract=contract, records=entries),
                tmp_path / "manifest.json")
    return GuiderSFTDataset(tmp_path, "train")


def test_cached_training_batch_contains_only_the_selected_text_and_same_video_camera(tmp_path):
    data = cached_dataset(tmp_path)
    sampler = BalancedAttributeBatchSampler(data.records, batch_size=4, accumulation=1)
    batch = next(iter(DataLoader(data, batch_sampler=sampler)))
    assert set(batch["embeddings"]) == {"selected"}
    assert Counter(batch["branch"]) == Counter(positive=2, negative=2)
    for i, branch in enumerate(batch["branch"]):
        assert (batch["embeddings"]["selected"][i] == (2 if branch == "positive" else -3)).all()
    p, n = data[(0, "positive")], data[(0, "negative")]
    for key in ("clean_latent", "viewmats", "Ks"):
        torch.testing.assert_close(p[key], n[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="empty"):
        data[(1, "negative")]


def test_resume_reproduces_actual_dataloader_stream_across_epoch_boundary(tmp_path):
    trainer = load_source("dropout_stream_test", ROOT / "Wan21/wan_trainer/camera_guider_sft.py")
    data = cached_dataset(tmp_path)

    def stream(epoch=0, offset=0):
        obj = trainer.Trainer.__new__(trainer.Trainer)
        obj.epoch, obj.batch_in_epoch = epoch, offset
        obj.sampler = BalancedAttributeBatchSampler(data.records, batch_size=1, accumulation=2,
                                                    world_size=2, rank=1)
        obj.loader = DataLoader(data, batch_sampler=obj.sampler)
        return obj, obj._batches()

    obj, original = stream()
    for _ in range(len(obj.loader) - 1):
        next(original)
    _, resumed = stream(obj.epoch, obj.batch_in_epoch)
    for _ in range(2 * len(obj.loader) + 3):
        a, b = next(original), next(resumed)
        assert a["sample_id"] == b["sample_id"] and a["branch"] == b["branch"]
        torch.testing.assert_close(a["embeddings"]["selected"], b["embeddings"]["selected"], rtol=0, atol=0)


def test_old_cache_rejected_and_legacy_config_resume_contract_preserved(tmp_path):
    cfg = load_config(ROOT / "Wan21/configs/guider2.yaml")
    assert cfg.attribute_dropout is True and cfg.initial_adapter is None
    with pytest.raises(ValueError, match="attribute_dropout"):
        load_config(ROOT / "Wan21/configs/guider2.yaml", ["attribute_dropout=0.5"])
    sample = dict(signature="test", clean_latent=torch.zeros(4, 1, 2, 4), embeddings={"full": torch.ones(3, 4)})
    with pytest.raises(ValueError, match="text conditions"):
        validate_cached_sample(sample, "test", (4, 1, 2, 4), controls=False, attribute_dropout=True)
    trainer = load_source("dropout_contract_test", ROOT / "Wan21/wan_trainer/camera_guider_sft.py")
    data = SimpleNamespace(sha256="cache")
    with patch.object(trainer, "file_identity", return_value={"path": "base"}):
        current = trainer.resume_contract(cfg, data, 2)
        assert current["config"]["attribute_dropout"] is True
        cfg.attribute_dropout = False
        legacy = trainer.resume_contract(cfg, data, 2)
        assert "attribute_dropout" not in legacy["config"] and current != legacy
    cfg.model_path = str(tmp_path / "model")
    model = Path(cfg.model_path)
    (model / "google/umt5-xxl").mkdir(parents=True)
    for relative in ("Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/tokenizer.json"):
        (model / relative).touch()
    cfg.annotation_path = str(tmp_path / "input.json")
    Path(cfg.annotation_path).write_text("[]")
    old_contract = cache_contract(cfg, [])
    assert "attribute_dropout" not in old_contract
    cfg.attribute_dropout = True
    assert digest(old_contract) != digest(cache_contract(cfg, []))


def test_supervised_and_anchor_forward_the_same_selected_condition(tmp_path):
    # A full-condition sentinel catches accidental use of the old combined input.
    from test_guider_sft import scheduler, tiny_model
    from test_guider_sft_anchor import config, models
    from wan_utils.guider_sft_anchor import OnPolicyAnchor

    data = cached_dataset(tmp_path)
    batch = next(iter(DataLoader(data, batch_size=2, sampler=[(0, "positive"), (0, "negative")])))
    model = tiny_model()
    observed = []
    hook = model.register_forward_pre_hook(
        lambda _m, _args, kwargs: observed.append(kwargs["conditional_dict"]["prompt_embeds"].clone()),
        with_kwargs=True)

    class AnchorSpy:
        def loss(self, batch, _generator, **kwargs):
            assert kwargs["condition"] == "selected"
            torch.testing.assert_close(batch["embeddings"]["selected"], observed[0])
            return torch.zeros(len(batch["clean_latent"]))

    backward_sft_with_anchor(model, batch, scheduler(), torch.Generator().manual_seed(3), AnchorSpy(),
                             torch.Generator().manual_seed(4), eta=.1, accumulation=1,
                             synchronize=True, condition="selected")
    hook.remove()
    assert len(observed) == 1
    # Also exercise the real on-policy anchor with only a selected embedding.
    student, reference = models()
    anchor = OnPolicyAnchor(student, reference, config(states=1))
    anchor.loss(batch, torch.Generator().manual_seed(5), condition="selected", state_indices=[0])
    for wrapper in (student, reference):
        assert wrapper.trace
        for row in wrapper.trace:
            torch.testing.assert_close(row["text"], batch["embeddings"]["selected"])


def test_real_trainer_loop_counts_selected_branches_across_epoch_boundary(tmp_path):
    from test_guider_sft import scheduler, tiny_model

    trainer = load_source("dropout_train_loop_test", ROOT / "Wan21/wan_trainer/camera_guider_sft.py")
    obj = trainer.Trainer.__new__(trainer.Trainer)
    obj.cfg = load_config(ROOT / "Wan21/configs/guider2.yaml", ["grad_accum_steps=4", "max_steps=4",
                          "val_every=0", "save_every=4", "anchor=false", "anchor_eta=0"])
    obj.train_data = cached_dataset(tmp_path)
    obj.val_data = []
    obj.sampler = BalancedAttributeBatchSampler(obj.train_data.records, batch_size=1, accumulation=4)
    obj.loader = DataLoader(obj.train_data, batch_sampler=obj.sampler)
    obj.device = torch.device("cpu")
    obj.rank, obj.world = 0, 1
    obj.model = obj.raw = tiny_model()
    obj.scheduler = scheduler()
    obj.parameters = [p for p in obj.model.parameters() if p.requires_grad]
    obj.optimizer = torch.optim.AdamW(obj.parameters, lr=obj.cfg.lr)
    obj.noise_rng = torch.Generator().manual_seed(3)
    obj.anchor_rng = torch.Generator().manual_seed(4)
    obj.anchor = None
    obj.step = obj.epoch = obj.batch_in_epoch = obj.samples_seen = 0
    obj.branch_samples_seen = dict(positive=0, negative=0)
    obj.timing = dict(train=None, validation=None, save=None)
    logs, saves, observed = [], [], []
    obj._log = logs.append
    obj._save = lambda: saves.append(obj.step)
    obj.model.register_forward_pre_hook(
        lambda _m, _args, kwargs: observed.append(kwargs["conditional_dict"]["prompt_embeds"].clone()),
        with_kwargs=True)
    obj.train()
    assert saves == [4] and obj.samples_seen == 16
    assert obj.branch_samples_seen == dict(positive=8, negative=8)
    assert all(log["branch_samples"] == dict(positive=2, negative=2)
               for log in logs if log["event"] == "train")
    assert len(observed) == 16 and all(((x == 2) | (x == -3)).all() for x in observed)


def test_validation_evaluates_only_nonempty_single_branches(tmp_path):
    from test_guider_sft import scheduler, tiny_model

    trainer = load_source("dropout_validation_test", ROOT / "Wan21/wan_trainer/camera_guider_sft.py")
    obj = trainer.Trainer.__new__(trainer.Trainer)
    obj.cfg = load_config(ROOT / "Wan21/configs/guider2.yaml", ["validation_controls=false"])
    obj.val_data = cached_dataset(tmp_path)
    obj.raw = tiny_model()
    obj.scheduler = scheduler()
    obj.rank, obj.world = 0, 1
    obj.device, obj.anchor = torch.device("cpu"), None
    obj._to_device = lambda batch: batch
    obj._record_duration = lambda *_args: None
    logs = []
    obj._log = logs.append
    with patch.object(trainer.torch, "autocast", side_effect=lambda *_args, **_kwargs: nullcontext()):
        obj.validate()
    metrics = logs[-1]["conditions"]
    assert metrics["positive"]["count"] == 9 and metrics["negative"]["count"] == 2
    assert "full" not in metrics
