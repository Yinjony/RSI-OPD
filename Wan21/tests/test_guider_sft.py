"""CPU regression tests, including two-process Gloo LoRA gradient synchronization.

Run: python -m unittest discover -s Wan21/tests -p 'test_guider_sft.py' -v
"""
import copy
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT)]
from wan_utils.biopd_lora import get_adapter_state, inject_lora, set_adapter_state
from wan_utils.guider_sft_config import load_config
from wan_utils.guider_sft_dataset import (
    FORMAT, GuiderSFTDataset, atomic_json, atomic_save, digest, file_sha256, format_prompt,
    load_records, split_records, validate_token_lengths,
)
from wan_utils.guider_sft_training import (
    adapter_payload, backward_microbatch, make_noisy_batch, merge_adapter_state,
    restore_adapter_payload, sft_loss,
)
from wan_utils.scheduler import FlowMatchScheduler

TARGETS = ["q", "k", "v", "o", "prope_o"]


class TinyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({name: nn.Linear(4, 4) for name in TARGETS})

    def forward(self, noisy_image_or_video, timestep, conditional_dict, clean_x, viewmats, Ks):
        # Every target participates, and all production condition arguments reach the wrapper.
        x = noisy_image_or_video
        context = conditional_dict["prompt_embeds"].mean() + viewmats.mean() + Ks.mean() + clean_x.mean()
        pred = sum(layer(x) for layer in self.model.values()) / len(TARGETS) + context * 0.01
        return pred, x - timestep[..., None, None, None] / 1000 * pred


def tiny_model():
    torch.manual_seed(7)
    model = TinyWrapper()
    inject_lora(model.model, 64, 64, TARGETS)
    return model


def scheduler():
    result = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
    result.set_timesteps(1000, training=True)
    return result


def batch(seed):
    g = torch.Generator().manual_seed(seed)
    return dict(clean_latent=torch.randn(1, 8, 1, 2, 4, generator=g),
                embeddings={"full": torch.randn(1, 3, 4, generator=g)},
                viewmats=torch.eye(4).expand(1, 8, 4, 4), Ks=torch.eye(3).expand(1, 8, 3, 3))


def ddp_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        raw = tiny_model()
        model = DDP(raw, broadcast_buffers=False)
        rng = torch.Generator().manual_seed(100 + rank)
        for micro in range(2):
            backward_microbatch(model, batch(rank * 10 + micro), scheduler(), rng,
                                accumulation=2, synchronize=micro == 1)
        grads = {n: p.grad.clone() for n, p in raw.named_parameters() if p.requires_grad}
        torch.save(grads, Path(directory) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class GuiderSFTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_eta_counts_future_validation_and_final_checkpoint(self):
        spec = importlib.util.spec_from_file_location("sft_eta_test", ROOT / "Wan21/wan_trainer/camera_guider_sft.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.pending_events(0, 1425, 475), 3)
        self.assertEqual(mod.pending_events(475, 1425, 475), 2)
        self.assertEqual(mod.pending_events(1425, 1425, 475), 0)
        self.assertEqual(mod.pending_events(1425, 1425, 250), 1)
        self.assertEqual(mod.format_duration(3661.1), "01:01:02")
        trainer = mod.Trainer.__new__(mod.Trainer)
        trainer.cfg = SimpleNamespace(max_steps=1425, val_every=475, save_every=250)
        trainer.val_data = list(range(100))
        trainer.step = 475
        trainer.timing = dict(train=10.0, validation=120.0, save=5.0)
        self.assertEqual(trainer._estimated_remaining(), 950 * 10 + 2 * 120 + 5 * 5)

    def test_config_defaults_and_rejection(self):
        cfg = load_config(ROOT / "Wan21/configs/guider1.yaml")
        self.assertEqual((cfg.lora_rank, cfg.lora_alpha), (64, 64))
        self.assertEqual(cfg.loss_weighting, "uniform")
        with self.assertRaises(ValueError):
            load_config(ROOT / "Wan21/configs/guider1.yaml", ["grad_accum_steps=0"])

    def test_attributes_preserve_observed_negative_and_empty_semantics(self):
        attrs = [{"attribute_id": 1, "name": "Missing walking motion"}]
        text = format_prompt("Robot walking", [], attrs)
        self.assertIn("Observed negative video attributes:\nMissing walking motion.", text)
        self.assertNotIn("positive", text)
        self.assertEqual(format_prompt("Robot walking", [], []), "Robot walking")
        self.assertNotIn("attribute_id", text)

    def test_paths_actions_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video.mp4").touch()
            record = dict(sample_id="one", video_path="video.mp4", prompt="Forest", action="j*8,a*11",
                          positive_attributes=[], negative_attributes=[])
            path = root / "input.json"
            atomic_json([record], path)
            rs = load_records(path, dict(fx=.5, fy=.9, cx=.5, cy=.5))
            self.assertEqual(rs[0]["video_path"], str(root / "video.mp4"))
            record["action"] = "j*8"
            atomic_json([record], path)
            with self.assertRaises(ValueError):
                load_records(path, dict(fx=.5, fy=.9, cx=.5, cy=.5))
        records = [dict(video_path=str(i // 2), negative_attributes=[{}] if i < 8 else []) for i in range(24)]
        splits = split_records(records, .2, 42)
        self.assertEqual(splits, split_records(records, .2, 42))
        for i in range(0, 24, 2):
            self.assertEqual(splits[i], splits[i + 1])
        self.assertIn("val", splits[:8])
        self.assertIn("train", splits[:8])

    def test_token_overflow_is_error(self):
        def fake_tokenizer(text, **kwargs):
            self.assertFalse(kwargs["truncation"])
            return torch.zeros(1, len(text), dtype=torch.long)
        self.assertEqual(validate_token_lengths(fake_tokenizer, {"full": "abc"}), {"full": 3})
        with self.assertRaises(ValueError):
            validate_token_lengths(fake_tokenizer, {"full": "x" * 513})

    def test_cache_hash_detects_changed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {"test": True}
            signature = digest(contract)
            sample = dict(signature=signature, sample_id="one", clean_latent=torch.zeros(8, 1, 2, 4),
                          embeddings={"full": torch.ones(3, 4)}, viewmats=torch.zeros(8, 4, 4), Ks=torch.zeros(8, 3, 3))
            atomic_save(sample, root / "sample.pt")
            atomic_json(dict(format=FORMAT, signature=signature, contract=contract,
                             records=[dict(split="train", sample_id="one", cache_file="sample.pt",
                                           negative_attributes=[], sha256=file_sha256(root / "sample.pt"))]), root / "manifest.json")
            data = GuiderSFTDataset(root, "train")
            self.assertEqual(data[0]["sample_id"], "one")
            atomic_save({"different": True}, root / "sample.pt")
            with self.assertRaises(ValueError):
                GuiderSFTDataset(root, "train")

    def test_flow_target_and_block_times(self):
        clean = batch(0)["clean_latent"]
        sch = scheduler()
        noisy, t, target = make_noisy_batch(clean, sch, torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(t[:, :4], t[:, :1].expand(-1, 4)))
        self.assertTrue(torch.equal(t[:, 4:], t[:, 4:5].expand(-1, 4)))
        torch.testing.assert_close(noisy - (t / 1000)[..., None, None, None] * target, clean)

    def test_lora_initial_parity_frozen_base_and_merge(self):
        torch.manual_seed(7)
        original = TinyWrapper()
        base = copy.deepcopy(original.state_dict())
        model = tiny_model()
        args = dict(noisy_image_or_video=batch(1)["clean_latent"], timestep=torch.ones(1, 8),
                    conditional_dict={"prompt_embeds": batch(1)["embeddings"]["full"]},
                    clean_x=batch(1)["clean_latent"], viewmats=batch(1)["viewmats"], Ks=batch(1)["Ks"])
        torch.testing.assert_close(original(**args)[0], model(**args)[0], rtol=0, atol=0)
        cfg = SimpleNamespace(lora_rank=64, lora_alpha=64, lora_target_modules=TARGETS, model_path="tiny")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01, weight_decay=0)
        losses = []
        for _ in range(12):
            opt.zero_grad()
            loss = sft_loss(model, batch(1), scheduler(), torch.Generator().manual_seed(10)).mean()
            loss.backward()
            for n, p in model.named_parameters():
                self.assertEqual(p.requires_grad, ".lora_" in n)
                if not p.requires_grad:
                    self.assertIsNone(p.grad)
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])
        payload = adapter_payload(model, cfg, {"path": "tiny"})
        loaded = tiny_model()
        restore_adapter_payload(loaded, payload, cfg, {"path": "tiny"})
        torch.testing.assert_close(loaded(**args)[0], model(**args)[0], rtol=0, atol=0)
        merged = merge_adapter_state(base, payload)
        original.load_state_dict(merged, strict=True)
        torch.testing.assert_close(original(**args)[0], model(**args)[0], atol=2e-6, rtol=2e-6)
        for n, p in model.state_dict().items():
            if ".base_layer." in n:
                torch.testing.assert_close(p, base[n.replace(".base_layer.", ".")], rtol=0, atol=0)
        corrupt = copy.deepcopy(payload)
        corrupt["adapter"].pop(next(iter(corrupt["adapter"])))
        with self.assertRaises((ValueError, KeyError)):
            merge_adapter_state(base, corrupt)

    def test_resume_optimizer_rng_and_actual_sample_stream(self):
        spec = importlib.util.spec_from_file_location("sft_trainer_test", ROOT / "Wan21/wan_trainer/camera_guider_sft.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def stream(epoch=0, offset=0):
            obj = mod.Trainer.__new__(mod.Trainer)
            obj.epoch, obj.batch_in_epoch = epoch, offset
            obj.sampler = DistributedSampler(list(range(7)), num_replicas=1, rank=0, seed=42)
            obj.loader = DataLoader(list(range(7)), sampler=obj.sampler, batch_size=2, drop_last=True,
                                    generator=torch.Generator().manual_seed(5))
            return obj, obj._batches()

        obj, it = stream()
        for _ in range(3):  # exact epoch boundary
            next(it)
        restored, restored_it = stream(obj.epoch, obj.batch_in_epoch)
        for _ in range(8):
            torch.testing.assert_close(next(it), next(restored_it))
        model = tiny_model()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        rng = torch.Generator().manual_seed(22)
        def step(m, o, g):
            o.zero_grad()
            sft_loss(m, batch(3), scheduler(), g).mean().backward()
            o.step()
        step(model, opt, rng)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "state.pt"
            atomic_save(dict(adapter=get_adapter_state(model.model), optimizer=opt.state_dict(),
                             noise=rng.get_state(), python=random.getstate()), p)
            state = torch.load(p, weights_only=True)
            resumed = tiny_model()
            set_adapter_state(resumed.model, state["adapter"])
            opt2 = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=.01)
            opt2.load_state_dict(state["optimizer"])
            rng2 = torch.Generator().set_state(state["noise"])
            for _ in range(3):
                step(model, opt, rng)
                step(resumed, opt2, rng2)
            for k, v in get_adapter_state(model.model).items():
                torch.testing.assert_close(v, get_adapter_state(resumed.model)[k], rtol=0, atol=0)

    def test_two_rank_ddp_accumulation_matches_global_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker, args=(str(Path(directory) / "rendezvous"), directory), nprocs=2, join=True)
            reference = tiny_model()
            for rank in range(2):
                rng = torch.Generator().manual_seed(100 + rank)
                for micro in range(2):
                    backward_microbatch(reference, batch(rank * 10 + micro), scheduler(), rng,
                                        accumulation=4, synchronize=True)
            for rank in range(2):
                grads = torch.load(Path(directory) / f"rank{rank}.pt", weights_only=True)
                for n, p in reference.named_parameters():
                    if p.requires_grad:
                        torch.testing.assert_close(grads[n], p.grad, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
