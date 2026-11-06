"""Independent single-GPU / DDP LoRA causal teacher-forcing SFT trainer."""
import json
import math
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from wan_utils.guider_sft_dataset import atomic_json, atomic_save, file_identity
from wan_utils.guider_sft_sampling import BalancedAttributeBatchSampler
from wan_utils.guider_sft_config import OBJECTIVE_DEFAULTS
from wan_utils.guider_sft_anchor import (
    OnPolicyAnchor, anchor_enabled, backward_sft_with_anchor, load_frozen_reference,
)
from wan_utils.guider_sft_training import (
    adapter_payload, load_lora_generator, restore_adapter_payload, sft_loss,
)


def resume_contract(cfg, dataset, world_size):
    ignored = {"max_steps", "save_every", "log_every", "output_dir", "resume", "num_workers", "val_every"}
    if not cfg.get("attribute_dropout", False):
        ignored.add("attribute_dropout")
    if cfg.sft_loss_type == "supervised":
        # Preserve compatibility with original supervised-only training states.
        ignored.update(OBJECTIVE_DEFAULTS)
    return dict(config={k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items() if k not in ignored},
                cache_sha256=dataset.sha256, world_size=world_size, base_identity=file_identity(cfg.generator_ckpt))


def pending_events(step, max_steps, interval):
    """Scheduled future events plus the final event for an incomplete interval."""
    if interval <= 0 or step > max_steps:
        return 0
    return max_steps // interval - step // interval + int(max_steps % interval != 0)


def format_duration(seconds):
    total = max(0, math.ceil(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Trainer:
    def __init__(self, cfg, train_data, val_data):
        self.cfg, self.train_data, self.val_data = cfg, train_data, val_data
        self.rank, self.world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        for dataset in (train_data, val_data):
            if bool(dataset.manifest["contract"].get("attribute_dropout", False)) != cfg.attribute_dropout:
                raise ValueError("Attribute dropout setting differs from cache; preencode to a new cache_dir")
        if int(os.environ.get("SP_SIZE", 1)) != 1:
            raise ValueError("Guider SFT uses DDP, requires SP_SIZE=1")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; source/cache checks and CPU tests remain available")
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This Wan training entry requires a CUDA GPU with BF16 support")
        self.output = Path(cfg.output_dir)
        if not cfg.resume and (self.output / "config.yaml").exists():
            raise FileExistsError("Existing SFT run: use --resume or a new output_dir")
        if not len(train_data):
            raise ValueError("Training split is empty")
        loader_options = dict(num_workers=cfg.num_workers, pin_memory=True,
                              generator=torch.Generator().manual_seed(cfg.seed))
        if cfg.attribute_dropout:
            self.sampler = BalancedAttributeBatchSampler(
                train_data.records, batch_size=cfg.batch_size, accumulation=cfg.grad_accum_steps,
                world_size=self.world, rank=self.rank, seed=cfg.seed)
            self.loader = DataLoader(train_data, batch_sampler=self.sampler, **loader_options)
        else:
            self.sampler = DistributedSampler(train_data, num_replicas=self.world, rank=self.rank,
                                              shuffle=True, seed=cfg.seed, drop_last=False)
            self.loader = DataLoader(train_data, batch_size=cfg.batch_size, sampler=self.sampler,
                                     drop_last=True, **loader_options)
        if not len(self.loader):
            raise ValueError("Per-rank data smaller than batch_size; reduce batch_size")
        if self.world > 1:
            dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        self.raw = load_lora_generator(cfg, self.device)
        self.reference = None
        if anchor_enabled(cfg):
            from wan_utils.biopd_causal_adapter import install_biopd_attention
            install_biopd_attention(self.raw.model)
            self.reference = load_frozen_reference(cfg, self.device)
        self.scheduler = self.raw.get_scheduler()
        self.parameters = [p for p in self.raw.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                          weight_decay=cfg.weight_decay)
        self.contract = resume_contract(cfg, train_data, self.world)
        self.base_identity = self.contract["base_identity"]
        self.noise_rng = torch.Generator(device=self.device).manual_seed(cfg.seed + self.rank)
        self.anchor_rng = torch.Generator(device=self.device).manual_seed(cfg.seed + 100000 + self.rank)
        self.step = self.epoch = self.batch_in_epoch = self.samples_seen = 0
        self.branch_samples_seen = dict(positive=0, negative=0)
        self.timing = dict(train=None, validation=None, save=None)
        restore_rng = None
        if cfg.resume:
            state = torch.load(cfg.resume, map_location="cpu", weights_only=True)
            if state.get("format") != "wan_guider_sft_training_v1" or state["contract"] != self.contract:
                raise ValueError("Resume requires matching base model, cache, world size, optimizer and objective config")
            restore_adapter_payload(self.raw, state["adapter_payload"], cfg, self.base_identity)
            self.optimizer.load_state_dict(state["optimizer"])
            for key in ("step", "epoch", "batch_in_epoch", "samples_seen"):
                setattr(self, key, state[key])
            self.branch_samples_seen.update(state.get("branch_samples_seen", {}))
            self.timing.update(state.get("timing", {}))
            restore_rng = state["rng_by_rank"][self.rank]
            del state
        self.model = DDP(self.raw, device_ids=[self.local_rank], broadcast_buffers=False,
                         find_unused_parameters=False) if self.world > 1 else self.raw
        self.anchor = OnPolicyAnchor(self.model, self.reference, cfg) if self.reference is not None else None
        if self.rank == 0:
            self.output.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, self.output / "config.yaml")
            atomic_json(dict(contract=self.contract, train_ids=[r["sample_id"] for r in train_data.records],
                             val_ids=[r["sample_id"] for r in val_data.records]), self.output / "run_manifest.json")
            self._log(dict(event="initialized", world_size=self.world, train_samples=len(train_data),
                           val_samples=len(val_data), trainable_parameters=sum(p.numel() for p in self.parameters),
                           effective_batch_size=cfg.batch_size * cfg.grad_accum_steps * self.world,
                           attribute_dropout=cfg.attribute_dropout,
                           attribute_pools={k: len(v) for k, v in self.sampler.pools.items()}
                           if cfg.attribute_dropout else None,
                           sft_loss_type=cfg.sft_loss_type, anchor_eta=cfg.anchor_eta,
                           anchor_states_per_batch=cfg.anchor_states_per_batch if self.anchor else 0,
                           anchor_reference=self.base_identity if self.anchor else None,
                           anchor_timesteps=self.anchor.times if self.anchor else None,
                           sampler_padding_per_epoch=None if cfg.attribute_dropout else
                           len(self.sampler) * self.world - len(train_data),
                           dropped_per_rank_per_epoch=0 if cfg.attribute_dropout else
                           len(self.sampler) % cfg.batch_size))
        if self.world > 1:
            dist.barrier()
        if restore_rng is not None:
            self.noise_rng.set_state(restore_rng["noise"])
            if self.anchor is not None:
                self.anchor_rng.set_state(restore_rng["anchor"])
            torch.set_rng_state(restore_rng["torch"])
            torch.cuda.set_rng_state(restore_rng["cuda"], self.device)
            random.setstate(restore_rng["python"])

    def _log(self, record):
        if self.rank == 0:
            record = dict(step=self.step, **record)
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            print(line, flush=True)
            with (self.output / "metrics.jsonl").open("a") as f:
                f.write(line + "\n")

    def _record_duration(self, name, seconds):
        previous = self.timing[name]
        self.timing[name] = seconds if previous is None else 0.8 * previous + 0.2 * seconds

    def _estimated_remaining(self):
        remaining_steps = max(0, self.cfg.max_steps - self.step)
        validations = (pending_events(self.step, self.cfg.max_steps, self.cfg.val_every)
                       if self.cfg.val_every and len(self.val_data) else 0)
        saves = pending_events(self.step, self.cfg.max_steps, self.cfg.save_every)
        return (remaining_steps * (self.timing["train"] or 0.0)
                + validations * (self.timing["validation"] or 0.0)
                + saves * (self.timing["save"] or 0.0))

    def _batches(self):
        while True:
            self.sampler.set_epoch(self.epoch)
            start = self.batch_in_epoch
            for i, batch in enumerate(self.loader):
                if i < start:
                    continue
                self.batch_in_epoch = i + 1
                yield batch
            self.epoch += 1
            self.batch_in_epoch = 0

    def _to_device(self, batch):
        return dict(clean_latent=batch["clean_latent"].to(self.device, dtype=torch.bfloat16, non_blocking=True),
                    embeddings={k: v.to(self.device, dtype=torch.bfloat16, non_blocking=True)
                                for k, v in batch["embeddings"].items()},
                    # Keep camera geometry in FP32 for the PRoPE transforms.
                    viewmats=batch["viewmats"].to(self.device, dtype=torch.float32, non_blocking=True),
                    Ks=batch["Ks"].to(self.device, dtype=torch.float32, non_blocking=True))

    def _save(self):
        started = time.monotonic()
        rng = dict(noise=self.noise_rng.get_state().cpu(), anchor=self.anchor_rng.get_state().cpu(),
                   torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state(self.device).cpu(), python=random.getstate())
        rngs = [None] * self.world
        if self.world > 1:
            dist.all_gather_object(rngs, rng)
        else:
            rngs[0] = rng
        if self.rank == 0:
            dest = self.output / f"checkpoint_{self.step:06d}"
            if dest.exists():
                raise FileExistsError(f"Checkpoint already exists: {dest}")
            temporary = self.output / f".checkpoint_{self.step:06d}.pending"
            temporary.mkdir(parents=True, exist_ok=True)
            payload = adapter_payload(self.raw, self.cfg, self.base_identity)
            payload["cache_template"] = self.train_data.manifest["contract"]["template_version"]
            payload["camera"] = OmegaConf.to_container(self.cfg.camera)
            atomic_save(payload, temporary / "adapter.pt")
            atomic_save(dict(format="wan_guider_sft_training_v1", contract=self.contract,
                             adapter_payload=payload, optimizer=self.optimizer.state_dict(), step=self.step,
                             epoch=self.epoch, batch_in_epoch=self.batch_in_epoch, samples_seen=self.samples_seen,
                             branch_samples_seen=self.branch_samples_seen,
                             rng_by_rank=rngs, timing=self.timing), temporary / "training_state.pt")
            temporary.rename(dest)
            atomic_json(dict(step=self.step, training_state=str(dest / "training_state.pt"),
                             adapter=str(dest / "adapter.pt")), self.output / "latest.json")
            self._log(dict(event="checkpoint", directory=str(dest)))
        if self.world > 1:
            dist.barrier()
        self._record_duration("save", time.monotonic() - started)

    @torch.no_grad()
    def validate(self):
        if not len(self.val_data):
            return
        started = time.monotonic()
        training_modes = ["positive", "negative"] if self.cfg.attribute_dropout else ["full"]
        modes = training_modes + (["no_attributes", "shuffled_attributes"] if self.cfg.validation_controls else [])
        self._log(dict(event="validation_start", samples=len(self.val_data), conditions=modes))
        totals = torch.zeros(len(modes), 4, device=self.device, dtype=torch.float64)
        anchor_totals = torch.zeros(len(modes), 2, device=self.device, dtype=torch.float64)
        self.raw.eval()
        for i in range(self.rank, len(self.val_data), self.world):
            item = self.val_data[i]
            batch = dict(clean_latent=item["clean_latent"][None],
                         embeddings={k: v[None] for k, v in item["embeddings"].items()},
                         viewmats=item["viewmats"][None], Ks=item["Ks"][None])
            batch = self._to_device(batch)
            for j, condition in enumerate(modes):
                if condition in ("positive", "negative") and not item["has_" + condition]:
                    continue
                # Same sample, timestep and noise for all conditions and every validation invocation.
                rng = torch.Generator(device=self.device).manual_seed(self.cfg.val_seed + i)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = sft_loss(self.raw, batch, self.scheduler, rng,
                                    block_size=self.cfg.num_frame_per_block, weighting="uniform", condition=condition)[0]
                totals[j, 0] += loss.double()
                totals[j, 1] += 1
                if item["has_negative"]:
                    totals[j, 2] += loss.double()
                    totals[j, 3] += 1
                if condition in training_modes and self.anchor is not None:
                    anchor_rng = torch.Generator(device=self.device).manual_seed(self.cfg.val_seed + 100000 + i)
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        anchor_loss = self.anchor.loss(batch, anchor_rng, condition=condition)[0]
                    anchor_totals[j, 0] += anchor_loss.double()
                    anchor_totals[j, 1] += 1
        self.raw.train()
        if self.world > 1:
            dist.all_reduce(totals)
            dist.all_reduce(anchor_totals)
        metrics = {}
        for name, values in zip(modes, totals.cpu().tolist()):
            total, n, neg_total, neg_n = values
            metrics[name] = dict(mse=total / n if n else None, count=int(n),
                                 negative_mse=neg_total / neg_n if neg_n else None, negative_count=int(neg_n))
        if self.anchor is not None:
            for name, (anchor_total, anchor_count) in zip(modes, anchor_totals.cpu().tolist()):
                if anchor_count:
                    metrics[name].update(anchor_mse=anchor_total / anchor_count,
                                         total_loss=metrics[name]["mse"] + self.cfg.anchor_eta * anchor_total / anchor_count,
                                         anchor_eta=self.cfg.anchor_eta)
        self._log(dict(event="validation", conditions=metrics))
        self._record_duration("validation", time.monotonic() - started)

    def train(self):
        session_started = time.monotonic()
        if self.cfg.val_every:
            self.validate()
        batches, last_saved = self._batches(), self.step if self.cfg.resume else -1
        while self.step < self.cfg.max_steps:
            started = time.monotonic()
            lr = self.cfg.lr * min(1.0, (self.step + 1) / max(1, self.cfg.warmup_steps))
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            stats = torch.zeros(6, device=self.device, dtype=torch.float64)
            for micro in range(self.cfg.grad_accum_steps):
                cpu_batch = next(batches)
                if self.cfg.attribute_dropout:
                    stats[4] += cpu_batch["branch"].count("positive")
                    stats[5] += cpu_batch["branch"].count("negative")
                batch = self._to_device(cpu_batch)
                losses = backward_sft_with_anchor(self.model, batch, self.scheduler, self.noise_rng,
                                                 self.anchor, self.anchor_rng, eta=self.cfg.anchor_eta,
                                                 accumulation=self.cfg.grad_accum_steps,
                                                 synchronize=micro == self.cfg.grad_accum_steps - 1,
                                                 amp=True, block_size=self.cfg.num_frame_per_block,
                                                 weighting=self.cfg.loss_weighting,
                                                 condition="selected" if self.cfg.attribute_dropout else "full")
                for j, name in enumerate(("loss", "supervised_loss", "anchor_loss")):
                    stats[j] += losses[name].double().sum()
                stats[3] += len(losses["loss"])
            if self.world > 1:
                dist.all_reduce(stats)
            if not torch.isfinite(stats).all():
                raise FloatingPointError("Nonfinite SFT loss; optimizer step aborted")
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.cfg.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            self.step += 1
            self.samples_seen += int(stats[3].item())
            for j, name in enumerate(("positive", "negative"), start=4):
                self.branch_samples_seen[name] += int(stats[j].item())
            train_seconds = time.monotonic() - started
            self._record_duration("train", train_seconds)
            if self.cfg.val_every and self.step % self.cfg.val_every == 0:
                self.validate()
            if self.step % self.cfg.save_every == 0:
                self._save()
                last_saved = self.step
            if self.step % self.cfg.log_every == 0:
                eta_seconds = self._estimated_remaining()
                self._log(dict(event="train", loss=(stats[0] / stats[3]).item(),
                               branch_samples={name: int(stats[j].item()) for j, name in
                                               enumerate(("positive", "negative"), start=4)},
                               branch_samples_seen=dict(self.branch_samples_seen),
                               supervised_loss=(stats[1] / stats[3]).item(),
                               anchor_loss=(stats[2] / stats[3]).item(),
                               weighted_anchor_loss=(self.cfg.anchor_eta * stats[2] / stats[3]).item(),
                               sft_loss_type=self.cfg.sft_loss_type, anchor_eta=self.cfg.anchor_eta, lr=lr,
                               grad_norm=float(grad_norm), samples_seen=self.samples_seen,
                               seconds=train_seconds, iteration_seconds=time.monotonic() - started,
                               session_elapsed_seconds=time.monotonic() - session_started,
                               eta_seconds=eta_seconds, eta=format_duration(eta_seconds),
                               estimated_finish=(datetime.now().astimezone()
                                                 + timedelta(seconds=eta_seconds)).isoformat(timespec="seconds")))
        if last_saved != self.step:
            self._save()
        if self.cfg.val_every and self.step % self.cfg.val_every:
            self.validate()
        self._log(dict(event="completed", eta_seconds=0, eta="00:00:00",
                       session_elapsed_seconds=time.monotonic() - session_started))
