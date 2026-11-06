"""Standalone DDP LoRA BiOPD trainer. No critic, rewards, or legacy DMD trainer."""

import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from wan_utils.biopd_dataset import collate_conditions
from wan_utils.biopd_io import (adapter_state, file_identity, load_source,
                               merged_generator_state, restore_adapter)
from wan_utils.biopd_lora import save_adapter
from wan_utils.biopd_progress import TrainingETA, format_duration, progress_message
from wan_utils.biopd_validation import (sample_noise_seed, validation_indices, validation_kind,
                                       validation_metric_names, validation_report)


def resume_contract(config, dataset, world_size):
    # These values define the objective, sample stream, or optimizer evolution.
    ignored = {"max_steps", "save_every", "log_every", "output_dir", "save_merged",
               "resume", "text_cache_size", "text_encoder_offload", "cache_device",
               "gradient_checkpointing", "text_cache_dir", "guider_branch_batch_size", "guider_text_kv_cache",
               "prope_transform_cache", "guider_video_kv_buffer",
               "val_every", "val_num_samples", "val_seed",
               "val_at_start", "val_full_at_end"}
    cfg = OmegaConf.to_container(config, resolve=True)
    contract = {k: v for k, v in cfg.items() if k not in ignored}
    contract["dataset_sha256"] = dataset.sha256
    if dataset.split_info:
        contract["dataset_partition"] = dataset.split_info
    contract["world_size"] = world_size
    contract["student_identity"] = file_identity(config.generator_ckpt)
    contract["guider_identity"] = file_identity(config.guider_ckpt)
    if config.get("initial_adapter"):
        from wan_utils.guider_sft_lora_init import read_initial_adapter, validate_initial_adapter
        payload, identities = read_initial_adapter(config.initial_adapter)
        validate_initial_adapter(payload, config)
        contract["initial_adapter_identity"] = identities
    if config.guider_adapter_ckpt:
        adapter_dir = Path(config.guider_adapter_ckpt)
        contract["guider_adapter_identity"] = [file_identity(p) for p in sorted(adapter_dir.iterdir()) if p.is_file()]
    return contract


class Trainer:
    def __init__(self, config, dataset, validation_dataset=None):
        if dataset.split_info and dataset.split_info["selected"] != "train":
            raise ValueError("BiOPD trainer accepts only the train partition")
        if config.get("val_every", 0):
            if validation_dataset is None or not validation_dataset.split_info:
                raise ValueError("Automatic validation requires the held-out val partition")
            if (validation_dataset.split_info["selected"] != "val" or not dataset.split_info
                    or dataset.sha256 != validation_dataset.sha256
                    or dataset.split_info["sha256"] != validation_dataset.split_info["sha256"]):
                raise ValueError("Train/val must come from the same annotations and split manifest")
            if set(x["sample_id"] for x in dataset.items) & set(x["sample_id"] for x in validation_dataset.items):
                raise ValueError("Train/val sample_ids overlap")
        self.config, self.dataset = config, dataset
        self.validation_dataset = validation_dataset
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if int(os.environ.get("SP_SIZE", "1")) != 1:
            raise ValueError("This entrypoint uses DDP; unset SP_SIZE or set SP_SIZE=1")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. --check-data and CPU tests can run without a GPU.")
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        if self.world_size > 1:
            dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        root = Path(__file__).resolve().parents[2]
        self.text = None
        if config.get("text_cache_dir"):
            from wan_utils.biopd_text_cache import PreencodedTextConditions
            self.text = PreencodedTextConditions(config, dataset, self.device)
            if validation_dataset is not None:
                PreencodedTextConditions(config, validation_dataset, self.device)
            print(f"[rank {self.rank}] Using preencoded text; T5 will not be loaded: {config.text_cache_dir}",
                  flush=True)
        model_module = load_source("biopd_camera_model", root / "Wan21/model/camera_biopd.py")
        pipeline_module = load_source("biopd_training_pipeline", root / "Wan21/pipeline/biopd_training.py")
        self.raw_student, self.guider = model_module.load_models(config, self.device)
        self.trainable = [p for p in self.raw_student.parameters() if p.requires_grad]
        if not self.trainable:
            raise RuntimeError("No student LoRA parameters were enabled")
        self.optimizer = torch.optim.AdamW(self.trainable, lr=config.lr,
                                          betas=(config.beta1, config.beta2),
                                          weight_decay=config.weight_decay)
        self.contract = resume_contract(config, dataset, self.world_size)
        self.step, self.epoch, self.batch_in_epoch = 0, 0, 0
        self.noise_generator = torch.Generator(device=self.device).manual_seed(config.seed + self.rank)
        self.resume_rng = None
        if config.resume:
            self._restore(config.resume)
        print(f"[rank {self.rank}] Preparing DDP / parameter synchronization", flush=True)
        self.student = (DDP(self.raw_student, device_ids=[self.local_rank],
                            broadcast_buffers=False, find_unused_parameters=False)
                        if self.world_size > 1 else self.raw_student)
        print(f"[rank {self.rank}] Student synchronization ready", flush=True)
        if self.text is None:
            self.text = model_module.FrozenTextConditions(config, self.device)
        self.pipeline = pipeline_module.BiOPDTrainingPipeline(self.student, self.guider, config)
        self.pipeline.noise_generator = self.noise_generator
        self.sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank,
                                          shuffle=True, seed=config.seed, drop_last=False)
        # Separate loader RNG: making/skipping iterators must not alter rollout noise.
        self.loader = DataLoader(dataset, batch_size=config.batch_size, sampler=self.sampler,
                                 collate_fn=collate_conditions, num_workers=0, drop_last=False,
                                 generator=torch.Generator().manual_seed(config.seed))
        self.output_dir = Path(config.output_dir)
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if (self.output_dir / "metrics.jsonl").exists() and not config.resume:
                raise FileExistsError("Output has an existing run. Use --resume or a new output_dir.")
            OmegaConf.save(config, self.output_dir / "config.yaml")
            if config.get("text_cache_dir"):
                (self.output_dir / "text_cache_report.json").write_text(json.dumps(self.text.report, indent=2))
            (self.output_dir / "data_report.json").write_text(json.dumps(dataset.report(), indent=2))
            if dataset.split_info:
                dataset.write_splits(self.output_dir / "data_splits")
            if config.get("val_every", 0):
                (self.output_dir / "validation_report.json").write_text(
                    json.dumps(validation_report(validation_dataset, config), ensure_ascii=False, indent=2))
            print(json.dumps(dict(event="initialized", world_size=self.world_size,
                                  guider_branch_batch_size=config.get("guider_branch_batch_size", 1),
                                  guider_text_kv_cache=config.get("guider_text_kv_cache", True),
                                  prope_transform_cache=config.get("prope_transform_cache", True),
                                  guider_video_kv_buffer=config.get("guider_video_kv_buffer", True),
                                  trainable_parameters=sum(p.numel() for p in self.trainable),
                                  effective_batch_size=config.batch_size * self.world_size * config.grad_accum_steps,
                                  times=self.pipeline.times, sigmas=self.pipeline.sigmas,
                                  data=dataset.report()), ensure_ascii=False), flush=True)
            if len(dataset) < self.world_size * config.batch_size:
                print("Dataset is smaller than one global batch; DistributedSampler repeats records. "
                      "This is suitable for a smoke test, not evidence of preference improvement.", flush=True)
        if self.world_size > 1:
            dist.barrier()
        if self.resume_rng is not None:
            torch.set_rng_state(self.resume_rng["torch"])
            torch.cuda.set_rng_state(self.resume_rng["cuda"], self.device)
            random.setstate(self.resume_rng["python"])

    def _restore(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != "wan_biopd_lora_v1":
            raise ValueError("Expected a BiOPD training_state.pt checkpoint")
        if state["contract"] != self.contract:
            changed = [k for k in self.contract if self.contract[k] != state["contract"].get(k)]
            raise ValueError(f"Resume requires the same training contract; changed: {changed}")
        restore_adapter(self.raw_student.model, state["adapter"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step, self.epoch, self.batch_in_epoch = state["step"], state["epoch"], state["batch_in_epoch"]
        self.resume_rng = state["rng_by_rank"][self.rank]
        self.noise_generator.set_state(self.resume_rng["noise"])

    def _batches(self):
        while True:
            self.sampler.set_epoch(self.epoch)
            start = self.batch_in_epoch
            for index, batch in enumerate(self.loader):
                if index < start:
                    continue
                self.batch_in_epoch = index + 1
                yield batch
            self.epoch += 1
            self.batch_in_epoch = 0

    def _save(self):
        rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(self.device).cpu(),
                   python=random.getstate(), noise=self.noise_generator.get_state().cpu())
        states = [None] * self.world_size
        if self.world_size > 1:
            dist.all_gather_object(states, rng)
        else:
            states[0] = rng
        if self.rank == 0:
            target = self.output_dir / f"step_{self.step:07d}"
            staging = self.output_dir / f"step_{self.step:07d}.incomplete"
            staging.mkdir(exist_ok=False)
            save_adapter(self.raw_student.model, staging / "adapter", rank=self.config.lora_rank,
                         alpha=self.config.lora_alpha, targets=self.config.lora_target_modules,
                         base_model_path=self.config.model_path)
            torch.save(dict(format="wan_biopd_lora_v1", step=self.step, epoch=self.epoch,
                            batch_in_epoch=self.batch_in_epoch, contract=self.contract,
                            adapter=adapter_state(self.raw_student.model),
                            optimizer=self.optimizer.state_dict(), rng_by_rank=states),
                       staging / "training_state.pt")
            if self.config.save_merged:
                torch.save({"generator": merged_generator_state(self.raw_student)}, staging / "model.pt")
            OmegaConf.save(self.config, staging / "config.yaml")
            (staging / "metadata.json").write_text(json.dumps(dict(step=self.step, contract=self.contract), indent=2))
            if target.exists():
                raise FileExistsError(target)
            staging.rename(target)
            (self.output_dir / "latest.txt").write_text(str(target / "training_state.pt") + "\n")
            print(f"Saved {target}", flush=True)
        if self.world_size > 1:
            dist.barrier()

    @torch.no_grad()
    def _validate(self, *, full=False):
        """Sample-weighted DDP evaluation; preserve training RNG and gradients."""
        cfg = self.config
        indices = validation_indices(self.validation_dataset, cfg, full=full)
        periodic = set(validation_indices(self.validation_dataset, cfg))
        names = validation_metric_names(len(self.pipeline.times))
        # Two rows: selected evaluation population and the fixed periodic subset.
        # Manual rank striding avoids DistributedSampler's padding/duplication.
        totals = torch.zeros((2, len(names) + 1), device=self.device, dtype=torch.float64)
        started = time.monotonic()
        if self.rank == 0:
            print(json.dumps(dict(event="validation_started", step=self.step, full=full,
                                  num_samples=len(indices))), flush=True)
        python_rng = random.getstate()
        devices = [self.device.index] if self.device.type == "cuda" else []
        local_indices = indices[self.rank::self.world_size]
        try:
            with torch.random.fork_rng(devices=devices):
                for position, index in enumerate(local_indices, 1):
                    batch = collate_conditions([self.validation_dataset[index]])
                    sample_id = batch["sample_ids"][0]
                    generator = torch.Generator(device=self.device).manual_seed(
                        sample_noise_seed(sample_id, cfg.get("val_seed", 12345)))
                    conditions = self.text.encode(batch["texts"])
                    viewmats = batch["viewmats"].to(device=self.device, dtype=torch.bfloat16)
                    ks = batch["Ks"].to(device=self.device, dtype=torch.bfloat16)
                    noise = torch.randn((1, cfg.num_training_frames, 16, cfg.latent_height, cfg.latent_width),
                                        device=self.device, dtype=torch.bfloat16, generator=generator)
                    with torch.autocast(self.device.type, dtype=torch.bfloat16, cache_enabled=False):
                        output, values = self.pipeline.run(conditions, viewmats, ks, noise,
                                                          backward=False, noise_generator=generator)
                    packed = torch.stack([values[name].detach().double() for name in names])
                    totals[0, :-1] += packed
                    totals[0, -1] += 1
                    if index in periodic:
                        totals[1, :-1] += packed
                        totals[1, -1] += 1
                    del output, values, conditions, noise, viewmats, ks, batch, packed
                    if self.rank == 0 and (position == 1 or position % 5 == 0 or position == len(local_indices)):
                        local_eta = (time.monotonic() - started) / position * (len(local_indices) - position)
                        print(json.dumps(dict(event="validation_progress", step=self.step, rank=0,
                                              local_samples_done=position,
                                              local_samples_total=len(local_indices),
                                              local_eta_seconds=local_eta, local_eta=format_duration(local_eta)),
                                         ensure_ascii=False), flush=True)
        finally:
            random.setstate(python_rng)
        if self.world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        count = int(totals[0, -1].item())
        if count != len(indices) or not count:
            raise RuntimeError(f"Validation sample count mismatch: {count} vs {len(indices)}")
        elapsed = torch.tensor(time.monotonic() - started, device=self.device, dtype=torch.float64)
        if self.world_size > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        row = dict(event="validation", step=self.step, full=full, num_samples=count,
                   seed=cfg.get("val_seed", 12345), seconds=elapsed.item(),
                   **dict(zip(names, (totals[0, :-1] / count).cpu().tolist())))
        if full:
            subset_count = int(totals[1, -1].item())
            row["periodic_subset"] = dict(num_samples=subset_count,
                                          **dict(zip(names, (totals[1, :-1] / subset_count).cpu().tolist())))
        if self.rank == 0:
            with (self.output_dir / "validation_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        return row

    def _show_progress(self, phase):
        snapshot = self.progress_timer.snapshot(self.step)
        print(json.dumps(dict(event="progress", phase=phase, **snapshot), ensure_ascii=False), flush=True)
        print(progress_message(phase, snapshot), flush=True)

    def _validation_with_progress(self, *, full=False):
        started = time.monotonic()
        self._validate(full=full)
        if self.rank == 0:
            count = len(validation_indices(self.validation_dataset, self.config, full=full))
            self.progress_timer.observe_validation(self.step, count, time.monotonic() - started)
            self._show_progress("验证完成")

    def train(self):
        cfg = self.config
        if self.rank == 0:
            self.progress_timer = TrainingETA(cfg, world_size=self.world_size, start_step=self.step,
                                              val_size=len(self.validation_dataset) if self.validation_dataset is not None else 0)
            self._show_progress("开始")
        batches = self._batches()
        if self.step == 0 and validation_kind(0, cfg):
            self._validation_with_progress()
        while self.step < cfg.max_steps:
            started = time.monotonic()
            self.optimizer.zero_grad(set_to_none=True)
            stats = {}
            for micro in range(cfg.grad_accum_steps):
                batch = next(batches)
                conditions = self.text.encode(batch["texts"])
                viewmats = batch["viewmats"].to(device=self.device, dtype=torch.bfloat16)
                ks = batch["Ks"].to(device=self.device, dtype=torch.bfloat16)
                noise = torch.randn((len(batch["sample_ids"]), cfg.num_training_frames, 16,
                                     cfg.latent_height, cfg.latent_width), device=self.device,
                                    dtype=torch.bfloat16, generator=self.noise_generator)
                # This rollout alternates grad forwards and no_grad clean-cache
                # commits on the SAME student. Cached casts from a no_grad
                # forward can detach the FP32 LoRA weights in later forwards.
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    video_latents, values = self.pipeline.run(
                        conditions, viewmats, ks, noise, loss_scale=1 / cfg.grad_accum_steps,
                        sync_gradients=micro == cfg.grad_accum_steps - 1)
                for name, value in values.items():
                    stats[name] = stats.get(name, 0) + value.detach() / cfg.grad_accum_steps
                del video_latents, conditions, noise, values
            grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, cfg.max_grad_norm,
                                                       error_if_nonfinite=True)
            self.optimizer.step()
            self.step += 1
            stats["grad_norm"] = grad_norm.detach()
            names = sorted(stats)
            packed = torch.stack([stats[name].float() for name in names])
            if self.world_size > 1:
                dist.all_reduce(packed)
                packed /= self.world_size
            if self.rank == 0:
                metric_values = packed.cpu().tolist()
                step_seconds = time.monotonic() - started
                self.progress_timer.observe_train(step_seconds)
                if self.step % cfg.log_every == 0 or self.step == 1 or self.step == cfg.max_steps:
                    estimate = self.progress_timer.snapshot(self.step)
                    row = dict(epoch=self.epoch, seconds=step_seconds, **estimate,
                               **dict(zip(names, metric_values)))
                    with (self.output_dir / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    print(progress_message("训练", estimate), flush=True)
            kind = validation_kind(self.step, cfg)
            if kind:
                self._validation_with_progress(full=kind == "full")
            if self.step % cfg.save_every == 0 or self.step == cfg.max_steps:
                save_started = time.monotonic()
                self._save()
                if self.rank == 0:
                    self.progress_timer.observe_checkpoint(self.step, time.monotonic() - save_started)
                    self._show_progress("保存完成")
        if self.rank == 0:
            self._show_progress("完成")
