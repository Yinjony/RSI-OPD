"""All-step, detached-history on-policy BiOPD with DDP accumulation."""

from contextlib import nullcontext

import torch

from shared.algorithms.biopd import biopd_loss, guider_targets, resolve_schedule
from wan_utils.biopd_causal_adapter import new_history, set_write
from wan_utils.biopd_camera_cache import set_chunk_camera
from wan_utils.biopd_guider import GuiderBranches


class BiOPDTrainingPipeline:
    def __init__(self, student, guider, config):
        self.student = student  # DDP or plain wrapper; use DDP only for gradient forwards.
        self.raw_student = getattr(student, "module", student)
        self.guider = guider
        self.config = config
        self.scheduler = self.raw_student.get_scheduler()
        self.times, self.sigmas = resolve_schedule(self.scheduler, config.denoising_step_list,
                                                   config.warp_denoising_step)
        if len(config.step_weights) != len(self.times):
            raise ValueError("step_weights must match the number of denoising predictions")
        self.layers = len(guider.model.blocks)
        self.frame_tokens = (config.latent_height // 2) * (config.latent_width // 2)

    def run(self, conditions, viewmats, ks, noise, *, loss_scale=1.0, sync_gradients=True,
            backward=True, noise_generator=None):
        """All-step rollout; backward=False evaluates without DDP forwards/gradients."""
        cfg = self.config
        generator = self.noise_generator if noise_generator is None else noise_generator
        predictor = self.student if backward else self.raw_student
        frames_per_chunk = cfg.num_frame_per_block
        num_chunks = noise.shape[1] // frames_per_chunk
        student_history = new_history(self.layers, cfg.cache_device, cache_text_kv=False)
        guider_branches = GuiderBranches(self.guider, conditions[1:],
                                        branches_per_call=getattr(cfg, "guider_branch_batch_size", 1),
                                        cache_device=cfg.cache_device,
                                        cache_text_kv=getattr(cfg, "guider_text_kv_cache", True),
                                        cache_prope_transforms=getattr(cfg, "prope_transform_cache", True),
                                        video_kv_buffer_tokens=(noise.shape[1] - frames_per_chunk) * self.frame_tokens
                                        if getattr(cfg, "guider_video_kv_buffer", True) else None)
        output = torch.empty_like(noise)
        metrics = {}

        for chunk in range(num_chunks):
            start = chunk * frames_per_chunk
            end = start + frames_per_chunk
            x = noise[:, start:end].detach()
            common = dict(current_start=start * self.frame_tokens,
                          viewmats=viewmats[:, start:end], Ks=ks[:, start:end])
            block = self.raw_student.model.blocks[0]
            if getattr(cfg, "prope_transform_cache", True) and hasattr(block, "self_attn"):
                set_chunk_camera(student_history, common, block.self_attn.head_dim)
            guider_branches.set_chunk(common)
            for step, time in enumerate(self.times):
                timestep = torch.full(x.shape[:2], time, device=x.device, dtype=torch.float32)
                with torch.no_grad():
                    predictions = guider_branches.predict(x, timestep)
                    positive, negative = guider_targets(predictions, cfg.omega_positive,
                                                        cfg.omega_negative)
                    del predictions
                last = chunk == num_chunks - 1 and step == len(self.times) - 1
                sync = sync_gradients and last
                context = (self.student.no_sync() if backward and hasattr(self.student, "no_sync") and not sync
                           else nullcontext())
                # Also protect callers that supply an outer autocast context
                # with caching enabled. Ignore casts cached by no_grad commits;
                # preserve the caller's mixed-precision enablement and dtype.
                with context, (nullcontext() if backward else torch.no_grad()), torch.autocast(
                    x.device.type, enabled=torch.is_autocast_enabled(x.device.type),
                    dtype=torch.get_autocast_dtype(x.device.type), cache_enabled=False,
                ):
                    flow, x0 = predictor(x, conditions[0], timestep, **student_history, **common)
                    clean = x0.detach()
                    loss, values = biopd_loss(flow, positive, negative,
                                             sigma=self.sigmas[step], next_sigma=self.sigmas[step + 1],
                                             weight=cfg.step_weights[step], lam=cfg.biopd_lambda)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at chunk={chunk}, step={step}")
                    if backward and not loss.requires_grad:
                        raise RuntimeError(
                            f"Student loss is detached at chunk={chunk}, step={step}; "
                            f"grad_enabled={torch.is_grad_enabled()}, flow_requires_grad={flow.requires_grad}. "
                            "Check student LoRA requires_grad and the caller's grad context."
                        )
                    if backward:
                        (loss * loss_scale).backward()
                for name, value in values.items():
                    metrics[name] = metrics.get(name, 0) + value
                metrics[f"step_{step}_signed_loss"] = metrics.get(f"step_{step}_signed_loss", 0) + values["signed_loss"]
                del loss, flow, x0, positive, negative, values
                with torch.no_grad():
                    if step < len(self.times) - 1:
                        next_time = torch.full((x.shape[0] * frames_per_chunk,), self.times[step + 1],
                                               device=x.device, dtype=torch.float32)
                        eps = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype,
                                          generator=generator)
                        x = self.scheduler.add_noise(clean.flatten(0, 1), eps.flatten(0, 1),
                                                     next_time).unflatten(0, clean.shape[:2])
            output[:, start:end] = clean
            # Final commit cannot affect any prediction, so skip student/guider commits.
            if chunk < num_chunks - 1:
                with torch.no_grad():
                    set_write(student_history, True)
                    try:
                        self.raw_student(clean, conditions[0], torch.zeros_like(timestep),
                                         **student_history, **common)
                    finally:
                        set_write(student_history, False)
                    guider_branches.commit(clean, torch.zeros_like(timestep))
        metrics["prediction_steps"] = noise.new_tensor(num_chunks * len(self.times), dtype=torch.float32)
        return output.detach(), metrics
