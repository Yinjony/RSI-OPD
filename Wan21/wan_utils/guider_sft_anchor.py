"""On-policy velocity anchor; the original supervised loss stays separate.

Uniformly sampled (chunk, step) states estimate the expectation over all rollout
states. State transitions and clean history are detached. Student and reference
consume the SAME generated clean chunks using separate model-specific K/V caches.
"""
from contextlib import nullcontext

import torch

from shared.algorithms.biopd import resolve_schedule
from wan_utils.biopd_causal_adapter import install_biopd_attention, new_history, set_write
from wan_utils.guider_sft_training import backward_microbatch


def anchor_enabled(cfg):
    return cfg.sft_loss_type == "supervised_anchor" and cfg.anchor_eta > 0


def load_frozen_reference(cfg, device):
    """Reload the training START checkpoint, never the resumed/current adapter."""
    from wan_utils.biopd_io import load_generator_state
    from wan_utils.wan_wrapper import WanDiffusionWrapper

    reference = WanDiffusionWrapper(model_path=cfg.model_path, is_causal=True, use_camera=True,
                                    local_attn_size=20, timestep_shift=cfg.timestep_shift, sink_size=0)
    reference.load_state_dict(load_generator_state(cfg.generator_ckpt, "generator"), strict=True)
    reference.model.num_frame_per_block = cfg.num_frame_per_block
    reference.model.independent_first_frame = False
    reference.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()
    install_biopd_attention(reference.model)
    return reference


class OnPolicyAnchor:
    def __init__(self, student, reference, cfg):
        self.student = student
        self.raw = getattr(student, "module", student)
        self.reference, self.cfg = reference, cfg
        if any(p.requires_grad for p in reference.parameters()):
            raise ValueError("Anchor reference must be frozen")
        self.scheduler = self.raw.get_scheduler()
        self.times, _ = resolve_schedule(self.scheduler, cfg.anchor_denoising_step_list,
                                         cfg.anchor_warp_denoising_step)
        self.layers = len(self.raw.model.blocks)

    def loss(self, batch, generator, *, condition="full", backward=False,
             loss_scale=1.0, synchronize=True, state_indices=None):
        """Return detached per-example mean MSE; optionally backward state by state.

        At most one activation graph is live. DDP synchronizes only the LAST
        selected state of the last accumulation microbatch, including gradients
        accumulated by the preceding supervised backward.
        """
        shape = batch["clean_latent"].shape  # shape only; no GT video in anchor rollout
        b, frames, _, height, width = shape
        block_size = self.cfg.num_frame_per_block
        if frames % block_size:
            raise ValueError("Anchor latent frames must be divisible by chunk size")
        num_states = frames // block_size * len(self.times)
        count = self.cfg.anchor_states_per_batch
        if state_indices is None:
            if isinstance(count, bool) or int(count) != count or not 1 <= count <= num_states:
                raise ValueError(f"anchor_states_per_batch must be an integer in [1, {num_states}]")
            state_indices = torch.randperm(num_states, device=batch["clean_latent"].device,
                                           generator=generator)[:count].sort().values.tolist()
        if (not state_indices or len(set(state_indices)) != len(state_indices)
                or any(type(i) is not int or not 0 <= i < num_states for i in state_indices)):
            raise ValueError("Anchor state indices must be distinct valid chunk/step indices")
        selected = set(state_indices)
        last_selected = max(selected)
        noise = torch.randn(shape, device=batch["clean_latent"].device,
                            dtype=batch["clean_latent"].dtype, generator=generator)
        conditions = dict(prompt_embeds=batch["embeddings"][condition])
        # Trainable student must recompute text K/V in every differentiable call.
        history = new_history(self.layers, self.cfg.anchor_cache_device, cache_text_kv=False)
        ref_history = new_history(self.layers, self.cfg.anchor_cache_device, cache_text_kv=True)
        predictor = self.student if backward else self.raw
        total = noise.new_zeros(b, dtype=torch.float32)
        was_training = self.raw.training
        self.raw.eval()
        self.reference.eval()
        try:
            for chunk in range(frames // block_size):
                start, end = chunk * block_size, (chunk + 1) * block_size
                x = noise[:, start:end].detach()
                common = dict(current_start=start * (height // 2) * (width // 2),
                              viewmats=batch["viewmats"][:, start:end], Ks=batch["Ks"][:, start:end])
                for step, time in enumerate(self.times):
                    state = chunk * len(self.times) + step
                    timestep = torch.full(x.shape[:2], time, device=x.device, dtype=torch.float32)
                    if state in selected:
                        sync = synchronize and state == last_selected
                        ddp_context = (self.student.no_sync() if backward
                                       and hasattr(self.student, "no_sync") and not sync else nullcontext())
                        # Avoid reusing casts made in no-grad rollout/clean commits
                        # across this grad forward or checkpoint recomputation.
                        with ddp_context, (nullcontext() if backward else torch.no_grad()), torch.autocast(
                            x.device.type, enabled=torch.is_autocast_enabled(x.device.type),
                            dtype=torch.get_autocast_dtype(x.device.type), cache_enabled=False,
                        ):
                            flow, x0 = predictor(x, conditions, timestep, **history, **common)
                            clean = x0.detach()
                            with torch.no_grad():
                                ref_flow, _ = self.reference(x, conditions, timestep, **ref_history, **common)
                            mse = (flow.float() - ref_flow.detach().float()).square().flatten(1).mean(1)
                            if not torch.isfinite(mse).all():
                                raise FloatingPointError("Nonfinite on-policy anchor loss")
                            if backward:
                                (mse.mean() * loss_scale / len(selected)).backward()
                            total += mse.detach() / len(selected)
                            del flow, x0, ref_flow, mse
                        if state == last_selected:
                            return total.detach()
                    else:
                        with torch.no_grad():
                            _, clean = self.raw(x, conditions, timestep, **history, **common)
                            clean = clean.detach()
                    if step < len(self.times) - 1:
                        with torch.no_grad():
                            eps = torch.randn(clean.shape, device=x.device, dtype=x.dtype, generator=generator)
                            next_time = torch.full((b * block_size,), self.times[step + 1],
                                                   device=x.device, dtype=torch.float32)
                            x = self.scheduler.add_noise(clean.flatten(0, 1), eps.flatten(0, 1),
                                                         next_time).unflatten(0, clean.shape[:2]).detach()
                # Both models encode the SAME student-generated clean history.
                with torch.no_grad():
                    zero = torch.zeros_like(timestep)
                    for model, caches in ((self.raw, history), (self.reference, ref_history)):
                        set_write(caches, True)
                        try:
                            model(clean, conditions, zero, **caches, **common)
                        finally:
                            set_write(caches, False)
        finally:
            self.raw.train(was_training)
        raise RuntimeError("No anchor state was evaluated")


# Keep imports used by older scripts and tests compatible with four-step runs.
FourStepAnchor = OnPolicyAnchor


def backward_sft_with_anchor(model, batch, scheduler, supervised_generator, anchor, anchor_generator,
                             *, eta, accumulation, synchronize, amp=False, **loss_kwargs):
    """L = original supervised MSE + eta * sampled on-policy reference MSE."""
    supervised = backward_microbatch(model, batch, scheduler, supervised_generator,
                                    accumulation=accumulation,
                                    synchronize=synchronize and anchor is None,
                                    amp=amp, **loss_kwargs)
    if anchor is None:
        anchor_loss = torch.zeros_like(supervised)
    else:
        with torch.autocast(device_type=batch["clean_latent"].device.type,
                            dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            anchor_loss = anchor.loss(batch, anchor_generator, backward=True,
                                      condition=loss_kwargs.get("condition", "full"),
                                      loss_scale=eta / accumulation, synchronize=synchronize)
    return dict(loss=supervised + eta * anchor_loss, supervised_loss=supervised, anchor_loss=anchor_loss)
