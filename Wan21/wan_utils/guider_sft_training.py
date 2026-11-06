"""Small, testable SFT objective and LoRA checkpoint helpers."""
from contextlib import nullcontext
import math

import torch

from wan_utils.biopd_lora import get_adapter_state, inject_lora, set_adapter_state


def load_lora_generator(cfg, device):
    from wan_utils.biopd_io import load_generator_state
    from wan_utils.wan_wrapper import WanDiffusionWrapper
    raw = WanDiffusionWrapper(model_path=cfg.model_path, is_causal=True, use_camera=True,
                              local_attn_size=20, timestep_shift=cfg.timestep_shift, sink_size=0)
    raw.load_state_dict(load_generator_state(cfg.generator_ckpt, "generator"), strict=True)
    raw.model.num_frame_per_block = cfg.num_frame_per_block
    raw.model.independent_first_frame = False
    raw.to(device=device, dtype=torch.bfloat16).requires_grad_(False)
    count = inject_lora(raw.model, cfg.lora_rank, cfg.lora_alpha, list(cfg.lora_target_modules))
    raw.train()
    if cfg.gradient_checkpointing:
        raw.enable_gradient_checkpointing()
    trainable = [(n, p) for n, p in raw.named_parameters() if p.requires_grad]
    if not trainable or any(".lora_" not in n for n, _ in trainable):
        raise RuntimeError("Expected only LoRA parameters to be trainable")
    print(f"LoRA: {count} layers, {sum(p.numel() for _, p in trainable):,} trainable parameters", flush=True)
    return raw


def make_noisy_batch(clean, scheduler, generator, block_size=4):
    b, f = clean.shape[:2]
    if f % block_size:
        raise ValueError("Latent frame count must be divisible by block size")
    index = torch.randint(len(scheduler.timesteps), (b, f // block_size),
                          device=clean.device, generator=generator).repeat_interleave(block_size, dim=1)
    # Keep scheduler times in FP32: BF16 rounding would change the selected sigma.
    timestep = scheduler.timesteps.to(clean.device)[index]
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    noisy = scheduler.add_noise(clean.flatten(0, 1), noise.flatten(0, 1), timestep.flatten()).unflatten(0, (b, f))
    target = scheduler.training_target(clean, noise, timestep)
    return noisy, timestep, target


def sft_loss(model, batch, scheduler, generator, *, block_size=4, weighting="uniform", condition="full"):
    clean = batch["clean_latent"]
    noisy, times, target = make_noisy_batch(clean, scheduler, generator, block_size)
    pred, _ = model(noisy_image_or_video=noisy, timestep=times,
                    conditional_dict={"prompt_embeds": batch["embeddings"][condition]},
                    clean_x=clean, viewmats=batch["viewmats"], Ks=batch["Ks"])
    if pred.shape != target.shape:
        raise ValueError(f"v shape mismatch: {pred.shape} vs {target.shape}")
    loss = (pred.float() - target.float()).square()
    if weighting == "scheduler":
        weight = scheduler.training_weight(times).reshape(*times.shape, 1, 1, 1)
        loss = loss * weight
    elif weighting != "uniform":
        raise ValueError(f"Unknown loss weighting: {weighting}")
    return loss.flatten(1).mean(1)


def backward_microbatch(model, batch, scheduler, generator, *, accumulation, synchronize,
                        amp=False, **loss_kwargs):
    # no_sync must enclose BOTH the DDP forward and backward.
    sync = model.no_sync() if hasattr(model, "no_sync") and not synchronize else nullcontext()
    with sync, torch.autocast(device_type=batch["clean_latent"].device.type, dtype=torch.bfloat16, enabled=amp):
        losses = sft_loss(model, batch, scheduler, generator, **loss_kwargs)
        (losses.mean() / accumulation).backward()
    return losses.detach()


def adapter_payload(raw, cfg, base_identity):
    return dict(format="wan_guider_sft_lora_v1", rank=int(cfg.lora_rank), alpha=float(cfg.lora_alpha),
                dropout=0.0, targets=list(cfg.lora_target_modules), base_identity=base_identity,
                model_path=str(cfg.model_path), adapter=get_adapter_state(raw.model))


def restore_adapter_payload(raw, payload, cfg, base_identity):
    expected = dict(format="wan_guider_sft_lora_v1", rank=int(cfg.lora_rank), alpha=float(cfg.lora_alpha),
                    dropout=0.0, targets=list(cfg.lora_target_modules), base_identity=base_identity)
    for k, v in expected.items():
        if payload.get(k) != v:
            raise ValueError(f"Incompatible adapter {k}: {payload.get(k)!r} != {v!r}")
    set_adapter_state(raw.model, payload["adapter"])


def merge_adapter_state(base_state, payload):
    """CPU merge without constructing a Wan model; validate every adapter tensor."""
    if payload.get("format") != "wan_guider_sft_lora_v1":
        raise ValueError("Expected a guider SFT adapter.pt")
    rank, alpha = payload["rank"], payload["alpha"]
    if not isinstance(rank, int) or rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid LoRA rank/alpha")
    adapter = payload["adapter"]
    result, consumed = dict(base_state), set()
    for key, a in adapter.items():
        prefix, suffix = "base_model.model.", ".lora_A.weight"
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        module = key[len(prefix):-len(suffix)]
        bkey = prefix + module + ".lora_B.weight"
        wkey = "model." + module + ".weight"
        b, w = adapter[bkey], base_state[wkey]
        if a.shape != (rank, w.shape[1]) or b.shape != (w.shape[0], rank):
            raise ValueError(f"Adapter shape mismatch: {module}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Nonfinite adapter: {module}")
        result[wkey] = (w.float() + (alpha / rank) * (b.float() @ a.float())).to(w.dtype)
        consumed.update((key, bkey))
    if consumed != set(adapter) or not consumed:
        raise ValueError("Unrecognized/incomplete adapter keys")
    expected_modules = {k[:-len(".weight")] for k, v in base_state.items()
                        if k.endswith(".weight") and v.ndim == 2 and k.split(".")[-2] in payload["targets"]}
    actual_modules = {"model." + k[len("base_model.model."):-len(".lora_A.weight")]
                      for k in adapter if k.endswith(".lora_A.weight")}
    if expected_modules != actual_modules:
        raise ValueError("Adapter does not cover all configured target layers")
    return result
