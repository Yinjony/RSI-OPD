"""Isolated, functional cache attention for detached-history BiOPD.

The original Wan classes/files and parameter names are unchanged. Only the
instances created by the BiOPD loader receive this attention subclass.
SP is deliberately unsupported here; multi-GPU training uses DDP.
"""

import torch


def new_history(num_layers, storage_device="cpu", *, cache_text_kv=False, buffer_tokens=None):
    """Opt in to text K/V reuse only for a frozen, no-grad guider rollout."""
    if buffer_tokens is not None and (type(buffer_tokens) is not int or buffer_tokens < 0):
        raise ValueError("buffer_tokens must be a nonnegative integer or None")

    def caches():
        # One transient K/V workspace per attention path, shared across layers.
        pool = {}
        return [dict(biopd=True, k=None, v=None, end=0, write=False,
                     storage_device=str(storage_device), buffer_tokens=buffer_tokens,
                     buffer_pool=pool) for _ in range(num_layers)]
    return dict(kv_cache=caches(), prope_kv_cache=caches(),
                # Student stays uncached so every backward sees its LoRA projections.
                # Guider text tensors remain on the device where they are computed.
                crossattn_cache=[dict(is_init=False, k=None, v=None) for _ in range(num_layers)]
                if cache_text_kv else [None] * num_layers)


def set_write(history, enabled):
    for kind in ("kv_cache", "prope_kv_cache"):
        for cache in history[kind]:
            cache["write"] = enabled


def history_kv(cache, key, value, current_start, max_tokens):
    """Current K/V keep gradients; past values are constant. Writes require no_grad.

    A cache stores only completed CLEAN chunks, never intermediate noisy chunks.
    This is equivalent to the original overwrite-at-current_start behavior.
    """
    if current_start != cache["end"]:
        raise ValueError(f"History ends at {cache['end']}, current_start={current_start}")
    if cache["write"] and torch.is_grad_enabled():
        raise RuntimeError("Persistent history commits must run under torch.no_grad()")
    storage = torch.device(cache["storage_device"])
    same_device = storage.type == key.device.type and storage.index in (None, key.device.index)
    if cache.get("buffer_tokens") is not None and same_device:
        return buffered_history_kv(cache, key, value, current_start, max_tokens)
    if cache["k"] is None:
        k, v = key, value
    else:
        k = torch.cat([cache["k"].detach().to(key), key], dim=1)
        v = torch.cat([cache["v"].detach().to(value), value], dim=1)
    k, v = k[:, -max_tokens:], v[:, -max_tokens:]
    if cache["write"]:
        # clone avoids retaining a full cat allocation behind a truncated view.
        cache["k"] = k.detach().to(cache["storage_device"]).clone()
        cache["v"] = v.detach().to(cache["storage_device"]).clone()
        cache["end"] = current_start + key.shape[1]
    return k, v


def buffered_history_kv(cache, key, value, current_start, max_tokens):
    """Frozen-only history storage plus transient, layer-shared attention inputs.

    Returned workspace views are consumed immediately by attention. They must
    never be retained for backward or used concurrently on separate streams.
    Only clean commits write the persistent per-layer history.
    """
    if torch.is_grad_enabled() or key.requires_grad or value.requires_grad:
        raise RuntimeError("Buffered video K/V requires frozen no-grad guider forwards")
    length = key.shape[1]
    keep = max(0, max_tokens - length)
    past = min(cache["k"].shape[1], keep) if cache["k"] is not None else 0
    count = past + min(length, max_tokens)
    if past:
        pool = cache["buffer_pool"]
        capacity = min(max_tokens, cache["buffer_tokens"] + length)
        signature = (key.shape[0], key.shape[2:], key.dtype, key.device, capacity)
        if pool.get("signature") != signature:
            shape = (key.shape[0], capacity, *key.shape[2:])
            pool.update(signature=signature, k=torch.empty(shape, device=key.device, dtype=key.dtype),
                        v=torch.empty(shape, device=value.device, dtype=value.dtype))
        k, v = pool["k"][:, :count], pool["v"][:, :count]
        k[:, :past].copy_(cache["k"][:, -past:])
        v[:, :past].copy_(cache["v"][:, -past:])
        k[:, past:].copy_(key[:, -min(length, max_tokens):])
        v[:, past:].copy_(value[:, -min(length, max_tokens):])
    else:
        k, v = key[:, -max_tokens:], value[:, -max_tokens:]
    if cache["write"]:
        capacity = min(cache["buffer_tokens"], max_tokens)
        if count > capacity:
            raise ValueError("Clean history exceeds reserved guider buffer; reserve all committed chunks")
        if "k_buffer" not in cache:
            shape = (key.shape[0], capacity, *key.shape[2:])
            cache["k_buffer"] = torch.empty(shape, device=key.device, dtype=key.dtype)
            cache["v_buffer"] = torch.empty(shape, device=value.device, dtype=value.dtype)
        # Append only the clean current chunk when no old tokens are evicted.
        old_count = cache["k"].shape[1] if cache["k"] is not None else 0
        if past == old_count:
            cache["k_buffer"][:, past:count].copy_(key[:, -min(length, max_tokens):])
            cache["v_buffer"][:, past:count].copy_(value[:, -min(length, max_tokens):])
        else:
            cache["k_buffer"][:, :count].copy_(k)
            cache["v_buffer"][:, :count].copy_(v)
        cache["k"], cache["v"] = cache["k_buffer"][:, :count], cache["v_buffer"][:, :count]
        cache["end"] = current_start + length
    return k, v


def install_biopd_attention(model):
    from wan.modules.attention import attention
    from wan.modules.causal_model import CausalWanSelfAttention, causal_rope_apply
    from wan.modules.prope import prope_qkv

    class BiOPDSelfAttention(CausalWanSelfAttention):
        def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None,
                    current_start=0, cache_start=None, viewmats=None, Ks=None,
                    prope_kv_cache=None):
            if kv_cache is None or not kv_cache.get("biopd", False):
                return super().forward(x, seq_lens, grid_sizes, freqs, block_mask,
                                       kv_cache, current_start, cache_start,
                                       viewmats, Ks, prope_kv_cache)
            if self.sink_size != 0:
                raise ValueError("BiOPD functional cache currently requires sink_size=0")
            batch, length = x.shape[:2]
            shape = (batch, length, self.num_heads, self.head_dim)
            q = self.norm_q(self.q(x)).view(shape)
            k = self.norm_k(self.k(x)).view(shape)
            v = self.v(x).view(shape)
            frame_tokens = int(grid_sizes[0, 1:].prod())
            if current_start % frame_tokens:
                raise ValueError("current_start must be aligned to a latent frame")
            start_frame = current_start // frame_tokens
            rq = causal_rope_apply(q, grid_sizes, freqs, start_frame).type_as(v)
            rk = causal_rope_apply(k, grid_sizes, freqs, start_frame).type_as(v)
            # Same attention window as the deployed model, including its full
            # context cap; the pipeline validates the supported frame count.
            hk, hv = history_kv(kv_cache, rk, v, current_start, self.max_attention_size)
            result = self.o(attention(rq, hk, hv).flatten(2))
            if viewmats is not None and hasattr(self, "prope_o"):
                if prope_kv_cache is None:
                    raise ValueError("Camera BiOPD requires a separate PRoPE cache")
                transforms = prope_kv_cache.get("camera_transforms")
                if transforms is None:
                    qp, kp, vp, output_transform = prope_qkv(
                        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                        v.permute(0, 2, 1, 3), viewmats=viewmats, Ks=Ks)
                else:
                    if prope_kv_cache["camera_start"] != current_start:
                        raise ValueError("PRoPE transforms must be refreshed at the start of each chunk")
                    apply_q, apply_kv, output_transform = transforms
                    qp, kp, vp = (apply_q(q.permute(0, 2, 1, 3)),
                                  apply_kv(k.permute(0, 2, 1, 3)), apply_kv(v.permute(0, 2, 1, 3)))
                pk, pv = history_kv(prope_kv_cache, kp.permute(0, 2, 1, 3),
                                    vp.permute(0, 2, 1, 3), current_start,
                                    self.max_attention_size)
                result_p = attention(qp.permute(0, 2, 1, 3), pk, pv)
                result_p = output_transform(result_p.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
                result = result + self.prope_o(result_p.flatten(2))
            return result

    count = 0
    for module in model.modules():
        if type(module) is CausalWanSelfAttention:
            module.__class__ = BiOPDSelfAttention
            count += 1
    if count != len(model.blocks):
        raise ValueError(f"Expected one causal attention per block, adapted {count}")
    return count
