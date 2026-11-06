# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import warnings
from importlib import metadata

import torch


def is_hopper_gpu():
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 9:
            return True
    except RuntimeError:
        pass
    device_name = torch.cuda.get_device_name(0).lower()
    return any(name in device_name for name in ("h100", "h200", "h800", "hopper"))


try:
    import flash_attn_interface
    FLASH_ATTN_3_IMPORT_ERROR = None
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except Exception as exc:
    flash_attn_interface = None
    FLASH_ATTN_3_IMPORT_ERROR = exc
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_IMPORT_ERROR = None
    FLASH_ATTN_2_AVAILABLE = True
except Exception as exc:
    flash_attn = None
    FLASH_ATTN_2_IMPORT_ERROR = exc
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

__all__ = [
    'flash_attention',
    'attention',
    'attention_backend',
    'attention_backend_info',
]

_ATTENTION_BACKEND_LOGGED = False


def attention_backend():
    if FLASH_ATTN_3_AVAILABLE:
        return "flash_attention_3"
    if FLASH_ATTN_2_AVAILABLE:
        return "flash_attention_2"
    return "torch_sdpa"


def _module_package_version(module, package_names):
    if module is None:
        return None
    for package_name in package_names:
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            pass
    return getattr(module, "__version__", "unknown")


def attention_backend_info():
    device = "cpu"
    capability = None
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        try:
            capability = torch.cuda.get_device_capability(0)
        except RuntimeError:
            capability = None

    fa3_version = _module_package_version(
        flash_attn_interface, ("flash-attn-3", "flash_attn_3"))
    fa2_version = _module_package_version(
        flash_attn, ("flash-attn", "flash_attn"))
    detail = (
        f"[Wan attention] backend={attention_backend()} "
        f"device={device} capability={capability} "
        f"fa3_available={FLASH_ATTN_3_AVAILABLE} fa3_version={fa3_version} "
        f"fa3_path={getattr(flash_attn_interface, '__file__', None)} "
        f"fa2_available={FLASH_ATTN_2_AVAILABLE} fa2_version={fa2_version} "
        f"fa2_path={getattr(flash_attn, '__file__', None)}"
    )
    if FLASH_ATTN_3_IMPORT_ERROR is not None:
        detail += (
            f" fa3_import_error={type(FLASH_ATTN_3_IMPORT_ERROR).__name__}: "
            f"{FLASH_ATTN_3_IMPORT_ERROR}"
        )
    if FLASH_ATTN_2_IMPORT_ERROR is not None:
        detail += (
            f" fa2_import_error={type(FLASH_ATTN_2_IMPORT_ERROR).__name__}: "
            f"{FLASH_ATTN_2_IMPORT_ERROR}"
        )
    return detail


def _log_attention_backend_once():
    global _ATTENTION_BACKEND_LOGGED
    if _ATTENTION_BACKEND_LOGGED:
        return
    _ATTENTION_BACKEND_LOGGED = True
    if os.environ.get("WAN_ATTN_BACKEND_LOG", "1") == "0":
        return

    warnings.warn(attention_backend_info())


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    flash_attn_3_dtype_supported = (
        q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
    )
    if os.environ.get("WAN_FA3_ALLOW_FP16", "0") == "1":
        flash_attn_3_dtype_supported = (
            q.dtype in half_dtypes
            and k.dtype in half_dtypes
            and v.dtype in half_dtypes
        )

    use_flash_attn_3 = (
        (version is None or version == 3)
        and FLASH_ATTN_3_AVAILABLE
        and flash_attn_3_dtype_supported
        and dropout_p == 0.
        and window_size == (-1, -1)
    )

    if version is not None and version == 3 and not use_flash_attn_3:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if use_flash_attn_3:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.unflatten(0, (b, lq))
    else:
        if not FLASH_ATTN_2_AVAILABLE:
            raise RuntimeError(
                "FlashAttention is unavailable. Install flash_attn or set up "
                "an attention path that supports padded variable lengths."
            ) from FLASH_ATTN_2_IMPORT_ERROR
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    _log_attention_backend_once()
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
