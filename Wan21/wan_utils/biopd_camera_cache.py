"""Constant, frame-level PRoPE transforms for one BiOPD chunk."""

import torch


def set_chunk_camera(history, common, head_dim):
    """Share transforms across layers without caching any learned Q/K/V values.

    Wan expands a frame's camera to every token before attention. Applying the
    same native transform with cameras=frames groups those tokens identically.
    The closures are immutable during the chunk, including checkpoint backward.
    """
    from wan.modules.prope import _prepare_apply_fns_all_dim

    viewmats, ks = common.get("viewmats"), common.get("Ks")
    transforms = None
    if viewmats is not None:
        if viewmats.requires_grad or (ks is not None and ks.requires_grad):
            raise ValueError("BiOPD camera transform caching requires constant camera inputs")
        with torch.no_grad():
            transforms = _prepare_apply_fns_all_dim(
                head_dim=head_dim, viewmats=viewmats, Ks=ks,
                patches_x=None, patches_y=None, image_width=None, image_height=None)
    for cache in history["prope_kv_cache"]:
        cache["camera_transforms"] = transforms
        cache["camera_start"] = common["current_start"]
