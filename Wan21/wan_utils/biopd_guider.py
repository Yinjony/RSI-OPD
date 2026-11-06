"""Batch independent guider conditions along B, keeping branch history rows separate."""

import torch

from wan_utils.biopd_causal_adapter import new_history, set_write
from wan_utils.biopd_camera_cache import set_chunk_camera


def repeat_batch(tensor, copies):
    return tensor if copies == 1 else tensor.repeat((copies,) + (1,) * (tensor.ndim - 1))


class GuiderBranches:
    """Order is always y+Apos, Aneg, y+Aneg, Apos, with branch-major batches.

    Each group owns one history whose batch rows retain independent K/V and
    PRoPE states for every (branch, sample). No branch shares another's values.
    """
    def __init__(self, guider, conditions, *, branches_per_call=1, cache_device="cpu", cache_text_kv=True,
                 cache_prope_transforms=True, video_kv_buffer_tokens=None):
        if type(branches_per_call) is not int or branches_per_call not in (1, 2, 4) or len(conditions) != 4:
            raise ValueError("Use four guider conditions and branches_per_call in {1,2,4}")
        if (cache_text_kv or video_kv_buffer_tokens is not None) and any(
                parameter.requires_grad for parameter in guider.parameters()):
            raise ValueError("Text/video K/V caching requires a fully frozen guider")
        self.guider = guider
        self.cache_prope_transforms = cache_prope_transforms
        self.groups = []
        for offset in range(0, 4, branches_per_call):
            selected = conditions[offset:offset + branches_per_call]
            if branches_per_call == 1:
                condition = selected[0]
            else:
                keys = set(selected[0])
                if any(set(branch) != keys for branch in selected):
                    raise ValueError("Guider condition dictionaries must have identical keys")
                condition = {key: torch.cat([branch[key] for branch in selected], dim=0) for key in keys}
            self.groups.append(dict(copies=len(selected), condition=condition,
                                    history=new_history(len(guider.model.blocks), cache_device,
                                                        cache_text_kv=cache_text_kv,
                                                        buffer_tokens=video_kv_buffer_tokens)))

    def set_chunk(self, common):
        for group in self.groups:
            group["common"] = {key: repeat_batch(value, group["copies"]) if torch.is_tensor(value) else value
                               for key, value in common.items()}
            block = self.guider.model.blocks[0]
            if self.cache_prope_transforms and hasattr(block, "self_attn"):
                set_chunk_camera(group["history"], group["common"], block.self_attn.head_dim)

    @torch.no_grad()
    def predict(self, x, timestep):
        predictions = []
        for group in self.groups:
            count = group["copies"]
            flow = self.guider(repeat_batch(x, count), group["condition"], repeat_batch(timestep, count),
                               **group["history"], **group["common"])[0]
            if flow.shape[0] != x.shape[0] * count:
                raise ValueError("Guider output batch does not match independent branch/sample rows")
            predictions.extend(flow.split(x.shape[0], dim=0))
        return predictions

    @torch.no_grad()
    def commit(self, clean, timestep):
        for group in self.groups:
            history, count = group["history"], group["copies"]
            set_write(history, True)
            try:
                self.guider(repeat_batch(clean, count), group["condition"], repeat_batch(timestep, count),
                            **history, **group["common"])
            finally:
                set_write(history, False)
