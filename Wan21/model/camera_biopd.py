"""Two causal camera models: LoRA student and a replaceable frozen guider."""

from collections import OrderedDict
import os

import torch

from wan_utils.biopd_causal_adapter import install_biopd_attention
from wan_utils.biopd_io import load_generator_state


def load_models(config, device):
    from wan_utils.biopd_lora import inject_lora, load_and_merge_adapter
    from wan_utils.wan_wrapper import WanDiffusionWrapper

    dtype = torch.bfloat16

    def base(path, checkpoint_key):
        print(f"[rank {os.environ.get('RANK', '0')}] Loading causal model: {path}", flush=True)
        model = WanDiffusionWrapper(model_path=config.model_path,
                                    timestep_shift=config.timestep_shift,
                                    is_causal=True, use_camera=True,
                                    local_attn_size=config.local_attn_size, sink_size=0)
        model.load_state_dict(load_generator_state(path, checkpoint_key), strict=True)
        model.requires_grad_(False)
        model.model.num_frame_per_block = config.num_frame_per_block
        install_biopd_attention(model.model)
        model = model.to(device=device, dtype=dtype).eval()
        print(f"[rank {os.environ.get('RANK', '0')}] Loaded on {device}: {path}", flush=True)
        return model

    student = base(config.generator_ckpt, config.generator_checkpoint_key)
    inject_lora(student.model, config.lora_rank, config.lora_alpha, list(config.lora_target_modules))
    if config.get("initial_adapter"):
        from wan_utils.guider_sft_lora_init import read_initial_adapter, restore_initial_adapter
        payload, _ = read_initial_adapter(config.initial_adapter)
        restore_initial_adapter(student, payload, config)
        del payload
        print(f"[rank {os.environ.get('RANK', '0')}] Continuing existing student LoRA A/B "
              f"without merging/resetting: {config.initial_adapter}", flush=True)
    for name, param in student.named_parameters():
        if param.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected trainable base parameter: {name}")
            param.data = param.data.float()
    student.eval()  # LoRA remains differentiable; dropout stays disabled for rollout parity.
    if config.gradient_checkpointing:
        student.enable_gradient_checkpointing()
    guider = base(config.guider_ckpt, config.guider_checkpoint_key)
    if config.guider_adapter_ckpt:
        load_and_merge_adapter(guider.model, config.guider_adapter_ckpt)
    guider.requires_grad_(False).eval()
    return student, guider


class FrozenTextConditions:
    """Bounded CPU embedding cache. T5 is offloaded between batches by default."""
    def __init__(self, config, device):
        from wan_utils.wan_wrapper import WanTextEncoder
        print(f"[rank {os.environ.get('RANK', '0')}] Loading frozen T5", flush=True)
        self.encoder = WanTextEncoder(model_path=config.model_path).requires_grad_(False).eval()
        self.encoder.to(dtype=torch.bfloat16)
        self.device = device
        self.offload = config.text_encoder_offload
        self.limit = config.text_cache_size
        self.cache = OrderedDict()
        if not self.offload:
            self.encoder.to(device)
        print(f"[rank {os.environ.get('RANK', '0')}] T5 ready", flush=True)

    @torch.no_grad()
    def encode(self, text_branches):
        # Keep this batch separately so eviction cannot remove a requested text.
        batch_cache = {}
        missing = []
        for text in dict.fromkeys(text for branch in text_branches for text in branch):
            if text in self.cache:
                batch_cache[text] = self.cache[text]
                self.cache.move_to_end(text)
            else:
                missing.append(text)
        if missing:
            self.encoder.to(self.device)
            for text in missing:
                ids, _ = self.encoder.tokenizer(text, return_mask=True, add_special_tokens=True,
                                                 truncation=False, padding=False)
                if ids.shape[-1] > 512:
                    raise ValueError("Text exceeds 512 tokens; shorten the prompt/attributes instead of truncating guidance")
                embedding = self.encoder([text])["prompt_embeds"][0].detach().cpu()
                batch_cache[text] = embedding
                if self.limit:
                    self.cache[text] = embedding
                    while len(self.cache) > self.limit:
                        self.cache.popitem(last=False)
            if self.offload:
                self.encoder.cpu()
                torch.cuda.empty_cache()
        return [dict(prompt_embeds=torch.stack([batch_cache[t] for t in branch]).to(self.device))
                for branch in text_branches]
