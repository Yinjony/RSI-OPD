"""Standard additive LoRA, isolated from optional PEFT/accelerate versions.

W_eff = W + (alpha/r) B A. Adapter directories use PEFT's ordinary LoRA
config/key convention; advanced PEFT variants are explicitly rejected.
"""

import json
import math
from pathlib import Path

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank, alpha):
        super().__init__()
        self.base_layer = base_layer.requires_grad_(False)
        self.scaling = float(alpha) / rank
        kwargs = dict(device=base_layer.weight.device, dtype=torch.float32, bias=False)
        self.lora_A = nn.Linear(base_layer.in_features, rank, **kwargs)
        self.lora_B = nn.Linear(rank, base_layer.out_features, **kwargs)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base = self.base_layer(x)
        delta = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype)))
        return base + (delta * self.scaling).to(base.dtype)


def inject_lora(model, rank, alpha, targets):
    if rank <= 0 or int(rank) != rank or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA requires a positive integer rank and positive finite alpha")
    model.requires_grad_(False)
    replacements = [(name, layer) for name, layer in model.named_modules()
                    if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in targets]
    found = {name.rsplit(".", 1)[-1] for name, _ in replacements}
    if found != set(targets):
        raise ValueError(f"Unmatched LoRA target modules: {set(targets) - found}")
    for name, layer in replacements:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, LoRALinear(layer, rank, alpha))
    return len(replacements)


def get_adapter_state(model):
    return {"base_model.model." + name: value.detach().cpu().clone()
            for name, value in model.state_dict().items() if ".lora_" in name}


def set_adapter_state(model, state):
    expected = get_adapter_state(model)
    if expected.keys() != state.keys():
        raise ValueError("Adapter keys do not match configured LoRA modules")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != expected[name].shape:
                raise ValueError(f"Adapter shape mismatch: {name}")
            parameters[name.removeprefix("base_model.model.")].copy_(value)


def save_adapter(model, path, *, rank, alpha, targets, base_model_path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    cfg = dict(peft_type="LORA", task_type=None, r=rank, lora_alpha=alpha,
               lora_dropout=0.0, bias="none", target_modules=list(targets),
               fan_in_fan_out=False, inference_mode=True,
               base_model_name_or_path=str(base_model_path))
    (path / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    torch.save(get_adapter_state(model), path / "adapter_model.bin")


def load_and_merge_adapter(model, path):
    path = Path(path)
    cfg = json.loads((path / "adapter_config.json").read_text())
    if cfg.get("peft_type") != "LORA" or cfg.get("bias", "none") != "none":
        raise ValueError("Only standard bias=none LoRA adapters are supported")
    for key in ("use_dora", "use_rslora", "fan_in_fan_out", "rank_pattern", "alpha_pattern",
                "modules_to_save", "lora_bias"):
        if cfg.get(key):
            raise ValueError(f"Unsupported LoRA extension {key}; export a merged guider instead")
    targets = cfg["target_modules"]
    if not isinstance(targets, list):
        raise ValueError("LoRA target_modules must be a list, not a regular expression")
    inject_lora(model, cfg["r"], cfg["lora_alpha"], targets)
    if (path / "adapter_model.safetensors").exists():
        from safetensors.torch import load_file
        state = load_file(str(path / "adapter_model.safetensors"))
    else:
        state = torch.load(path / "adapter_model.bin", map_location="cpu", weights_only=True)
    set_adapter_state(model, state)
    replacements = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, LoRALinear)]
    with torch.no_grad():
        for name, layer in replacements:
            delta = layer.lora_B.weight.float() @ layer.lora_A.weight.float()
            layer.base_layer.weight.copy_(layer.base_layer.weight.float() + layer.scaling * delta)
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, layer.base_layer)
    return model.requires_grad_(False).eval()
