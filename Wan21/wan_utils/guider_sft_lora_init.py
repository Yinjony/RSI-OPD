"""Read an existing SFT/BiOPD adapter without merging or resetting its A/B tensors."""
import json
from pathlib import Path

import torch

from wan_utils.biopd_lora import inject_lora
from wan_utils.guider_sft_dataset import file_identity
from wan_utils.guider_sft_training import restore_adapter_payload


def read_initial_adapter(source):
    source = Path(source).expanduser().resolve()
    if source.is_file():
        payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        if payload.get("format") != "wan_guider_sft_lora_v1":
            raise ValueError("Use an SFT adapter.pt, not model.pt or training_state.pt")
        return payload, [file_identity(source)]
    config_path = source / "adapter_config.json"
    metadata_path = source.parent / "metadata.json"
    config = json.loads(config_path.read_text())
    contract = json.loads(metadata_path.read_text())["contract"]
    if config.get("peft_type") != "LORA" or config.get("bias", "none") != "none":
        raise ValueError("Expected a standard BiOPD LoRA adapter")
    for key in ("use_dora", "use_rslora", "fan_in_fan_out", "rank_pattern", "alpha_pattern", "modules_to_save", "lora_bias"):
        if config.get(key):
            raise ValueError(f"Unsupported LoRA extension: {key}")
    if config.get("lora_dropout", 0.0) != 0.0 or not isinstance(config["target_modules"], list):
        raise ValueError("Expected dropout=0 and an explicit target module list")
    base_identity = contract["student_identity"]
    base_path = Path(base_identity["path"])
    recorded_base = Path(contract["generator_ckpt"])
    model_path = Path(contract["model_path"])
    if not model_path.is_absolute():
        roots = [p for p in base_path.parents if not recorded_base.is_absolute()
                 and (p / recorded_base).resolve() == base_path.resolve()]
        if len(roots) != 1:
            raise ValueError("Cannot resolve BiOPD model assets from recorded base provenance")
        model_path = (roots[0] / model_path).resolve()
    tensors_path = source / "adapter_model.safetensors"
    if tensors_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(tensors_path))
    else:
        tensors_path = source / "adapter_model.bin"
        state = torch.load(tensors_path, map_location="cpu", weights_only=True, mmap=True)
    payload = dict(format="wan_guider_sft_lora_v1", rank=config["r"], alpha=config["lora_alpha"],
                   dropout=0.0, targets=config["target_modules"], base_identity=base_identity,
                   model_path=str(model_path), adapter=state)
    return payload, [file_identity(p) for p in (config_path, tensors_path, metadata_path)]


def validate_initial_adapter(payload, cfg):
    expected = dict(format="wan_guider_sft_lora_v1", rank=int(cfg.lora_rank), alpha=float(cfg.lora_alpha),
                    dropout=0.0, targets=list(cfg.lora_target_modules),
                    base_identity=file_identity(cfg.generator_ckpt), model_path=str(cfg.model_path))
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Initial adapter {key} differs: {payload.get(key)!r} != {value!r}. "
                             "Use its original unmerged base; do not apply it to an already merged model.")
    state = payload.get("adapter")
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Initial adapter has no tensor state")


def restore_initial_adapter(raw, payload, cfg):
    validate_initial_adapter(payload, cfg)
    restore_adapter_payload(raw, payload, cfg, file_identity(cfg.generator_ckpt))
    return raw


def restore_initial_reference(reference, payload, cfg):
    inject_lora(reference.model, cfg.lora_rank, cfg.lora_alpha, list(cfg.lora_target_modules))
    restore_initial_adapter(reference, payload, cfg)
    return reference.requires_grad_(False).eval()
