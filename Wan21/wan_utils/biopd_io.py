"""Checkpoint and import helpers isolated from the legacy trainer registry."""

import importlib.util
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch


@contextmanager
def runtime_log(directory, rank):
    """Persist worker output/exceptions independently of the terminal session."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, original, log):
            self.original, self.log = original, log

        def write(self, text):
            self.log.write(text)
            self.log.flush()
            return self.original.write(text)

        def flush(self):
            self.log.flush()
            self.original.flush()

        def __getattr__(self, name):
            return getattr(self.original, name)

    old_stdout, old_stderr = sys.stdout, sys.stderr
    with (directory / f"runtime_rank{rank}.log").open("a", buffering=1) as handle:
        handle.write(f"\n=== invocation {datetime.now(timezone.utc).isoformat()} rank={rank} ===\n")
        sys.stdout, sys.stderr = Tee(old_stdout, handle), Tee(old_stderr, handle)
        try:
            yield
        except BaseException:
            traceback.print_exc(file=handle)
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def load_generator_state(path, key="generator"):
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if key in state and isinstance(state[key], dict):
        state = state[key]
    elif not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError(f"{path}: no tensor state dict under explicit key {key!r}")
    fixed = {}
    for name, value in state.items():
        name = name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        name = name.replace("_fsdp_wrapped_module.", "")
        if name in fixed:
            raise ValueError(f"Checkpoint key collision: {name}")
        fixed[name] = value
    return fixed


def adapter_state(model):
    from wan_utils.biopd_lora import get_adapter_state
    return get_adapter_state(model)


def restore_adapter(model, state):
    from wan_utils.biopd_lora import set_adapter_state
    set_adapter_state(model, state)


def merged_generator_state(wrapper):
    """Merge on CPU without mutating the live LoRA or allocating a second GPU model."""
    from wan_utils.biopd_lora import LoRALinear
    inner = wrapper.model
    state = {}
    for name, tensor in inner.state_dict().items():
        if ".lora_" in name:
            continue
        state["model." + name.replace(".base_layer.", ".")] = tensor.detach().cpu().clone()
    for name, layer in inner.named_modules():
        if isinstance(layer, LoRALinear):
            key = "model." + name + ".weight"
            delta = (layer.lora_B.weight.detach().float().cpu()
                     @ layer.lora_A.weight.detach().float().cpu())
            state[key] = (state[key].float() + delta * layer.scaling).to(state[key].dtype)
    return state
