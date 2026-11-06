#!/usr/bin/env python3
"""Four experiment recipes, shared training and inference launchers."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]

from omegaconf import OmegaConf
from wan_utils.experiment_options import accumulation, gpu_selection, inference_sampler, prepare_relora, training_config

MODELS = ("guider1", "guider2", "generator1", "generator2")
PATH_FIELDS = ("generator_ckpt", "guider_ckpt", "model_path", "annotation_path", "cache_dir",
               "text_cache_dir", "split_manifest", "initial_adapter", "output_dir", "resume")


def absolute(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_paths(cfg):
    for key in PATH_FIELDS:
        if cfg.get(key):
            cfg[key] = str(absolute(cfg[key]))
    return cfg


def check_initialization(cfg, guider):
    from wan_utils.guider_sft_config import OBJECTIVE_DEFAULTS, validate_config as validate_sft
    from wan_train_biopd import validate_config as validate_biopd
    if guider:
        cfg = OmegaConf.merge(OBJECTIVE_DEFAULTS, cfg)
        validate_sft(cfg)
        for key in ("generator_ckpt", "model_path", "annotation_path", "cache_dir"):
            if not Path(cfg[key]).exists():
                raise FileNotFoundError(f"{key}: {cfg[key]}")
    else:
        validate_biopd(cfg)
    result = {"initialization": "fresh_lora", "rank": cfg.lora_rank, "alpha": cfg.lora_alpha}
    if cfg.get("initial_adapter"):
        from wan_utils.guider_sft_lora_init import read_initial_adapter, validate_initial_adapter
        payload, sources = read_initial_adapter(cfg.initial_adapter)
        validate_initial_adapter(payload, cfg)
        result.update(initialization="continue_existing_lora", adapter_tensors=len(payload["adapter"]),
                      base_identity=payload["base_identity"], sources=sources)
    return result


def saved_config(state_path):
    state = absolute(state_path)
    if not state.is_file() or state.name != "training_state.pt":
        raise ValueError("--resume requires an existing training_state.pt from the SAME run")
    for path in (state.parent / "config.yaml", state.parent.parent / "config.yaml"):
        if path.is_file():
            cfg = OmegaConf.load(path)
            cfg.resume = str(state)
            return cfg
    raise FileNotFoundError(f"Cannot find saved config.yaml beside {state} or its run directory")


def inference_command(args, recipe, gpus, *, config_path=None):
    options = recipe.inference
    checkpoint = absolute(args.checkpoint or options.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    count = args.num_samples if args.num_samples is not None else options.num_samples
    if type(count) is not int or count <= 0:
        raise ValueError("num_samples must be positive")
    output = absolute(args.output_dir or f"outputs/{args.model}_first{count}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    sampler_path = str(config_path or output / "inference_config.yaml")
    common = ["--num-samples", str(count)] if args.model.startswith("guider") else []
    if args.model.startswith("guider"):
        command = [sys.executable, str(ROOT / "Wan21/infer_guider_sft_validation_cfg.py"),
                   "--model-source", "base", "--base", str(checkpoint),
                   "--sft-config", str(absolute(options.validation_config)),
                   "--inference-config", sampler_path,
                   "--output-dir", str(output), "--guidance-scale", str(options.guidance_scale),
                   "--base-seed", str(options.base_seed), "--fps", str(options.fps)] + common
    else:
        command = [sys.executable, str(ROOT / "Wan21/infer_rsi_biopd.py"),
                   "--config", sampler_path,
                   "--checkpoint", str(checkpoint), "--prompts", str(absolute(options.prompts)),
                   "--trajectories", str(absolute(options.trajectories)), "--output-root", str(output),
                   "--iteration", "0", "--samples-per-iteration", "2000",
                   "--start-index", "0", "--end-index", str(count), "--limit", str(count),
                   "--base-seed", str(options.base_seed), "--fps", str(options.fps)]
    if args.dry_run:
        command.append("--dry-run")
    return command, output, gpus[:1]


@contextmanager
def inference_config_file(cfg, output, *, dry_run):
    """Keep resume signatures stable; dry runs only write a temporary config."""
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="minwm_inference_") as temporary:
            path = Path(temporary) / "inference_config.yaml"
            OmegaConf.save(cfg, path)
            yield path
    else:
        path = output / "inference_config.yaml"
        if path.exists():
            if OmegaConf.to_container(OmegaConf.load(path), resolve=True) != OmegaConf.to_container(cfg, resolve=True):
                raise ValueError(f"Output contains a different inference schedule/config: {path}; select a new output directory")
        else:
            output.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, path)
        yield path


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "infer"))
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--gpus", help="Physical IDs, e.g. 0,1; overrides CUDA_VISIBLE_DEVICES and YAML")
    parser.add_argument("--override", action="append", default=[], help="Repeat YAML key=value overrides")
    parser.add_argument("--resume", help="Same-run training_state.pt; use its saved training configuration")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint", help="Inference: full merged model.pt, never adapter/training_state")
    parser.add_argument("--num-samples", type=int, help="Inference, default 10")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--dry-run", action="store_true", help="CPU validation/command preview; never starts CUDA")
    checks.add_argument("--check-init", action="store_true", help="CPU base/adapter provenance check")
    checks.add_argument("--check-data", action="store_true", help="CPU training data/cache inventory")
    checks.add_argument("--check-cache", action="store_true", help="CPU SFT cache contract validation")
    args = parser.parse_args()
    if args.action == "infer" and (args.resume or args.check_init or args.check_data or args.check_cache):
        parser.error("Inference supports --dry-run; training-only checks/resume are not accepted")
    if args.action == "train" and (args.checkpoint or args.num_samples is not None):
        parser.error("--checkpoint and --num-samples are inference-only")
    if args.resume and (args.override or args.output_dir):
        parser.error("Resume uses its saved parameters/output directory; omit overrides/output-dir")
    recipe_path = ROOT / f"Wan21/configs/{args.model}.yaml"
    recipe = OmegaConf.merge(OmegaConf.load(recipe_path), OmegaConf.from_dotlist(args.override))
    gpus = gpu_selection(args.gpus if args.gpus is not None else os.environ.get("CUDA_VISIBLE_DEVICES", recipe.gpus))
    checking = args.check_init or args.check_data or args.check_cache
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env["SP_SIZE"] = "1"
    if args.action == "infer":
        command, output, gpus = inference_command(args, recipe, gpus)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        sampler = inference_sampler(recipe, OmegaConf.load(ROOT / "Wan21/configs/_base/inference.yaml"))
        if sampler.model_kwargs.get("model_path"):
            sampler.model_kwargs.model_path = str(absolute(sampler.model_kwargs.model_path))
        with inference_config_file(sampler, output, dry_run=args.dry_run) as config_path:
            flag = "--inference-config" if args.model.startswith("guider") else "--config"
            command[command.index(flag) + 1] = str(config_path)
            print(json.dumps(dict(model=args.model, action="infer", output_dir=str(output),
                                  denoising_steps=len(sampler.denoising_step_list),
                                  denoising_step_list=list(sampler.denoising_step_list),
                                  CUDA_VISIBLE_DEVICES=env["CUDA_VISIBLE_DEVICES"], command=shlex.join(command)), indent=2), flush=True)
            return subprocess.run(command, cwd=ROOT, env=env, check=True)

    guider = args.model.startswith("guider")
    if not guider and "anchor" in recipe:
        parser.error("anchor is an SFT-guider option; BiOPD uses its own frozen guider objective")
    if args.check_cache and not guider:
        parser.error("BiOPD text-cache coverage is included in --check-data")
    relora = bool(recipe.get("relora", False)) and not args.resume
    if args.resume:
        # Keep historical relative paths exactly as saved: they participate in
        # the trainer's resume contract. Workers run from ROOT as before.
        cfg = saved_config(args.resume)
    else:
        cfg = resolve_paths(training_config(recipe, allow_relora=True, selected_gpus=gpus))
        cfg.grad_accum_steps = accumulation(recipe.effective_batch_size, cfg.batch_size, len(gpus))
        explicit_output = args.output_dir or (cfg.output_dir if any(x.startswith("output_dir=") for x in args.override) else None)
        cfg.output_dir = str(absolute(explicit_output or f"{cfg.output_dir}_{datetime.now():%Y%m%d_%H%M%S_%f}"))
        if cfg.get("resume"):
            parser.error("Use --resume training_state.pt for an exact resume")
    initialization = check_initialization(cfg, guider)
    output = Path(cfg.output_dir)
    if relora:
        initialization["initialization"] = "phase_boundary_merge_then_fresh_lora"
    report = dict(model=args.model, action="train", gpus=gpus, nproc_per_node=len(gpus),
                  grad_accum_steps=cfg.grad_accum_steps,
                  effective_batch_size=len(gpus) * cfg.batch_size * cfg.grad_accum_steps,
                  sft_loss_type=cfg.get("sft_loss_type"), relora=relora,
                  output_dir=str(output), **initialization)
    steps = cfg.get("denoising_step_list", cfg.get("anchor_denoising_step_list", []))
    report.update(denoising_step_list=list(steps),
                  training_rollout_steps=len(steps) if not guider else 0,
                  anchor_rollout_steps=len(cfg.anchor_denoising_step_list)
                  if guider and cfg.get("sft_loss_type") == "supervised_anchor" and cfg.anchor_eta > 0 else 0)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.check_init:
        return
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    entry = "wan_train_guider_sft.py" if guider else "wan_train_biopd.py"
    flags = ["--export-merged"] if guider else []
    if args.check_data:
        flags.append("--check-data")
    if args.check_cache:
        flags.append("--check-cache")
    if not args.dry_run and not checking:
        if not args.resume:
            output.mkdir(parents=True, exist_ok=False)
            OmegaConf.save(recipe, output / "recipe.yaml")
            (output / "launch.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            if relora:
                cfg = prepare_relora(cfg, output / "initialization")
        # Resumes keep the original prepared base; no repeated merge/reset.
        runtime = output / "runtime_config.yaml"
        OmegaConf.save(cfg, runtime)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc-per-node={len(gpus)}", str(ROOT / "Wan21" / entry), "--config", str(runtime)] + flags
        print(shlex.join(command), flush=True)
        return subprocess.run(command, cwd=ROOT, env=env, check=True)
    with tempfile.TemporaryDirectory(prefix="minwm_recipe_") as temporary:
        runtime = Path(temporary) / "config.yaml"
        OmegaConf.save(cfg, runtime)
        command = [sys.executable, str(ROOT / "Wan21" / entry), "--config", str(runtime)] + flags
        if args.dry_run:
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       f"--nproc-per-node={len(gpus)}"] + command[1:]
            print("Preview only; runtime config is temporary: " + shlex.join(command), flush=True)
        else:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
