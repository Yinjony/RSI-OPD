"""Deterministic held-out selection and noise seeds for BiOPD evaluation."""

import hashlib
import random


def validation_kind(step, config):
    """Return periodic/full, or None; optimizer steps define the schedule."""
    every = int(config.get("val_every", 0))
    if every <= 0:
        return None
    if step == 0:
        return "periodic" if config.get("val_at_start", True) else None
    if step == config.max_steps:
        return "full" if config.get("val_full_at_end", True) else "periodic"
    return "periodic" if step % every == 0 else None


def validation_indices(dataset, config, *, full=False):
    # Sort before shuffling so source JSON reordering and DDP rank assignment
    # cannot silently change the fixed periodic evaluation subset.
    indices = sorted(range(len(dataset)), key=lambda i: dataset[i]["sample_id"])
    random.Random(config.get("val_seed", 12345)).shuffle(indices)
    return indices if full else indices[:int(config.get("val_num_samples", 20))]


def sample_noise_seed(sample_id, base_seed):
    digest = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big")
    return (digest + int(base_seed)) % (2**63 - 1)


def validation_report(dataset, config):
    full = validation_indices(dataset, config, full=True)
    periodic = validation_indices(dataset, config)
    every = int(config.get("val_every", 0))
    maximum = int(config.max_steps)
    candidates = {0, maximum, *range(every, maximum, every)} if every else set()
    schedule = []
    for step in sorted(candidates):
        kind = validation_kind(step, config)
        if kind:
            schedule.append(dict(step=step, kind=kind, num_samples=len(full if kind == "full" else periodic)))
    seed = config.get("val_seed", 12345)
    return dict(data=dataset.report(), seed=seed, schedule=schedule,
                periodic_sample_ids=[dataset[i]["sample_id"] for i in periodic],
                full_sample_ids=[dataset[i]["sample_id"] for i in full],
                sample_noise_seeds={dataset[i]["sample_id"]: sample_noise_seed(dataset[i]["sample_id"], seed)
                                    for i in full})


def validation_metric_names(num_steps):
    return sorted(["loss", "signed_loss", "positive_mse", "negative_mse", "target_gap_mse", "prediction_steps",
                   *[f"step_{i}_signed_loss" for i in range(num_steps)]])
