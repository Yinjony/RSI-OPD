"""BiOPD for Wan's denoise-and-renoise sampler (not an Euler ODE loss)."""

import math

import torch


def biopd_loss(student, positive, negative, *, sigma, next_sigma, weight, lam):
    """Mean over batch/elements; callers SUM over chunks and denoising steps.

    The positive regression below and the signed MSE have identical gradients
    for detached states/teachers. They do not have identical scalar values.
    """
    values = (sigma, next_sigma, weight, lam)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("BiOPD coefficients must be finite")
    if not (0 <= lam < 1 and weight >= 0 and 0 <= next_sigma < sigma <= 1):
        raise ValueError("Require 0 <= lambda < 1, weight >= 0, 0 <= next_sigma < sigma <= 1")
    if student.shape != positive.shape or student.shape != negative.shape:
        raise ValueError("Student and both guider predictions must have identical shapes")
    positive, negative = positive.detach().float(), negative.detach().float()
    student = student.float()
    coefficient = weight * sigma**2 * (1 - next_sigma)**2 / 2
    target = (positive - lam * negative) / (1 - lam)
    loss = coefficient * (1 - lam) * (student - target).square().mean()
    with torch.no_grad():
        pos_mse = (student - positive).square().mean()
        neg_mse = (student - negative).square().mean()
        metrics = {
            "loss": loss.detach(),
            "signed_loss": coefficient * (pos_mse - lam * neg_mse),
            "positive_mse": pos_mse,
            "negative_mse": neg_mse,
            "target_gap_mse": (positive - negative).square().mean(),
        }
    return loss, metrics


def guider_targets(predictions, omega_positive, omega_negative):
    """Order: y+Apos, Aneg, y+Aneg, Apos. Never insert y in branches 2/4."""
    yp, n, yn, p = [value.detach().float() for value in predictions]
    return yp + omega_positive * (yp - n), yn + omega_negative * (yn - p)


def resolve_schedule(scheduler, steps, warp=True):
    """Resolve the SAME sparse model timesteps as CausalInferencePipeline."""
    steps = list(steps)
    if steps and steps[-1] == 0:
        steps.pop()
    if not isinstance(warp, bool):
        raise ValueError("warp_denoising_step must be boolean")
    if not steps or any(isinstance(x, bool) or not isinstance(x, (int, float))
                        or not math.isfinite(x) or int(x) != x or not 0 < x <= 1000 for x in steps):
        raise ValueError("denoising_step_list must contain integers in [1, 1000]")
    steps = [int(x) for x in steps]
    dense_times = scheduler.timesteps.detach().cpu()
    if warp:
        if len(dense_times) != 1000:
            raise ValueError("Wan's warped sparse schedule requires 1000 scheduler entries")
        times = torch.cat([dense_times, torch.zeros(1)])[1000 - torch.tensor(steps)]
    else:
        times = torch.tensor(steps, dtype=dense_times.dtype)
    indices = (dense_times[:, None] - times[None, :]).abs().argmin(dim=0)
    sigmas = scheduler.sigmas.detach().cpu()[indices].tolist() + [0.0]
    if not all(0 <= b < a <= 1 for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError(f"Noise levels must strictly decrease: {sigmas}")
    return times.tolist(), sigmas
