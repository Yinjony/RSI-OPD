"""Camera transform reuse and frozen video buffers preserve predictions/gradients."""

import copy
from unittest.mock import patch

import pytest
import torch

import test_biopd_guider_batch as fixtures
from wan_utils.biopd_camera_cache import set_chunk_camera
from wan_utils.biopd_causal_adapter import history_kv, new_history, set_write
from wan_utils.biopd_guider import GuiderBranches
from wan_utils.biopd_lora import inject_lora
from shared.algorithms.biopd import biopd_loss, guider_targets

camera_guider = fixtures.camera_guider


def test_buffered_history_matches_concat_and_sliding_window():
    old, new = new_history(2), new_history(2, buffer_tokens=6)
    rng = torch.Generator().manual_seed(45)
    pointer = None
    with torch.no_grad():
        for chunk in range(5):
            for write in (False, False, True):
                set_write(old, write)
                set_write(new, write)
                for layer in range(2):
                    key = torch.randn(2, 2, 1, 4, generator=rng)
                    value = torch.randn(2, 2, 1, 4, generator=rng)
                    expected = history_kv(old["kv_cache"][layer], key, value, chunk * 2, 6)
                    # The optimized path cannot allocate by cat or clone.
                    with patch("torch.cat", side_effect=AssertionError("Unexpected concatenation")):
                        actual = history_kv(new["kv_cache"][layer], key, value, chunk * 2, 6)
                    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
                pool = new["kv_cache"][0]["buffer_pool"]
                if "k" in pool:
                    current = pool["k"].data_ptr(), pool["v"].data_ptr()
                    if pointer is None:
                        pointer = current
                    assert pointer == current
                if write:
                    for a, b in zip(old["kv_cache"], new["kv_cache"]):
                        assert a["end"] == b["end"]
                        assert torch.equal(a["k"], b["k"])
                        assert torch.equal(a["v"], b["v"])
    assert new["kv_cache"][0]["buffer_pool"] is new["kv_cache"][1]["buffer_pool"]
    assert new["kv_cache"][0]["buffer_pool"] is not new["prope_kv_cache"][0]["buffer_pool"]


def test_video_buffer_rejects_student_gradients_and_capacity_overrun():
    cache = new_history(1, buffer_tokens=1)["kv_cache"][0]
    key = torch.ones(1, 2, 1, 4, requires_grad=True)
    with pytest.raises(RuntimeError, match="no-grad"):
        history_kv(cache, key, key, 0, 8)
    cache["write"] = True
    with torch.no_grad(), pytest.raises(ValueError, match="reserved"):
        history_kv(cache, key.detach(), key.detach(), 0, 8)


@pytest.mark.parametrize("packing", [1, 2, 4])
def test_real_wan_five_chunk_predictions_targets_and_clean_history(camera_guider, packing, monkeypatch):
    import wan.modules.prope as prope

    original = prope._prepare_apply_fns_all_dim
    calls = []

    def counted(**kwargs):
        calls.append(kwargs["viewmats"].shape)
        return original(**kwargs)

    monkeypatch.setattr(prope, "_prepare_apply_fns_all_dim", counted)
    conditions = [dict(prompt_embeds=torch.randn(2, 5, 64)) for _ in range(4)]
    noisy = torch.randn(5, 2, 2, 4, 4, 4, 4)
    student = torch.randn(2, 4, 4, 4, 4)
    vm = torch.eye(4).expand(2, 4, 4, 4).clone()
    vm[1, :, 0, 3] = .23
    ks = torch.eye(3).expand(2, 4, 3, 3).clone()
    ks[:, :, 0, 0] = .7
    results, losses = [], []
    for camera_cache, video_buffer in ((False, False), (True, False), (False, True), (True, True)):
        calls.clear()
        predictor = GuiderBranches(camera_guider, conditions, branches_per_call=packing,
                                   cache_prope_transforms=camera_cache,
                                   video_kv_buffer_tokens=64 if video_buffer else None)
        output, loss_sum = [], 0
        for chunk in range(5):
            common = dict(current_start=chunk * 16, viewmats=vm + chunk * .01, Ks=ks)
            predictor.set_chunk(common)
            before = [cache["k"].clone() if cache["k"] is not None else None
                      for group in predictor.groups for cache in group["history"]["kv_cache"]]
            for step in range(2):
                prediction = predictor.predict(noisy[chunk, step], torch.full((2, 4), 625.))
                output.extend(prediction)
                positive, negative = guider_targets(prediction, 1., 1.)
                loss_sum += biopd_loss(student, positive, negative, sigma=.625, next_sigma=0,
                                      weight=1., lam=.25)[0]
            after = [cache for group in predictor.groups for cache in group["history"]["kv_cache"]]
            assert all(cache["end"] == chunk * 16 for cache in after)
            assert all(a is None or torch.equal(a, b["k"]) for a, b in zip(before, after))
            if chunk < 4:
                predictor.commit(noisy[chunk, 0] * .1, torch.zeros(2, 4))
        multiplier = 4 // packing
        assert len(calls) == (5 if camera_cache else 14 * 2) * multiplier
        if camera_cache:
            assert all(shape == (2 * packing, 4, 4, 4) for shape in calls)
        results.append(output)
        losses.append(loss_sum)
    assert all(torch.allclose(a, b, atol=3e-6, rtol=3e-5) for row in results[1:] for a, b in zip(results[0], row))
    assert all(torch.allclose(losses[0], loss, atol=3e-6, rtol=3e-5) for loss in losses[1:])


@pytest.mark.parametrize("checkpointing", [False, True])
def test_cached_camera_preserves_real_student_lora_gradients(camera_guider, checkpointing):
    student = copy.deepcopy(camera_guider)
    inject_lora(student.model, 2, 2, ["q", "k", "v", "o", "prope_o"])
    student.model.gradient_checkpointing = checkpointing
    condition = dict(prompt_embeds=torch.randn(2, 5, 64))
    vm = torch.eye(4).expand(2, 4, 4, 4).clone()
    vm[1, :, 1, 3] = .2
    ks = torch.eye(3).expand(2, 4, 3, 3).clone()
    noise = torch.randn(2, 2, 4, 4, 4, 4)
    gradients = []
    for enabled in (False, True):
        student.zero_grad(set_to_none=True)
        history = new_history(2)
        for chunk in range(2):
            common = dict(current_start=chunk * 16, viewmats=vm, Ks=ks)
            if enabled:
                set_chunk_camera(history, common, 128)
                assert history["prope_kv_cache"][0]["camera_transforms"] is history["prope_kv_cache"][1]["camera_transforms"]
            flow = student(noise[chunk], condition, torch.full((2, 4), 625.), **history, **common)[0]
            flow.square().mean().backward()
            if chunk == 0:
                set_write(history, True)
                with torch.no_grad():
                    student(noise[chunk] * .1, condition, torch.zeros(2, 4), **history, **common)
                set_write(history, False)
        gradients.append({name: p.grad.clone() for name, p in student.named_parameters() if p.requires_grad})
    assert any(grad.abs().max() > 0 for grad in gradients[1].values())
    assert all(torch.allclose(grad, gradients[1][name], atol=3e-6, rtol=3e-5)
               for name, grad in gradients[0].items())
