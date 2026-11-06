"""ETA arithmetic across parallel validation, boundary jobs and resume."""
# ruff: noqa: E402

from datetime import datetime, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Wan21"))
from wan_utils.biopd_progress import TrainingETA, format_duration, progress_message


def config():
    return OmegaConf.create(dict(max_steps=475, save_every=100, val_every=100,
                                 val_num_samples=20, val_at_start=True, val_full_at_end=True))


def test_eta_includes_parallel_validation_and_current_boundary_jobs():
    clock = [0.]
    base = datetime(2026, 10, 2, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
    tracker = TrainingETA(config(), world_size=4, val_size=100, clock=lambda: clock[0],
                          wall_clock=lambda: base + timedelta(seconds=clock[0]))
    assert tracker.snapshot(0)["eta_seconds"] is None
    assert tracker.snapshot(0)["remaining_validation_runs"] == 6
    tracker.observe_validation(0, 20, 200)  # 5 samples/card, 40 seconds per sample per card
    tracker.observe_train(300)
    clock[0] = 500
    first = tracker.snapshot(1)
    assert first["eta_train_seconds"] == 474 * 300
    assert first["eta_validation_seconds"] == (4 * 5 + 25) * 40
    assert first["remaining_checkpoints"] == 5
    assert first["eta_seconds"] == 144000 and first["estimated_total_seconds"] == 144500
    assert first["eta_missing_components"] == ["checkpoint"] and not first["eta_complete"]
    assert first["estimated_finish_time"] == "2026-10-04 04:08:20+08:00"
    assert "尚未计入 ETA" in progress_message("训练", first)

    before = tracker.snapshot(100)
    assert before["remaining_validation_runs"] == 5 and before["remaining_checkpoints"] == 5
    tracker.observe_validation(100, 20, 200)
    tracker.observe_checkpoint(100, 10)
    after = tracker.snapshot(100)
    assert after["remaining_validation_runs"] == 4 and after["remaining_checkpoints"] == 4
    assert after["eta_seconds"] == 375 * 300 + (3 * 5 + 25) * 40 + 4 * 10
    assert after["eta_complete"]

    # The final optimizer step still has full validation and saving pending.
    assert tracker.snapshot(475)["eta_seconds"] == 25 * 40 + 10
    tracker.observe_validation(475, 100, 1000)
    assert tracker.snapshot(475)["eta_seconds"] == 10
    tracker.observe_checkpoint(475, 10)
    final = tracker.snapshot(475)
    assert final["eta_seconds"] == 0 and final["eta_complete"] and final["progress_percent"] == 100


def test_resume_and_recent_step_window_do_not_count_past_jobs():
    tracker = TrainingETA(config(), world_size=3, val_size=100, start_step=100, clock=lambda: 0)
    initial = tracker.snapshot(100)
    assert initial["remaining_validation_runs"] == 4 and initial["remaining_checkpoints"] == 4
    assert initial["elapsed_seconds"] == 0 and initial["eta_seconds"] is None
    for seconds in range(1, 22):
        tracker.observe_train(seconds)
    tracker.observe_validation(200, 20, 70)  # ceil(20/3) = 7; ceil(100/3) = 34
    tracker.observe_checkpoint(200, 10)
    result = tracker.snapshot(201)
    assert result["average_step_seconds"] == pytest.approx(11.5)  # newest 20 observations
    assert result["eta_validation_seconds"] == (7 + 7 + 34) * 10
    assert result["remaining_validation_runs"] == 3 and result["remaining_checkpoints"] == 3


def test_disabled_validation_and_duration_formatting():
    cfg = config()
    cfg.val_every = 0
    tracker = TrainingETA(cfg, world_size=4, clock=lambda: 0)
    tracker.observe_train(10)
    assert tracker.snapshot(1)["remaining_validation_runs"] == 0
    assert tracker.snapshot(1)["eta_validation_seconds"] == 0
    assert format_duration(None) == "估算中"
    assert format_duration(3600.1) == "01:00:01"
    assert format_duration(25 * 3600) == "1天 01:00:00"
