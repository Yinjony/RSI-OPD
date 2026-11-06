"""Wall-clock ETA for optimizer updates, scheduled validation and checkpoints."""

from collections import deque
from datetime import datetime, timedelta
import math
import time
from zoneinfo import ZoneInfo

from wan_utils.biopd_validation import validation_kind


def format_duration(seconds):
    if seconds is None:
        return "估算中"
    total = math.ceil(max(0, seconds))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02}:{minutes:02}:{seconds:02}"
    return f"{days}天 {clock}" if days else clock


class TrainingETA:
    def __init__(self, config, *, world_size, val_size=0, start_step=0, clock=time.monotonic,
                 wall_clock=None):
        self.config = config
        self.world_size = world_size
        self.val_size = val_size
        self.start_step = start_step
        self.clock = clock
        self.wall_clock = wall_clock or (lambda: datetime.now(ZoneInfo("Asia/Shanghai")))
        self.started = clock()
        self.train_times = deque(maxlen=20)
        self.validation_rates = deque(maxlen=5)
        self.checkpoint_times = deque(maxlen=5)
        self.validated = set()
        self.saved = set()

    def observe_train(self, seconds):
        self.train_times.append(max(0, seconds))

    def observe_validation(self, step, samples, seconds):
        # Validation is distributed over ranks, not samples processed serially
        # on a single GPU. Estimate wall time from the busiest rank's work.
        units = math.ceil(samples / self.world_size)
        if units:
            self.validation_rates.append(max(0, seconds) / units)
        self.validated.add(step)

    def observe_checkpoint(self, step, seconds):
        self.checkpoint_times.append(max(0, seconds))
        self.saved.add(step)

    def snapshot(self, step):
        maximum = int(self.config.max_steps)
        remaining_steps = max(0, maximum - step)
        every = int(self.config.get("val_every", 0))
        val_candidates = {0, maximum, *range(every, maximum, every)} if every else set()
        val_steps = [s for s in val_candidates if s >= step and s not in self.validated
                     and (self.start_step == 0 or s > self.start_step) and validation_kind(s, self.config)]
        periodic_size = min(self.val_size, int(self.config.get("val_num_samples", 20)))
        val_units = sum(math.ceil((self.val_size if validation_kind(s, self.config) == "full"
                                   else periodic_size) / self.world_size) for s in val_steps)
        save_every = int(self.config.save_every)
        save_candidates = {maximum, *range(save_every, maximum, save_every)}
        saves = [s for s in save_candidates if s >= step and s not in self.saved and s > self.start_step]

        missing = []
        estimates = {}
        for name, work, history in (("train", remaining_steps, self.train_times),
                                    ("validation", val_units, self.validation_rates),
                                    ("checkpoint", len(saves), self.checkpoint_times)):
            estimates[name] = work * sum(history) / len(history) if history else (0 if not work else None)
            if estimates[name] is None:
                missing.append(name)
        # Do not imply a useful total before measuring even one training update.
        eta = (sum(value for value in estimates.values() if value is not None)
               if estimates["train"] is not None else None)
        elapsed = max(0, self.clock() - self.started)
        total = elapsed + eta if eta is not None else None
        finish = (self.wall_clock() + timedelta(seconds=eta)).isoformat(sep=" ", timespec="seconds") if eta is not None else None
        return dict(step=step, max_steps=maximum, progress_percent=round(min(step / maximum, 1) * 100, 2),
                    elapsed_seconds=elapsed, elapsed=format_duration(elapsed),
                    average_step_seconds=sum(self.train_times) / len(self.train_times) if self.train_times else None,
                    eta_train_seconds=estimates["train"], eta_validation_seconds=estimates["validation"],
                    eta_checkpoint_seconds=estimates["checkpoint"], remaining_validation_runs=len(val_steps),
                    remaining_checkpoints=len(saves), eta_seconds=eta, eta=format_duration(eta),
                    estimated_total_seconds=total, estimated_total=format_duration(total),
                    eta_complete=not missing, eta_missing_components=missing, estimated_finish_time=finish)


def progress_message(phase, snapshot):
    message = (f"[BiOPD {phase}] {snapshot['step']}/{snapshot['max_steps']} "
               f"({snapshot['progress_percent']:.1f}%) | 已用 {snapshot['elapsed']} "
               f"| 预计剩余 {snapshot['eta']} | 预计总用时 {snapshot['estimated_total']}")
    if snapshot["estimated_finish_time"]:
        message += f" | 预计完成 {snapshot['estimated_finish_time']}"
    labels = dict(train="训练", validation="验证", checkpoint="保存")
    if snapshot["eta_missing_components"]:
        message += " | 待实测（尚未计入 ETA）: " + ", ".join(labels[name] for name in snapshot["eta_missing_components"])
    return message
