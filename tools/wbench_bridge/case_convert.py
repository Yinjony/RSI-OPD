"""Convert WBench navigation cases into minWM bridge manifest records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TRAJ_RE = re.compile(r"^([a-z]+)\*(\d+)$", re.IGNORECASE)

NAV_ACTION_TO_LETTER = {
    "W": "w",
    "S": "s",
    "A": "a",
    "D": "d",
    "w": "w",
    "s": "s",
    "a": "a",
    "d": "d",
    "forward": "w",
    "back": "s",
    "backward": "s",
    "left": "j",
    "right": "l",
    "up": "i",
    "down": "k",
    "←": "j",
    "→": "l",
    "↑": "i",
    "↓": "k",
}


def load_wbench_case(case_path: Path) -> dict[str, Any]:
    with case_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _navigation_letter(action: str) -> str:
    key = action.strip()
    if key not in NAV_ACTION_TO_LETTER:
        raise ValueError(f"Unsupported navigation action: {action!r}")
    return NAV_ACTION_TO_LETTER[key]


def _turn_trajectory(action: str, expected_latent_frames: int) -> str:
    steps = max(int(expected_latent_frames) - 1, 1)
    parts = [part.strip() for part in action.split("+") if part.strip()]
    if not parts:
        raise ValueError(f"Empty navigation action: {action!r}")
    if len(parts) == 1:
        return f"{_navigation_letter(parts[0])}*{steps}"

    letters = [_navigation_letter(part) for part in parts]
    minor = steps // len(letters)
    extra = steps - minor * len(letters)
    counts = [minor + (1 if index < extra else 0) for index in range(len(letters))]
    return ",".join(f"{letter}*{count}" for letter, count in zip(letters, counts))


def _parse_trajectory_segment(segment: str) -> tuple[str, int]:
    match = TRAJ_RE.match(segment.strip())
    if not match:
        raise ValueError(f"Invalid trajectory segment: {segment!r}")
    return match.group(1).lower(), int(match.group(2))


def _parse_trajectory_string(trajectory: str) -> list[tuple[str, int]]:
    segments = []
    for part in trajectory.strip().split(","):
        part = part.strip()
        if part:
            segments.append(_parse_trajectory_segment(part))
    if not segments:
        raise ValueError(f"Empty trajectory string: {trajectory!r}")
    return segments


def _allocate_steps(total: int, weights: list[int]) -> list[int]:
    if not weights:
        raise ValueError("weights must be non-empty")
    if sum(weights) <= 0:
        raise ValueError("weights must be positive")
    if total < 0:
        raise ValueError("total must be non-negative")
    if total == 0:
        return [0] * len(weights)

    if len(set(weights)) == 1:
        base, rem = divmod(total, len(weights))
        return [base + (1 if index < rem else 0) for index in range(len(weights))]

    weight_sum = sum(weights)
    alloc: list[int] = []
    remaining = total
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            alloc.append(remaining)
            continue
        share = int(total * weight / weight_sum)
        alloc.append(share)
        remaining -= share
    if sum(alloc) != total:
        raise ValueError(f"Allocation mismatch: total={total}, weights={weights}, alloc={alloc}")
    return alloc


def merge_turns_to_single_trajectory(
    turns: list[dict[str, Any]], expected_latent_frames: int
) -> str:
    """Compress per-turn trajectories into one string summing to latent_frames - 1 steps."""
    total_steps = max(int(expected_latent_frames) - 1, 1)
    if not turns:
        raise ValueError("merge_turns_to_single_trajectory requires at least one turn")

    turn_segments = [_parse_trajectory_string(str(turn["trajectory"])) for turn in turns]
    turn_weights = [sum(count for _, count in segments) for segments in turn_segments]
    turn_alloc = _allocate_steps(total_steps, turn_weights)

    parts: list[str] = []
    for segments, steps in zip(turn_segments, turn_alloc):
        if steps <= 0:
            continue
        if len(segments) == 1:
            letter, _ = segments[0]
            parts.append(f"{letter}*{steps}")
            continue
        seg_weights = [count for _, count in segments]
        seg_alloc = _allocate_steps(steps, seg_weights)
        for (letter, _), seg_steps in zip(segments, seg_alloc):
            if seg_steps > 0:
                parts.append(f"{letter}*{seg_steps}")

    merged_steps = sum(_parse_trajectory_segment(part)[1] for part in parts)
    if merged_steps != total_steps:
        raise ValueError(f"Merged steps must equal {total_steps}, got {parts}")
    return ",".join(parts)


def _build_prompt(case: dict[str, Any]) -> str:
    chunks = [
        case.get("environment_prompt", "").strip(),
        case.get("character_prompt", "").strip(),
        case.get("perspective_prompt", "").strip(),
    ]
    prompt = " ".join(part for part in chunks if part)
    if not prompt:
        raise ValueError(f"Case {case.get('id')} has empty prompt fields")
    return prompt


def _navigation_move_yaw(action: str) -> dict[str, int]:
    primary = action.split("+", 1)[0].strip()
    letter = _navigation_letter(primary)
    if letter == "w":
        return {"move": [1, 0], "yaw": 0, "pitch": 0}
    if letter == "s":
        return {"move": [-1, 0], "yaw": 0, "pitch": 0}
    if letter == "a":
        return {"move": [0, -1], "yaw": 0, "pitch": 0}
    if letter == "d":
        return {"move": [0, 1], "yaw": 0, "pitch": 0}
    if letter == "j":
        return {"move": [0, 0], "yaw": -1, "pitch": 0}
    if letter == "l":
        return {"move": [0, 0], "yaw": 1, "pitch": 0}
    if letter == "i":
        return {"move": [0, 0], "yaw": 0, "pitch": 1}
    if letter == "k":
        return {"move": [0, 0], "yaw": 0, "pitch": -1}
    raise ValueError(f"Unsupported navigation letter: {letter}")


def convert_case(
    case: dict[str, Any],
    images_dir: Path,
    *,
    expected_latent_frames: int = 20,
) -> dict[str, Any] | None:
    case_id = str(case.get("id", "")).strip()
    if not case_id:
        return None

    nav_turns = [
        item
        for item in case.get("interactions", [])
        if isinstance(item, dict) and item.get("type") == "navigation"
    ]
    if not nav_turns:
        return None

    initial_image = case.get("settings", {}).get("initial_image", "")
    if not initial_image:
        return None
    image_path = images_dir / Path(str(initial_image)).name
    if not image_path.is_file():
        return None

    turns: list[dict[str, Any]] = []
    for item in nav_turns:
        action = str(item.get("action", "")).strip()
        if not action:
            return None
        turn_no = int(item.get("turn", len(turns) + 1))
        turns.append(
            {
                "turn": turn_no,
                "action": action,
                "navigation": _navigation_move_yaw(action),
                "trajectory": _turn_trajectory(action, expected_latent_frames),
            }
        )

    return {
        "case_id": case_id,
        "name": f"case_{case_id}",
        "perspective": case.get("settings", {}).get("perspective", ""),
        "prompt": _build_prompt(case),
        "image_path": str(image_path.resolve()),
        "expected_latent_frames": int(expected_latent_frames),
        "turns": turns,
    }


def discover_navigation_cases(cases_dir: Path) -> list[Path]:
    return sorted(cases_dir.glob("case_*.json"))


def convert_cases_dir(
    wbench_data_dir: Path,
    *,
    expected_latent_frames: int = 20,
) -> tuple[list[dict[str, Any]], list[str]]:
    cases_dir = wbench_data_dir / "cases"
    images_dir = wbench_data_dir / "images"
    samples: list[dict[str, Any]] = []
    skipped: list[str] = []
    for case_path in discover_navigation_cases(cases_dir):
        case = load_wbench_case(case_path)
        sample = convert_case(
            case, images_dir, expected_latent_frames=expected_latent_frames
        )
        if sample is None:
            skipped.append(case_path.stem)
            continue
        sample["wbench_case_path"] = str(case_path.resolve())
        samples.append(sample)
    samples.sort(key=lambda item: int(item["case_id"]))
    return samples, skipped
