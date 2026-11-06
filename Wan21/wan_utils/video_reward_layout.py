"""Video reward adapters for GRPO training.

The public interface intentionally accepts a full video tensor so downstream
projects can swap in task-specific video rewards without changing trainers.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import posixpath
import re
import shutil
import shlex
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


@dataclass
class RewardOutput:
    rewards: torch.Tensor
    log_dict: dict


class VideoRewardFn:
    def __call__(
        self,
        prompts: List[str],
        videos: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        raise NotImplementedError


def _config_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class MeanPixelVideoReward(VideoRewardFn):
    """Tiny deterministic reward used for smoke tests and CPU-only checks."""

    def __call__(
        self,
        prompts: List[str],
        videos: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        del prompts, metadata
        rewards = videos.float().mean(dim=tuple(range(1, videos.ndim)))
        return RewardOutput(rewards=rewards, log_dict={"reward_mean_pixel": rewards.mean()})


DEFAULT_LAYOUT_VIDEO_PATH = "/root/data/ltl/minWM/Wan21/prompts/test/video"
DEFAULT_LAYOUT_OUTPUT_PATH = "/root/data/ltl/minWM/Wan21/prompts/test"
DEFAULT_LAYOUT_PROMPT_DIR = "/root/data/ltl/minWM/Wan21/prompts/test/prompts"
DEFAULT_LAYOUT_MODEL_PATH = "/root/data/ltl/llava-v1.6-34b"
DEFAULT_LAYOUT_VIDEO_SUFFIXES = (".mp4",)
DEFAULT_LAYOUT_EVAL_RESULTS = "t2v_compbench_eval_results.json"
DEFAULT_LAYOUT_STATISTICS = "t2v_compbench_statistics.json"


@dataclass(frozen=True)
class LayoutDimensionSpec:
    name: str
    csv_name: str
    id_column: str
    score_column: str
    score_transform: Callable[[float], float] = lambda value: value


LAYOUT_DIMENSIONS = (
    LayoutDimensionSpec(
        name="physical_interactions",
        csv_name="bidirectional_object_interactions_score.csv",
        id_column="name",
        score_column="Score",
        score_transform=lambda value: (value - 1.0) / 9.0,
    ),
    LayoutDimensionSpec(
        name="spatial_relationship",
        csv_name="bidirectional_spatial_score.csv",
        id_column="id",
        score_column="Score",
    ),
    LayoutDimensionSpec(
        name="object_count",
        csv_name="bidirectional_numeracy_video.csv",
        id_column="id",
        score_column="Score_1",
    ),
)
LAYOUT_DIMENSION_NAMES = tuple(spec.name for spec in LAYOUT_DIMENSIONS)
LAYOUT_TASK_TO_DIMENSION = {
    "interaction": "physical_interactions",
    "physical": "physical_interactions",
    "physical_interaction": "physical_interactions",
    "physical_interactions": "physical_interactions",
    "spatial": "spatial_relationship",
    "spatial_relation": "spatial_relationship",
    "spatial_relationship": "spatial_relationship",
    "numeracy": "object_count",
    "object": "object_count",
    "object_count": "object_count",
    "count": "object_count",
}


def _layout_dimension_names_for_tasks(tasks: Sequence[str] | None) -> tuple[str, ...]:
    if not tasks:
        return LAYOUT_DIMENSION_NAMES

    names: list[str] = []
    for task in tasks:
        key = str(task).strip().lower()
        dimension = LAYOUT_TASK_TO_DIMENSION.get(key)
        if dimension is None and key in LAYOUT_DIMENSION_NAMES:
            dimension = key
        if dimension is not None and dimension not in names:
            names.append(dimension)
    return tuple(names or LAYOUT_DIMENSION_NAMES)


def _layout_is_absolute_arg(path: str) -> bool:
    return path.startswith("/") or Path(path).is_absolute()


def _layout_join_arg_path(parent: str, child: str) -> str:
    if parent.startswith("/"):
        return posixpath.join(parent, child)
    return str(Path(parent) / child)


def _layout_child_path(parent: str, child: str) -> str:
    if _layout_is_absolute_arg(child):
        return child
    return _layout_join_arg_path(parent, child)


def _layout_finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _layout_extract_number(raw_id: str) -> str:
    match = re.search(r"\d+", raw_id)
    if not match:
        return raw_id.strip()
    return match.group(0)


def _layout_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def _layout_read_video_order_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict):
            for key in ("videos", "files", "filenames"):
                value = data.get(key)
                if isinstance(value, list):
                    return [str(item) for item in value]
        raise TypeError(f"Video list JSON must be a list or contain videos/files/filenames: {path}")

    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _layout_discover_video_files(
    video_dir: Path,
    suffixes: Sequence[str],
    order_file: str | None,
) -> list[Path]:
    if not video_dir.is_dir():
        raise NotADirectoryError(f"video path is not a directory: {video_dir}")

    if order_file:
        names = _layout_read_video_order_file(Path(order_file))
        files = [video_dir / name for name in names]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError("Video list contains missing files:\n" + "\n".join(missing[:20]))
        return files

    normalized_suffixes = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in suffixes
    }
    files = [
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in normalized_suffixes
    ]
    files.sort(key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"No videos with suffixes {sorted(normalized_suffixes)} found in {video_dir}")
    return files


def _layout_copy_numbered_video(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        os.symlink(source, destination)
    else:
        raise ValueError(f"Unknown video copy mode: {mode}")


def _layout_conda_python_command(
    env_name: str,
    script: str,
    args: Sequence[str],
    python: str,
    conda_exe: str,
    no_conda: bool,
) -> list[str]:
    command = [python, script, *args]
    if no_conda:
        return command
    return [conda_exe, "run", "--no-capture-output", "-n", env_name, *command]


def _layout_task_python_command(
    env_name: str,
    script: str,
    args: Sequence[str],
    python: str,
    conda_exe: str,
    no_conda: bool,
    direct_python: str | None,
) -> list[str]:
    if direct_python:
        return [direct_python, script, *args]
    return _layout_conda_python_command(env_name, script, args, python, conda_exe, no_conda)


def _layout_load_video_mapping(mapping_path: Path) -> tuple[str | None, dict[str, str]]:
    if not mapping_path.exists():
        return None, {}

    data = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    videos = data.get("videos", [])
    number_to_path: dict[str, str] = {}
    if isinstance(videos, list):
        for item in videos:
            if not isinstance(item, dict):
                continue
            number = str(item.get("number", "")).strip()
            source_path = item.get("source_path")
            source_name = item.get("source_name")
            mapped_path = str(source_path or source_name or "")
            if not number or not mapped_path:
                continue
            number_to_path[number] = mapped_path
            number_to_path[number.zfill(4)] = mapped_path
            number_to_path[number.lstrip("0") or "0"] = mapped_path

    source_video_path = data.get("source_video_path")
    return str(source_video_path) if source_video_path else None, number_to_path


def _layout_lookup_video_path(number_to_path: dict[str, str], raw_id: str) -> str:
    number = _layout_extract_number(raw_id)
    return (
        number_to_path.get(number)
        or number_to_path.get(number.zfill(4))
        or number_to_path.get(number.lstrip("0") or "0")
        or raw_id
    )


def _layout_iter_dimension_scores(
    csv_path: Path,
    spec: LayoutDimensionSpec,
    number_to_path: dict[str, str],
) -> Iterable[dict[str, object]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        missing_columns = {spec.id_column, spec.score_column} - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required column(s): {missing}")

        for row in reader:
            raw_score = _layout_finite_float(row.get(spec.score_column))
            if raw_score is None:
                continue

            raw_id = str(row.get(spec.id_column, "")).strip()
            yield {
                "video_path": _layout_lookup_video_path(number_to_path, raw_id),
                "video_results": spec.score_transform(raw_score),
            }


def _layout_rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _layout_build_dimension_statistics(
    scores: list[float],
    official_mean: float,
) -> dict[str, float | int]:
    if not scores:
        raise ValueError("Cannot calculate statistics for an empty score list")

    calculated_mean = statistics.mean(scores)
    variance = statistics.variance(scores) if len(scores) > 1 else 0.0
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

    return {
        "calculated_mean": _layout_rounded(calculated_mean),
        "official_mean": official_mean,
        "difference_from_official": _layout_rounded(calculated_mean - official_mean),
        "variance": _layout_rounded(variance),
        "std_dev": _layout_rounded(std_dev),
        "min": _layout_rounded(min(scores)),
        "max": _layout_rounded(max(scores)),
        "median": _layout_rounded(statistics.median(scores)),
        "video_count": len(scores),
    }


def summarize_t2v_compbench_results(
    result_dir: str | Path,
    video_mapping: str | Path | None,
    output_eval_results: str | Path,
    output_statistics: str | Path,
    dimensions: Sequence[str] | None = None,
) -> dict[str, str]:
    result_dir = Path(result_dir)
    mapping_path = Path(video_mapping) if video_mapping else result_dir / "video_name_mapping.json"
    eval_results_path = Path(output_eval_results)
    statistics_path = Path(output_statistics)

    source_video_path, number_to_path = _layout_load_video_mapping(mapping_path)
    eval_results: dict[str, list[object]] = {}
    dimension_statistics: dict[str, dict[str, float | int]] = {}
    all_scores: list[float] = []
    dimensions_evaluated: list[str] = []
    total_videos = 0
    requested_dimensions = set(dimensions or LAYOUT_DIMENSION_NAMES)

    for spec in LAYOUT_DIMENSIONS:
        if spec.name not in requested_dimensions:
            continue
        csv_path = result_dir / spec.csv_name
        records = list(_layout_iter_dimension_scores(csv_path, spec, number_to_path))
        scores = [float(record["video_results"]) for record in records]
        if not scores:
            raise ValueError(f"No valid scores found in {csv_path}")

        mean_score = statistics.mean(scores)
        official_mean = round(mean_score, 4)
        eval_results[spec.name] = [mean_score, records]
        dimension_statistics[spec.name] = _layout_build_dimension_statistics(scores, official_mean)
        all_scores.extend(scores)
        dimensions_evaluated.append(spec.name)
        total_videos = max(total_videos, len(scores))

    overall_mean = statistics.mean(all_scores)
    overall_std_dev = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
    statistics_json = {
        "summary_file": str(eval_results_path.resolve()),
        "evaluation_time": datetime.now().isoformat(),
        "video_directory": source_video_path or str(result_dir.resolve()),
        "total_videos": total_videos,
        "dimensions_evaluated": dimensions_evaluated,
        "overall_average_score": round(
            statistics.mean([eval_results[name][0] for name in dimensions_evaluated]),
            4,
        ),
        "dimension_statistics": dimension_statistics,
        "calculated_overall": {
            "mean": _layout_rounded(overall_mean),
            "std_dev": _layout_rounded(overall_std_dev),
            "total_scores": len(all_scores),
        },
    }

    _layout_write_json(eval_results_path, eval_results)
    _layout_write_json(statistics_path, statistics_json)

    print(f"Wrote {eval_results_path}")
    print(f"Wrote {statistics_path}")
    for dimension in dimensions_evaluated:
        stats = dimension_statistics[dimension]
        print(
            f"{dimension}: mean={stats['calculated_mean']}, "
            f"variance={stats['variance']}, n={stats['video_count']}"
        )

    return {
        "eval_results": str(eval_results_path),
        "statistics": str(statistics_path),
    }


class LayoutCompBenchEvaluate:
    """Callable Layout/T2V-CompBench evaluation pipeline."""

    DEFAULT_TASKS = ("interaction", "spatial", "numeracy")

    def __init__(
        self,
        video_path: str = DEFAULT_LAYOUT_VIDEO_PATH,
        output_path: str = DEFAULT_LAYOUT_OUTPUT_PATH,
        prompt_dir: str = DEFAULT_LAYOUT_PROMPT_DIR,
        interaction_prompt_file: str = "layout_prompts.json",
        spatial_prompt_file: str = "layout_spatial_relations.json",
        numeracy_prompt_file: str = "layout_object_counts.json",
        t2v_model: str = "bidirectional",
        model_path: str = DEFAULT_LAYOUT_MODEL_PATH,
        interaction_script: str = "/root/data/ltl/T2V-CompBench/LLaVA/llava/eval/compbench_eval_interaction.py",
        spatial_script: str = "/root/data/ltl/T2V-CompBench/Grounded-Segment-Anything/compbench_eval_spatial_relationships.py",
        numeracy_script: str = "/root/data/ltl/T2V-CompBench/Grounded-Segment-Anything/GroundingDINO/demo/compbench_eval_numeracy.py",
        spatial_depth_folder: str = "output_spatial_depth",
        spatial_output_dir: str = "output_spatial",
        numeracy_output_dir: str = "output_numeracy",
        llava_env: str = "llava",
        compbench_env: str = "compbench",
        conda_exe: str = "conda",
        python: str = "python",
        llava_python: str | None = None,
        compbench_python: str | None = None,
        compbench_extra_pythonpath: str | None = None,
        compbench_extra_ld_library_path: str | None = None,
        no_conda: bool = True,
        cuda_visible_devices: str | None = "2",
        task_cuda_visible_devices: dict[str, str] | None = None,
        parallel_tasks: bool = False,
        workdir: str | None = None,
        only: Sequence[str] | None = None,
        no_prepare_numbered_videos: bool = False,
        video_list_file: str | None = None,
        video_suffixes: Sequence[str] = DEFAULT_LAYOUT_VIDEO_SUFFIXES,
        number_width: int = 4,
        video_copy_mode: str = "copy",
        temp_parent_dir: str | None = "/root/data/ltl/minWM/Wan21/scripts/evaluate/results/temp_parent",
        temp_video_dir: str | None = "/root/data/ltl/minWM/Wan21/scripts/evaluate/results/temp_videos",
        keep_temp_videos: bool = False,
        video_name_map_output: str = "video_name_mapping.json",
        dry_run: bool = False,
        skip_file_checks: bool = False,
        create_output_dirs: bool = True,
        summarize: bool = True,
        summary_eval_results_output: str = DEFAULT_LAYOUT_EVAL_RESULTS,
        summary_statistics_output: str = DEFAULT_LAYOUT_STATISTICS,
        timeout: int | None = None,
        command_stdout_to_stderr: bool = False,
        interaction_runner: Callable[[str, str, str], str] | None = None,
        compbench_runner: Callable[[Sequence[str], dict[str, list[str]]], None] | None = None,
    ) -> None:
        self.video_path = video_path
        self.output_path = output_path
        self.prompt_dir = prompt_dir
        self.interaction_prompt_file = _layout_child_path(prompt_dir, interaction_prompt_file)
        self.spatial_prompt_file = _layout_child_path(prompt_dir, spatial_prompt_file)
        self.numeracy_prompt_file = _layout_child_path(prompt_dir, numeracy_prompt_file)
        self.t2v_model = t2v_model
        self.model_path = model_path
        self.interaction_script = interaction_script
        self.spatial_script = spatial_script
        self.numeracy_script = numeracy_script
        self.spatial_depth_folder = _layout_child_path(output_path, spatial_depth_folder)
        self.spatial_output_dir = _layout_child_path(output_path, spatial_output_dir)
        self.numeracy_output_dir = _layout_child_path(output_path, numeracy_output_dir)
        self.llava_env = llava_env
        self.compbench_env = compbench_env
        self.conda_exe = conda_exe
        self.python = python
        self.llava_python = llava_python
        self.compbench_python = compbench_python
        self.compbench_extra_pythonpath = compbench_extra_pythonpath
        self.compbench_extra_ld_library_path = compbench_extra_ld_library_path
        self.no_conda = no_conda
        self.cuda_visible_devices = cuda_visible_devices
        self.task_cuda_visible_devices = task_cuda_visible_devices or {}
        self.parallel_tasks = bool(parallel_tasks)
        self.workdir = workdir
        self.only = list(only or self.DEFAULT_TASKS)
        self.no_prepare_numbered_videos = no_prepare_numbered_videos
        self.video_list_file = video_list_file
        self.video_suffixes = tuple(video_suffixes)
        self.number_width = int(number_width)
        self.video_copy_mode = video_copy_mode
        self.temp_parent_dir = temp_parent_dir
        self.temp_video_dir = temp_video_dir
        self.keep_temp_videos = keep_temp_videos
        self.video_name_map_output = _layout_child_path(output_path, video_name_map_output)
        self.dry_run = dry_run
        self.skip_file_checks = skip_file_checks
        self.create_output_dirs = create_output_dirs
        self.summarize = summarize
        self.summary_eval_results_output = _layout_child_path(output_path, summary_eval_results_output)
        self.summary_statistics_output = _layout_child_path(output_path, summary_statistics_output)
        self.timeout = timeout
        self.command_stdout_to_stderr = bool(command_stdout_to_stderr)
        self.interaction_runner = interaction_runner
        self.compbench_runner = compbench_runner

    def check_required_files(self) -> None:
        required_files = {
            "interaction prompt file": self.interaction_prompt_file,
            "spatial prompt file": self.spatial_prompt_file,
            "numeracy prompt file": self.numeracy_prompt_file,
        }
        for label, path in required_files.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"{label} not found: {path}")

    def create_directories(self) -> None:
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.spatial_depth_folder).mkdir(parents=True, exist_ok=True)
        Path(self.spatial_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.numeracy_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_name_map_output).parent.mkdir(parents=True, exist_ok=True)
        if self.temp_parent_dir:
            Path(self.temp_parent_dir).mkdir(parents=True, exist_ok=True)
        if self.temp_video_dir:
            Path(self.temp_video_dir).parent.mkdir(parents=True, exist_ok=True)

    def write_video_mapping(
        self,
        source_video_path: str,
        temp_video_path: str,
        videos: list[dict[str, object]],
    ) -> None:
        mapping = {
            "source_video_path": source_video_path,
            "numbered_video_path": temp_video_path,
            "videos": videos,
            "mapping": {str(item["number"]): item["source_name"] for item in videos},
            "file_mapping": {item["numbered_name"]: item["source_name"] for item in videos},
        }
        path = Path(self.video_name_map_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Video name mapping written to: {path}", flush=True)

    def create_temp_video_dir(self) -> tempfile.TemporaryDirectory[str] | None:
        if self.temp_video_dir:
            temp_path = Path(self.temp_video_dir)
            if temp_path.exists() and any(temp_path.iterdir()):
                raise RuntimeError(f"Temporary video directory is not empty: {temp_path}")
            temp_path.mkdir(parents=True, exist_ok=True)
            return None
        return tempfile.TemporaryDirectory(prefix="compbench_numbered_videos_", dir=self.temp_parent_dir)

    @contextmanager
    def prepared_video_path(self) -> Iterator[str]:
        if self.no_prepare_numbered_videos:
            if not self.dry_run:
                video_files = _layout_discover_video_files(
                    Path(self.video_path),
                    self.video_suffixes,
                    self.video_list_file,
                )
                number_width = max(self.number_width, len(str(len(video_files))))
                videos = []
                for index, source in enumerate(video_files, start=1):
                    number = f"{index:0{number_width}d}"
                    videos.append(
                        {
                            "index": index,
                            "number": number,
                            "numbered_name": source.name,
                            "source_name": source.name,
                            "source_path": str(source),
                            "temp_path": str(source),
                        }
                    )
                self.write_video_mapping(self.video_path, self.video_path, videos)
            yield self.video_path
            return

        if self.dry_run:
            temp_path = self.temp_video_dir or "<temporary-numbered-video-dir>"
            print(
                "Dry run: would copy videos from "
                f"{self.video_path} to {temp_path} as 0001.mp4, 0002.mp4, ... "
                f"and write {self.video_name_map_output}",
                flush=True,
            )
            yield temp_path
            return

        source_video_path = self.video_path
        source_dir = Path(source_video_path)
        video_files = _layout_discover_video_files(source_dir, self.video_suffixes, self.video_list_file)
        number_width = max(self.number_width, len(str(len(video_files))))

        temp_context = self.create_temp_video_dir()
        temp_dir = Path(temp_context.name if temp_context is not None else str(self.temp_video_dir))
        videos: list[dict[str, object]] = []

        try:
            for index, source in enumerate(video_files, start=1):
                number = f"{index:0{number_width}d}"
                numbered_name = f"{number}.mp4"
                destination = temp_dir / numbered_name
                _layout_copy_numbered_video(source, destination, self.video_copy_mode)
                videos.append(
                    {
                        "index": index,
                        "number": number,
                        "numbered_name": numbered_name,
                        "source_name": source.name,
                        "source_path": str(source),
                        "temp_path": str(destination),
                    }
                )

            self.write_video_mapping(source_video_path, str(temp_dir), videos)
            print(f"Prepared {len(videos)} numbered videos in: {temp_dir}", flush=True)
            yield str(temp_dir)
        finally:
            if self.keep_temp_videos:
                print(f"Keeping temporary numbered videos in: {temp_dir}", flush=True)
            elif temp_context is not None:
                temp_context.cleanup()
                print(f"Removed temporary numbered videos from: {temp_dir}", flush=True)
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"Removed temporary numbered videos from: {temp_dir}", flush=True)

    def build_interaction_command(self, video_path: str) -> list[str]:
        args = [
            "--video-path",
            video_path,
            "--output-path",
            self.output_path,
            "--output-root",
            self.output_path,
            "--read-prompt-file",
            self.interaction_prompt_file,
            "--t2v-model",
            self.t2v_model,
            "--model-path",
            self.model_path,
        ]
        return _layout_task_python_command(
            self.llava_env,
            self.interaction_script,
            args,
            self.python,
            self.conda_exe,
            self.no_conda,
            self.llava_python,
        )

    def build_spatial_command(self, video_path: str) -> list[str]:
        args = [
            "--video-path",
            video_path,
            "--output-root",
            self.output_path,
            "--depth_folder",
            self.spatial_depth_folder,
            "--output-path",
            self.output_path,
            "--read-prompt-file",
            self.spatial_prompt_file,
            "--t2v-model",
            self.t2v_model,
            "--output_dir",
            self.spatial_output_dir,
        ]
        return _layout_task_python_command(
            self.compbench_env,
            self.spatial_script,
            args,
            self.python,
            self.conda_exe,
            self.no_conda,
            self.compbench_python,
        )

    def build_numeracy_command(self, video_path: str) -> list[str]:
        args = [
            "--video-path",
            video_path,
            "--output-path",
            self.output_path,
            "--read-prompt-file",
            self.numeracy_prompt_file,
            "--t2v-model",
            self.t2v_model,
            "--output_dir",
            self.numeracy_output_dir,
        ]
        return _layout_task_python_command(
            self.compbench_env,
            self.numeracy_script,
            args,
            self.python,
            self.conda_exe,
            self.no_conda,
            self.compbench_python,
        )

    @staticmethod
    def _prepend_env_path(env: dict, key: str, value: str | None) -> None:
        if not value:
            return
        existing = env.get(key, "")
        env[key] = f"{value}:{existing}" if existing else value

    def _command_env(self, cuda_visible_devices: str | None, task_name: str | None = None) -> tuple[dict, str]:
        env = os.environ.copy()
        display_env = ""
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
            display_env = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
        if task_name in {"spatial", "numeracy"}:
            self._prepend_env_path(env, "PYTHONPATH", self.compbench_extra_pythonpath)
            self._prepend_env_path(env, "LD_LIBRARY_PATH", self.compbench_extra_ld_library_path)
        return env, display_env

    def _task_cuda(self, task_name: str) -> str | None:
        return self.task_cuda_visible_devices.get(task_name, self.cuda_visible_devices)

    def run_command(self, command: list[str], task_name: str | None = None) -> None:
        env, display_env = self._command_env(self._task_cuda(task_name or ""), task_name=task_name)

        print(f"\n$ {display_env}{shlex.join(command)}", flush=True)
        if self.dry_run:
            return

        stdout = sys.stderr if self.command_stdout_to_stderr else None
        stderr = sys.stderr if self.command_stdout_to_stderr else None

        completed = subprocess.run(
            command,
            check=False,
            env=env,
            cwd=self.workdir,
            timeout=self.timeout,
            stdout=stdout,
            stderr=stderr,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Layout/T2V-CompBench command failed.\n"
                f"Command: {shlex.join(command)}\n"
                f"returncode={completed.returncode}"
            )

    def run_commands_parallel(self, commands: list[tuple[str, list[str]]]) -> None:
        if self.dry_run:
            for task_name, command in commands:
                env, display_env = self._command_env(self._task_cuda(task_name), task_name=task_name)
                del env
                print(f"\n$ {display_env}{shlex.join(command)}", flush=True)
            return

        processes = []
        stdout = sys.stderr if self.command_stdout_to_stderr else None
        stderr = sys.stderr if self.command_stdout_to_stderr else None
        try:
            for task_name, command in commands:
                env, display_env = self._command_env(self._task_cuda(task_name), task_name=task_name)
                print(f"\n$ {display_env}{shlex.join(command)}", flush=True)
                processes.append(
                    (
                        task_name,
                        command,
                        subprocess.Popen(
                            command,
                            env=env,
                            cwd=self.workdir,
                            stdout=stdout,
                            stderr=stderr,
                        ),
                    )
                )

            deadline = None if self.timeout is None else time.monotonic() + float(self.timeout)
            for task_name, command, process in processes:
                wait_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                returncode = process.wait(timeout=wait_timeout)
                if returncode != 0:
                    raise RuntimeError(
                        "Layout/T2V-CompBench command failed.\n"
                        f"Task: {task_name}\n"
                        f"Command: {shlex.join(command)}\n"
                        f"returncode={returncode}"
                    )
        except Exception:
            for _, _, process in processes:
                if process.poll() is None:
                    process.terminate()
            raise

    def run_interaction_and_commands_parallel(
        self,
        numbered_video_path: str,
        commands: dict[str, list[str]],
    ) -> None:
        if self.interaction_runner is None:
            raise RuntimeError("interaction_runner is required for interaction parallel mode.")

        errors: list[tuple[str, str]] = []

        def run_interaction() -> None:
            try:
                print("\n== interaction ==", flush=True)
                self.interaction_runner(
                    numbered_video_path,
                    self.output_path,
                    self.interaction_prompt_file,
                )
            except Exception:
                errors.append(("interaction", traceback.format_exc()))

        interaction_thread = threading.Thread(
            target=run_interaction,
            name="layoutreward-interaction",
            daemon=False,
        )
        interaction_thread.start()

        compbench_tasks = [
            str(task_name).strip().lower()
            for task_name in self.only
            if str(task_name).strip().lower() != "interaction"
        ]
        if compbench_tasks:
            try:
                self.run_compbench_tasks(compbench_tasks, commands, allow_parallel=False)
            except Exception:
                errors.append(("compbench", traceback.format_exc()))

        interaction_thread.join()

        if errors:
            details = "\n\n".join(
                f"[{task_name}]\n{traceback_text}" for task_name, traceback_text in errors
            )
            raise RuntimeError("Parallel Layout/T2V-CompBench task failed.\n" + details)

    def run_compbench_tasks(
        self,
        task_names: Sequence[str],
        commands: dict[str, list[str]],
        allow_parallel: bool = True,
    ) -> None:
        normalized = [str(task_name).strip().lower() for task_name in task_names]
        selected = [task_name for task_name in normalized if task_name in {"spatial", "numeracy"}]
        if not selected:
            return

        if self.compbench_runner is not None:
            print(f"\n== persistent compbench tasks: {','.join(selected)} ==", flush=True)
            self.compbench_runner(
                selected,
                {task_name: commands[task_name] for task_name in selected},
            )
            return

        if allow_parallel and self.parallel_tasks and len(selected) > 1:
            self.run_commands_parallel([(task_name, commands[task_name]) for task_name in selected])
            return

        for task_name in selected:
            print(f"\n== {task_name} ==", flush=True)
            self.run_command(commands[task_name], task_name=task_name)

    def evaluate(self) -> dict[str, str] | None:
        if not self.skip_file_checks and not self.dry_run:
            self.check_required_files()

        if self.create_output_dirs and not self.dry_run:
            self.create_directories()

        with self.prepared_video_path() as numbered_video_path:
            commands = {
                "interaction": self.build_interaction_command(numbered_video_path),
                "spatial": self.build_spatial_command(numbered_video_path),
                "numeracy": self.build_numeracy_command(numbered_video_path),
            }

            for task_name in self.only:
                task_key = str(task_name).strip().lower()
                if task_key not in commands:
                    raise ValueError(f"Unknown evaluation task: {task_name}")

            normalized_only = [str(task_name).strip().lower() for task_name in self.only]
            active_task_set = set(normalized_only)
            compbench_tasks = [task_name for task_name in normalized_only if task_name in {"spatial", "numeracy"}]

            if self.parallel_tasks and len(self.only) > 1:
                if self.interaction_runner is not None and "interaction" in active_task_set:
                    print("\n== parallel layout tasks: interaction + compbench ==", flush=True)
                    self.run_interaction_and_commands_parallel(numbered_video_path, commands)
                else:
                    print("\n== parallel layout tasks ==", flush=True)
                    if self.compbench_runner is not None and active_task_set <= {"spatial", "numeracy"}:
                        self.run_compbench_tasks(compbench_tasks, commands, allow_parallel=True)
                    else:
                        self.run_commands_parallel(
                            [
                                (task_name, commands[task_name])
                                for task_name in normalized_only
                            ]
                        )
            else:
                if "interaction" in active_task_set:
                    print("\n== interaction ==", flush=True)
                    if self.interaction_runner is not None:
                        self.interaction_runner(
                            numbered_video_path,
                            self.output_path,
                            self.interaction_prompt_file,
                        )
                    else:
                        self.run_command(commands["interaction"], task_name="interaction")
                if compbench_tasks:
                    self.run_compbench_tasks(compbench_tasks, commands, allow_parallel=False)

        if not self.summarize:
            return None

        if self.dry_run:
            print("\nDry run: would summarize CSV outputs into JSON files.")
            return None

        print("\n== summarize ==")
        return summarize_t2v_compbench_results(
            result_dir=self.output_path,
            video_mapping=self.video_name_map_output,
            output_eval_results=self.summary_eval_results_output,
            output_statistics=self.summary_statistics_output,
            dimensions=_layout_dimension_names_for_tasks(self.only),
        )


class AestheticReward(VideoRewardFn):
    """VBench-based aesthetic/quality reward for generated videos.

    The reward writes the current rollout batch to temporary mp4 files, runs
    VBench in a clean subprocess, then reads the per-video dimension scores.
    Running VBench out-of-process avoids accidental interaction with the
    training process group used by torchrun/FSDP.
    """

    DEFAULT_DIMENSIONS = [
        "aesthetic_quality",
        "imaging_quality",
        "motion_smoothness",
        "temporal_flickering",
    ]

    def __init__(
        self,
        device: torch.device,
        dimensions: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        full_json_dir: Optional[str] = None,
        fps: int = 16,
        timeout: int = 900,
        load_ckpt_from_local: bool = True,
        read_frame: bool = False,
        imaging_quality_preprocessing_mode: str = "longer",
        use_mock: bool = False,
        keep_videos: bool = False,
        viewable_videos: bool = True,
        persistent_worker: bool = False,
    ) -> None:
        self.device = device
        self.dimensions = list(dimensions or self.DEFAULT_DIMENSIONS)
        self.fps = int(fps)
        self.timeout = int(timeout)
        self.load_ckpt_from_local = bool(load_ckpt_from_local)
        self.read_frame = bool(read_frame)
        self.imaging_quality_preprocessing_mode = str(imaging_quality_preprocessing_mode)
        self.use_mock = bool(use_mock)
        self.keep_videos = bool(keep_videos)
        self.viewable_videos = bool(viewable_videos)
        self.persistent_worker = bool(persistent_worker)
        self._worker_process = None
        self._worker_log_file = None
        self._worker_log_path = None
        self._owns_output_dir = not output_dir
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="aesreward_vbench_")
        os.makedirs(self.output_dir, exist_ok=True)

        if full_json_dir:
            self.full_json_dir = str(full_json_dir)
        else:
            try:
                import vbench

                self.full_json_dir = os.path.join(
                    os.path.dirname(vbench.__file__),
                    "VBench_full_info.json",
                )
            except ImportError:
                self.full_json_dir = ""

        if not self.use_mock:
            try:
                import cv2  # noqa: F401
                import vbench  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "AestheticReward requires VBench and opencv-python in the active env. "
                    "Install/activate VBench first, or set aesreward_use_mock=true only for smoke tests."
                ) from exc
            if self.persistent_worker:
                self._start_persistent_worker()

    @staticmethod
    def _to_bool_arg(value: bool) -> str:
        return "true" if bool(value) else "false"

    @staticmethod
    def _tensor_to_video(video: torch.Tensor, output_path: str, fps: int) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("AestheticReward requires opencv-python to write mp4 files.") from exc

        video_np = video.detach().float().clamp(0, 1).cpu()
        video_np = (video_np * 255.0).round().byte().permute(0, 2, 3, 1).numpy()
        height, width = int(video_np.shape[1]), int(video_np.shape[2])
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        try:
            for frame in video_np:
                writer.write(np.ascontiguousarray(frame[..., ::-1]))
        finally:
            writer.release()

    @staticmethod
    def _find_ffmpeg_binary() -> Optional[str]:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            return ffmpeg_bin
        try:
            import imageio_ffmpeg

            bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            if bundled_ffmpeg and os.path.exists(bundled_ffmpeg):
                return bundled_ffmpeg
        except Exception:
            return None
        return None

    @staticmethod
    def _make_saved_video_viewable(video_paths: List[str]) -> int:
        ffmpeg_bin = AestheticReward._find_ffmpeg_binary()
        if not ffmpeg_bin:
            raise RuntimeError(
                "aesreward_viewable_videos=true requires ffmpeg so saved VBench "
                "mp4 files can be transcoded to H.264 for local viewing. "
                "Install system ffmpeg or the Python package imageio-ffmpeg."
            )

        converted = 0
        for video_path in video_paths:
            tmp_path = f"{video_path}.h264_tmp.mp4"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            cmd = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                tmp_path,
            ]
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise RuntimeError(
                    "Failed to transcode saved VBench video to H.264.\n"
                    f"Video: {video_path}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            os.replace(tmp_path, video_path)
            converted += 1
        return converted

    def _make_subprocess_env(self) -> dict:
        env = os.environ.copy()
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env["MASTER_PORT"] = str(30000 + (os.getpid() % 20000))
        if self.device.type == "cuda":
            env["LOCAL_RANK"] = str(self.device.index or 0)
        else:
            env["LOCAL_RANK"] = "0"
        return env

    def _persistent_worker_script(self) -> str:
        return r"""
import contextlib
import json
import os
import sys
import traceback

import torch

output_dir, full_json_dir = sys.argv[1:3]
dimensions = json.loads(sys.argv[3])
load_local = sys.argv[4].lower() == "true"
read_frame = sys.argv[5].lower() == "true"
imaging_mode = sys.argv[6]

_aes_cache = {}
_imaging_cache = {}
_motion_cache = {}

def _install_vbench_dimension_cache():
    import vbench.aesthetic_quality as aq
    import vbench.imaging_quality as iq
    import vbench.motion_smoothness as ms

    def _device_key(device):
        return str(device)

    def _get_aesthetic_cached(submodules_list, device):
        vit_path = submodules_list[0]
        aes_path = submodules_list[1]
        key = (str(vit_path), str(aes_path), _device_key(device))
        if key not in _aes_cache:
            aesthetic_model = aq.get_aesthetic_model(aes_path).to(device)
            clip_model, _ = aq.clip.load(vit_path, device=device)
            aesthetic_model.eval()
            clip_model.eval()
            _aes_cache[key] = (aesthetic_model, clip_model)
        return _aes_cache[key]

    def compute_aesthetic_quality_cached(json_dir, device, submodules_list, **kwargs):
        aesthetic_model, clip_model = _get_aesthetic_cached(submodules_list, device)
        video_list, _ = aq.load_dimension_info(json_dir, dimension="aesthetic_quality", lang="en")
        video_list = aq.distribute_list_to_rank(video_list)
        all_results, video_results = aq.laion_aesthetic(aesthetic_model, clip_model, video_list, device)
        if aq.get_world_size() > 1:
            video_results = aq.gather_list_of_dict(video_results)
            all_results = sum([d["video_results"] for d in video_results]) / len(video_results)
        return all_results, video_results

    def _get_imaging_cached(submodules_list, device):
        model_path = submodules_list["model_path"]
        key = (str(model_path), _device_key(device))
        if key not in _imaging_cache:
            model = iq.MUSIQ(pretrained_model_path=model_path)
            model.to(device)
            model.training = False
            if hasattr(model, "eval"):
                model.eval()
            _imaging_cache[key] = model
        return _imaging_cache[key]

    def compute_imaging_quality_cached(json_dir, device, submodules_list, **kwargs):
        model = _get_imaging_cached(submodules_list, device)
        video_list, _ = iq.load_dimension_info(json_dir, dimension="imaging_quality", lang="en")
        video_list = iq.distribute_list_to_rank(video_list)
        all_results, video_results = iq.technical_quality(model, video_list, device, **kwargs)
        if iq.get_world_size() > 1:
            video_results = iq.gather_list_of_dict(video_results)
            all_results = sum([d["video_results"] for d in video_results]) / len(video_results)
            all_results = all_results / 100.0
        return all_results, video_results

    def _get_motion_cached(submodules_list, device):
        config = submodules_list["config"]
        ckpt = submodules_list["ckpt"]
        key = (str(config), str(ckpt), _device_key(device))
        if key not in _motion_cache:
            _motion_cache[key] = ms.MotionSmoothness(config, ckpt, device)
        return _motion_cache[key]

    def compute_motion_smoothness_cached(json_dir, device, submodules_list, **kwargs):
        motion = _get_motion_cached(submodules_list, device)
        video_list, _ = ms.load_dimension_info(json_dir, dimension="motion_smoothness", lang="en")
        video_list = ms.distribute_list_to_rank(video_list)
        all_results, video_results = ms.motion_smoothness(motion, video_list)
        if ms.get_world_size() > 1:
            video_results = ms.gather_list_of_dict(video_results)
            all_results = sum([d["video_results"] for d in video_results]) / len(video_results)
        return all_results, video_results

    aq.compute_aesthetic_quality = compute_aesthetic_quality_cached
    iq.compute_imaging_quality = compute_imaging_quality_cached
    ms.compute_motion_smoothness = compute_motion_smoothness_cached

    return {
        "aesthetic_quality": _get_aesthetic_cached,
        "imaging_quality": _get_imaging_cached,
        "motion_smoothness": _get_motion_cached,
    }

def _warm_vbench_dimension_cache(dimensions, device, local, read_frame):
    cache_getters = _install_vbench_dimension_cache()
    from vbench import init_submodules

    submodules_dict = init_submodules(dimensions, local=local, read_frame=read_frame)
    warmed = []
    for dimension in dimensions:
        getter = cache_getters.get(dimension)
        if getter is None:
            continue
        getter(submodules_dict[dimension], device)
        warmed.append(dimension)
    return warmed

try:
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with contextlib.redirect_stdout(sys.stderr):
        from vbench import VBench
        bench = VBench(device, full_json_dir, output_dir)
        warmed_dimensions = _warm_vbench_dimension_cache(dimensions, device, load_local, read_frame)

    print(json.dumps({"status": "ready", "warmed_dimensions": warmed_dimensions}), flush=True)

    for line in sys.stdin:
        if not line:
            break
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "shutdown":
                print(json.dumps({"status": "shutdown"}), flush=True)
                break
            if command != "score":
                raise RuntimeError(f"Unknown command: {command}")

            videos_dir = request["videos_dir"]
            prompt_file = request["prompt_file"]
            eval_name = request["eval_name"]
            result_path = os.path.join(output_dir, f"{eval_name}_eval_results.json")

            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_map = json.load(f)

            with contextlib.redirect_stdout(sys.stderr):
                bench.evaluate(
                    videos_path=videos_dir,
                    name=eval_name,
                    prompt_list=prompt_map,
                    dimension_list=dimensions,
                    local=load_local,
                    read_frame=read_frame,
                    mode="custom_input",
                    imaging_quality_preprocessing_mode=imaging_mode,
                )

            if not os.path.exists(result_path):
                raise RuntimeError(f"VBench finished but did not write {result_path}")
            print(json.dumps({"status": "ok", "result_path": result_path}), flush=True)
        except Exception:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )
except Exception:
    print(
        json.dumps(
            {
                "status": "fatal",
                "traceback": traceback.format_exc(),
            }
        ),
        flush=True,
    )
"""

    def _read_worker_message(self, timeout: Optional[float] = None) -> dict:
        if self._worker_process is None or self._worker_process.stdout is None:
            raise RuntimeError("Persistent VBench worker is not running.")

        import select

        stdout = self._worker_process.stdout
        fd = stdout.fileno()
        if timeout is not None:
            readable, _, _ = select.select([fd], [], [], timeout)
            if not readable:
                raise TimeoutError("Timed out waiting for persistent VBench worker.")

        while True:
            line = stdout.readline()
            if line == "":
                returncode = self._worker_process.poll()
                raise RuntimeError(
                    "Persistent VBench worker exited before returning a result. "
                    f"returncode={returncode}"
                )
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # VBench should be redirected to stderr, but skip any stray stdout.
                continue

    def _start_persistent_worker(self) -> None:
        if self._worker_process is not None and self._worker_process.poll() is None:
            return

        cmd = [
            sys.executable,
            "-c",
            self._persistent_worker_script(),
            self.output_dir,
            self.full_json_dir,
            json.dumps(self.dimensions),
            self._to_bool_arg(self.load_ckpt_from_local),
            self._to_bool_arg(self.read_frame),
            self.imaging_quality_preprocessing_mode,
        ]
        self._worker_log_path = os.path.join(
            self.output_dir,
            f"aesreward_worker_{os.getpid()}_{int(time.time() * 1000)}.log",
        )
        self._worker_log_file = open(self._worker_log_path, "a", encoding="utf-8", buffering=1)
        self._worker_process = subprocess.Popen(
            cmd,
            env=self._make_subprocess_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._worker_log_file,
            text=True,
            bufsize=1,
        )
        message = self._read_worker_message(timeout=self.timeout)
        if message.get("status") != "ready":
            traceback_text = message.get("traceback", "")
            log_path = self._worker_log_path
            self._stop_persistent_worker()
            raise RuntimeError(
                "Persistent VBench worker failed to start.\n"
                f"status={message.get('status')}\n"
                f"{traceback_text}\n"
                f"worker_log={log_path}"
            )

    def _stop_persistent_worker(self) -> None:
        process = self._worker_process
        self._worker_process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=10)
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
        finally:
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass
            try:
                if process.stdout:
                    process.stdout.close()
            except Exception:
                pass
            try:
                if self._worker_log_file:
                    self._worker_log_file.close()
            except Exception:
                pass
            self._worker_log_file = None

    def _run_vbench_persistent(self, videos_dir: str, prompt_file: str, eval_name: str) -> str:
        self._start_persistent_worker()
        if self._worker_process is None or self._worker_process.stdin is None:
            raise RuntimeError("Persistent VBench worker is not writable.")
        request = {
            "cmd": "score",
            "videos_dir": videos_dir,
            "prompt_file": prompt_file,
            "eval_name": eval_name,
        }
        try:
            self._worker_process.stdin.write(json.dumps(request) + "\n")
            self._worker_process.stdin.flush()
        except BrokenPipeError as exc:
            self._stop_persistent_worker()
            raise RuntimeError("Persistent VBench worker pipe broke while sending a request.") from exc

        message = self._read_worker_message(timeout=self.timeout)
        status = message.get("status")
        if status != "ok":
            traceback_text = message.get("traceback", "")
            if status in {"fatal", None}:
                self._stop_persistent_worker()
            raise RuntimeError(
                "Persistent VBench aesthetic reward failed.\n"
                f"status={status}\n"
                f"{traceback_text}\n"
                f"worker_log={self._worker_log_path}"
            )
        result_path = message.get("result_path")
        if not result_path or not os.path.exists(result_path):
            raise RuntimeError(f"Persistent VBench worker returned missing result_path={result_path!r}")
        return result_path

    def _run_vbench(self, videos_dir: str, prompt_file: str, eval_name: str) -> str:
        if self.persistent_worker:
            return self._run_vbench_persistent(videos_dir, prompt_file, eval_name)

        result_path = os.path.join(self.output_dir, f"{eval_name}_eval_results.json")
        script = r"""
import json
import os
import sys

import torch
from vbench import VBench

videos_dir, prompt_file, output_dir, full_json_dir, eval_name = sys.argv[1:6]
dimensions = json.loads(sys.argv[6])
load_local = sys.argv[7].lower() == "true"
read_frame = sys.argv[8].lower() == "true"
imaging_mode = sys.argv[9]

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

with open(prompt_file, "r", encoding="utf-8") as f:
    prompt_map = json.load(f)

bench = VBench(device, full_json_dir, output_dir)
bench.evaluate(
    videos_path=videos_dir,
    name=eval_name,
    prompt_list=prompt_map,
    dimension_list=dimensions,
    local=load_local,
    read_frame=read_frame,
    mode="custom_input",
    imaging_quality_preprocessing_mode=imaging_mode,
)
"""
        cmd = [
            sys.executable,
            "-c",
            script,
            videos_dir,
            prompt_file,
            self.output_dir,
            self.full_json_dir,
            eval_name,
            json.dumps(self.dimensions),
            self._to_bool_arg(self.load_ckpt_from_local),
            self._to_bool_arg(self.read_frame),
            self.imaging_quality_preprocessing_mode,
        ]
        completed = subprocess.run(
            cmd,
            env=self._make_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "VBench aesthetic reward failed.\n"
                f"Command: {' '.join(cmd[:3])} ...\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        if not os.path.exists(result_path):
            raise RuntimeError(
                f"VBench finished but did not write {result_path}.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return result_path

    @staticmethod
    def _extract_scores(result_data: dict, video_paths: List[str], dimensions: List[str]) -> tuple[list[float], dict]:
        path_to_index = {
            os.path.abspath(path): idx
            for idx, path in enumerate(video_paths)
        }
        per_video_scores = [[] for _ in video_paths]
        log_values = {}

        for dimension in dimensions:
            if dimension not in result_data:
                raise RuntimeError(f"VBench result is missing dimension '{dimension}'")
            dimension_result = result_data[dimension]
            if (
                isinstance(dimension_result, (list, tuple))
                and len(dimension_result) >= 2
            ):
                mean_score = float(dimension_result[0])
                video_results = dimension_result[1]
            else:
                raise RuntimeError(
                    f"Unexpected VBench result format for {dimension}: {dimension_result!r}"
                )
            log_values[f"reward_aes_{dimension}"] = mean_score
            for item in video_results:
                video_path = os.path.abspath(str(item.get("video_path", "")))
                score = item.get("video_results", item.get("score", None))
                if score is None:
                    continue
                if video_path in path_to_index:
                    score_value = float(score)
                    # VBench reports imaging_quality mean in [0, 1], but some
                    # versions store per-video scores in [0, 100].
                    if dimension == "imaging_quality" and mean_score <= 1.0 and score_value > 1.0:
                        score_value /= 100.0
                    per_video_scores[path_to_index[video_path]].append(score_value)

        scores = []
        for path, values in zip(video_paths, per_video_scores):
            if not values:
                raise RuntimeError(f"VBench result did not contain a score for {path}")
            scores.append(sum(values) / len(values))
        return scores, log_values

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        videos: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        del metadata
        if videos.ndim != 5:
            raise ValueError(f"Expected videos with shape [B,F,C,H,W], got {tuple(videos.shape)}")
        if len(prompts) != videos.shape[0]:
            raise ValueError(f"Prompt/video batch mismatch: {len(prompts)} vs {videos.shape[0]}")

        if self.use_mock:
            rewards = videos.float().mean(dim=tuple(range(1, videos.ndim)))
            return RewardOutput(
                rewards=rewards,
                log_dict={"reward_aes_mock": rewards.mean()},
            )

        eval_name = f"aesreward_{os.getpid()}_{int(time.time() * 1000)}"
        batch_dir = tempfile.mkdtemp(prefix=f"{eval_name}_videos_", dir=self.output_dir)
        try:
            video_paths = []
            prompt_map = {}
            for idx, (prompt, video) in enumerate(zip(prompts, videos)):
                video_path = os.path.abspath(os.path.join(batch_dir, f"video_{idx:05d}.mp4"))
                self._tensor_to_video(video, video_path, self.fps)
                video_paths.append(video_path)
                prompt_map[video_path] = prompt

            prompt_file = os.path.join(batch_dir, "prompts.json")
            with open(prompt_file, "w", encoding="utf-8") as f:
                json.dump(prompt_map, f, ensure_ascii=False)

            result_path = self._run_vbench(batch_dir, prompt_file, eval_name)
            with open(result_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            scores, log_values = self._extract_scores(result_data, video_paths, self.dimensions)
            if self.keep_videos and self.viewable_videos:
                converted = self._make_saved_video_viewable(video_paths)
                log_values["reward_aes_viewable_video_count"] = float(converted)
        finally:
            if not self.keep_videos:
                shutil.rmtree(batch_dir, ignore_errors=True)

        rewards = torch.tensor(scores, dtype=torch.float32, device=videos.device)
        log_dict = {
            key: torch.tensor(value, dtype=torch.float32, device=videos.device)
            for key, value in log_values.items()
        }
        log_dict["reward_aes_video"] = rewards.mean()
        log_dict["reward_aes_dimension_count"] = torch.tensor(
            len(self.dimensions),
            dtype=torch.float32,
            device=videos.device,
        )
        return RewardOutput(rewards=rewards, log_dict=log_dict)

    def __del__(self):
        try:
            self._stop_interaction_worker()
        except Exception:
            pass
        try:
            self._stop_persistent_worker()
        except Exception:
            pass
        try:
            if getattr(self, "_owns_output_dir", False):
                shutil.rmtree(getattr(self, "output_dir", ""), ignore_errors=True)
        except Exception:
            pass


class LayoutReward(VideoRewardFn):
    """Layout/T2V-CompBench reward over interaction, spatial, and numeracy scores."""

    DEFAULT_WEIGHTS = {
        "physical_interactions": 1.0,
        "spatial_relationship": 1.0,
        "object_count": 1.0,
    }
    WEIGHT_ALIASES = {
        "interaction": "physical_interactions",
        "physical": "physical_interactions",
        "physical_interaction": "physical_interactions",
        "physical_interactions": "physical_interactions",
        "spatial": "spatial_relationship",
        "spatial_relation": "spatial_relationship",
        "spatial_relationship": "spatial_relationship",
        "numeracy": "object_count",
        "object": "object_count",
        "object_count": "object_count",
        "count": "object_count",
    }

    def __init__(
        self,
        device: torch.device,
        weights: object = None,
        output_dir: Optional[str] = None,
        prompt_dir: str = DEFAULT_LAYOUT_PROMPT_DIR,
        interaction_prompt_file: str = "layout_prompts.json",
        spatial_prompt_file: str = "layout_spatial_relations.json",
        numeracy_prompt_file: str = "layout_object_counts.json",
        t2v_model: str = "bidirectional",
        model_path: str = DEFAULT_LAYOUT_MODEL_PATH,
        interaction_script: str = "/root/data/ltl/T2V-CompBench/LLaVA/llava/eval/compbench_eval_interaction.py",
        spatial_script: str = "/root/data/ltl/T2V-CompBench/Grounded-Segment-Anything/compbench_eval_spatial_relationships.py",
        numeracy_script: str = "/root/data/ltl/T2V-CompBench/Grounded-Segment-Anything/GroundingDINO/demo/compbench_eval_numeracy.py",
        llava_env: str = "llava",
        compbench_env: str = "compbench",
        conda_exe: str = "conda",
        python: str = "python",
        llava_python: str | None = None,
        compbench_python: str | None = None,
        compbench_extra_pythonpath: str | None = None,
        compbench_extra_ld_library_path: str | None = None,
        no_conda: bool = True,
        cuda_visible_devices: str | None = None,
        task_cuda_visible_devices: object = None,
        parallel_tasks: bool = False,
        workdir: str | None = None,
        only: Sequence[str] | None = None,
        fps: int = 16,
        timeout: int | None = None,
        use_mock: bool = False,
        keep_videos: bool = False,
        viewable_videos: bool = True,
        number_width: int = 4,
        video_copy_mode: str = "copy",
        direct_numbered_videos: bool = True,
        persistent_worker: bool = False,
        interaction_persistent_worker: bool = False,
        compbench_persistent_worker: bool = False,
        interaction_load_4bit: bool = True,
        interaction_load_8bit: bool = False,
        interaction_device_map: str = "auto",
        interaction_repeats: int = 3,
        interaction_max_new_tokens: int = 512,
        interaction_temperature: float = 0.1,
        interaction_top_p: object = None,
        interaction_num_beams: int = 1,
    ) -> None:
        self.device = device
        self.weights = self._coerce_weights(weights)
        self.reference_prompt_dir = prompt_dir
        self.reference_interaction_prompt_file = interaction_prompt_file
        self.reference_spatial_prompt_file = spatial_prompt_file
        self.reference_numeracy_prompt_file = numeracy_prompt_file
        self.batch_interaction_prompt_file = os.path.basename(interaction_prompt_file) or "layout_prompts.json"
        self.batch_spatial_prompt_file = os.path.basename(spatial_prompt_file) or "layout_spatial_relations.json"
        self.batch_numeracy_prompt_file = os.path.basename(numeracy_prompt_file) or "layout_object_counts.json"
        self.t2v_model = t2v_model
        self.model_path = model_path
        self.interaction_script = interaction_script
        self.spatial_script = spatial_script
        self.numeracy_script = numeracy_script
        self.llava_env = llava_env
        self.compbench_env = compbench_env
        self.conda_exe = conda_exe
        self.python = python
        self.llava_python = llava_python
        self.compbench_python = compbench_python
        self.compbench_extra_pythonpath = compbench_extra_pythonpath
        self.compbench_extra_ld_library_path = compbench_extra_ld_library_path
        self.no_conda = bool(no_conda)
        self.workdir = workdir
        self.only = list(only or LayoutCompBenchEvaluate.DEFAULT_TASKS)
        self.fps = int(fps)
        self.timeout = timeout
        self.use_mock = bool(use_mock)
        self.keep_videos = bool(keep_videos)
        self.viewable_videos = bool(viewable_videos)
        self.number_width = int(number_width)
        self.video_copy_mode = video_copy_mode
        self.direct_numbered_videos = bool(direct_numbered_videos)
        self.persistent_worker = bool(persistent_worker)
        self.interaction_persistent_worker = bool(interaction_persistent_worker)
        self.compbench_persistent_worker = bool(compbench_persistent_worker)
        self.interaction_load_4bit = bool(interaction_load_4bit)
        self.interaction_load_8bit = bool(interaction_load_8bit)
        self.interaction_device_map = str(interaction_device_map or "auto")
        self.interaction_repeats = max(1, int(interaction_repeats))
        self.interaction_max_new_tokens = int(interaction_max_new_tokens)
        self.interaction_temperature = float(interaction_temperature)
        self.interaction_top_p = interaction_top_p
        self.interaction_num_beams = int(interaction_num_beams)
        self.task_cuda_visible_devices = self._coerce_task_cuda_visible_devices(task_cuda_visible_devices)
        self.parallel_tasks = bool(parallel_tasks)
        self._reference_spatial_by_prompt: dict[str, dict] | None = None
        self._reference_numeracy_by_prompt: dict[str, dict] | None = None
        self._worker_process = None
        self._worker_log_file = None
        self._worker_log_path = None
        self._interaction_worker_process = None
        self._interaction_worker_log_file = None
        self._interaction_worker_log_path = None
        self._compbench_worker_process = None
        self._compbench_worker_log_file = None
        self._compbench_worker_log_path = None
        self._owns_output_dir = not output_dir
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="layoutreward_compbench_")
        os.makedirs(self.output_dir, exist_ok=True)

        if cuda_visible_devices is not None:
            self.cuda_visible_devices = str(cuda_visible_devices)
        elif os.environ.get("CUDA_VISIBLE_DEVICES"):
            self.cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        elif device.type == "cuda":
            self.cuda_visible_devices = str(device.index or 0)
        else:
            self.cuda_visible_devices = None

        if self.persistent_worker and not self.use_mock:
            try:
                self._start_persistent_worker()
            except Exception:
                self._stop_persistent_worker()
                raise

    @classmethod
    def _coerce_weights(cls, weights: object) -> dict[str, float]:
        if weights is None:
            parsed = dict(cls.DEFAULT_WEIGHTS)
        elif isinstance(weights, str):
            stripped = weights.strip()
            if not stripped:
                parsed = dict(cls.DEFAULT_WEIGHTS)
            elif stripped[0] in "[{":
                parsed = cls._coerce_weights(json.loads(stripped))
            elif "=" in stripped or ":" in stripped:
                parsed = {}
                for item in stripped.split(","):
                    if not item.strip():
                        continue
                    separator = "=" if "=" in item else ":"
                    key, value = item.split(separator, 1)
                    parsed[key.strip()] = float(value.strip())
            else:
                values = [float(item.strip()) for item in stripped.split(",") if item.strip()]
                if len(values) != len(LAYOUT_DIMENSION_NAMES):
                    raise ValueError(
                        "layoutReward weights string must have three comma-separated values "
                        "or key=value entries."
                    )
                parsed = dict(zip(LAYOUT_DIMENSION_NAMES, values))
        elif isinstance(weights, dict):
            parsed = {}
            for key, value in weights.items():
                parsed[str(key)] = float(value)
        elif isinstance(weights, (list, tuple)):
            values = [float(item) for item in weights]
            if len(values) != len(LAYOUT_DIMENSION_NAMES):
                raise ValueError("layoutReward weights sequence must contain exactly three values.")
            parsed = dict(zip(LAYOUT_DIMENSION_NAMES, values))
        else:
            raise TypeError(f"Unsupported layoutReward weights type: {type(weights).__name__}")

        normalized = {}
        for key, value in parsed.items():
            normalized_key = cls.WEIGHT_ALIASES.get(str(key).strip(), str(key).strip())
            if normalized_key not in LAYOUT_DIMENSION_NAMES:
                raise ValueError(f"Unknown layoutReward weight dimension: {key}")
            normalized[normalized_key] = float(value)

        for name, value in cls.DEFAULT_WEIGHTS.items():
            normalized.setdefault(name, float(value))

        total = sum(normalized[name] for name in LAYOUT_DIMENSION_NAMES)
        if total <= 0:
            raise ValueError("layoutReward weights must sum to a positive value.")
        return normalized

    def _coerce_task_cuda_visible_devices(self, value: object) -> dict[str, str]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {str(key).strip(): str(item).strip() for key, item in value.items() if str(item).strip()}
        if isinstance(value, (list, tuple)):
            tasks = list(LayoutCompBenchEvaluate.DEFAULT_TASKS)
            return {
                task: str(item).strip()
                for task, item in zip(tasks, value)
                if str(item).strip()
            }
        text = str(value).strip()
        if not text:
            return {}
        if "=" in text or ":" in text:
            parsed = {}
            for item in text.split(","):
                item = item.strip()
                if not item:
                    continue
                separator = "=" if "=" in item else ":"
                key, gpu = item.split(separator, 1)
                parsed[key.strip()] = gpu.strip()
            return parsed
        values = [item.strip() for item in text.split(",") if item.strip()]
        if len(values) == 1:
            return {task: values[0] for task in LayoutCompBenchEvaluate.DEFAULT_TASKS}
        return {
            task: gpu
            for task, gpu in zip(LayoutCompBenchEvaluate.DEFAULT_TASKS, values)
            if gpu
        }

    @staticmethod
    def _indexed_value(value: object, index: int) -> object:
        if isinstance(value, (list, tuple)) and not isinstance(value, str):
            if index >= len(value):
                return None
            return value[index]
        return value

    @staticmethod
    def _metadata_sample(metadata: object, index: int) -> dict | None:
        if isinstance(metadata, (list, tuple)):
            item = LayoutReward._indexed_value(metadata, index)
            return item if isinstance(item, dict) else None
        if not isinstance(metadata, dict):
            return None
        for key in ("samples", "records", "items", "metadata"):
            item = LayoutReward._indexed_value(metadata.get(key), index)
            if isinstance(item, dict):
                return item
        return None

    @staticmethod
    def _record_value(record: dict | None, key: str) -> str | None:
        if not record:
            return None
        value = record.get(key)
        if value is None:
            return None
        return str(value)

    @classmethod
    def _metadata_record(cls, metadata: object, index: int, keys: Sequence[str]) -> dict | None:
        sources: list[dict] = []
        sample = cls._metadata_sample(metadata, index)
        if sample is not None:
            sources.append(sample)
        if isinstance(metadata, dict):
            sources.append(metadata)

        for source in sources:
            for key in keys:
                value = cls._indexed_value(source.get(key), index)
                if isinstance(value, dict):
                    return value
        return None

    @classmethod
    def _metadata_value(cls, metadata: object, index: int, keys: Sequence[str]) -> str | None:
        sources: list[dict] = []
        sample = cls._metadata_sample(metadata, index)
        if sample is not None:
            sources.append(sample)
        if isinstance(metadata, dict):
            sources.append(metadata)

        for source in sources:
            for key in keys:
                if key not in source:
                    continue
                value = cls._indexed_value(source.get(key), index)
                if value is None or isinstance(value, dict):
                    continue
                return str(value)
        return None

    @staticmethod
    def _first_text(*values: object) -> str:
        for value in values:
            if value is None:
                continue
            text = str(value)
            if text:
                return text
        return ""

    @staticmethod
    def _read_prompt_record_map(path: str) -> dict[str, dict]:
        prompt_map: dict[str, dict] = {}
        file_path = Path(path)
        if not file_path.exists():
            return prompt_map
        data = json.loads(file_path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            for key in ("prompts", "data", "records"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            return prompt_map
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("prompt"), str):
                prompt_map.setdefault(item["prompt"], item)
        return prompt_map

    def _load_reference_prompt_records(self) -> None:
        if self._reference_spatial_by_prompt is not None and self._reference_numeracy_by_prompt is not None:
            return
        spatial_path = _layout_child_path(self.reference_prompt_dir, self.reference_spatial_prompt_file)
        numeracy_path = _layout_child_path(self.reference_prompt_dir, self.reference_numeracy_prompt_file)
        self._reference_spatial_by_prompt = self._read_prompt_record_map(spatial_path)
        self._reference_numeracy_by_prompt = self._read_prompt_record_map(numeracy_path)

    def _build_prompt_records(
        self,
        prompts: List[str],
        metadata: object,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
        self._load_reference_prompt_records()
        spatial_by_prompt = self._reference_spatial_by_prompt or {}
        numeracy_by_prompt = self._reference_numeracy_by_prompt or {}

        interaction_records: list[dict[str, str]] = []
        spatial_records: list[dict[str, str]] = []
        numeracy_records: list[dict[str, str]] = []

        for idx, prompt in enumerate(prompts):
            metadata_spatial = self._metadata_record(
                metadata,
                idx,
                (
                    "layout_spatial_record",
                    "layout_spatial_records",
                    "spatial_record",
                    "spatial_records",
                    "spatial_relation_record",
                    "spatial_relation_records",
                ),
            )
            metadata_numeracy = self._metadata_record(
                metadata,
                idx,
                (
                    "layout_object_count_record",
                    "layout_object_count_records",
                    "object_count_record",
                    "object_count_records",
                    "numeracy_record",
                    "numeracy_records",
                ),
            )
            reward_prompt = self._first_text(
                self._metadata_value(metadata, idx, ("layout_prompt", "layout_prompts", "reward_prompt", "reward_prompts")),
                self._record_value(metadata_spatial, "prompt"),
                self._record_value(metadata_numeracy, "prompt"),
                prompt,
            )
            reference_spatial = spatial_by_prompt.get(reward_prompt) or spatial_by_prompt.get(prompt, {})
            reference_numeracy = numeracy_by_prompt.get(reward_prompt) or numeracy_by_prompt.get(prompt, {})

            spatial_record = {
                "prompt": reward_prompt,
                "spatial": self._first_text(
                        self._record_value(metadata_spatial, "spatial"),
                        self._metadata_value(metadata, idx, ("layout_spatial", "spatial")),
                        self._record_value(reference_spatial, "spatial"),
                ),
                "object_1": self._first_text(
                        self._record_value(metadata_spatial, "object_1"),
                        self._metadata_value(metadata, idx, ("layout_object_1", "object_1")),
                        self._record_value(reference_spatial, "object_1"),
                ),
                "object_2": self._first_text(
                        self._record_value(metadata_spatial, "object_2"),
                        self._metadata_value(metadata, idx, ("layout_object_2", "object_2")),
                        self._record_value(reference_spatial, "object_2"),
                ),
            }
            numeracy_record = {
                "prompt": reward_prompt,
                "objects": self._first_text(
                        self._record_value(metadata_numeracy, "objects"),
                        self._metadata_value(metadata, idx, ("layout_objects", "objects")),
                        self._record_value(reference_numeracy, "objects"),
                ),
                "numbers": self._first_text(
                        self._record_value(metadata_numeracy, "numbers"),
                        self._metadata_value(metadata, idx, ("layout_numbers", "numbers")),
                        self._record_value(reference_numeracy, "numbers"),
                ),
            }
            active_tasks = {str(task).strip().lower() for task in self.only}
            if "spatial" in active_tasks:
                missing = [key for key in ("spatial", "object_1", "object_2") if not spatial_record[key]]
                if missing:
                    raise ValueError(
                        f"Layout reward spatial metadata is missing {missing} for sample {idx}: {reward_prompt!r}"
                    )
            if "numeracy" in active_tasks:
                missing = [key for key in ("objects", "numbers") if not numeracy_record[key]]
                if missing:
                    raise ValueError(
                        f"Layout reward numeracy metadata is missing {missing} for sample {idx}: {reward_prompt!r}"
                    )

            interaction_records.append({"prompt": reward_prompt})
            spatial_records.append(spatial_record)
            numeracy_records.append(numeracy_record)

        return interaction_records, spatial_records, numeracy_records

    def _write_batch_prompt_files(
        self,
        prompt_dir: str,
        prompts: List[str],
        metadata: object,
    ) -> None:
        os.makedirs(prompt_dir, exist_ok=True)
        interaction_records, spatial_records, numeracy_records = self._build_prompt_records(prompts, metadata)
        files = {
            self.batch_interaction_prompt_file: interaction_records,
            self.batch_spatial_prompt_file: spatial_records,
            self.batch_numeracy_prompt_file: numeracy_records,
        }
        for file_name, records in files.items():
            with open(os.path.join(prompt_dir, file_name), "w", encoding="utf-8") as file_obj:
                json.dump(records, file_obj, ensure_ascii=False, indent=2)

    def _interaction_cuda_visible_devices(self) -> str | None:
        return self.task_cuda_visible_devices.get("interaction", self.cuda_visible_devices)

    def _interaction_worker_config(self) -> dict:
        return {
            "model_path": self.model_path,
            "cuda_visible_devices": self._interaction_cuda_visible_devices(),
            "load_4bit": self.interaction_load_4bit,
            "load_8bit": self.interaction_load_8bit,
            "device_map": self.interaction_device_map,
            "repeats": self.interaction_repeats,
            "max_new_tokens": self.interaction_max_new_tokens,
            "temperature": self.interaction_temperature,
            "top_p": self.interaction_top_p,
            "num_beams": self.interaction_num_beams,
        }

    @staticmethod
    def _prepend_env_path(env: dict, key: str, value: str | None) -> None:
        if not value:
            return
        existing = env.get(key, "")
        env[key] = f"{value}:{existing}" if existing else value

    def _interaction_worker_script(self) -> str:
        return r"""
import csv
import json
import os
import sys
import traceback

_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def _send(payload):
    print(json.dumps(payload), file=_REAL_STDOUT, flush=True)


try:
    config = json.loads(sys.argv[1])
    if config.get("cuda_visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["cuda_visible_devices"])

    import numpy as np
    import torch

    sys.path.insert(0, "/root/data/ltl/T2V-CompBench/LLaVA")

    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.eval.compbench_eval_interaction import (
        Video_preprocess,
        extract_json,
        load_images,
        model_score,
        set_seed,
    )

    def _patch_clip_vision_tower():
        from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
        import llava.model.multimodal_encoder.clip_encoder as clip_encoder

        def _init(self, vision_tower, args, delay_load=False):
            torch.nn.Module.__init__(self)
            self.is_loaded = False
            self.vision_tower_name = vision_tower
            self.select_layer = args.mm_vision_select_layer
            self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
            if not delay_load:
                self.load_model()
            else:
                self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

        def _load_model(self, device_map=None):
            del device_map
            if self.is_loaded:
                print(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
                return
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=False,
            )
            if torch.cuda.is_available():
                self.vision_tower.to(device="cuda", dtype=torch.float16)
            self.vision_tower.requires_grad_(False)
            self.is_loaded = True

        def _load_model_s2(self, device_map=None):
            _load_model(self, device_map=device_map)
            self.image_processor.size["shortest_edge"] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

        clip_encoder.CLIPVisionTower.__init__ = _init
        clip_encoder.CLIPVisionTower.load_model = _load_model
        clip_encoder.CLIPVisionTowerS2.load_model = _load_model_s2

    _patch_clip_vision_tower()
    disable_torch_init()
    model_name = get_model_name_from_path(config["model_path"])
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        config["model_path"],
        None,
        model_name,
        load_8bit=bool(config.get("load_8bit", False)),
        load_4bit=bool(config.get("load_4bit", True)),
        device_map=str(config.get("device_map") or "auto"),
    )
    del context_len
    model.eval()

    def _model_device():
        device = getattr(model, "device", None)
        if device is None or str(device) == "meta":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device

    def _generate(input_ids, images_tensor, image_sizes):
        with torch.inference_mode():
            return model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if float(config.get("temperature", 0.1)) > 0 else False,
                temperature=float(config.get("temperature", 0.1)),
                top_p=config.get("top_p", None),
                num_beams=int(config.get("num_beams", 1)),
                max_new_tokens=int(config.get("max_new_tokens", 512)),
                use_cache=True,
            )

    def _score_interaction(video_path, output_path, prompt_file, t2v_model):
        image_grid_path = Video_preprocess(output_root=output_path).convert_video_to_grid(video_path)
        with open(prompt_file, "r", encoding="utf-8") as json_data:
            prompts = json.load(json_data)

        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(output_path, f"{t2v_model}_object_interactions_score.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="", encoding="utf-8-sig") as csvreader:
                line_count = len(list(csv.reader(csvreader)))
        else:
            line_count = 0

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            if line_count == 0:
                csv_writer.writerow([
                    "name",
                    "prompt",
                    "seed0_answer1",
                    "seed0_answer2",
                    "seed0_answer3",
                    "seed0_score",
                    "seed1_answer1",
                    "seed1_answer2",
                    "seed1_answer3",
                    "seed1_score",
                    "seed2_answer1",
                    "seed2_answer2",
                    "seed2_answer3",
                    "seed2_score",
                    "seed_score",
                    "Score",
                ])

            grid_images = [name for name in os.listdir(image_grid_path) if name and name[0].isdigit()]
            grid_images = sorted(grid_images)
            print(len(grid_images), flush=True)
            evaluated = max(line_count - 1, 0)
            repeats = max(1, int(config.get("repeats", 3)))

            for i in range(evaluated, len(grid_images)):
                grid_image_name = grid_images[i]
                num = int(grid_image_name[0:4]) - 1
                this_prompt = prompts[num]["prompt"]

                images = load_images([os.path.join(image_grid_path, grid_images[i])])
                image_sizes = [image.size for image in images]
                images_tensor = process_images(images, image_processor, model.config).to(
                    _model_device(),
                    dtype=torch.float16,
                )

                q1 = (
                    "The provided image arranges 6 key frames from an AI generated video in a 3-row by "
                    "2-column grid layout.  Describe the video, focusing on the interactions between "
                    "the characters or objects that appear throughout the frames."
                )
                qs1 = DEFAULT_IMAGE_TOKEN + "\n" + q1
                conv_mode = "chatml_direct"

                conv_init = conv_templates[conv_mode].copy()
                conv_init.append_message(conv_init.roles[0], qs1)
                conv_init.append_message(conv_init.roles[1], None)
                prompt_init = conv_init.get_prompt()
                input_ids_init = tokenizer_image_token(
                    prompt_init,
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).to(_model_device())

                outputs_1 = []
                outputs_2 = []
                outputs_3 = []
                scores_tmp = []

                for iteration in range(repeats):
                    set_seed(int(config.get("seed", 0)) + iteration)

                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], qs1)
                    conv.append_message(conv.roles[1], None)

                    output_ids = _generate(input_ids_init, images_tensor, image_sizes)
                    output_1 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    outputs_1.append(output_1)
                    conv.messages[-1][-1] = output_1

                    q2 = f"To evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please carefully answer the following question.\n \
Question: \n \
A: All the objects involved in the interaction are clearly portrayed in the video. \n \
B: Some objects involved in the interaction are not depicted very clear. \n \
C: Some objects involved in the interaction are missing. \n \
D: None of the objects involved in the interaction are present. \n \
Select the most suitable option according to the video and your previous description. \
Provide your answer in JSON format with the following keys: option (e.g., B), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., A)."

                    conv.append_message(conv.roles[0], q2)
                    conv.append_message(conv.roles[1], None)
                    input_ids = tokenizer_image_token(
                        conv.get_prompt(),
                        tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    ).unsqueeze(0).to(_model_device())

                    output_ids = _generate(input_ids, images_tensor, image_sizes)
                    output_2 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    outputs_2.append(output_2)
                    conv.messages[-1][-1] = output_2

                    json_obj_2 = extract_json(output_2) or {}
                    option_value_2 = json_obj_2.get("adjust", "bad reply 1")
                    print("option_value_2 ", option_value_2, flush=True)

                    q3_a = f"To evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please carefully examine the interaction and select the most suitable option.\n \
A: The interaction process is clearly shown, with the objects involved engaging in a dynamic manner. The outcome logically follow from the preceding actions and aligns accurately with what the prompt indicated, if mentioned. \n \
B: The interaction process is mostly clear, with objects engaging actively. The outcome generally aligns with what the prompt indicated, if mentioned. \n \
C: The interaction process is somewhat clear, but the objects show limited engagement. The development of process might be unclear, and while the outcome aligns with the prompt but with no previous actions. \n \
D: The interaction process is unclear, with minimal engagement from the objects. There is little to no development of the process. The outcome, if mentioned in prompt, is vague. \n \
E: The interaction process is virtually nonexistent, with no visible engagement from the objects. The outcome in the prompt, if mentioned, is absent or completely irrelevant.\n \
Provide your answer in a JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."
                    q3_b = f"Following the previous question, to evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please select the most suitable option.\n \
A: the interaction of the clearly presented object(s) is highly dynamic and compensates effectively for the unclear ones, with a logical outcome that implies the interaction. \n \
B: the interaction of the clearly presented object(s) is weak, with confusing relationship to the unclear ones and an irrelevant outcome to the prompt.  \n \
Provide your answer in a JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."
                    q3_c = f"Following the previous question, to evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please select the most suitable option.\n \
A: the interaction of the clearly presented object(s) is highly dynamic and compensates effectively for the missing ones, with a logical outcome that implies the interaction. \n \
B: the interaction of the clearly presented object(s) is weak, with confusing relationship to the missing ones and an irrelevant outcome to the prompt.  \n \
Provide your answer in a JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."

                    if option_value_2 == "A":
                        q3 = q3_a
                        ask_q3 = True
                    elif option_value_2 == "B":
                        q3 = q3_b
                        ask_q3 = True
                    elif option_value_2 == "C":
                        q3 = q3_c
                        ask_q3 = True
                    elif option_value_2 == "D":
                        score_tmp = 1
                        ask_q3 = False
                    else:
                        score_tmp = "bad reply"
                        ask_q3 = False

                    output_3 = ""
                    if ask_q3:
                        conv.append_message(conv.roles[0], q3)
                        conv.append_message(conv.roles[1], None)
                        input_ids = tokenizer_image_token(
                            conv.get_prompt(),
                            tokenizer,
                            IMAGE_TOKEN_INDEX,
                            return_tensors="pt",
                        ).unsqueeze(0).to(_model_device())

                        output_ids = _generate(input_ids, images_tensor, image_sizes)
                        output_3 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                        json_obj_3 = extract_json(output_3) or {}
                        option_value_3 = json_obj_3.get("adjust", "bad reply 2")
                        print("option_value_3 ", option_value_3, flush=True)

                        if option_value_2 == "A":
                            score_tmp = {"A": 10, "B": 9, "C": 8, "D": 7, "E": 6}.get(option_value_3, "bad reply ?")
                        elif option_value_2 == "B":
                            score_tmp = {"A": 5, "B": 3}.get(option_value_3, "bad reply ?")
                        elif option_value_2 == "C":
                            score_tmp = {"A": 4, "B": 2}.get(option_value_3, "bad reply ?")

                    outputs_3.append(output_3)
                    scores_tmp.append(score_tmp)
                    print("score for", grid_images[i], score_tmp, flush=True)

                if all(isinstance(score, int) for score in scores_tmp):
                    score_avg = sum(scores_tmp) / len(scores_tmp)
                else:
                    score_avg = "bad reply"

                def _item(values, index):
                    return values[index] if index < len(values) else ""

                csv_writer.writerow([
                    grid_image_name,
                    this_prompt,
                    _item(outputs_1, 0),
                    _item(outputs_2, 0),
                    _item(outputs_3, 0),
                    _item(scores_tmp, 0),
                    _item(outputs_1, 1),
                    _item(outputs_2, 1),
                    _item(outputs_3, 1),
                    _item(scores_tmp, 1),
                    _item(outputs_1, 2),
                    _item(outputs_2, 2),
                    _item(outputs_3, 2),
                    _item(scores_tmp, 2),
                    scores_tmp,
                    score_avg,
                ])
                csvfile.flush()

        model_score(csv_path)
        return csv_path

    _send({"status": "ready"})

    for line in sys.stdin:
        if not line:
            break
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "shutdown":
                _send({"status": "shutdown"})
                break
            if command != "score":
                raise RuntimeError(f"Unknown command: {command}")
            csv_path = _score_interaction(
                request["video_path"],
                request["output_path"],
                request["prompt_file"],
                request["t2v_model"],
            )
            _send({"status": "ok", "csv_path": csv_path})
        except Exception:
            _send({"status": "error", "traceback": traceback.format_exc()})
except Exception:
    _send({"status": "fatal", "traceback": traceback.format_exc()})
"""

    def _read_interaction_worker_message(self, timeout: Optional[float] = None) -> dict:
        if self._interaction_worker_process is None or self._interaction_worker_process.stdout is None:
            raise RuntimeError("Persistent LLaVA interaction worker is not running.")

        import select

        stdout = self._interaction_worker_process.stdout
        fd = stdout.fileno()
        if timeout is not None:
            readable, _, _ = select.select([fd], [], [], timeout)
            if not readable:
                raise TimeoutError("Timed out waiting for persistent LLaVA interaction worker.")

        while True:
            line = stdout.readline()
            if line == "":
                returncode = self._interaction_worker_process.poll()
                raise RuntimeError(
                    "Persistent LLaVA interaction worker exited before returning a result. "
                    f"returncode={returncode}"
                )
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def _start_interaction_worker(self) -> None:
        if self._interaction_worker_process is not None and self._interaction_worker_process.poll() is None:
            return

        python_bin = self.llava_python or self.python
        cmd = [
            python_bin,
            "-c",
            self._interaction_worker_script(),
            json.dumps(self._interaction_worker_config()),
        ]
        if not self.llava_python and not self.no_conda:
            cmd = [self.conda_exe, "run", "--no-capture-output", "-n", self.llava_env, *cmd]

        self._interaction_worker_log_path = os.path.join(
            self.output_dir,
            f"layoutreward_llava_interaction_worker_{os.getpid()}_{int(time.time() * 1000)}.log",
        )
        self._interaction_worker_log_file = open(
            self._interaction_worker_log_path,
            "a",
            encoding="utf-8",
            buffering=1,
        )
        env = os.environ.copy()
        cuda_visible_devices = self._interaction_cuda_visible_devices()
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        llava_root = "/root/data/ltl/T2V-CompBench/LLaVA"
        env["PYTHONPATH"] = f"{llava_root}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else llava_root
        self._interaction_worker_process = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._interaction_worker_log_file,
            text=True,
            bufsize=1,
        )
        message = self._read_interaction_worker_message(timeout=self.timeout)
        if message.get("status") != "ready":
            traceback_text = message.get("traceback", "")
            log_path = self._interaction_worker_log_path
            self._stop_interaction_worker()
            raise RuntimeError(
                "Persistent LLaVA interaction worker failed to start.\n"
                f"status={message.get('status')}\n"
                f"{traceback_text}\n"
                f"worker_log={log_path}"
            )

    def _stop_interaction_worker(self) -> None:
        process = self._interaction_worker_process
        self._interaction_worker_process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=10)
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
        finally:
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass
            try:
                if process.stdout:
                    process.stdout.close()
            except Exception:
                pass
            try:
                if self._interaction_worker_log_file:
                    self._interaction_worker_log_file.close()
            except Exception:
                pass
            self._interaction_worker_log_file = None

    def _run_interaction_persistent(
        self,
        video_path: str,
        output_path: str,
        prompt_file: str,
    ) -> str:
        self._start_interaction_worker()
        if self._interaction_worker_process is None or self._interaction_worker_process.stdin is None:
            raise RuntimeError("Persistent LLaVA interaction worker is not writable.")

        request = {
            "cmd": "score",
            "video_path": video_path,
            "output_path": output_path,
            "prompt_file": prompt_file,
            "t2v_model": self.t2v_model,
        }
        try:
            self._interaction_worker_process.stdin.write(json.dumps(request) + "\n")
            self._interaction_worker_process.stdin.flush()
        except BrokenPipeError as exc:
            self._stop_interaction_worker()
            raise RuntimeError("Persistent LLaVA interaction worker pipe broke while sending a request.") from exc

        message = self._read_interaction_worker_message(timeout=self.timeout)
        status = message.get("status")
        if status != "ok":
            traceback_text = message.get("traceback", "")
            if status in {"fatal", None}:
                self._stop_interaction_worker()
            raise RuntimeError(
                "Persistent LLaVA interaction worker failed.\n"
                f"status={status}\n"
                f"{traceback_text}\n"
                f"worker_log={self._interaction_worker_log_path}"
            )
        csv_path = message.get("csv_path")
        if not csv_path or not os.path.exists(csv_path):
            raise RuntimeError(f"Persistent LLaVA interaction worker returned missing csv_path={csv_path!r}")
        return str(csv_path)

    def _compbench_cuda_visible_devices(self) -> str | None:
        active = {str(task).strip().lower() for task in self.only} & {"spatial", "numeracy"}
        values: list[str] = []
        for task_name in ("spatial", "numeracy"):
            if task_name not in active:
                continue
            value = self.task_cuda_visible_devices.get(task_name, self.cuda_visible_devices)
            if value is not None:
                values.append(str(value))

        unique_values: list[str] = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        if len(unique_values) > 1:
            raise ValueError(
                "layoutreward_compbench_persistent_worker requires spatial and numeracy "
                f"to use the same CUDA_VISIBLE_DEVICES, got {unique_values}."
            )
        if unique_values:
            return unique_values[0]
        return self.cuda_visible_devices

    def _compbench_worker_config(self) -> dict:
        return {
            "spatial_script": self.spatial_script,
            "numeracy_script": self.numeracy_script,
            "workdir": self.workdir,
            "cuda_visible_devices": self._compbench_cuda_visible_devices(),
        }

    def _compbench_worker_script(self) -> str:
        return r"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import traceback

_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def _send(payload):
    print(json.dumps(payload), file=_REAL_STDOUT, flush=True)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepend_sys_path(path):
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _args_after_script(command, script_path):
    command = [str(item) for item in command]
    normalized_script = os.path.normpath(script_path)
    for index, token in enumerate(command):
        if os.path.normpath(token) == normalized_script:
            return command[index + 1 :]
    for index, token in enumerate(command):
        if token.startswith("--"):
            return command[index:]
    return []


def _option(argv, *names, default=None, cast=str):
    for index, token in enumerate(argv):
        if token not in names:
            continue
        if index + 1 >= len(argv):
            return default
        value = argv[index + 1]
        if value is None:
            return default
        if cast is None:
            return value
        return cast(value)
    return default


def _bool_flag(argv, *names):
    return any(token in names for token in argv)


def _spatial_args(command, script_path):
    argv = _args_after_script(command, script_path)
    return argparse.Namespace(
        config=_option(
            argv,
            "--config",
            default="/root/data/ltl/T2V-CompBench/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ),
        grounded_checkpoint=_option(
            argv,
            "--grounded_checkpoint",
            default="/root/data/ltl/T2V-CompBench/Grounded-Segment-Anything/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        ),
        box_threshold=_option(argv, "--box_threshold", default=0.35, cast=float),
        text_threshold=_option(argv, "--text_threshold", default=0.25, cast=float),
        device=_option(argv, "--device", default="cuda"),
        iou_threshold_2d=_option(argv, "--iou_threshold_2d", default=0.9, cast=float),
        sam_version=_option(argv, "--sam_version", default="vit_h"),
        sam_checkpoint=_option(
            argv,
            "--sam_checkpoint",
            default="/root/data/ltl/T2V-CompBench/Grounded-Segment-Anything/sam_vit_h_4b8939.pth",
        ),
        sam_hq_checkpoint=_option(argv, "--sam_hq_checkpoint", default=None),
        use_sam_hq=_bool_flag(argv, "--use_sam_hq"),
        depth_folder=_option(argv, "--depth_folder", default=None),
        iou_threshold_3d=_option(argv, "--iou_threshold_3d", default=0.95, cast=float),
        output_path=_option(argv, "--output-path", default="csv_spatial"),
        output_root=_option(argv, "--output-root", default=None),
        read_prompt_file=_option(argv, "--read-prompt-file", default="meta_data/spatial_relationships.json"),
        video_path=_option(argv, "--video-path", default=None),
        frame_folder=_option(argv, "--frame_folder", "--frame-folder", default=None),
        t2v_model=_option(argv, "--t2v-model", default=None),
        output_dir=_option(argv, "--output_dir", default="output_spatial/"),
    )


def _numeracy_args(command, script_path):
    argv = _args_after_script(command, script_path)
    return argparse.Namespace(
        config_file=_option(
            argv,
            "--config_file",
            "-c",
            default="/root/data/ltl/T2V-CompBench/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ),
        checkpoint_path=_option(
            argv,
            "--checkpoint_path",
            "-p",
            default="/root/data/ltl/T2V-CompBench/Grounded-Segment-Anything/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        ),
        output_dir=_option(argv, "--output_dir", "-o", default="output_numeracy"),
        output_path=_option(argv, "--output-path", default="csv_numeracy"),
        iou_threshold=_option(argv, "--iou_threshold", default=0.9, cast=float),
        box_threshold=_option(argv, "--box_threshold", default=0.4, cast=float),
        text_threshold=_option(argv, "--text_threshold", default=0.25, cast=float),
        token_spans=_option(argv, "--token_spans", default=None),
        cpu_only=_bool_flag(argv, "--cpu-only"),
        read_prompt_file=_option(argv, "--read-prompt-file", default="meta_data/generative_numeracy.json"),
        video_path=_option(argv, "--video-path", default=None),
        frame_folder=_option(argv, "--frame-folder", "--frame_folder", default=None),
        t2v_model=_option(argv, "--t2v-model", default=None),
    )


try:
    config = json.loads(sys.argv[1])
    if config.get("cuda_visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["cuda_visible_devices"])
    if config.get("workdir"):
        os.chdir(str(config["workdir"]))

    spatial_script = str(config["spatial_script"])
    numeracy_script = str(config["numeracy_script"])
    gsa_root = os.path.dirname(spatial_script)
    groundingdino_root = os.path.join(gsa_root, "GroundingDINO")
    depth_root = os.path.join(gsa_root, "Depth-Anything")
    segment_root = os.path.join(gsa_root, "segment_anything")
    for path in (
        depth_root,
        groundingdino_root,
        segment_root,
        gsa_root,
        os.path.dirname(numeracy_script),
    ):
        _prepend_sys_path(path)

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    spatial_mod = _load_module("layout_compbench_spatial", spatial_script)
    numeracy_mod = _load_module("layout_compbench_numeracy", numeracy_script)
    depth_mod = sys.modules.get("compbench_run_depth")
    if depth_mod is None:
        depth_mod = _load_module("layout_compbench_depth", os.path.join(depth_root, "compbench_run_depth.py"))

    spatial_model_cache = {}
    numeracy_model_cache = {}
    sam_model_cache = {}
    depth_cache = {}

    original_spatial_load_model = spatial_mod.load_model

    def cached_spatial_load_model(model_config_path, model_checkpoint_path, device):
        key = (os.path.abspath(model_config_path), os.path.abspath(model_checkpoint_path), str(device))
        if key not in spatial_model_cache:
            print(f"[compbench worker] loading spatial GroundingDINO: {key}", flush=True)
            spatial_model_cache[key] = original_spatial_load_model(
                model_config_path,
                model_checkpoint_path,
                device,
            )
        return spatial_model_cache[key]

    spatial_mod.load_model = cached_spatial_load_model

    original_numeracy_load_model = numeracy_mod.load_model

    def cached_numeracy_load_model(model_config_path, model_checkpoint_path, cpu_only=False):
        key = (os.path.abspath(model_config_path), os.path.abspath(model_checkpoint_path), bool(cpu_only))
        if key not in numeracy_model_cache:
            print(f"[compbench worker] loading numeracy GroundingDINO: {key}", flush=True)
            numeracy_model_cache[key] = original_numeracy_load_model(
                model_config_path,
                model_checkpoint_path,
                cpu_only=cpu_only,
            )
        return numeracy_model_cache[key]

    numeracy_mod.load_model = cached_numeracy_load_model

    def _patch_sam_registry(registry, label):
        for name, constructor in list(registry.items()):
            def cached_constructor(*args, _name=name, _constructor=constructor, **kwargs):
                checkpoint = kwargs.get("checkpoint")
                if checkpoint is None and args:
                    checkpoint = args[0]
                key = (label, _name, str(checkpoint))
                if key not in sam_model_cache:
                    print(f"[compbench worker] loading {label} SAM {_name}: {checkpoint}", flush=True)
                    sam_model_cache[key] = _constructor(*args, **kwargs)
                return sam_model_cache[key]

            registry[name] = cached_constructor

    _patch_sam_registry(spatial_mod.sam_model_registry, "sam")
    _patch_sam_registry(spatial_mod.sam_hq_model_registry, "sam_hq")

    def _depth_components(encoder):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        key = (str(encoder), device)
        if key in depth_cache:
            return depth_cache[key]

        print(f"[compbench worker] loading Depth-Anything encoder={encoder} device={device}", flush=True)
        try:
            depth_anything = depth_mod.DepthAnything.from_pretrained(
                "LiheYoung/depth_anything_{}14".format(encoder),
                encoder=encoder,
            ).to(device).eval()
        except Exception as exc:
            print(f"Failed to load from HuggingFace Hub: {exc}", flush=True)
            print("Attempting to load with local hub...", flush=True)
            depth_anything = depth_mod.DepthAnything(
                config={"encoder": encoder, "localhub": True}
            ).to(device).eval()

        total_params = sum(param.numel() for param in depth_anything.parameters())
        print("Total parameters: {:.2f}M".format(total_params / 1e6), flush=True)
        transform = depth_mod.Compose(
            [
                depth_mod.Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                depth_mod.NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                depth_mod.PrepareForNet(),
            ]
        )
        depth_cache[key] = (depth_anything, transform, device)
        return depth_cache[key]

    def cached_run_depth(
        video_path,
        t2v_model,
        output_dir,
        meta_file,
        encoder="vitl",
        pred_only=True,
        grayscale=True,
        output_root=None,
    ):
        depth_anything, transform, device = _depth_components(encoder)
        vid_process = depth_mod.Video_preprocess(output_root=output_root)
        with open(meta_file, "r", encoding="utf-8") as json_data:
            prompts = json.load(json_data)

        frame_folder = vid_process.convert_video_to_frames(video_path)
        print("frame_folder: ", frame_folder, flush=True)

        output_dir = os.path.join(output_dir, t2v_model)
        videos = os.listdir(frame_folder)
        os.makedirs(output_dir, exist_ok=True)

        for vid in videos:
            images = os.listdir(os.path.join(frame_folder, vid))
            images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            num = int(vid) - 1
            spatial = prompts[num]["spatial"]
            if spatial not in ["left", "right", "above", "on", "under", "below", "in front of", "behind"]:
                print(spatial, "spatial not included!!!, index: ", vid, flush=True)
                break
            if spatial not in ["behind", "in front of"]:
                continue

            for frame in images:
                filename = os.path.join(frame_folder, vid, frame)
                raw_image = cv2.imread(filename)
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
                h, w = image.shape[:2]
                image = transform({"image": image})["image"]
                image = torch.from_numpy(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    depth = depth_anything(image)

                depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.cpu().numpy().astype(np.uint8)

                if grayscale:
                    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                else:
                    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

                if pred_only:
                    os.makedirs(os.path.join(output_dir, vid), exist_ok=True)
                    cv2.imwrite(os.path.join(output_dir, vid, os.path.basename(filename)), depth)
        return frame_folder

    depth_mod.run_depth = cached_run_depth
    spatial_mod.run_depth = cached_run_depth

    def run_spatial(command):
        args = _spatial_args(command, spatial_script)
        if not args.video_path or not args.depth_folder or not args.t2v_model:
            raise RuntimeError(f"Invalid spatial command args: {command}")

        frame_folder = spatial_mod.run_depth(
            args.video_path,
            args.t2v_model,
            args.depth_folder,
            args.read_prompt_file,
            output_root=args.output_root or args.output_path,
        )
        args.frame_folder = frame_folder
        csv_2d = spatial_mod.spatial_2d(args)
        csv_3d = spatial_mod.spatial_3d(args)
        output_csv = f"{args.output_path}/{args.t2v_model}_spatial_score.csv"
        spatial_mod.combine_csv_and_cal_model_score(csv_2d, csv_3d, output_csv)
        return {"csv_path": output_csv, "frame_folder": frame_folder}

    def run_numeracy(command, frame_folder_override=None):
        args = _numeracy_args(command, numeracy_script)
        if not args.video_path or not args.t2v_model:
            raise RuntimeError(f"Invalid numeracy command args: {command}")

        config_file = args.config_file
        checkpoint_path = args.checkpoint_path
        output_dir = os.path.join(args.output_dir, args.t2v_model)
        output_path = args.output_path
        video_path = args.video_path
        box_threshold = args.box_threshold
        text_threshold = args.text_threshold
        token_spans = args.token_spans
        iou_threshold = args.iou_threshold

        model = numeracy_mod.load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

        with open(args.read_prompt_file, "r", encoding="utf-8") as json_data:
            prompts = json.load(json_data)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        vid2frame = numeracy_mod.Video_preprocess()
        if frame_folder_override is not None:
            frame_folder = frame_folder_override
        elif args.frame_folder is None:
            frame_folder = vid2frame.convert_video_to_frames(video_path)
        else:
            frame_folder = args.frame_folder

        videos = os.listdir(frame_folder)
        videos.sort()

        csv_path = f"{output_path}/{args.t2v_model}_numeracy_frame.csv"
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="", encoding="utf-8-sig") as csvreader:
                line_count = len(list(csv.reader(csvreader)))
        else:
            line_count = 0

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            if line_count == 0:
                csv_writer.writerow(["video_name", "image_name", "prompt", "objects", "numbers", "actual_numbers", "score"])

            evaluated = max(line_count - 1, 0)
            for i in range(evaluated, len(videos)):
                os.makedirs(os.path.join(output_dir, videos[i]), exist_ok=True)
                video_frame_path = os.path.join(frame_folder, videos[i])
                images = os.listdir(video_frame_path)
                images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

                video_name = videos[i]
                num = int(video_name[0:4]) - 1
                prompt = prompts[num]["prompt"]
                objects = prompts[num]["objects"]
                numbers = prompts[num]["numbers"]
                objs = objects.split(",")
                nums = numbers.split(",")

                if len(objs) != len(nums):
                    print("video ", i, " parse wrong, objects and quantities not match", flush=True)
                    break
                for j in range(len(objs)):
                    objs[j] = objs[j].strip()
                    try:
                        nums[j] = int(nums[j].strip())
                    except Exception:
                        print("object number not int", flush=True)

                print(objs, nums, flush=True)

                for image_name in images:
                    image_path = os.path.join(frame_folder, videos[i], image_name)
                    image_pil, image = numeracy_mod.load_image(image_path)

                    if token_spans is not None:
                        text_threshold = None
                        print("Using token_spans. Set the text_threshold to None.", flush=True)

                    image_with_box = image_pil
                    all_box = []
                    all_prob = []
                    all_phrase = []
                    for j in range(len(objs)):
                        boxes_filt_0, pred_phrases_0, prob_0 = numeracy_mod.get_grounding_output(
                            model,
                            image,
                            objs[j],
                            box_threshold,
                            text_threshold,
                            cpu_only=args.cpu_only,
                            token_spans=eval(f"{token_spans}") if token_spans is not None else None,
                        )
                        size = image_pil.size
                        all_box = all_box + [boxes_filt_0]
                        all_prob = all_prob + prob_0
                        all_phrase = all_phrase + pred_phrases_0

                    all_box = torch.cat(all_box, dim=0)
                    all_box, all_phrase, all_prob = numeracy_mod.filter_box(all_box, all_phrase, all_prob, iou_threshold)

                    pred_dict_0 = {
                        "boxes": all_box,
                        "size": [size[1], size[0]],
                        "labels": all_phrase,
                    }
                    image_with_box = numeracy_mod.plot_boxes_to_image(image_with_box, pred_dict_0, objs)[0]

                    if len(objs) == 1:
                        real_obj1_num = len(
                            [phrase.split("(")[1] for phrase in all_phrase if phrase.split("(")[0] == objs[0]]
                        )
                        real_obj_num = [real_obj1_num]
                        score = 1 if nums[0] == real_obj1_num else 0
                    else:
                        real_obj1_num = len(
                            [phrase.split("(")[1] for phrase in all_phrase if phrase.split("(")[0] == objs[0]]
                        )
                        real_obj2_num = len(
                            [phrase.split("(")[1] for phrase in all_phrase if phrase.split("(")[0] == objs[1]]
                        )
                        real_obj_num = [real_obj1_num, real_obj2_num]
                        score = 0
                        if nums[0] == real_obj1_num:
                            score += 0.5
                        if nums[1] == real_obj2_num:
                            score += 0.5

                    csv_writer.writerow([videos[i], image_name, prompt, objs, nums, real_obj_num, score])
                    csvfile.flush()
                    image_with_box.save(os.path.join(output_dir, videos[i], f"{image_name.split('.')[0]}.jpg"))

        output_csv = f"{output_path}/{args.t2v_model}_numeracy_video.csv"
        numeracy_mod.combine_frame(csv_path, output_csv)
        return {"csv_path": output_csv, "frame_folder": frame_folder}

    _send({"status": "ready"})

    for line in sys.stdin:
        if not line:
            break
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "shutdown":
                _send({"status": "shutdown"})
                break
            if command != "score":
                raise RuntimeError(f"Unknown command: {command}")

            tasks = [str(task).strip().lower() for task in request.get("tasks", [])]
            commands = request.get("commands", {})
            results = {}
            shared_frame_folder = None

            if "spatial" in tasks:
                results["spatial"] = run_spatial(commands["spatial"])
                shared_frame_folder = results["spatial"].get("frame_folder")
            if "numeracy" in tasks:
                results["numeracy"] = run_numeracy(commands["numeracy"], shared_frame_folder)

            _send({"status": "ok", "results": results})
        except Exception:
            _send({"status": "error", "traceback": traceback.format_exc()})
except Exception:
    _send({"status": "fatal", "traceback": traceback.format_exc()})
"""

    def _read_compbench_worker_message(self, timeout: Optional[float] = None) -> dict:
        if self._compbench_worker_process is None or self._compbench_worker_process.stdout is None:
            raise RuntimeError("Persistent CompBench worker is not running.")

        import select

        stdout = self._compbench_worker_process.stdout
        fd = stdout.fileno()
        if timeout is not None:
            readable, _, _ = select.select([fd], [], [], timeout)
            if not readable:
                raise TimeoutError("Timed out waiting for persistent CompBench worker.")

        while True:
            line = stdout.readline()
            if line == "":
                returncode = self._compbench_worker_process.poll()
                raise RuntimeError(
                    "Persistent CompBench worker exited before returning a result. "
                    f"returncode={returncode}"
                )
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def _start_compbench_worker(self) -> None:
        if self._compbench_worker_process is not None and self._compbench_worker_process.poll() is None:
            return

        python_bin = self.compbench_python or self.python
        cmd = [
            python_bin,
            "-c",
            self._compbench_worker_script(),
            json.dumps(self._compbench_worker_config()),
        ]
        if not self.compbench_python and not self.no_conda:
            cmd = [self.conda_exe, "run", "--no-capture-output", "-n", self.compbench_env, *cmd]

        self._compbench_worker_log_path = os.path.join(
            self.output_dir,
            f"layoutreward_compbench_worker_{os.getpid()}_{int(time.time() * 1000)}.log",
        )
        self._compbench_worker_log_file = open(
            self._compbench_worker_log_path,
            "a",
            encoding="utf-8",
            buffering=1,
        )
        env = os.environ.copy()
        cuda_visible_devices = self._compbench_cuda_visible_devices()
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        self._prepend_env_path(env, "PYTHONPATH", self.compbench_extra_pythonpath)
        self._prepend_env_path(env, "LD_LIBRARY_PATH", self.compbench_extra_ld_library_path)
        self._compbench_worker_process = subprocess.Popen(
            cmd,
            env=env,
            cwd=self.workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._compbench_worker_log_file,
            text=True,
            bufsize=1,
        )
        message = self._read_compbench_worker_message(timeout=self.timeout)
        if message.get("status") != "ready":
            traceback_text = message.get("traceback", "")
            log_path = self._compbench_worker_log_path
            self._stop_compbench_worker()
            raise RuntimeError(
                "Persistent CompBench worker failed to start.\n"
                f"status={message.get('status')}\n"
                f"{traceback_text}\n"
                f"worker_log={log_path}"
            )

    def _stop_compbench_worker(self) -> None:
        process = self._compbench_worker_process
        self._compbench_worker_process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=10)
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
        finally:
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass
            try:
                if process.stdout:
                    process.stdout.close()
            except Exception:
                pass
            try:
                if self._compbench_worker_log_file:
                    self._compbench_worker_log_file.close()
            except Exception:
                pass
            self._compbench_worker_log_file = None

    def _run_compbench_persistent(
        self,
        task_names: Sequence[str],
        commands: dict[str, list[str]],
    ) -> None:
        self._start_compbench_worker()
        if self._compbench_worker_process is None or self._compbench_worker_process.stdin is None:
            raise RuntimeError("Persistent CompBench worker is not writable.")
        request = {
            "cmd": "score",
            "tasks": [str(task_name).strip().lower() for task_name in task_names],
            "commands": commands,
        }
        try:
            self._compbench_worker_process.stdin.write(json.dumps(request) + "\n")
            self._compbench_worker_process.stdin.flush()
        except BrokenPipeError as exc:
            self._stop_compbench_worker()
            raise RuntimeError("Persistent CompBench worker pipe broke while sending a request.") from exc

        message = self._read_compbench_worker_message(timeout=self.timeout)
        status = message.get("status")
        if status != "ok":
            traceback_text = message.get("traceback", "")
            if status in {"fatal", None}:
                self._stop_compbench_worker()
            raise RuntimeError(
                "Persistent CompBench worker failed.\n"
                f"status={status}\n"
                f"{traceback_text}\n"
                f"worker_log={self._compbench_worker_log_path}"
            )

    def _persistent_worker_config(self) -> dict:
        return {
            "weights": self.weights,
            "output_dir": self.output_dir,
            "prompt_dir": self.reference_prompt_dir,
            "interaction_prompt_file": self.reference_interaction_prompt_file,
            "spatial_prompt_file": self.reference_spatial_prompt_file,
            "numeracy_prompt_file": self.reference_numeracy_prompt_file,
            "t2v_model": self.t2v_model,
            "model_path": self.model_path,
            "interaction_script": self.interaction_script,
            "spatial_script": self.spatial_script,
            "numeracy_script": self.numeracy_script,
            "llava_env": self.llava_env,
            "compbench_env": self.compbench_env,
            "conda_exe": self.conda_exe,
            "python": self.python,
            "llava_python": self.llava_python,
            "compbench_python": self.compbench_python,
            "compbench_extra_pythonpath": self.compbench_extra_pythonpath,
            "compbench_extra_ld_library_path": self.compbench_extra_ld_library_path,
            "no_conda": self.no_conda,
            "cuda_visible_devices": self.cuda_visible_devices,
            "task_cuda_visible_devices": self.task_cuda_visible_devices,
            "parallel_tasks": self.parallel_tasks,
            "workdir": self.workdir,
            "only": self.only,
            "fps": self.fps,
            "timeout": self.timeout,
            "use_mock": False,
            "keep_videos": self.keep_videos,
            "viewable_videos": self.viewable_videos,
            "number_width": self.number_width,
            "video_copy_mode": self.video_copy_mode,
            "direct_numbered_videos": self.direct_numbered_videos,
            "persistent_worker": False,
            "interaction_persistent_worker": self.interaction_persistent_worker,
            "compbench_persistent_worker": self.compbench_persistent_worker,
            "interaction_load_4bit": self.interaction_load_4bit,
            "interaction_load_8bit": self.interaction_load_8bit,
            "interaction_device_map": self.interaction_device_map,
            "interaction_repeats": self.interaction_repeats,
            "interaction_max_new_tokens": self.interaction_max_new_tokens,
            "interaction_temperature": self.interaction_temperature,
            "interaction_top_p": self.interaction_top_p,
            "interaction_num_beams": self.interaction_num_beams,
        }

    def _persistent_worker_script(self) -> str:
        return r"""
import contextlib
import json
import sys
import traceback

import torch

config = json.loads(sys.argv[1])

try:
    from wan_utils.video_reward_layout import LayoutReward

    with contextlib.redirect_stdout(sys.stderr):
        reward = LayoutReward(
            device=torch.device("cpu"),
            **config,
        )

    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        if not line:
            break
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "shutdown":
                print(json.dumps({"status": "shutdown"}), flush=True)
                break
            if command != "score":
                raise RuntimeError(f"Unknown command: {command}")

            with contextlib.redirect_stdout(sys.stderr):
                result_path = reward._run_layout_compbench_once(
                    request["video_dir"],
                    request["prompt_dir"],
                    request["eval_output_dir"],
                    command_stdout_to_stderr=True,
                )
            print(json.dumps({"status": "ok", "result_path": result_path}), flush=True)
        except Exception:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )
except Exception:
    print(
        json.dumps(
            {
                "status": "fatal",
                "traceback": traceback.format_exc(),
            }
        ),
        flush=True,
    )
"""

    def _read_worker_message(self, timeout: Optional[float] = None) -> dict:
        if self._worker_process is None or self._worker_process.stdout is None:
            raise RuntimeError("Persistent LayoutReward worker is not running.")

        import select

        stdout = self._worker_process.stdout
        fd = stdout.fileno()
        if timeout is not None:
            readable, _, _ = select.select([fd], [], [], timeout)
            if not readable:
                raise TimeoutError("Timed out waiting for persistent LayoutReward worker.")

        while True:
            line = stdout.readline()
            if line == "":
                returncode = self._worker_process.poll()
                raise RuntimeError(
                    "Persistent LayoutReward worker exited before returning a result. "
                    f"returncode={returncode}"
                )
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def _start_persistent_worker(self) -> None:
        if self._worker_process is not None and self._worker_process.poll() is None:
            return

        cmd = [
            sys.executable,
            "-c",
            self._persistent_worker_script(),
            json.dumps(self._persistent_worker_config()),
        ]
        self._worker_log_path = os.path.join(
            self.output_dir,
            f"layoutreward_worker_{os.getpid()}_{int(time.time() * 1000)}.log",
        )
        self._worker_log_file = open(self._worker_log_path, "a", encoding="utf-8", buffering=1)
        self._worker_process = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._worker_log_file,
            text=True,
            bufsize=1,
        )
        message = self._read_worker_message(timeout=self.timeout)
        if message.get("status") != "ready":
            traceback_text = message.get("traceback", "")
            log_path = self._worker_log_path
            self._stop_persistent_worker()
            raise RuntimeError(
                "Persistent LayoutReward worker failed to start.\n"
                f"status={message.get('status')}\n"
                f"{traceback_text}\n"
                f"worker_log={log_path}"
            )

    def _stop_persistent_worker(self) -> None:
        process = self._worker_process
        self._worker_process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=10)
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
        finally:
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass
            try:
                if process.stdout:
                    process.stdout.close()
            except Exception:
                pass
            try:
                if self._worker_log_file:
                    self._worker_log_file.close()
            except Exception:
                pass
            self._worker_log_file = None

    def _run_layout_compbench_persistent(self, video_dir: str, prompt_dir: str, eval_output_dir: str) -> str:
        self._start_persistent_worker()
        if self._worker_process is None or self._worker_process.stdin is None:
            raise RuntimeError("Persistent LayoutReward worker is not writable.")
        request = {
            "cmd": "score",
            "video_dir": video_dir,
            "prompt_dir": prompt_dir,
            "eval_output_dir": eval_output_dir,
        }
        try:
            self._worker_process.stdin.write(json.dumps(request) + "\n")
            self._worker_process.stdin.flush()
        except BrokenPipeError as exc:
            self._stop_persistent_worker()
            raise RuntimeError("Persistent LayoutReward worker pipe broke while sending a request.") from exc

        message = self._read_worker_message(timeout=self.timeout)
        status = message.get("status")
        if status != "ok":
            traceback_text = message.get("traceback", "")
            if status in {"fatal", None}:
                self._stop_persistent_worker()
            raise RuntimeError(
                "Persistent LayoutReward failed.\n"
                f"status={status}\n"
                f"{traceback_text}\n"
                f"worker_log={self._worker_log_path}"
            )
        result_path = message.get("result_path")
        if not result_path or not os.path.exists(result_path):
            raise RuntimeError(f"Persistent LayoutReward returned missing result_path={result_path!r}")
        return result_path

    def _run_layout_compbench_once(
        self,
        video_dir: str,
        prompt_dir: str,
        eval_output_dir: str,
        command_stdout_to_stderr: bool = False,
    ) -> str:
        evaluator = LayoutCompBenchEvaluate(
            video_path=video_dir,
            output_path=eval_output_dir,
            prompt_dir=prompt_dir,
            interaction_prompt_file=self.batch_interaction_prompt_file,
            spatial_prompt_file=self.batch_spatial_prompt_file,
            numeracy_prompt_file=self.batch_numeracy_prompt_file,
            t2v_model=self.t2v_model,
            model_path=self.model_path,
            interaction_script=self.interaction_script,
            spatial_script=self.spatial_script,
            numeracy_script=self.numeracy_script,
            llava_env=self.llava_env,
            compbench_env=self.compbench_env,
            conda_exe=self.conda_exe,
            python=self.python,
            llava_python=self.llava_python,
            compbench_python=self.compbench_python,
            compbench_extra_pythonpath=self.compbench_extra_pythonpath,
            compbench_extra_ld_library_path=self.compbench_extra_ld_library_path,
            no_conda=self.no_conda,
            cuda_visible_devices=self.cuda_visible_devices,
            task_cuda_visible_devices=self.task_cuda_visible_devices,
            parallel_tasks=self.parallel_tasks,
            workdir=self.workdir,
            only=self.only,
            no_prepare_numbered_videos=self.direct_numbered_videos,
            number_width=self.number_width,
            video_copy_mode=self.video_copy_mode,
            temp_parent_dir=os.path.join(eval_output_dir, "temp_parent"),
            temp_video_dir=os.path.join(eval_output_dir, "temp_videos"),
            keep_temp_videos=self.keep_videos,
            video_name_map_output="video_name_mapping.json",
            summarize=True,
            summary_eval_results_output=DEFAULT_LAYOUT_EVAL_RESULTS,
            summary_statistics_output=DEFAULT_LAYOUT_STATISTICS,
            timeout=self.timeout,
            command_stdout_to_stderr=command_stdout_to_stderr,
            interaction_runner=(
                self._run_interaction_persistent
                if self.interaction_persistent_worker and "interaction" in {str(task).strip().lower() for task in self.only}
                else None
            ),
            compbench_runner=(
                self._run_compbench_persistent
                if self.compbench_persistent_worker and {"spatial", "numeracy"} & {str(task).strip().lower() for task in self.only}
                else None
            ),
        )
        result_paths = evaluator.evaluate()
        if not result_paths or "eval_results" not in result_paths:
            raise RuntimeError("Layout/T2V-CompBench did not return an eval_results path.")
        return result_paths["eval_results"]

    def _run_layout_compbench(self, video_dir: str, prompt_dir: str, eval_output_dir: str) -> str:
        if self.persistent_worker:
            return self._run_layout_compbench_persistent(video_dir, prompt_dir, eval_output_dir)
        return self._run_layout_compbench_once(video_dir, prompt_dir, eval_output_dir)

    @staticmethod
    def _path_key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def _extract_weighted_scores(
        self,
        result_data: dict,
        video_paths: List[str],
        dimensions: Sequence[str] | None = None,
    ) -> tuple[list[float], dict[str, float]]:
        path_to_index = {self._path_key(path): idx for idx, path in enumerate(video_paths)}
        basename_to_index = {os.path.basename(path): idx for idx, path in enumerate(video_paths)}
        per_video_scores: list[dict[str, float]] = [dict() for _ in video_paths]
        log_values: dict[str, float] = {}
        active_dimensions = tuple(dimensions or LAYOUT_DIMENSION_NAMES)
        dimension_fallbacks: dict[str, float] = {}
        dimension_present_counts: dict[str, int] = {}

        for dimension in active_dimensions:
            if dimension not in result_data:
                raise RuntimeError(f"Layout/T2V-CompBench result is missing dimension '{dimension}'")
            dimension_result = result_data[dimension]
            if isinstance(dimension_result, (list, tuple)) and len(dimension_result) >= 2:
                mean_score = float(dimension_result[0])
                video_results = dimension_result[1]
            else:
                raise RuntimeError(
                    f"Unexpected Layout/T2V-CompBench result format for {dimension}: {dimension_result!r}"
                )
            log_values[f"reward_layout_{dimension}"] = mean_score
            dimension_values: list[float] = []

            for item in video_results:
                if not isinstance(item, dict):
                    continue
                score = _layout_finite_float(item.get("video_results", item.get("score")))
                if score is None:
                    continue
                raw_path = str(item.get("video_path", ""))
                index = path_to_index.get(self._path_key(raw_path))
                if index is None:
                    index = basename_to_index.get(os.path.basename(raw_path))
                if index is not None:
                    score = float(score)
                    per_video_scores[index][dimension] = score
                    dimension_values.append(score)

            dimension_fallbacks[dimension] = float(
                statistics.mean(dimension_values) if dimension_values else mean_score
            )
            dimension_present_counts[dimension] = len(dimension_values)

        weight_total = sum(self.weights[name] for name in active_dimensions)
        scores = []
        missing_total = 0
        missing_by_dimension = {name: 0 for name in active_dimensions}
        missing_examples: list[str] = []
        for path, dimension_scores in zip(video_paths, per_video_scores):
            missing = [name for name in active_dimensions if name not in dimension_scores]
            if missing:
                missing_total += len(missing)
                if len(missing_examples) < 3:
                    missing_examples.append(f"{os.path.basename(path)}:{','.join(missing)}")
                for name in missing:
                    missing_by_dimension[name] += 1
                    dimension_scores[name] = dimension_fallbacks[name]
            weighted_score = sum(
                dimension_scores[name] * self.weights[name]
                for name in active_dimensions
            ) / weight_total
            scores.append(weighted_score)

        if missing_total:
            log_values["reward_layout_missing_score_count"] = float(missing_total)
            for name in active_dimensions:
                log_values[f"reward_layout_missing_{name}"] = float(missing_by_dimension[name])
            details = ", ".join(
                f"{name}={missing_by_dimension[name]}/{len(video_paths)}"
                for name in active_dimensions
                if missing_by_dimension[name]
            )
            print(
                "[LayoutReward] filled missing per-video scores with dimension fallbacks: "
                f"total={missing_total}, {details}; present="
                + ", ".join(
                    f"{name}={dimension_present_counts[name]}/{len(video_paths)}"
                    for name in active_dimensions
                )
                + f"; examples={missing_examples}",
                flush=True,
            )

        for name in active_dimensions:
            log_values[f"reward_layout_weight_{name}"] = self.weights[name]
        return scores, log_values

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        videos: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        if videos.ndim != 5:
            raise ValueError(f"Expected videos with shape [B,F,C,H,W], got {tuple(videos.shape)}")
        if len(prompts) != videos.shape[0]:
            raise ValueError(f"Prompt/video batch mismatch: {len(prompts)} vs {videos.shape[0]}")

        if self.use_mock:
            rewards = videos.float().mean(dim=tuple(range(1, videos.ndim)))
            return RewardOutput(
                rewards=rewards,
                log_dict={"reward_layout_mock": rewards.mean()},
            )

        eval_name = f"layoutreward_{os.getpid()}_{int(time.time() * 1000)}"
        eval_output_dir = os.path.join(self.output_dir, eval_name)
        video_dir = os.path.join(eval_output_dir, "videos")
        prompt_dir = os.path.join(eval_output_dir, "prompts")
        os.makedirs(video_dir, exist_ok=True)

        video_paths: list[str] = []
        try:
            for idx, video in enumerate(videos):
                if self.direct_numbered_videos:
                    video_name = f"{idx + 1:0{self.number_width}d}.mp4"
                else:
                    video_name = f"video_{idx:05d}.mp4"
                video_path = os.path.abspath(os.path.join(video_dir, video_name))
                AestheticReward._tensor_to_video(video, video_path, self.fps)
                video_paths.append(video_path)

            self._write_batch_prompt_files(prompt_dir, prompts, metadata)
            result_path = self._run_layout_compbench(video_dir, prompt_dir, eval_output_dir)
            with open(result_path, "r", encoding="utf-8") as file_obj:
                result_data = json.load(file_obj)
            active_dimensions = _layout_dimension_names_for_tasks(self.only)
            scores, log_values = self._extract_weighted_scores(
                result_data,
                video_paths,
                dimensions=active_dimensions,
            )

            if self.keep_videos and self.viewable_videos:
                converted = AestheticReward._make_saved_video_viewable(video_paths)
                log_values["reward_layout_viewable_video_count"] = float(converted)
        finally:
            if not self.keep_videos:
                shutil.rmtree(video_dir, ignore_errors=True)

        rewards = torch.tensor(scores, dtype=torch.float32, device=videos.device)
        log_dict = {
            key: torch.tensor(value, dtype=torch.float32, device=videos.device)
            for key, value in log_values.items()
        }
        log_dict["reward_layout_video"] = rewards.mean()
        log_dict["reward_layout_dimension_count"] = torch.tensor(
            len(_layout_dimension_names_for_tasks(self.only)),
            dtype=torch.float32,
            device=videos.device,
        )
        return RewardOutput(rewards=rewards, log_dict=log_dict)

    def __del__(self):
        try:
            self._stop_persistent_worker()
        except Exception:
            pass
        try:
            self._stop_interaction_worker()
        except Exception:
            pass
        try:
            self._stop_compbench_worker()
        except Exception:
            pass
        try:
            if getattr(self, "_owns_output_dir", False):
                shutil.rmtree(getattr(self, "output_dir", ""), ignore_errors=True)
        except Exception:
            pass

class HPSVideoReward(VideoRewardFn):
    """Frame-averaged HPS-v2 reward over a full generated video."""

    def __init__(
        self,
        device: torch.device,
        frame_count: int = 8,
        hps_ckpt_path: str = "./hps_ckpt/HPS_v2.1_compressed.pt",
        clip_ckpt_path: str = "./hps_ckpt/open_clip_pytorch_model.bin",
    ) -> None:
        try:
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        except ImportError as exc:
            raise ImportError(
                "HPSVideoReward requires HPSv2. Install it first, e.g. "
                "`git clone https://github.com/tgxs002/HPSv2.git && cd HPSv2 && pip install -e .`."
            ) from exc

        self.device = device
        self.frame_count = frame_count
        model, _, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            clip_ckpt_path,
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        checkpoint = torch.load(hps_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        self.model = model.eval().to(device)
        self.preprocess_val = preprocess_val
        self.tokenizer = get_tokenizer("ViT-H-14")

    def _select_frame_ids(self, num_frames: int) -> List[int]:
        frame_count = min(self.frame_count, num_frames)
        return torch.linspace(0, num_frames - 1, frame_count).round().long().tolist()

    @staticmethod
    def _frame_to_pil(frame: torch.Tensor) -> Image.Image:
        frame = frame.detach().float().clamp(0, 1).cpu()
        frame = frame.permute(1, 2, 0).numpy()
        frame = (frame * 255.0).round().astype(np.uint8)
        return Image.fromarray(frame)

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        videos: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        del metadata
        if videos.ndim != 5:
            raise ValueError(f"Expected videos with shape [B,F,C,H,W], got {tuple(videos.shape)}")
        if len(prompts) != videos.shape[0]:
            raise ValueError(f"Prompt/video batch mismatch: {len(prompts)} vs {videos.shape[0]}")

        frame_ids = self._select_frame_ids(videos.shape[1])
        rewards = []
        for prompt, video in zip(prompts, videos):
            frame_scores = []
            text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
            for frame_id in frame_ids:
                image = self._frame_to_pil(video[frame_id])
                image = self.preprocess_val(image).unsqueeze(0).to(
                    device=self.device, non_blocking=True
                )
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = self.model(image, text)
                    image_features = outputs["image_features"]
                    text_features = outputs["text_features"]
                    frame_scores.append(torch.diagonal(image_features @ text_features.T))
            rewards.append(torch.stack(frame_scores).mean())
        rewards = torch.stack(rewards).to(videos.device)
        return RewardOutput(
            rewards=rewards,
            log_dict={
                "reward_hps_video": rewards.mean(),
                "reward_frame_count": torch.tensor(len(frame_ids), device=videos.device),
            },
        )


def build_video_reward(config, device: torch.device) -> VideoRewardFn:
    reward_type = str(getattr(config, "reward_type", "hps")).lower()
    if reward_type in {"mean_pixel", "dummy"}:
        return MeanPixelVideoReward()
    if reward_type in {"aesreward", "aesthetic", "vbench"}:
        dimensions = getattr(config, "aesreward_dimensions", None)
        if isinstance(dimensions, str):
            dimensions = [item.strip() for item in dimensions.split(",") if item.strip()]
        return AestheticReward(
            device=device,
            dimensions=dimensions,
            output_dir=getattr(config, "aesreward_output_dir", None),
            full_json_dir=getattr(config, "aesreward_full_json_dir", None),
            fps=int(getattr(config, "aesreward_fps", 16)),
            timeout=int(getattr(config, "aesreward_timeout", 900)),
            load_ckpt_from_local=_config_bool(
                getattr(config, "aesreward_load_ckpt_from_local", True),
                default=True,
            ),
            read_frame=_config_bool(getattr(config, "aesreward_read_frame", False)),
            imaging_quality_preprocessing_mode=str(
                getattr(config, "aesreward_imaging_quality_preprocessing_mode", "longer")
            ),
            use_mock=_config_bool(getattr(config, "aesreward_use_mock", False)),
            keep_videos=_config_bool(getattr(config, "aesreward_keep_videos", False)),
            viewable_videos=_config_bool(
                getattr(config, "aesreward_viewable_videos", True),
                default=True,
            ),
            persistent_worker=_config_bool(
                getattr(config, "aesreward_persistent_worker", False),
                default=False,
            ),
        )
    if reward_type in {"layout", "layoutreward", "layout_reward", "t2v_compbench"}:
        weights = getattr(config, "layoutreward_weights", None)
        if weights is None:
            weights = {
                "physical_interactions": float(
                    getattr(
                        config,
                        "layoutreward_physical_interactions_weight",
                        getattr(config, "layoutreward_interaction_weight", 1.0),
                    )
                ),
                "spatial_relationship": float(
                    getattr(
                        config,
                        "layoutreward_spatial_relationship_weight",
                        getattr(config, "layoutreward_spatial_weight", 1.0),
                    )
                ),
                "object_count": float(
                    getattr(
                        config,
                        "layoutreward_object_count_weight",
                        getattr(config, "layoutreward_numeracy_weight", 1.0),
                    )
                ),
            }
        only = getattr(config, "layoutreward_only", None)
        if isinstance(only, str):
            only = [item.strip() for item in only.split(",") if item.strip()]
        timeout = getattr(config, "layoutreward_timeout", None)
        timeout = int(timeout) if timeout is not None else None
        cuda_visible_devices = getattr(config, "layoutreward_cuda_visible_devices", None)
        if cuda_visible_devices is not None:
            cuda_visible_devices = str(cuda_visible_devices)
        return LayoutReward(
            device=device,
            weights=weights,
            output_dir=getattr(config, "layoutreward_output_dir", None),
            prompt_dir=str(getattr(config, "layoutreward_prompt_dir", DEFAULT_LAYOUT_PROMPT_DIR)),
            interaction_prompt_file=str(
                getattr(config, "layoutreward_interaction_prompt_file", "layout_prompts.json")
            ),
            spatial_prompt_file=str(
                getattr(config, "layoutreward_spatial_prompt_file", "layout_spatial_relations.json")
            ),
            numeracy_prompt_file=str(
                getattr(config, "layoutreward_numeracy_prompt_file", "layout_object_counts.json")
            ),
            t2v_model=str(getattr(config, "layoutreward_t2v_model", "bidirectional")),
            model_path=str(getattr(config, "layoutreward_model_path", DEFAULT_LAYOUT_MODEL_PATH)),
            interaction_script=str(
                getattr(
                    config,
                    "layoutreward_interaction_script",
                    "/root/data/ltl/T2V-CompBench/LLaVA/llava/eval/compbench_eval_interaction.py",
                )
            ),
            spatial_script=str(
                getattr(
                    config,
                    "layoutreward_spatial_script",
                    "/root/data/ltl/T2V-CompBench/Grounded-Segment-Anything/compbench_eval_spatial_relationships.py",
                )
            ),
            numeracy_script=str(
                getattr(
                    config,
                    "layoutreward_numeracy_script",
                    "/root/data/ltl/T2V-CompBench/Grounded-Segment-Anything/GroundingDINO/demo/compbench_eval_numeracy.py",
                )
            ),
            llava_env=str(getattr(config, "layoutreward_llava_env", "llava")),
            compbench_env=str(getattr(config, "layoutreward_compbench_env", "compbench")),
            conda_exe=str(getattr(config, "layoutreward_conda_exe", "conda")),
            python=str(getattr(config, "layoutreward_python", "python")),
            llava_python=getattr(config, "layoutreward_llava_python", None),
            compbench_python=getattr(config, "layoutreward_compbench_python", None),
            compbench_extra_pythonpath=getattr(config, "layoutreward_compbench_extra_pythonpath", None),
            compbench_extra_ld_library_path=getattr(
                config,
                "layoutreward_compbench_extra_ld_library_path",
                None,
            ),
            no_conda=_config_bool(getattr(config, "layoutreward_no_conda", True), default=True),
            cuda_visible_devices=cuda_visible_devices,
            task_cuda_visible_devices=getattr(config, "layoutreward_task_cuda_visible_devices", None),
            parallel_tasks=_config_bool(getattr(config, "layoutreward_parallel_tasks", False)),
            workdir=getattr(config, "layoutreward_workdir", None),
            only=only,
            fps=int(getattr(config, "layoutreward_fps", 16)),
            timeout=timeout,
            use_mock=_config_bool(getattr(config, "layoutreward_use_mock", False)),
            keep_videos=_config_bool(getattr(config, "layoutreward_keep_videos", False)),
            viewable_videos=_config_bool(
                getattr(config, "layoutreward_viewable_videos", True),
                default=True,
            ),
            number_width=int(getattr(config, "layoutreward_number_width", 4)),
            video_copy_mode=str(getattr(config, "layoutreward_video_copy_mode", "copy")),
            direct_numbered_videos=_config_bool(
                getattr(config, "layoutreward_direct_numbered_videos", True),
                default=True,
            ),
            persistent_worker=_config_bool(
                getattr(config, "layoutreward_persistent_worker", False),
                default=False,
            ),
            interaction_persistent_worker=_config_bool(
                getattr(config, "layoutreward_interaction_persistent_worker", False),
                default=False,
            ),
            compbench_persistent_worker=_config_bool(
                getattr(config, "layoutreward_compbench_persistent_worker", False),
                default=False,
            ),
            interaction_load_4bit=_config_bool(
                getattr(config, "layoutreward_interaction_load_4bit", True),
                default=True,
            ),
            interaction_load_8bit=_config_bool(
                getattr(config, "layoutreward_interaction_load_8bit", False),
                default=False,
            ),
            interaction_device_map=str(
                getattr(config, "layoutreward_interaction_device_map", "auto")
            ),
            interaction_repeats=int(getattr(config, "layoutreward_interaction_repeats", 3)),
            interaction_max_new_tokens=int(
                getattr(config, "layoutreward_interaction_max_new_tokens", 512)
            ),
            interaction_temperature=float(
                getattr(config, "layoutreward_interaction_temperature", 0.1)
            ),
            interaction_top_p=getattr(config, "layoutreward_interaction_top_p", None),
            interaction_num_beams=int(
                getattr(config, "layoutreward_interaction_num_beams", 1)
            ),
        )
    if reward_type == "hps":
        return HPSVideoReward(
            device=device,
            frame_count=int(getattr(config, "reward_frame_count", 8)),
            hps_ckpt_path=str(getattr(config, "hps_ckpt_path", "./hps_ckpt/HPS_v2.1_compressed.pt")),
            clip_ckpt_path=str(getattr(config, "hps_clip_ckpt_path", "./hps_ckpt/open_clip_pytorch_model.bin")),
        )
    raise ValueError(f"Unknown reward_type: {reward_type}")
