from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

import torch

from wan_utils.video_reward import AestheticReward, RewardOutput, VideoRewardFn

DEFAULT_ALIGNMENT_WEIGHTS: Dict[str, float] = {
    "motion_smoothness": 0.25,
    "temporal_flickering": 0.25,
    "frame_consistency": 0.25,
    "overall_consistency": 0.25,
}

VBENCH_ALIGNMENT_DIMENSIONS = [
    "motion_smoothness",
    "temporal_flickering",
    "overall_consistency",
]

DEFAULT_MINWM_EVAL_ROOT = "/root/minWM_eval"


class AlignmentReward(VideoRewardFn):
    def __init__(
        self,
        device: torch.device,
        weights: Optional[Dict[str, float]] = None,
        output_dir: Optional[str] = None,
        full_json_dir: Optional[str] = None,
        minwm_eval_root: Optional[str] = None,
        wbench_root: Optional[str] = None,
        weights_dir: Optional[str] = None,
        fps: int = 16,
        timeout: int = 900,
        load_ckpt_from_local: bool = True,
        read_frame: bool = False,
        imaging_quality_preprocessing_mode: str = "longer",
        use_mock: bool = False,
        keep_videos: bool = False,
        viewable_videos: bool = True,
        persistent_worker: bool = False,
        frame_consistency_persistent_worker: Optional[bool] = None,
        active_dimensions: Optional[List[str]] = None,
    ) -> None:
        self.device = device
        self.weights = dict(weights or DEFAULT_ALIGNMENT_WEIGHTS)
        self._normalize_weights()
        self.active_components = self._resolve_active_components(active_dimensions)
        self.active_vbench_dimensions = [
            dimension
            for dimension in VBENCH_ALIGNMENT_DIMENSIONS
            if dimension in self.active_components
        ]
        self.use_frame_consistency = "frame_consistency" in self.active_components
        self.fps = int(fps)
        self.timeout = int(timeout)
        self.use_mock = bool(use_mock)
        self.keep_videos = bool(keep_videos)
        self.viewable_videos = bool(viewable_videos)
        self.frame_consistency_persistent_worker = (
            bool(persistent_worker)
            if frame_consistency_persistent_worker is None
            else bool(frame_consistency_persistent_worker)
        )
        self._frame_worker_process = None
        self._frame_worker_log_file = None
        self._frame_worker_log_path = None

        self.minwm_eval_root = os.path.abspath(
            minwm_eval_root or os.environ.get("MINWM_EVAL_ROOT", DEFAULT_MINWM_EVAL_ROOT)
        )
        self.wbench_root = os.path.abspath(
            wbench_root
            or os.environ.get("WBENCH_ROOT", os.path.join(self.minwm_eval_root, "wbench"))
        )
        self.weights_dir = os.path.abspath(
            weights_dir
            or os.environ.get("WBENCH_WEIGHTS_DIR", os.path.join(self.minwm_eval_root, "weights"))
        )

        self._owns_output_dir = not output_dir
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="alignmentreward_")
        os.makedirs(self.output_dir, exist_ok=True)

        self._vbench = None
        if self.active_vbench_dimensions:
            self._vbench = AestheticReward(
                device=device,
                dimensions=list(self.active_vbench_dimensions),
                output_dir=self.output_dir,
                full_json_dir=full_json_dir,
                fps=self.fps,
                timeout=self.timeout,
                load_ckpt_from_local=load_ckpt_from_local,
                read_frame=read_frame,
                imaging_quality_preprocessing_mode=imaging_quality_preprocessing_mode,
                use_mock=False,
                keep_videos=False,
                viewable_videos=False,
                persistent_worker=persistent_worker,
            )
        if self.use_frame_consistency and self.frame_consistency_persistent_worker and not self.use_mock:
            self._start_frame_consistency_worker()
        print(
            "[AlignmentReward] "
            f"active_components={self.active_components} "
            f"vbench_dimensions={self.active_vbench_dimensions} "
            f"frame_consistency={self.use_frame_consistency}",
            flush=True,
        )

    def _normalize_weights(self) -> None:
        total = sum(float(v) for v in self.weights.values())
        if total <= 0:
            raise ValueError(f"Alignment reward weights must sum to > 0, got {self.weights}")
        if abs(total - 1.0) > 1e-6:
            self.weights = {key: float(value) / total for key, value in self.weights.items()}

    def _resolve_active_components(self, active_dimensions: Optional[List[str]]) -> List[str]:
        valid_components = set(VBENCH_ALIGNMENT_DIMENSIONS) | {"frame_consistency"}
        unknown_weights = [key for key in self.weights if key not in valid_components]
        if unknown_weights:
            raise KeyError(f"Unknown alignment reward component(s): {unknown_weights}")

        positive_weight_components = [
            key for key, value in self.weights.items() if abs(float(value)) > 1e-12
        ]
        if not positive_weight_components:
            raise ValueError(f"No active alignment reward components in weights={self.weights}")

        if active_dimensions is None:
            return positive_weight_components

        requested = [str(item).strip() for item in active_dimensions if str(item).strip()]
        unknown_requested = [key for key in requested if key not in valid_components]
        if unknown_requested:
            raise KeyError(f"Unknown requested alignment reward dimension(s): {unknown_requested}")

        missing = [key for key in positive_weight_components if key not in requested]
        if missing:
            raise ValueError(
                "alignmentreward_dimensions must include every component with non-zero "
                f"weight. Missing {missing} for weights={self.weights}"
            )
        return [key for key in requested if key in positive_weight_components]

    @staticmethod
    def _extract_dimension_scores(
        result_data: dict,
        video_paths: List[str],
        dimensions: List[str],
    ) -> Dict[str, List[float]]:
        path_to_index = {os.path.abspath(path): idx for idx, path in enumerate(video_paths)}
        per_dimension = {dimension: [float("nan")] * len(video_paths) for dimension in dimensions}

        for dimension in dimensions:
            if dimension not in result_data:
                raise RuntimeError(f"VBench result is missing dimension '{dimension}'")
            dimension_result = result_data[dimension]
            if not (
                isinstance(dimension_result, (list, tuple))
                and len(dimension_result) >= 2
            ):
                raise RuntimeError(
                    f"Unexpected VBench result format for {dimension}: {dimension_result!r}"
                )
            for item in dimension_result[1]:
                video_path = os.path.abspath(str(item.get("video_path", "")))
                score = item.get("video_results", item.get("score", None))
                if score is None or video_path not in path_to_index:
                    continue
                per_dimension[dimension][path_to_index[video_path]] = float(score)

        for dimension in dimensions:
            for idx, score in enumerate(per_dimension[dimension]):
                if score != score:
                    raise RuntimeError(
                        f"VBench result did not contain a score for dimension={dimension} "
                        f"video={video_paths[idx]}"
                    )
        return per_dimension

    def _frame_consistency_script(self) -> str:
        return r"""
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

real_stdout = sys.stdout
sys.stdout = sys.stderr
root, video_paths_json, device = sys.argv[1:4]
os.environ["WBENCH_ROOT"] = os.path.join(root, "wbench")
os.environ["WBENCH_WEIGHTS_DIR"] = os.path.join(root, "weights")
eval_run = os.path.join(root, "eval_run")
for path in (
    eval_run,
    os.environ["WBENCH_ROOT"],
    os.path.join(os.environ["WBENCH_ROOT"], "third_party", "sam2"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.evaluate import evaluate_background_consistency
from src.metrics.consistency.background_consistency import BackgroundConsistencyMetric

try:
    metric = BackgroundConsistencyMetric(device=device)
    video_paths = json.loads(video_paths_json)
    scores = []
    for video_path in video_paths:
        with contextlib.redirect_stdout(sys.stderr):
            result = evaluate_background_consistency(
                str(Path(video_path)),
                device=device,
                metric=metric,
            )
        score = result.get("score")
        if score is None:
            raise RuntimeError(f"frame_consistency failed for {video_path}: {result}")
        scores.append(float(score))
    print(json.dumps(scores), file=real_stdout, flush=True)
except Exception:
    print(traceback.format_exc(), file=sys.stderr, flush=True)
    raise
"""

    def _frame_consistency_worker_script(self) -> str:
        return r"""
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

real_stdout = sys.stdout
sys.stdout = sys.stderr
root, device = sys.argv[1:3]
os.environ["WBENCH_ROOT"] = os.path.join(root, "wbench")
os.environ["WBENCH_WEIGHTS_DIR"] = os.path.join(root, "weights")
eval_run = os.path.join(root, "eval_run")
for path in (
    eval_run,
    os.environ["WBENCH_ROOT"],
    os.path.join(os.environ["WBENCH_ROOT"], "third_party", "sam2"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from src.evaluate import evaluate_background_consistency
    from src.metrics.consistency.background_consistency import BackgroundConsistencyMetric

    metric = BackgroundConsistencyMetric(device=device)
    print(json.dumps({"status": "ready"}), file=real_stdout, flush=True)

    for line in sys.stdin:
        if not line:
            break
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "shutdown":
                print(json.dumps({"status": "shutdown"}), file=real_stdout, flush=True)
                break
            if command != "score":
                raise RuntimeError(f"Unknown command: {command}")

            scores = []
            for video_path in request.get("video_paths", []):
                with contextlib.redirect_stdout(sys.stderr):
                    result = evaluate_background_consistency(
                        str(Path(video_path)),
                        device=device,
                        metric=metric,
                    )
                score = result.get("score")
                if score is None:
                    raise RuntimeError(f"frame_consistency failed for {video_path}: {result}")
                scores.append(float(score))
            print(
                json.dumps({"status": "ok", "scores": scores}),
                file=real_stdout,
                flush=True,
            )
        except Exception:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "traceback": traceback.format_exc(),
                    }
                ),
                file=real_stdout,
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
        file=real_stdout,
        flush=True,
    )
"""

    def _frame_consistency_device(self) -> str:
        if self.device.type == "cuda" and self.device.index is not None:
            return f"cuda:{self.device.index}"
        return str(self.device)

    def _frame_consistency_env(self) -> dict:
        env = self._vbench._make_subprocess_env()
        env["WBENCH_ROOT"] = self.wbench_root
        env["WBENCH_WEIGHTS_DIR"] = self.weights_dir
        return env

    def _read_frame_worker_message(self, timeout: Optional[float] = None) -> dict:
        if self._frame_worker_process is None or self._frame_worker_process.stdout is None:
            raise RuntimeError("Persistent frame-consistency worker is not running.")

        import select

        stdout = self._frame_worker_process.stdout
        fd = stdout.fileno()
        if timeout is not None:
            readable, _, _ = select.select([fd], [], [], timeout)
            if not readable:
                raise TimeoutError("Timed out waiting for persistent frame-consistency worker.")

        while True:
            line = stdout.readline()
            if line == "":
                returncode = self._frame_worker_process.poll()
                raise RuntimeError(
                    "Persistent frame-consistency worker exited before returning a result. "
                    f"returncode={returncode}"
                )
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def _start_frame_consistency_worker(self) -> None:
        if self._frame_worker_process is not None and self._frame_worker_process.poll() is None:
            return

        cmd = [
            sys.executable,
            "-c",
            self._frame_consistency_worker_script(),
            self.minwm_eval_root,
            self._frame_consistency_device(),
        ]
        self._frame_worker_log_path = os.path.join(
            self.output_dir,
            f"alignment_frame_worker_{os.getpid()}_{int(time.time() * 1000)}.log",
        )
        self._frame_worker_log_file = open(
            self._frame_worker_log_path,
            "a",
            encoding="utf-8",
            buffering=1,
        )
        self._frame_worker_process = subprocess.Popen(
            cmd,
            env=self._frame_consistency_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._frame_worker_log_file,
            text=True,
            bufsize=1,
        )
        message = self._read_frame_worker_message(timeout=self.timeout)
        if message.get("status") != "ready":
            traceback_text = message.get("traceback", "")
            log_path = self._frame_worker_log_path
            self._stop_frame_consistency_worker()
            raise RuntimeError(
                "Persistent frame-consistency worker failed to start.\n"
                f"status={message.get('status')}\n"
                f"{traceback_text}\n"
                f"worker_log={log_path}"
            )

    def _stop_frame_consistency_worker(self) -> None:
        process = self._frame_worker_process
        self._frame_worker_process = None
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
                if self._frame_worker_log_file:
                    self._frame_worker_log_file.close()
            except Exception:
                pass
            self._frame_worker_log_file = None

    def _run_frame_consistency_persistent(self, video_paths: List[str]) -> List[float]:
        self._start_frame_consistency_worker()
        if self._frame_worker_process is None or self._frame_worker_process.stdin is None:
            raise RuntimeError("Persistent frame-consistency worker is not writable.")

        request = {
            "cmd": "score",
            "video_paths": [os.path.abspath(path) for path in video_paths],
        }
        try:
            self._frame_worker_process.stdin.write(json.dumps(request) + "\n")
            self._frame_worker_process.stdin.flush()
        except BrokenPipeError as exc:
            self._stop_frame_consistency_worker()
            raise RuntimeError(
                "Persistent frame-consistency worker pipe broke while sending a request."
            ) from exc

        message = self._read_frame_worker_message(timeout=self.timeout)
        status = message.get("status")
        if status != "ok":
            traceback_text = message.get("traceback", "")
            if status in {"fatal", None}:
                self._stop_frame_consistency_worker()
            raise RuntimeError(
                "Persistent frame-consistency worker failed.\n"
                f"status={status}\n"
                f"{traceback_text}\n"
                f"worker_log={self._frame_worker_log_path}"
            )
        scores = message.get("scores", [])
        if len(scores) != len(video_paths):
            raise RuntimeError(
                f"frame_consistency returned {len(scores)} scores for {len(video_paths)} videos"
            )
        return [float(score) for score in scores]

    def _score_frame_consistency(self, video_paths: List[str]) -> List[float]:
        if self.frame_consistency_persistent_worker:
            return self._run_frame_consistency_persistent(video_paths)

        device_str = self._frame_consistency_device()
        cmd = [
            sys.executable,
            "-c",
            self._frame_consistency_script(),
            self.minwm_eval_root,
            json.dumps([os.path.abspath(path) for path in video_paths]),
            device_str,
        ]
        env = self._frame_consistency_env()
        completed = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Alignment frame_consistency subprocess failed.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        scores = json.loads(completed.stdout.strip() or "[]")
        if len(scores) != len(video_paths):
            raise RuntimeError(
                f"frame_consistency returned {len(scores)} scores for {len(video_paths)} videos"
            )
        return [float(score) for score in scores]

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
                log_dict={"reward_alignment_mock": rewards.mean()},
            )

        eval_name = f"alignmentreward_{os.getpid()}_{int(time.time() * 1000)}"
        batch_dir = tempfile.mkdtemp(prefix=f"{eval_name}_videos_", dir=self.output_dir)
        try:
            video_paths: List[str] = []
            prompt_map = {}
            for idx, (prompt, video) in enumerate(zip(prompts, videos)):
                video_path = os.path.abspath(os.path.join(batch_dir, f"video_{idx:05d}.mp4"))
                AestheticReward._tensor_to_video(video, video_path, self.fps)
                video_paths.append(video_path)
                prompt_map[video_path] = prompt

            prompt_file = os.path.join(batch_dir, "prompts.json")
            with open(prompt_file, "w", encoding="utf-8") as handle:
                json.dump(prompt_map, handle, ensure_ascii=False)

            vbench_scores = {}
            if self.active_vbench_dimensions:
                result_path = self._vbench._run_vbench(batch_dir, prompt_file, eval_name)
                with open(result_path, "r", encoding="utf-8") as handle:
                    result_data = json.load(handle)

                vbench_scores = self._extract_dimension_scores(
                    result_data,
                    video_paths,
                    self.active_vbench_dimensions,
                )
            frame_scores = (
                self._score_frame_consistency(video_paths)
                if self.use_frame_consistency
                else None
            )

            combined_scores = []
            log_values: Dict[str, float] = {
                key: 0.0 for key in self.active_components
            }
            for idx in range(len(video_paths)):
                components = {
                    key: vbench_scores[key][idx] for key in self.active_vbench_dimensions
                }
                if frame_scores is not None:
                    components["frame_consistency"] = frame_scores[idx]
                total = 0.0
                for key in self.active_components:
                    weight = self.weights[key]
                    if key not in components:
                        raise KeyError(f"Unknown alignment reward component: {key}")
                    total += float(weight) * float(components[key])
                    log_values[key] += float(components[key])
                combined_scores.append(total)

            if self.keep_videos and self.viewable_videos:
                converted = AestheticReward._make_saved_video_viewable(video_paths)
                log_values["reward_alignment_viewable_video_count"] = float(converted)
        finally:
            if not self.keep_videos:
                shutil.rmtree(batch_dir, ignore_errors=True)

        batch_size = max(len(video_paths), 1)
        rewards = torch.tensor(combined_scores, dtype=torch.float32, device=videos.device)
        log_dict = {
            f"reward_alignment_{key}": torch.tensor(
                value / batch_size,
                dtype=torch.float32,
                device=videos.device,
            )
            for key, value in log_values.items()
        }
        log_dict["reward_alignment_video"] = rewards.mean()
        return RewardOutput(rewards=rewards, log_dict=log_dict)

    def __del__(self):
        try:
            self._stop_frame_consistency_worker()
        except Exception:
            pass
        try:
            if hasattr(self, "_vbench"):
                del self._vbench
        except Exception:
            pass
        try:
            if getattr(self, "_owns_output_dir", False):
                shutil.rmtree(getattr(self, "output_dir", ""), ignore_errors=True)
        except Exception:
            pass
