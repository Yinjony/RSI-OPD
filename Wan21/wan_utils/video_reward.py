"""Video reward adapters for GRPO training.

The public interface intentionally accepts a full video tensor so downstream
projects can swap in task-specific video rewards without changing trainers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

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
_overall_cache = {}

def _install_vbench_dimension_cache():
    import vbench.aesthetic_quality as aq
    import vbench.imaging_quality as iq
    import vbench.motion_smoothness as ms
    import vbench.overall_consistency as oc

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

    def _get_overall_cached(submodules_list, device):
        key = (json.dumps(submodules_list, sort_keys=True), _device_key(device))
        if key not in _overall_cache:
            tokenizer = oc.SimpleTokenizer(os.path.join(oc.CACHE_DIR, "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
            viclip = oc.ViCLIP(tokenizer=tokenizer, **submodules_list).to(device)
            if hasattr(viclip, "eval"):
                viclip.eval()
            _overall_cache[key] = (viclip, tokenizer)
        return _overall_cache[key]

    def compute_overall_consistency_cached(json_dir, device, submodules_list, **kwargs):
        viclip, tokenizer = _get_overall_cached(submodules_list, device)
        _, video_dict = oc.load_dimension_info(json_dir, dimension="overall_consistency", lang="en")
        video_dict = oc.distribute_list_to_rank(video_dict)
        all_results, video_results = oc.overall_consistency(viclip, video_dict, tokenizer, device)
        if oc.get_world_size() > 1:
            video_results = oc.gather_list_of_dict(video_results)
            all_results = sum([d["video_results"] for d in video_results]) / len(video_results)
        return all_results, video_results

    aq.compute_aesthetic_quality = compute_aesthetic_quality_cached
    iq.compute_imaging_quality = compute_imaging_quality_cached
    ms.compute_motion_smoothness = compute_motion_smoothness_cached
    oc.compute_overall_consistency = compute_overall_consistency_cached

    return {
        "aesthetic_quality": _get_aesthetic_cached,
        "imaging_quality": _get_imaging_cached,
        "motion_smoothness": _get_motion_cached,
        "overall_consistency": _get_overall_cached,
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
            self._stop_persistent_worker()
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
        from wan_utils.video_reward_layout import (
            DEFAULT_LAYOUT_MODEL_PATH,
            DEFAULT_LAYOUT_PROMPT_DIR,
            LayoutReward,
        )

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
    if reward_type in {"alignment", "alignmentreward", "alignment_reward"}:
        from wan_utils.video_reward_alignment import (
            DEFAULT_ALIGNMENT_WEIGHTS,
            AlignmentReward,
        )

        weights = getattr(config, "alignmentreward_weights", None)
        if weights is None:
            weights = {
                "motion_smoothness": float(
                    getattr(config, "alignmentreward_motion_smoothness_weight", 0.25)
                ),
                "temporal_flickering": float(
                    getattr(config, "alignmentreward_temporal_flickering_weight", 0.25)
                ),
                "frame_consistency": float(
                    getattr(config, "alignmentreward_frame_consistency_weight", 0.25)
                ),
                "overall_consistency": float(
                    getattr(config, "alignmentreward_overall_consistency_weight", 0.25)
                ),
            }
        elif isinstance(weights, str):
            parsed = {}
            for item in weights.split(","):
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                parsed[key.strip()] = float(value.strip())
            weights = parsed or dict(DEFAULT_ALIGNMENT_WEIGHTS)
        dimensions = getattr(config, "alignmentreward_dimensions", None)
        if isinstance(dimensions, str):
            dimensions = [item.strip() for item in dimensions.split(",") if item.strip()]
        return AlignmentReward(
            device=device,
            weights=weights,
            active_dimensions=dimensions,
            output_dir=getattr(config, "alignmentreward_output_dir", None),
            full_json_dir=getattr(config, "alignmentreward_full_json_dir", None),
            minwm_eval_root=getattr(config, "alignmentreward_minwm_eval_root", None),
            wbench_root=getattr(config, "alignmentreward_wbench_root", None),
            weights_dir=getattr(config, "alignmentreward_weights_dir", None),
            fps=int(getattr(config, "alignmentreward_fps", 16)),
            timeout=int(getattr(config, "alignmentreward_timeout", 900)),
            load_ckpt_from_local=_config_bool(
                getattr(config, "alignmentreward_load_ckpt_from_local", True),
                default=True,
            ),
            read_frame=_config_bool(getattr(config, "alignmentreward_read_frame", False)),
            imaging_quality_preprocessing_mode=str(
                getattr(
                    config,
                    "alignmentreward_imaging_quality_preprocessing_mode",
                    "longer",
                )
            ),
            use_mock=_config_bool(getattr(config, "alignmentreward_use_mock", False)),
            keep_videos=_config_bool(getattr(config, "alignmentreward_keep_videos", False)),
            viewable_videos=_config_bool(
                getattr(config, "alignmentreward_viewable_videos", True),
                default=True,
            ),
            persistent_worker=_config_bool(
                getattr(config, "alignmentreward_persistent_worker", False),
                default=False,
            ),
            frame_consistency_persistent_worker=_config_bool(
                getattr(
                    config,
                    "alignmentreward_frame_consistency_persistent_worker",
                    getattr(config, "alignmentreward_persistent_worker", False),
                ),
                default=False,
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
