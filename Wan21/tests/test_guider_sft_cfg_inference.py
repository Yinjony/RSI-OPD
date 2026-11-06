"""Small CPU check of the DMD CFG branch and its independent temporal caches."""
import sys
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Wan21"), str(ROOT), str(ROOT / "shared")]
from wan_utils.scheduler import FlowMatchScheduler

wrapper = ModuleType("wan_utils.wan_wrapper")
wrapper.WanDiffusionWrapper = wrapper.WanTextEncoder = wrapper.WanVAEWrapper = object
memory = ModuleType("demo_utils.memory")
memory.gpu = torch.device("cpu")
memory.get_cuda_free_memory_gb = lambda _device: 0
memory.DynamicSwapInstaller = object
memory.move_model_to_device_with_memory_preservation = lambda *_args, **_kwargs: None
with patch.dict(sys.modules, {"wan_utils.wan_wrapper": wrapper, "demo_utils.memory": memory}):
    spec = importlib.util.spec_from_file_location("causal_inference_cfg_test", ROOT / "Wan21/pipeline/causal_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
CausalInferencePipeline = module.CausalInferencePipeline


class FakeGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(local_attn_size=1)
        self.calls = []

    def get_scheduler(self):
        scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
        scheduler.set_timesteps(1000, training=True)
        scheduler.add_noise = lambda x, _noise, _t: x
        return scheduler

    def _convert_flow_pred_to_x0(self, flow, xt, timestep):
        return xt - 0.5 * flow

    def forward(self, noisy_image_or_video, conditional_dict, timestep, *, kv_cache,
                crossattn_cache, prope_kv_cache, **_kwargs):
        branch = conditional_dict["branch"]
        self.calls.append((branch, kv_cache, crossattn_cache, prope_kv_cache,
                           noisy_image_or_video.clone()))
        for cache in (kv_cache, prope_kv_cache):
            cache[0]["global_end_index"] += 1
        crossattn_cache[0]["is_init"] = True
        flow = torch.full_like(noisy_image_or_video, 2 if branch == "positive" else 1)
        return flow, noisy_image_or_video - 0.5 * flow


class FakeTextEncoder(torch.nn.Module):
    def forward(self, text_prompts):
        return {"branch": text_prompts[0]}


class FakeVAE(torch.nn.Module):
    def decode_to_pixel(self, latents, use_cache=False):
        return latents[:, :, :1]


class TestDmdCfg(unittest.TestCase):
    def test_shared_asset_directory_reaches_all_model_components(self):
        args = SimpleNamespace(denoising_step_list=[10, 5], warp_denoising_step=False,
                               num_frame_per_block=4, independent_first_frame=False,
                               model_kwargs={"model_path": "/shared/wan", "use_camera": True})
        with patch.object(module, "WanDiffusionWrapper", return_value=FakeGenerator()) as generator, \
             patch.object(module, "WanTextEncoder", return_value=FakeTextEncoder()) as text, \
             patch.object(module, "WanVAEWrapper", return_value=FakeVAE()) as vae:
            CausalInferencePipeline(args, torch.device("cpu"))
        generator.assert_called_once_with(model_path="/shared/wan", use_camera=True, is_causal=True)
        text.assert_called_once_with(model_path="/shared/wan")
        vae.assert_called_once_with(model_path="/shared/wan")

    def test_guided_v_and_separate_reusable_caches(self):
        args = SimpleNamespace(denoising_step_list=[10, 5], warp_denoising_step=False,
                               num_frame_per_block=4, independent_first_frame=False,
                               context_noise=0)
        generator = FakeGenerator()
        pipeline = CausalInferencePipeline(args, torch.device("cpu"), generator=generator,
                                           text_encoder=FakeTextEncoder(), vae=FakeVAE())
        pipeline.num_transformer_blocks = 1
        pipeline.frame_seq_length = 1

        def initialise_kv(batch_size, dtype, device):
            pipeline.kv_cache1 = [dict(global_end_index=torch.tensor([0]),
                                       local_end_index=torch.tensor([0]))]

        def initialise_cross(batch_size, dtype, device):
            pipeline.crossattn_cache = [dict(is_init=False)]

        def initialise_prope(batch_size, dtype, device):
            pipeline.prope_kv_cache1 = [dict(global_end_index=torch.tensor([0]),
                                             local_end_index=torch.tensor([0]))]

        pipeline._initialize_kv_cache = initialise_kv
        pipeline._initialize_crossattn_cache = initialise_cross
        pipeline._initialize_prope_kv_cache = initialise_prope
        with patch("torch.cuda.synchronize"):
            for _ in range(2):
                generator.calls.clear()
                _, latent = pipeline.inference(
                    noise=torch.zeros(1, 4, 1, 2, 2), text_prompts=["positive"],
                    negative_text_prompts=["negative"], guidance_scale=3,
                    return_latents=True)
                # v_pos=2, v_neg=1 => v_cfg=4. Each of two denoising steps
                # subtracts 0.5*4 from the current input.
                self.assertTrue(torch.allclose(latent, torch.full_like(latent, -4)))
                self.assertEqual([x[0] for x in generator.calls],
                                 ["positive", "negative", "positive", "negative",
                                  "positive", "negative"])
                positive, negative = generator.calls[:2]
                for cache_index in (1, 2, 3):
                    self.assertIsNot(positive[cache_index], negative[cache_index])
                self.assertTrue(torch.equal(positive[4], negative[4]))
            generator.calls.clear()
            _, latent = pipeline.inference(
                noise=torch.zeros(1, 4, 1, 2, 2), text_prompts=["positive"],
                return_latents=True)
            self.assertTrue(torch.allclose(latent, torch.full_like(latent, -2)))
            self.assertEqual([x[0] for x in generator.calls], ["positive"] * 3)


if __name__ == "__main__":
    unittest.main()
