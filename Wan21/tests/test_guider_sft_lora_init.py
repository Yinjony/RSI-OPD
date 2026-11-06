"""CPU checks of old A/B preservation, optimizer updates and base provenance."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Wan21"))
from wan_utils.biopd_lora import get_adapter_state, inject_lora, save_adapter
from wan_utils.guider_sft_dataset import file_identity
from wan_utils.guider_sft_lora_init import (
    read_initial_adapter, restore_initial_adapter, restore_initial_reference, validate_initial_adapter,
)
from wan_utils.guider_sft_training import adapter_payload
from wan_train_guider_sft_continue_lora import export_final_checkpoint

TARGETS = ["q", "k", "v", "o", "prope_o"]


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({name: nn.Linear(4, 4) for name in TARGETS})

    def forward(self, x):
        return sum(layer(x) for layer in self.model.values())


class ContinueLoraTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.base_path = self.root / "ckpts/base.pt"
        self.base_path.parent.mkdir()
        self.base_path.write_bytes(b"original base identity")
        self.cfg = SimpleNamespace(lora_rank=2, lora_alpha=2, lora_target_modules=TARGETS,
                                   generator_ckpt=str(self.base_path), model_path=str(self.root / "Wan21/models"))
        self.dense = Tiny()
        self.old = copy.deepcopy(self.dense)
        inject_lora(self.old.model, 2, 2, TARGETS)
        with torch.no_grad():
            for name, parameter in self.old.named_parameters():
                if "lora_B" in name:
                    parameter.normal_()
        self.payload = adapter_payload(self.old, self.cfg, file_identity(self.base_path))

    def test_preserve_old_adapter_and_continue_updating_only_it(self):
        continued = copy.deepcopy(self.dense)
        inject_lora(continued.model, 2, 2, TARGETS)
        restore_initial_adapter(continued, self.payload, self.cfg)
        for key, value in get_adapter_state(continued.model).items():
            self.assertTrue(torch.equal(value, self.payload["adapter"][key]))
        x = torch.randn(3, 4)
        self.assertTrue(torch.equal(continued(x), self.old(x)))
        frozen = {n: p.detach().clone() for n, p in continued.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in continued.parameters() if p.requires_grad], lr=.01)
        continued(x).square().mean().backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(v, self.payload["adapter"][k])
                            for k, v in get_adapter_state(continued.model).items()))
        for name, parameter in continued.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(parameter, frozen[name]))

    def test_anchor_reference_is_initial_composite_and_frozen(self):
        reference = restore_initial_reference(copy.deepcopy(self.dense), self.payload, self.cfg)
        x = torch.randn(2, 4)
        self.assertTrue(torch.equal(reference(x), self.old(x)))
        self.assertFalse(any(p.requires_grad for p in reference.parameters()))

    def test_reject_applying_old_adapter_to_another_or_merged_base(self):
        other = self.root / "already_merged.pt"
        other.write_bytes(b"merged base is not the old base")
        cfg = copy.copy(self.cfg)
        cfg.generator_ckpt = str(other)
        with self.assertRaisesRegex(ValueError, "base_identity"):
            validate_initial_adapter(self.payload, cfg)

    def test_read_sft_and_biopd_formats(self):
        sft = self.root / "adapter.pt"
        torch.save(self.payload, sft)
        parsed, sources = read_initial_adapter(sft)
        validate_initial_adapter(parsed, self.cfg)
        self.assertEqual(len(sources), 1)
        step = self.root / "step_0000475"
        step.mkdir()
        save_adapter(self.old.model, step / "adapter", rank=2, alpha=2, targets=TARGETS,
                     base_model_path="Wan21/models")
        (step / "metadata.json").write_text(json.dumps(dict(contract=dict(
            student_identity=file_identity(self.base_path), generator_ckpt="ckpts/base.pt", model_path="Wan21/models"))))
        parsed, sources = read_initial_adapter(step / "adapter")
        validate_initial_adapter(parsed, self.cfg)
        self.assertEqual(len(sources), 3)
        for key, value in parsed["adapter"].items():
            self.assertTrue(torch.equal(value, self.payload["adapter"][key]))

    def test_export_uses_final_trained_adapter_and_preserves_sources(self):
        torch.save(dict(generator=self.dense.state_dict()), self.base_path)
        initial = adapter_payload(self.old, self.cfg, file_identity(self.base_path))
        source = self.root / "initial_adapter.pt"
        torch.save(initial, source)
        with torch.no_grad():
            for name, parameter in self.old.named_parameters():
                if "lora_B" in name:
                    parameter.add_(0.25)
        final = adapter_payload(self.old, self.cfg, file_identity(self.base_path))
        self.cfg.output_dir = str(self.root / "output")
        checkpoint = Path(self.cfg.output_dir) / "checkpoint_000950"
        checkpoint.mkdir(parents=True)
        adapter_path = checkpoint / "adapter.pt"
        torch.save(final, adapter_path)
        (checkpoint.parent / "latest.json").write_text(json.dumps(dict(step=950, adapter=str(adapter_path))))
        sources_before = (source.read_bytes(), adapter_path.read_bytes(), self.base_path.read_bytes())
        exported = export_final_checkpoint(self.cfg, [file_identity(source)])
        model = copy.deepcopy(self.dense)
        model.load_state_dict(torch.load(exported, weights_only=True)["generator"], strict=True)
        x = torch.randn(3, 4)
        torch.testing.assert_close(model(x), self.old(x))
        self.assertEqual(sources_before, (source.read_bytes(), adapter_path.read_bytes(), self.base_path.read_bytes()))
        provenance = json.loads(exported.with_suffix(".pt.merge.json").read_text())
        self.assertEqual(provenance["sources"]["adapter"], file_identity(adapter_path))
        with self.assertRaises(FileExistsError):
            export_final_checkpoint(self.cfg, [file_identity(source)])

    def test_standard_entry_delegates_adapter_initialization_before_loading_data(self):
        from omegaconf import OmegaConf
        from wan_train_guider_sft import main
        cfg = OmegaConf.create(dict(initial_adapter="original_guider1_adapter.pt"))
        argv = ["wan_train_guider_sft.py", "--config_path", "config.yaml", "--check-init", "--export-merged"]
        with patch.object(sys, "argv", argv), \
                patch("wan_utils.guider_sft_config.load_config", return_value=cfg), \
                patch("wan_utils.guider_sft_dataset.load_records", side_effect=AssertionError("too early")), \
                patch("wan_train_guider_sft_continue_lora.main") as continued:
            main()
        continued.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
