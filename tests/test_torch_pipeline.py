from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchPipelineTests(unittest.TestCase):
    def test_counterfactual_changes_only_border(self) -> None:
        from wsp_vision.data import DatasetConfig, render_example

        neutral, neutral_label, _ = render_example(
            17, DatasetConfig(seed=4, shortcut_mode="neutral")
        )
        counterfactual, counterfactual_label, _ = render_example(
            17, DatasetConfig(seed=4, shortcut_mode="counterfactual")
        )
        self.assertEqual(neutral_label, counterfactual_label)
        self.assertTrue(torch.equal(neutral[:, 2:-2, 2:-2], counterfactual[:, 2:-2, 2:-2]))
        self.assertFalse(torch.equal(neutral, counterfactual))

    def test_channel_permutation_preserves_logits(self) -> None:
        from wsp_vision.model import SmallCNN, parameter_count
        from wsp_vision.symmetry import maximum_logit_error, permute_conv1_channels

        model = SmallCNN()
        transformed = permute_conv1_channels(model, torch.tensor([3, 2, 1, 0, 7, 6, 5, 4]))
        inputs = torch.randn(16, 3, 16, 16)
        error = maximum_logit_error(model, transformed, inputs, device=torch.device("cpu"))
        self.assertEqual(parameter_count(model), 6164)
        self.assertLess(error, 1e-6)

    def test_patch_application_is_sparse(self) -> None:
        from wsp_vision.model import SmallCNN
        from wsp_vision.patches import PatchProposal, apply_patch

        model = SmallCNN()
        patch = PatchProposal.from_mapping(
            {
                "diagnosis": "test",
                "evidence_summary": "test only",
                "edits": [
                    {
                        "tensor": "classifier.bias",
                        "flat_indices": [0],
                        "deltas": [0.25],
                    }
                ],
            }
        )
        updated, manifest = apply_patch(model.state_dict(), patch)
        difference = updated["classifier.bias"] - model.state_dict()["classifier.bias"]
        self.assertAlmostEqual(float(difference[0]), 0.25, places=6)
        self.assertEqual(int(torch.count_nonzero(difference)), 1)
        self.assertEqual(manifest["change_count"], 1)


if __name__ == "__main__":
    unittest.main()
