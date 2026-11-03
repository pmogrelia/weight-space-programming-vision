from __future__ import annotations

import math
import unittest

from wsp_vision.patches import PatchProposal, matched_random_patch


class PatchSchemaTests(unittest.TestCase):
    def test_parses_valid_patch(self) -> None:
        patch = PatchProposal.from_mapping(
            {
                "diagnosis": "border shortcut",
                "evidence_summary": "Channel 3 changes most across paired borders.",
                "edits": [
                    {
                        "tensor": "classifier.weight",
                        "flat_indices": [3, 35],
                        "deltas": [-0.1, 0.1],
                    }
                ],
            }
        )
        self.assertEqual(patch.change_count, 2)
        self.assertEqual(patch.edits[0].tensor, "classifier.weight")

    def test_rejects_duplicate_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            PatchProposal.from_mapping(
                {
                    "diagnosis": "border shortcut",
                    "evidence_summary": "duplicate",
                    "edits": [
                        {
                            "tensor": "classifier.weight",
                            "flat_indices": [3, 3],
                            "deltas": [-0.1, 0.1],
                        }
                    ],
                }
            )

    def test_rejects_non_finite_delta(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            PatchProposal.from_mapping(
                {
                    "diagnosis": "border shortcut",
                    "evidence_summary": "bad delta",
                    "edits": [
                        {
                            "tensor": "classifier.weight",
                            "flat_indices": [3],
                            "deltas": [math.inf],
                        }
                    ],
                }
            )

    def test_random_control_avoids_cross_edit_collisions(self) -> None:
        class FakeTensor:
            def numel(self) -> int:
                return 4

        patch = PatchProposal.from_mapping(
            {
                "diagnosis": "border shortcut",
                "evidence_summary": "two edits in one tensor",
                "edits": [
                    {
                        "tensor": "classifier.bias",
                        "flat_indices": [0, 1],
                        "deltas": [0.1, 0.1],
                    },
                    {
                        "tensor": "classifier.bias",
                        "flat_indices": [2, 3],
                        "deltas": [0.2, 0.2],
                    },
                ],
            }
        )
        random_patch = matched_random_patch(patch, {"classifier.bias": FakeTensor()}, seed=7)
        indices = [index for edit in random_patch.edits for index in edit.flat_indices]
        self.assertEqual(len(indices), len(set(indices)))


if __name__ == "__main__":
    unittest.main()
