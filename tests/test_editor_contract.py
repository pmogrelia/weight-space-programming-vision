from __future__ import annotations

from unittest.mock import patch

from wsp_vision.editor import propose_patch
from wsp_vision.model import SmallCNN


class _FakeUsage:
    def model_dump(self):
        return {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


class _FakeResponses:
    def parse(self, **kwargs):
        output_type = kwargs["text_format"]
        parsed = output_type(
            diagnosis="border shortcut",
            evidence_summary="A classifier channel is sensitive to paired border changes.",
            edits=[
                {
                    "tensor": "classifier.bias",
                    "flat_indices": [0],
                    "deltas": [0.01],
                }
            ],
        )
        return type(
            "FakeResponse",
            (),
            {
                "output_parsed": parsed,
                "usage": _FakeUsage(),
                "id": "resp_test",
                "_request_id": "req_test",
            },
        )()


class _FakeOpenAI:
    def __init__(self):
        self.responses = _FakeResponses()


def test_structured_editor_contract_without_network() -> None:
    model = SmallCNN()
    metrics = {
        "correlated": {"accuracy": 0.95},
        "counterfactual": {"accuracy": 0.25},
        "neutral": {"accuracy": 0.50},
    }
    diagnostics = {
        "paired_examples": 8,
        "mean_absolute_feature_shift": [0.0] * 32,
        "mean_signed_feature_shift": [0.0] * 32,
        "classifier_weight_l2_by_channel": [0.1] * 32,
    }

    with patch("openai.OpenAI", _FakeOpenAI):
        proposal, metadata = propose_patch(
            model.state_dict(),
            metrics,
            diagnostics,
            model="gpt-6-astra",
            max_changes=4,
        )

    assert proposal.change_count == 1
    assert proposal.edits[0].tensor == "classifier.bias"
    assert metadata["request_id"] == "req_test"
    assert metadata["usage"]["total_tokens"] == 120
