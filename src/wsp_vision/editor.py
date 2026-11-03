"""OpenAI-backed, structured weight patch proposal."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .patches import PatchProposal


def serialize_state_dict(state_dict: Mapping[str, Any], decimals: int = 5) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for name, tensor in state_dict.items():
        values = tensor.detach().cpu().reshape(-1).tolist()
        serialized[name] = {
            "shape": list(tensor.shape),
            "values_flat": [round(float(value), decimals) for value in values],
        }
    return serialized


def build_editor_prompt(
    state_dict: Mapping[str, Any],
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    max_changes: int,
) -> str:
    payload = {
        "architecture": [
            "conv1: Conv2d(3,8,3,padding=1) + ReLU",
            "conv2: Conv2d(8,16,3,padding=1) + ReLU",
            "conv3: Conv2d(16,32,3,padding=1) + ReLU",
            "global average pooling",
            "classifier: Linear(32,4)",
        ],
        "task": (
            "Classify four centered shapes. Training has a 95%-correlated two-pixel border "
            "color. Improve counterfactual-border accuracy while preserving neutral-border "
            "and correlated accuracy. Do not use gradients, optimization, or relabel outputs."
        ),
        "constraints": {
            "operation": "Sparse additive deltas to flattened named tensors",
            "maximum_changed_scalars": max_changes,
            "maximum_absolute_delta": 2.0,
            "preferred_scope": "classifier weights, unless upstream evidence is compelling",
        },
        "evaluation_metrics": metrics,
        "paired_channel_diagnostics": diagnostics,
        "state_dict": serialize_state_dict(state_dict),
    }
    return (
        "Propose one conservative weight-space patch. Use the measurements as evidence, "
        "keep the evidence summary brief, and return only the requested structured object.\n\n"
        + json.dumps(payload, separators=(",", ":"))
    )


def propose_patch(
    state_dict: Mapping[str, Any],
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    model: str = "gpt-6-astra",
    reasoning_effort: str = "high",
    max_changes: int = 64,
) -> tuple[PatchProposal, dict[str, Any]]:
    """Ask an OpenAI reasoning model for a bounded, schema-validated patch."""

    try:
        from openai import OpenAI
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise RuntimeError(
            "Install the editor dependencies with: pip install -e '.[editor]'"
        ) from exc

    class TensorEditOutput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        tensor: str
        flat_indices: list[int] = Field(min_length=1)
        deltas: list[float] = Field(min_length=1)

    class PatchOutput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        diagnosis: str
        evidence_summary: str
        edits: list[TensorEditOutput] = Field(min_length=1)

    client = OpenAI()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        input=[
            {
                "role": "system",
                "content": (
                    "You are a careful weight-space programming researcher. Make direct, "
                    "auditable parameter edits without gradients or automated search."
                ),
            },
            {
                "role": "user",
                "content": build_editor_prompt(
                    state_dict,
                    metrics,
                    diagnostics,
                    max_changes=max_changes,
                ),
            },
        ],
        text_format=PatchOutput,
    )
    if response.output_parsed is None:
        raise RuntimeError("The editor returned no parsed patch")
    raw = response.output_parsed.model_dump()
    raw["editor_model"] = model
    proposal = PatchProposal.from_mapping(raw)
    if proposal.change_count > max_changes:
        raise ValueError(f"Editor returned {proposal.change_count} changes; limit is {max_changes}")
    usage = response.usage.model_dump() if response.usage is not None else None
    metadata = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "request_id": getattr(response, "_request_id", None),
        "response_id": response.id,
        "usage": usage,
    }
    return proposal, metadata
