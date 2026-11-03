"""Validated, auditable sparse patches for model state dictionaries."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TensorEdit:
    """Add deltas to flattened positions in one named tensor."""

    tensor: str
    flat_indices: tuple[int, ...]
    deltas: tuple[float, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TensorEdit:
        try:
            tensor = str(value["tensor"])
            indices = tuple(int(item) for item in value["flat_indices"])
            deltas = tuple(float(item) for item in value["deltas"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed tensor edit") from exc

        if not tensor:
            raise ValueError("Tensor name cannot be empty")
        if not indices:
            raise ValueError("Each tensor edit must change at least one value")
        if len(indices) != len(deltas):
            raise ValueError("flat_indices and deltas must have equal lengths")
        if any(index < 0 for index in indices):
            raise ValueError("Flat indices must be non-negative")
        if len(set(indices)) != len(indices):
            raise ValueError("Flat indices must be unique within an edit")
        if any(not math.isfinite(delta) for delta in deltas):
            raise ValueError("Deltas must be finite")
        return cls(tensor=tensor, flat_indices=indices, deltas=deltas)


@dataclass(frozen=True)
class PatchProposal:
    """Model-independent representation of an editor's proposed patch."""

    diagnosis: str
    evidence_summary: str
    edits: tuple[TensorEdit, ...]
    editor_model: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PatchProposal:
        try:
            diagnosis = str(value["diagnosis"]).strip()
            evidence = str(value["evidence_summary"]).strip()
            raw_edits = value["edits"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed patch proposal") from exc
        if not diagnosis:
            raise ValueError("A diagnosis is required")
        if not evidence:
            raise ValueError("An evidence summary is required")
        if not isinstance(raw_edits, (list, tuple)):
            raise ValueError("edits must be a list")
        edits = tuple(TensorEdit.from_mapping(item) for item in raw_edits)
        if not edits:
            raise ValueError("A patch must contain at least one edit")
        return cls(
            diagnosis=diagnosis,
            evidence_summary=evidence,
            edits=edits,
            editor_model=value.get("editor_model"),
        )

    @property
    def change_count(self) -> int:
        return sum(len(edit.flat_indices) for edit in self.edits)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_patch(proposal: PatchProposal, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(proposal.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_patch(path: str | Path) -> PatchProposal:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return PatchProposal.from_mapping(raw)


def apply_patch(
    state_dict: Mapping[str, Any],
    proposal: PatchProposal,
    *,
    max_changes: int = 128,
    max_abs_delta: float = 2.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a sparse additive patch after validating every requested change."""

    import torch

    if proposal.change_count > max_changes:
        raise ValueError(f"Patch requests {proposal.change_count} changes; limit is {max_changes}")

    output = {name: tensor.detach().clone() for name, tensor in state_dict.items()}
    manifest_edits: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for edit in proposal.edits:
        if edit.tensor not in output:
            raise ValueError(f"Unknown tensor: {edit.tensor}")
        tensor = output[edit.tensor]
        if not torch.is_floating_point(tensor):
            raise ValueError(f"Tensor is not floating point: {edit.tensor}")
        flat = tensor.reshape(-1)
        if any(index >= flat.numel() for index in edit.flat_indices):
            raise ValueError(f"Index outside tensor bounds: {edit.tensor}")
        if any(abs(delta) > max_abs_delta for delta in edit.deltas):
            raise ValueError(f"Delta exceeds absolute limit for {edit.tensor}")
        for index in edit.flat_indices:
            key = (edit.tensor, index)
            if key in seen:
                raise ValueError(f"Duplicate patch location: {edit.tensor}[{index}]")
            seen.add(key)

        index_tensor = torch.tensor(edit.flat_indices, dtype=torch.long, device=flat.device)
        delta_tensor = torch.tensor(edit.deltas, dtype=flat.dtype, device=flat.device)
        flat[index_tensor] += delta_tensor
        manifest_edits.append(
            {
                "tensor": edit.tensor,
                "changes": len(edit.flat_indices),
                "delta_l2": float(delta_tensor.norm().item()),
                "delta_max_abs": float(delta_tensor.abs().max().item()),
            }
        )

    total_parameters = sum(tensor.numel() for tensor in output.values())
    manifest = {
        "diagnosis": proposal.diagnosis,
        "change_count": proposal.change_count,
        "changed_fraction": proposal.change_count / total_parameters,
        "tensor_edits": manifest_edits,
    }
    return output, manifest


def matched_random_patch(
    proposal: PatchProposal,
    state_dict: Mapping[str, Any],
    *,
    seed: int,
) -> PatchProposal:
    """Create a random edit matched by tensor, sparsity, and per-tensor delta norm."""

    rng = random.Random(seed)
    random_edits: list[TensorEdit] = []
    used_indices: dict[str, set[int]] = {}
    for edit in proposal.edits:
        if edit.tensor not in state_dict:
            raise ValueError(f"Unknown tensor: {edit.tensor}")
        count = len(edit.flat_indices)
        size = int(state_dict[edit.tensor].numel())
        used = used_indices.setdefault(edit.tensor, set())
        available = [index for index in range(size) if index not in used]
        if count > len(available):
            raise ValueError(f"Too many changes requested for {edit.tensor}")
        indices = tuple(sorted(rng.sample(available, count)))
        used.update(indices)
        target_norm = math.sqrt(sum(delta * delta for delta in edit.deltas))
        direction = [rng.gauss(0.0, 1.0) for _ in range(count)]
        direction_norm = math.sqrt(sum(value * value for value in direction)) or 1.0
        deltas = tuple(value * target_norm / direction_norm for value in direction)
        random_edits.append(TensorEdit(edit.tensor, indices, deltas))
    return PatchProposal(
        diagnosis="matched-random-control",
        evidence_summary="Random control matched by tensor, sparsity, and delta L2 norm.",
        edits=tuple(random_edits),
        editor_model=None,
    )
