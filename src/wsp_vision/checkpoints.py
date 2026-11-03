"""Checkpoint I/O with explicit format and metadata."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .model import SmallCNN, parameter_count

FORMAT_VERSION = "wsp-vision-v1"


def state_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save({name: tensor.detach().cpu() for name, tensor in state_dict.items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def save_checkpoint(
    model: SmallCNN,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    payload = {
        "format": FORMAT_VERSION,
        "architecture": "SmallCNN",
        "parameter_count": parameter_count(model),
        "state_hash": state_hash(state),
        "state_dict": state,
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, target)
    return {key: value for key, value in payload.items() if key != "state_dict"}


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SmallCNN, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format')}")
    if payload.get("architecture") != "SmallCNN":
        raise ValueError(f"Unsupported architecture: {payload.get('architecture')}")
    expected_hash = payload.get("state_hash")
    actual_hash = state_hash(payload["state_dict"])
    if expected_hash != actual_hash:
        raise ValueError("Checkpoint state hash does not match its manifest")
    model = SmallCNN()
    model.load_state_dict(payload["state_dict"], strict=True)
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata
