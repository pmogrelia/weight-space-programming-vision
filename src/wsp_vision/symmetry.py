"""Function-preserving channel permutations for the symmetry stress test."""

from __future__ import annotations

import copy
from typing import Any

import torch


def permute_conv1_channels(model: Any, permutation: torch.Tensor) -> Any:
    """Permute conv1 outputs and matching conv2 inputs without changing the function."""

    if permutation.ndim != 1 or permutation.numel() != model.conv1.out_channels:
        raise ValueError("Permutation has the wrong shape")
    expected = torch.arange(permutation.numel(), device=permutation.device)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("Values must form a complete permutation")

    transformed = copy.deepcopy(model)
    with torch.no_grad():
        transformed.conv1.weight.copy_(model.conv1.weight[permutation])
        transformed.conv1.bias.copy_(model.conv1.bias[permutation])
        transformed.conv2.weight.copy_(model.conv2.weight[:, permutation])
    return transformed


@torch.inference_mode()
def maximum_logit_error(
    first: Any,
    second: Any,
    inputs: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    first.eval().to(device)
    second.eval().to(device)
    return float((first(inputs.to(device)) - second(inputs.to(device))).abs().max().item())
