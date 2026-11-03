"""Deterministic synthetic shapes with a controllable colored-border shortcut."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

PALETTE = torch.tensor(
    [
        [0.95, 0.10, 0.10],
        [0.10, 0.90, 0.20],
        [0.10, 0.25, 0.95],
        [0.95, 0.85, 0.10],
    ],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class DatasetConfig:
    size: int = 4096
    seed: int = 0
    image_size: int = 16
    border_width: int = 2
    shortcut_correlation: float = 0.95
    shortcut_mode: str = "correlated"


def _shape_mask(label: int, image_size: int, generator: torch.Generator) -> torch.Tensor:
    coords = torch.arange(image_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    jitter_x = int(torch.randint(-1, 2, (1,), generator=generator).item())
    jitter_y = int(torch.randint(-1, 2, (1,), generator=generator).item())
    radius = int(torch.randint(3, 5, (1,), generator=generator).item())
    cx = (image_size - 1) / 2 + jitter_x
    cy = (image_size - 1) / 2 + jitter_y
    dx = xx - cx
    dy = yy - cy

    if label == 0:
        return dx.square() + dy.square() <= radius**2
    if label == 1:
        return torch.maximum(dx.abs(), dy.abs()) <= radius
    if label == 2:
        vertical = (dx.abs() <= 1.0) & (dy.abs() <= radius)
        horizontal = (dy.abs() <= 1.0) & (dx.abs() <= radius)
        return vertical | horizontal
    diagonal_a = (dx - dy).abs() <= 1.0
    diagonal_b = (dx + dy).abs() <= 1.0
    return (diagonal_a | diagonal_b) & (dx.abs() <= radius) & (dy.abs() <= radius)


def _shortcut_index(
    label: int,
    mode: str,
    correlation: float,
    generator: torch.Generator,
) -> int | None:
    if mode == "neutral":
        return None
    if mode == "counterfactual":
        return (label + 1) % len(PALETTE)
    if mode == "random":
        return int(torch.randint(0, len(PALETTE), (1,), generator=generator).item())
    if mode != "correlated":
        raise ValueError(f"Unknown shortcut mode: {mode}")
    if float(torch.rand((), generator=generator).item()) < correlation:
        return label
    alternatives = [index for index in range(len(PALETTE)) if index != label]
    choice = int(torch.randint(0, len(alternatives), (1,), generator=generator).item())
    return alternatives[choice]


def render_example(index: int, config: DatasetConfig) -> tuple[torch.Tensor, int, int]:
    """Render one example; object pixels are invariant across shortcut modes."""

    label = index % 4
    generator = torch.Generator().manual_seed(config.seed * 1_000_003 + index)
    image = torch.rand((3, config.image_size, config.image_size), generator=generator) * 0.08
    mask = _shape_mask(label, config.image_size, generator)
    foreground = 0.70 + 0.25 * torch.rand((3, 1), generator=generator)
    image[:, mask] = foreground

    shortcut = _shortcut_index(
        label,
        config.shortcut_mode,
        config.shortcut_correlation,
        generator,
    )
    width = config.border_width
    border_color = torch.full((3,), 0.5) if shortcut is None else PALETTE[shortcut]
    image[:, :width, :] = border_color[:, None, None]
    image[:, -width:, :] = border_color[:, None, None]
    image[:, :, :width] = border_color[:, None, None]
    image[:, :, -width:] = border_color[:, None, None]
    return image.clamp(0.0, 1.0), label, -1 if shortcut is None else shortcut


class ShortcutShapes(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config

    def __len__(self) -> int:
        return self.config.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        return render_example(index, self.config)


def make_loader(
    config: DatasetConfig,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed + 17)
    return DataLoader(
        ShortcutShapes(config),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )
