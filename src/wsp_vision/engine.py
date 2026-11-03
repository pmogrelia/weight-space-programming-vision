"""Training, evaluation, diagnostics, and the gradient baseline."""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .data import DatasetConfig, make_loader


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 12
    learning_rate: float = 3e-3
    batch_size: int = 128
    seed: int = 0
    train_size: int = 4096
    shortcut_correlation: float = 0.95


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    model: nn.Module,
    config: TrainConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    seed_everything(config.seed)
    model.to(device)
    loader = make_loader(
        DatasetConfig(
            size=config.train_size,
            seed=config.seed,
            shortcut_correlation=config.shortcut_correlation,
            shortcut_mode="correlated",
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += labels.numel()
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": loss_sum / count,
                "accuracy": correct / count,
            }
        )
    return history


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval().to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    count = 0
    correct = 0
    per_class_correct = [0, 0, 0, 0]
    per_class_count = [0, 0, 0, 0]
    predictions: list[int] = []
    labels_all: list[int] = []
    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        predicted = logits.argmax(dim=1)
        loss_sum += float(criterion(logits, labels).item())
        correct += int((predicted == labels).sum().item())
        count += labels.numel()
        predictions.extend(int(value) for value in predicted.cpu().tolist())
        labels_all.extend(int(value) for value in labels.cpu().tolist())
        for class_index in range(4):
            class_mask = labels == class_index
            per_class_correct[class_index] += int(((predicted == labels) & class_mask).sum().item())
            per_class_count[class_index] += int(class_mask.sum().item())
    class_accuracy = [
        hits / total if total else 0.0
        for hits, total in zip(per_class_correct, per_class_count, strict=True)
    ]
    return {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "worst_class_accuracy": min(class_accuracy),
        "per_class_accuracy": class_accuracy,
        "predictions": predictions,
        "labels": labels_all,
    }


def evaluate_suite(
    model: nn.Module,
    *,
    device: torch.device,
    size: int = 1024,
    seed: int = 10_000,
    batch_size: int = 256,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mode in ("correlated", "counterfactual", "neutral"):
        loader = make_loader(
            DatasetConfig(size=size, seed=seed, shortcut_mode=mode),
            batch_size=batch_size,
            shuffle=False,
        )
        results[mode] = evaluate_loader(model, loader, device)
    return results


def summarize_suite(results: dict[str, Any]) -> dict[str, Any]:
    return {
        mode: {
            "loss": values["loss"],
            "accuracy": values["accuracy"],
            "worst_class_accuracy": values["worst_class_accuracy"],
            "per_class_accuracy": values["per_class_accuracy"],
        }
        for mode, values in results.items()
    }


def regression_rate(original: dict[str, Any], edited: dict[str, Any]) -> float:
    original_correct = [
        prediction == label
        for prediction, label in zip(original["predictions"], original["labels"], strict=True)
    ]
    regressions = sum(
        was_correct and new_prediction != label
        for was_correct, new_prediction, label in zip(
            original_correct,
            edited["predictions"],
            original["labels"],
            strict=True,
        )
    )
    denominator = sum(original_correct)
    return regressions / denominator if denominator else 0.0


@torch.inference_mode()
def collect_channel_diagnostics(
    model: Any,
    *,
    device: torch.device,
    size: int = 512,
    seed: int = 20_000,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Measure feature sensitivity on paired neutral/counterfactual examples."""

    neutral = make_loader(
        DatasetConfig(size=size, seed=seed, shortcut_mode="neutral"),
        batch_size=batch_size,
        shuffle=False,
    )
    counterfactual = make_loader(
        DatasetConfig(size=size, seed=seed, shortcut_mode="counterfactual"),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval().to(device)
    absolute_shift = torch.zeros(32, device=device)
    signed_shift = torch.zeros(32, device=device)
    count = 0
    for neutral_batch, counterfactual_batch in zip(neutral, counterfactual, strict=True):
        neutral_inputs = neutral_batch[0].to(device)
        counterfactual_inputs = counterfactual_batch[0].to(device)
        neutral_features = model.forward_features(neutral_inputs)
        counterfactual_features = model.forward_features(counterfactual_inputs)
        difference = counterfactual_features - neutral_features
        absolute_shift += difference.abs().sum(dim=0)
        signed_shift += difference.sum(dim=0)
        count += difference.shape[0]
    classifier = model.classifier.weight.detach().cpu()
    return {
        "paired_examples": count,
        "mean_absolute_feature_shift": (absolute_shift / count).cpu().tolist(),
        "mean_signed_feature_shift": (signed_shift / count).cpu().tolist(),
        "classifier_weight_l2_by_channel": classifier.norm(dim=0).tolist(),
    }


def fine_tune_baseline(
    model: nn.Module,
    *,
    device: torch.device,
    steps: int = 100,
    learning_rate: float = 1e-3,
    seed: int = 30_000,
    batch_size: int = 128,
) -> nn.Module:
    """Gradient control trained on independently randomized border colors."""

    seed_everything(seed)
    edited = copy.deepcopy(model).to(device)
    loader = make_loader(
        DatasetConfig(size=max(steps * batch_size, 1024), seed=seed, shortcut_mode="random"),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(edited.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    edited.train()
    iterator = iter(loader)
    for _ in range(steps):
        try:
            inputs, labels, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, labels, _ = next(iterator)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(edited(inputs), labels)
        loss.backward()
        optimizer.step()
    return edited.cpu()
