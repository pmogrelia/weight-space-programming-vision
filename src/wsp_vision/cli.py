"""Command-line entry point for the complete MVP workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_checkpoint, save_checkpoint
from .data import DatasetConfig, make_loader
from .editor import propose_patch
from .engine import (
    TrainConfig,
    collect_channel_diagnostics,
    evaluate_suite,
    fine_tune_baseline,
    regression_rate,
    resolve_device,
    seed_everything,
    summarize_suite,
    train_model,
)
from .model import SmallCNN, parameter_count
from .patches import apply_patch, load_patch, matched_random_patch, save_patch
from .symmetry import maximum_logit_error, permute_conv1_channels


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2))


def _train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    seed_everything(args.seed)
    model = SmallCNN()
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        train_size=args.train_size,
        shortcut_correlation=args.shortcut_correlation,
    )
    history = train_model(model, config, device)
    metadata = save_checkpoint(
        model.cpu(),
        args.output,
        metadata={"training": vars(config), "history": history},
    )
    _print_json(
        {
            "checkpoint": str(args.output),
            "device": str(device),
            "parameter_count": parameter_count(model),
            "last_epoch": history[-1],
            "manifest": metadata,
        }
    )


def _evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint)
    detailed = evaluate_suite(
        model,
        device=device,
        size=args.size,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "state_hash": metadata["state_hash"],
        "metrics": summarize_suite(detailed),
    }
    if args.compare_to:
        original, original_metadata = load_checkpoint(args.compare_to)
        original_detailed = evaluate_suite(
            original,
            device=device,
            size=args.size,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        report["comparison"] = {
            "original_checkpoint": str(args.compare_to),
            "original_state_hash": original_metadata["state_hash"],
            "accuracy_delta": {
                mode: detailed[mode]["accuracy"] - original_detailed[mode]["accuracy"]
                for mode in detailed
            },
            "regression_rate": {
                mode: regression_rate(original_detailed[mode], detailed[mode]) for mode in detailed
            },
        }
    if args.output:
        _write_json(args.output, report)
    _print_json(report)


def _propose(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint)
    detailed = evaluate_suite(
        model,
        device=device,
        size=args.dev_size,
        seed=args.dev_seed,
        batch_size=args.batch_size,
    )
    diagnostics = collect_channel_diagnostics(
        model,
        device=device,
        size=args.dev_size,
        seed=args.dev_seed,
        batch_size=args.batch_size,
    )
    proposal, editor_metadata = propose_patch(
        model.state_dict(),
        summarize_suite(detailed),
        diagnostics,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_changes=args.max_changes,
    )
    save_patch(proposal, args.output)
    record = {
        "base_checkpoint": str(args.checkpoint),
        "base_state_hash": metadata["state_hash"],
        "development_seed": args.dev_seed,
        "development_size": args.dev_size,
        "max_changes": args.max_changes,
        "editor": editor_metadata,
    }
    metadata_path = Path(args.output).with_suffix(".metadata.json")
    _write_json(metadata_path, record)
    _print_json(
        {
            "patch": str(args.output),
            "metadata": str(metadata_path),
            "diagnosis": proposal.diagnosis,
            "change_count": proposal.change_count,
        }
    )


def _apply(args: argparse.Namespace) -> None:
    model, base_metadata = load_checkpoint(args.checkpoint)
    proposal = load_patch(args.patch)
    updated_state, manifest = apply_patch(
        model.state_dict(),
        proposal,
        max_changes=args.max_changes,
        max_abs_delta=args.max_abs_delta,
    )
    model.load_state_dict(updated_state, strict=True)
    saved = save_checkpoint(
        model,
        args.output,
        metadata={
            "parent_state_hash": base_metadata["state_hash"],
            "patch_file": str(args.patch),
            "patch_manifest": manifest,
        },
    )
    _print_json({"checkpoint": str(args.output), "patch_manifest": manifest, "saved": saved})


def _random_baseline(args: argparse.Namespace) -> None:
    model, base_metadata = load_checkpoint(args.checkpoint)
    source = load_patch(args.patch)
    random_proposal = matched_random_patch(source, model.state_dict(), seed=args.seed)
    save_patch(random_proposal, args.output_patch)
    updated_state, manifest = apply_patch(
        model.state_dict(),
        random_proposal,
        max_changes=args.max_changes,
        max_abs_delta=args.max_abs_delta,
    )
    model.load_state_dict(updated_state, strict=True)
    save_checkpoint(
        model,
        args.output_checkpoint,
        metadata={
            "parent_state_hash": base_metadata["state_hash"],
            "source_patch": str(args.patch),
            "random_seed": args.seed,
            "patch_manifest": manifest,
        },
    )
    _print_json(
        {
            "patch": str(args.output_patch),
            "checkpoint": str(args.output_checkpoint),
            "patch_manifest": manifest,
        }
    )


def _fine_tune(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, base_metadata = load_checkpoint(args.checkpoint)
    tuned = fine_tune_baseline(
        model,
        device=device,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    saved = save_checkpoint(
        tuned,
        args.output,
        metadata={
            "parent_state_hash": base_metadata["state_hash"],
            "baseline": "random-border-fine-tune",
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
    )
    _print_json({"checkpoint": str(args.output), "manifest": saved})


def _symmetry(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, base_metadata = load_checkpoint(args.checkpoint)
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(model.conv1.out_channels, generator=generator)
    transformed = permute_conv1_channels(model, permutation)
    batch = next(
        iter(
            make_loader(
                DatasetConfig(size=args.examples, seed=args.seed, shortcut_mode="random"),
                batch_size=args.examples,
                shuffle=False,
            )
        )
    )[0]
    error = maximum_logit_error(model, transformed, batch, device=device)
    if error > args.tolerance:
        raise RuntimeError(f"Symmetry transform failed: max logit error {error}")
    saved = save_checkpoint(
        transformed.cpu(),
        args.output,
        metadata={
            "parent_state_hash": base_metadata["state_hash"],
            "transform": "conv1-channel-permutation",
            "permutation": permutation.tolist(),
            "verified_examples": args.examples,
            "maximum_logit_error": error,
        },
    )
    _print_json({"checkpoint": str(args.output), "maximum_logit_error": error, "saved": saved})


def _local_study(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / "base.pt"
    fine_tuned_path = output_dir / "fine_tuned.pt"
    symmetry_path = output_dir / "symmetry.pt"
    device = resolve_device(args.device)

    seed_everything(args.seed)
    base = SmallCNN()
    train_config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        train_size=args.train_size,
        shortcut_correlation=args.shortcut_correlation,
    )
    history = train_model(base, train_config, device)
    base = base.cpu()
    base_manifest = save_checkpoint(
        base,
        base_path,
        metadata={"training": vars(train_config), "history": history},
    )
    base_detailed = evaluate_suite(base, device=device, size=args.eval_size, seed=args.eval_seed)

    tuned = fine_tune_baseline(
        base,
        device=device,
        steps=args.fine_tune_steps,
        learning_rate=args.fine_tune_learning_rate,
        seed=args.seed + 1,
        batch_size=args.batch_size,
    )
    fine_tuned_manifest = save_checkpoint(
        tuned,
        fine_tuned_path,
        metadata={
            "parent_state_hash": base_manifest["state_hash"],
            "baseline": "random-border-fine-tune",
            "steps": args.fine_tune_steps,
        },
    )
    tuned_detailed = evaluate_suite(tuned, device=device, size=args.eval_size, seed=args.eval_seed)

    permutation = torch.randperm(8, generator=torch.Generator().manual_seed(args.seed + 2))
    symmetric = permute_conv1_channels(base, permutation)
    probe = next(
        iter(
            make_loader(
                DatasetConfig(size=128, seed=args.eval_seed, shortcut_mode="random"),
                batch_size=128,
                shuffle=False,
            )
        )
    )[0]
    symmetry_error = maximum_logit_error(base, symmetric, probe, device=device)
    symmetry_manifest = save_checkpoint(
        symmetric,
        symmetry_path,
        metadata={
            "parent_state_hash": base_manifest["state_hash"],
            "transform": "conv1-channel-permutation",
            "permutation": permutation.tolist(),
            "maximum_logit_error": symmetry_error,
        },
    )

    report = {
        "configuration": {key: value for key, value in vars(args).items() if key != "handler"},
        "base": {
            "checkpoint": str(base_path),
            "manifest": base_manifest,
            "metrics": summarize_suite(base_detailed),
        },
        "fine_tuned": {
            "checkpoint": str(fine_tuned_path),
            "manifest": fine_tuned_manifest,
            "metrics": summarize_suite(tuned_detailed),
            "regression_rate": {
                mode: regression_rate(base_detailed[mode], tuned_detailed[mode])
                for mode in base_detailed
            },
        },
        "symmetry": {
            "checkpoint": str(symmetry_path),
            "manifest": symmetry_manifest,
            "maximum_logit_error": symmetry_error,
        },
    }
    report_path = output_dir / "local_study.json"
    _write_json(report_path, report)
    _print_json({"report": str(report_path), **report})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsp-vision",
        description="Weight-Space Programming for Vision research MVP",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a shortcut-biased base checkpoint")
    train.add_argument("--output", default="artifacts/base.pt")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--learning-rate", type=float, default=3e-3)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--train-size", type=int, default=4096)
    train.add_argument("--shortcut-correlation", type=float, default=0.95)
    train.add_argument("--device", default="auto")
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate three shortcut conditions")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--compare-to")
    evaluate.add_argument("--output")
    evaluate.add_argument("--size", type=int, default=1024)
    evaluate.add_argument("--seed", type=int, default=10_000)
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(handler=_evaluate)

    propose = subparsers.add_parser("propose", help="Request a structured OpenAI weight patch")
    propose.add_argument("--checkpoint", required=True)
    propose.add_argument("--output", default="patches/llm.json")
    propose.add_argument("--model", default="gpt-6-astra")
    propose.add_argument("--reasoning-effort", default="high")
    propose.add_argument("--max-changes", type=int, default=64)
    propose.add_argument("--dev-size", type=int, default=512)
    propose.add_argument("--dev-seed", type=int, default=20_000)
    propose.add_argument("--batch-size", type=int, default=128)
    propose.add_argument("--device", default="auto")
    propose.set_defaults(handler=_propose)

    apply_command = subparsers.add_parser("apply", help="Validate and apply a JSON patch")
    apply_command.add_argument("--checkpoint", required=True)
    apply_command.add_argument("--patch", required=True)
    apply_command.add_argument("--output", default="artifacts/llm_edited.pt")
    apply_command.add_argument("--max-changes", type=int, default=128)
    apply_command.add_argument("--max-abs-delta", type=float, default=2.0)
    apply_command.set_defaults(handler=_apply)

    random_baseline = subparsers.add_parser(
        "random-baseline", help="Build the matched random-edit control"
    )
    random_baseline.add_argument("--checkpoint", required=True)
    random_baseline.add_argument("--patch", required=True)
    random_baseline.add_argument("--output-patch", default="patches/random.json")
    random_baseline.add_argument("--output-checkpoint", default="artifacts/random_edited.pt")
    random_baseline.add_argument("--seed", type=int, default=7)
    random_baseline.add_argument("--max-changes", type=int, default=128)
    random_baseline.add_argument("--max-abs-delta", type=float, default=2.0)
    random_baseline.set_defaults(handler=_random_baseline)

    fine_tune = subparsers.add_parser("fine-tune", help="Run the gradient repair baseline")
    fine_tune.add_argument("--checkpoint", required=True)
    fine_tune.add_argument("--output", default="artifacts/fine_tuned.pt")
    fine_tune.add_argument("--steps", type=int, default=100)
    fine_tune.add_argument("--learning-rate", type=float, default=1e-3)
    fine_tune.add_argument("--batch-size", type=int, default=128)
    fine_tune.add_argument("--seed", type=int, default=30_000)
    fine_tune.add_argument("--device", default="auto")
    fine_tune.set_defaults(handler=_fine_tune)

    symmetry = subparsers.add_parser("symmetry", help="Create and verify an equivalent network")
    symmetry.add_argument("--checkpoint", required=True)
    symmetry.add_argument("--output", default="artifacts/symmetry.pt")
    symmetry.add_argument("--seed", type=int, default=11)
    symmetry.add_argument("--examples", type=int, default=256)
    symmetry.add_argument("--tolerance", type=float, default=1e-5)
    symmetry.add_argument("--device", default="auto")
    symmetry.set_defaults(handler=_symmetry)

    local = subparsers.add_parser(
        "local-study", help="Train, evaluate, fine-tune, and verify symmetry without an API"
    )
    local.add_argument("--output-dir", default="results/local-study")
    local.add_argument("--seed", type=int, default=0)
    local.add_argument("--epochs", type=int, default=12)
    local.add_argument("--learning-rate", type=float, default=3e-3)
    local.add_argument("--batch-size", type=int, default=128)
    local.add_argument("--train-size", type=int, default=4096)
    local.add_argument("--shortcut-correlation", type=float, default=0.95)
    local.add_argument("--eval-size", type=int, default=1024)
    local.add_argument("--eval-seed", type=int, default=10_000)
    local.add_argument("--fine-tune-steps", type=int, default=100)
    local.add_argument("--fine-tune-learning-rate", type=float, default=1e-3)
    local.add_argument("--device", default="auto")
    local.set_defaults(handler=_local_study)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
