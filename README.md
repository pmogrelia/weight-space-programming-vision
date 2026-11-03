# Weight-Space Programming for Vision

An auditable research MVP that asks a narrow question:

> Can a reasoning model repair a shortcut-biased image classifier by directly changing a small number of weights, without gradients or an automated search loop?

The project deliberately uses one synthetic image task, one 6,164-parameter CNN, one structured LLM editor, three controls, and one parameter-symmetry test. That scope is small enough for a student to finish and rigorous enough to discuss in research-engineering interviews.

## What this repository demonstrates

- A deterministic four-class shape dataset with a 95%-correlated colored-border shortcut.
- Counterfactual and neutral test distributions generated from the same underlying objects.
- A complete small CNN whose weights fit in one model request.
- A schema-validated sparse patch format with bounds on changed scalars and delta magnitude.
- A one-shot OpenAI editor that receives weights, metrics, and paired activation diagnostics.
- Unedited, matched-random-edit, and standard fine-tuning comparisons.
- A verified function-preserving channel permutation to expose coordinate dependence.
- Checkpoint hashes, patch manifests, regression metrics, API usage metadata, and deterministic seeds.

This is a harness for an honest experiment, not a claim that LLM weight editing already works.

## System design

    correlated shapes ──> train 6K CNN ──> frozen checkpoint
                                              │
                 paired forward diagnostics ──┤
                                              v
                                      structured LLM patch
                                              │
                       validate + apply sparse deltas
                                              │
                                              v
                  correlated / counterfactual / neutral evaluation
                         │              │              │
                    unedited      random patch     fine-tuning

The OpenAI request is the only paid step. Training, evaluation, controls, and symmetry verification run locally.

## Setup

Use Python 3.11 or 3.12. The computer that created this repository only had the macOS system Python 3.9, so dependencies are intentionally not vendored.

With uv:

    uv venv --python 3.11
    source .venv/bin/activate
    uv pip install -e ".[dev,editor]"

With standard Python:

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev,editor]"

## Run the no-API study first

This trains the biased model, evaluates it, runs the gradient baseline, creates a symmetry-equivalent checkpoint, and writes one JSON report.

    wsp-vision local-study --output-dir results/seed-0 --seed 0

Inspect results/seed-0/local_study.json. A useful base checkpoint should score well on the correlated split and materially worse on the counterfactual split. If it does not, change the data/model design before spending API credits.

Run additional independent checkpoints:

    wsp-vision local-study --output-dir results/seed-1 --seed 1
    wsp-vision local-study --output-dir results/seed-2 --seed 2
    wsp-vision local-study --output-dir results/seed-3 --seed 3
    wsp-vision local-study --output-dir results/seed-4 --seed 4

## Request and evaluate a weight-space patch

Set OPENAI_API_KEY in your shell, then:

    wsp-vision propose \
      --checkpoint results/seed-0/base.pt \
      --output patches/seed-0-llm.json \
      --model gpt-6-astra \
      --reasoning-effort high \
      --max-changes 64

The request sends the 6K model weights, development metrics, and paired channel diagnostics to the API. It does not send hidden evaluation results.

Validate and apply the returned patch:

    wsp-vision apply \
      --checkpoint results/seed-0/base.pt \
      --patch patches/seed-0-llm.json \
      --output artifacts/seed-0-llm.pt

Create the matched random control:

    wsp-vision random-baseline \
      --checkpoint results/seed-0/base.pt \
      --patch patches/seed-0-llm.json \
      --output-patch patches/seed-0-random.json \
      --output-checkpoint artifacts/seed-0-random.pt \
      --seed 7

Evaluate both against exactly the same hidden seed:

    wsp-vision evaluate \
      --checkpoint artifacts/seed-0-llm.pt \
      --compare-to results/seed-0/base.pt \
      --seed 40000 \
      --output results/seed-0/llm-evaluation.json

    wsp-vision evaluate \
      --checkpoint artifacts/seed-0-random.pt \
      --compare-to results/seed-0/base.pt \
      --seed 40000 \
      --output results/seed-0/random-evaluation.json

The development seed defaults to 20000 and the example hidden seed above is 40000. Freeze the hidden seed before looking at LLM results.

## Experimental matrix

| Condition | Gradients during repair? | Uses dev examples? | Purpose |
|---|---:|---:|---|
| Unedited checkpoint | No | No | Base failure mode |
| Matched random patch | No | No | Tests whether any same-sized perturbation helps |
| LLM WSP patch | No | Forward diagnostics only | Main treatment |
| Fine-tuning on randomized borders | Yes | Yes | Conventional repair ceiling |

Primary metric: counterfactual accuracy change from the unedited checkpoint.

Guardrails:

- Report correlated and neutral accuracy so repair cannot hide destructive regressions.
- Report regression rate on examples the base model classified correctly.
- Run all preregistered seeds, including failures.
- Keep the editor prompt, patch budget, model version, and hidden seed fixed.
- Repeat the editor on a channel-permuted but functionally identical checkpoint.
- Do not describe the LLM condition as gradient-free unless no gradient-derived data enters its prompt.

## Repository map

| Path | Role |
|---|---|
| src/wsp_vision/data.py | Deterministic shortcut dataset |
| src/wsp_vision/model.py | 6,164-parameter CNN |
| src/wsp_vision/editor.py | Structured OpenAI editor request |
| src/wsp_vision/patches.py | Patch schema, validation, application, and random control |
| src/wsp_vision/engine.py | Training, evaluation, diagnostics, and fine-tuning |
| src/wsp_vision/symmetry.py | Function-preserving channel permutation |
| src/wsp_vision/cli.py | Reproducible command-line workflow |
| tests | Offline schema tests and PyTorch integration tests |
| docs/VERIFICATION.md | Executed baseline and runtime verification record |
| MVP_EXECUTION.md | Eight-week completion plan and stop rules |
| PROJECT_PLAN.md | Larger publication-oriented roadmap; explicitly not MVP scope |

## Quality checks

    ruff check .
    pytest

The test suite never calls the OpenAI API. Patch files and trained artifacts are ignored by Git by default; publish selected aggregate JSON results after reviewing them.

## What makes this resume-worthy

The strongest result is not necessarily a successful edit. The employable signal is the combination of:

- research judgment: a falsifiable question and controlled scope;
- ML systems work: deterministic data, checkpointing, evaluation, and experiment metadata;
- API engineering: strict structured output and validation before mutation;
- scientific controls: random, gradient, counterfactual, and symmetry comparisons;
- honest communication: full seed-level results and clear failure analysis.

Suggested resume bullet after results exist:

> Built Weight-Space Programming for Vision, a reproducible PyTorch/OpenAI evaluation harness for sparse, gradient-free LLM edits to 6K-parameter image classifiers; benchmarked edits across counterfactual distributions, matched random controls, fine-tuning, and function-preserving network reparameterizations.

Replace that sentence with measured outcomes only after completing all seeds. Do not claim improved robustness unless the aggregate results support it.

## GitHub strategy

Publish this as an original repository rather than only a fork; original work is clearer to recruiters. Add an experiment table, one architecture figure, a short demo GIF, and a two-page findings report after the first five-seed run. Pin the repository on your GitHub profile. If you later build on an upstream WSP harness, preserve attribution and link this repository as the vision-specific extension.

## Research status

Current status: implementation-complete MVP with the local baseline, gradient control, tests, and symmetry check executed successfully. The verified seed-0 base reached 96.78% correlated accuracy and 25.68% counterfactual accuracy. The OpenAI patch study remains intentionally unrun because no API key was supplied; see docs/VERIFICATION.md.

The extended research design lives in PROJECT_PLAN.md. It contains worthwhile follow-on ideas, but adding CIFAR-10, more editors, an interactive tool loop, or a larger architecture before the core experiment is complete would overcomplicate this side project.

