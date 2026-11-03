# Weight-Space Programming for Vision

## Benchmarking LLM-Guided Diagnosis and Gradient-Free Repair of Image Classifiers

**Plan version:** 1.1
**Date:** November 3, 2026  
**Target duration:** up to 12 weeks at 10–12 focused hours per week, with explicit stop gates after the 6K MVP and 23.8K scale bridge
**Primary outcome:** A public, reproducible research repository that remains valuable whether the central hypothesis succeeds or fails.

## 1. Executive Summary

This project will test whether a general-purpose reasoning model can diagnose and directly edit the weights of image classifiers from a natural-language objective, without gradients or a training loop. It will not claim that weight-space analysis or vision-model editing is itself new. Existing work already studies weight-space learning, spuriousness detection from weights, parameter symmetries, and vision-model editing.

The original contribution will be:

1. **WSP-VisionBench:** an open, auditable benchmark for LLM-native weight-space programming of visual classifiers;
2. **A controlled editor harness:** strict enforcement of no gradients, limited forward passes, blind evaluation, patch manifests, and complete run logging;
3. **Symmetry stress tests:** function-preserving channel permutations and ReLU rescalings that test whether editing depends on arbitrary parameter coordinates;
4. **A rigorous empirical result:** positive or negative evidence about diagnosis, repair, locality, robustness, cost, and reproducibility across independently trained checkpoints.

The original Weight-Space Programming authors have announced that they are building their own benchmark and harness. Therefore, the durable differentiation is not the existence of a benchmark: it is the combination of **vision-specific counterfactual evaluation, parameter-symmetry invariance, edit locality, and an independently auditable no-gradient protocol**. Re-check related work before protocol freeze and again before publication; revise naming and claims if the landscape changes.

The project will use a small synthetic benchmark for mechanistic analysis and a controlled real-image benchmark for external validity. A larger natural-spuriousness dataset is a stretch goal, not part of the minimum viable study.

## 2. Research Positioning

### 2.1 Central question

> Can a general-purpose LLM, operating under a strict no-gradient protocol, diagnose and repair visual classifiers by directly editing their weights—and do those edits remain effective under function-preserving reparameterizations and unseen distribution shifts?

### 2.2 What is actually novel

The novelty is **not** “looking at visual-model weights” or “editing a vision model.” The novelty candidate is evaluating a *general-purpose, natural-language-directed LLM as the weight editor*, without training a task-specific hypernetwork or using gradients, under a reproducible and adversarially controlled protocol.

This distinction must appear in the README, report, résumé, and interviews.

### 2.3 Prior art that constrains the claims

- Takehi et al., *Sparks of Weight-Space Programming* (2026): the direct inspiration and protocol precedent.  
  <https://rikiyat.github.io/blog/weight-space-programming/>
- Păduraru et al., *WASP: A Weight-Space Approach to Detecting Learned Spuriousness* (2024/2025): already detects learned spurious correlations through weight analysis.  
  <https://arxiv.org/abs/2410.18970>
- Huang et al., *Model Editing for Vision Transformers* (NeurIPS 2025): already edits visual transformers and evaluates edit success, generalization, and locality.  
  <https://papers.neurips.cc/paper_files/paper/2025/file/6155cfb4e650d41167848fe2ead30ab8-Paper-Conference.pdf>
- Ainsworth et al., *Git Re-Basin* (2022): establishes why functionally equivalent networks can occupy different weight coordinates.  
  <https://arxiv.org/abs/2209.04836>
- Han et al., *A Survey of Weight Space Learning* (2026): broader taxonomy and related work.  
  <https://arxiv.org/abs/2603.10090>

The project must compare against relevant baselines and avoid state-of-the-art claims unless the experiments truly support them.

## 3. Falsifiable Hypotheses

The hypotheses will be frozen before the hidden test evaluation.

### H1 — Diagnosis

Given weights, architecture metadata, and no examples, the LLM editor identifies the planted shortcut from a frozen answer vocabulary (`none`, `background-color`, `border`, `corner-patch`, `texture`) above chance across independently trained checkpoints. Open-ended explanations are scored separately and never substitute for the preregistered classification metric.

### H2 — Repair

Given a fixed development set and a limited forward-pass budget, the submitted weight patch improves hidden worst-group accuracy relative to the unedited checkpoint and matched random edits.

### H3 — Locality

The patch preserves clean-distribution accuracy and predictions on examples outside the intended edit scope.

### H4 — Symmetry robustness

Editor performance remains stable after function-preserving channel permutations and positive ReLU-channel rescalings. Failure of H4 is still a meaningful result: it would show coordinate dependence rather than function-level understanding.

### H5a — Controlled scaling

Under the same task, topology, data, and editor protocol, targeted repair remains distinguishable from matched random editing when the CNN grows from 6,164 to 23,844 parameters.

### H5b — Tool-mediated scaling

The tool-mediated editor can transfer from a raw-text small-CNN setting to a larger classifier whose full weights do not fit naturally into the prompt.

## 4. Scope

### 4.1 Tier A: mechanistic benchmark

**Dataset:** `SynthShapes-16`, generated entirely by the repository.

- 16×16 RGB images;
- four classes: circle, square, triangle, cross;
- randomized position, size, rotation, stroke, foreground color, background color, texture, and noise;
- configurable class-correlated shortcuts such as background color, corner patch, border texture, or watermark;
- counterfactual test sets that change one attribute while holding the object constant.

**Model:** approximately 6,000-parameter CNN.

```text
3×16×16
→ Conv(8, 3×3) + ReLU
→ Conv(16, 3×3) + ReLU
→ Conv(32, 3×3) + ReLU
→ global average pooling
→ Linear(32 → 4)
```

The full weights can be serialized as text for the most constrained experiment.

### 4.2 Tier A2: controlled scale bridge

Before changing datasets or editor interfaces, repeat the Tier A protocol with a width-scaled CNN:

```text
3×16×16
→ Conv(16, 3×3) + ReLU
→ Conv(32, 3×3) + ReLU
→ Conv(64, 3×3) + ReLU
→ global average pooling
→ Linear(64 → 4)
```

The model has exactly 23,844 trainable parameters, approximately 3.87 times the Tier A model. The primary condition retains the fixed 64-scalar edit budget; a preregistered 256-scalar subset separates model-scale failure from shrinking editable fraction. Full protocol, gates, call ceilings, and stop rules are defined in `SCALE_BRIDGE_PLAN.md`.

This stage must finish before Tier B. It isolates parameter scale while full serialization remains feasible; it does not count as real-image validation.

### 4.3 Tier B: controlled real-image benchmark

**Dataset:** CIFAR-10 with a controlled shortcut injected only into the training distribution.

Candidate shortcuts:

- class-correlated 3×3 corner patches;
- class-correlated border colors;
- frequency-domain or texture overlays.

Use one primary shortcut and one preregistered transfer shortcut. Do not keep trying shortcuts until one produces a favorable result.

**Model:** ResNet-20 or another compact residual CNN. The editor accesses its weights through the sandbox API rather than receiving every parameter in the prompt.

### 4.4 Tier C: stretch validation

Waterbirds with a publicly documented ResNet or ViT checkpoint. This tier is attempted only after the core study is complete and reproducible. If a ViT is used, RefineViT becomes a required comparison where protocols align.

## 5. Weight-Space Programming Protocol

### 5.1 Editor interface

The LLM receives a natural-language objective and a sandbox exposing only audited operations:

- `describe_model()` — architecture, tensor names, shapes, and dtypes;
- `load_tensor(name)` — read a selected tensor;
- `save_edit(name, tensor)` — save a candidate replacement;
- `forward(batch_id, probes)` — consume one forward-pass unit and return approved outputs or activations;
- `compare_original_and_edit(batch_id)` — paired development comparison, also consuming budget;
- `patch_manifest()` — changed tensors, elements, norms, and hashes;
- `submit()` — irrevocably finalize the patch.

One forward-pass unit contains at most 256 preregistered development examples. A run may not combine units into an unrestricted dataset dump. Activation probes are restricted to named layers and bounded summaries or a capped number of examples; their exact schema is frozen before hidden evaluation.

The implementation uses structured tool calls and an allowlisted tensor-expression layer by default, not arbitrary shell access. Any Python analysis mode runs in an ephemeral Linux container with a read-only base checkpoint, no network, explicit CPU/memory/time limits, and only the submitted patch directory writable. The sandbox has no direct access to hidden evaluation data.

### 5.2 Enforced prohibitions

- no autograd or backward passes;
- no optimizer objects;
- no numerical gradient estimation or automated black-box optimizer;
- no programmatic parameter sweeps, evolutionary search, or repeated perturbations generated by a search algorithm;
- no unrestricted dataset iteration;
- no access to hidden test labels or test metrics;
- no editing embeddings or output labels merely to relabel the task unless that condition is explicitly being studied.

The LLM may form a hypothesis, make a reasoned edit, inspect an allowed forward-pass result, and revise the edit in the interactive conditions. That feedback loop is part of Weight-Space Programming, but it must remain model-directed rather than delegated to an optimizer. Every proposed edit and observation is logged. A run that violates the protocol is invalid rather than silently repaired.

### 5.3 Editor conditions

- **Primary diagnosis — Direct-W0:** weights and architecture only, one answer, zero forward passes;
- **Primary repair — Interactive-F20+A:** up to twenty model-directed cycles with selected activations available;
- **Ablation — Direct-F1:** one fixed diagnostic forward result, then one submitted patch with no edit-evaluate loop;
- **Ablation — Interactive-F5:** up to five model-directed edit/inspection cycles;
- **Ablation — Interactive-F20:** twenty cycles without activations;
- **Portability comparison:** the frozen Tier A subset is repeated with at least one separately versioned editor, preferably an open-weight model, to expose vendor/model dependence.

Only the two primary conditions run across the powered checkpoint set. The three ablation conditions and portability comparison run on a frozen representative subset selected before their outcomes are known. Tier B runs only the primary repair condition plus the required baselines unless the resource envelope explicitly supports more.

The primary editor and reasoning effort are fixed before hidden testing. The returned provider model identifier, request ID, date, token usage, latency, and cost are logged because model aliases can change. “Gradient-free” always describes the editor protocol, not the comparison baselines and not the original classifier training.

## 6. Experimental Controls

### 6.1 Required baselines

1. Unedited checkpoint;
2. Random patch matched to the LLM patch's tensor locations, sparsity, and delta norm;
3. Activation-guided channel ablation without an LLM;
4. Final-layer-only repair;
5. Standard fine-tuning using the same development examples;
6. Full or partial fine-tuning with an explicitly reported compute budget;
7. Relevant published editor where architecture, dataset, and protocol make the comparison valid.

Gradient baselines are not falsely described as having the same information budget: backward passes provide additional information. Report data, forward, backward, wall-clock, and dollar budgets separately.

### 6.2 Symmetry controls

For every selected checkpoint, construct verified function-preserving variants:

- channel permutation with the inverse permutation applied to the next layer;
- positive rescaling of a ReLU channel's incoming weights with inverse rescaling of its outgoing weights;
- tensor serialization order changes;
- optional canonicalized/re-basined version.

Before using a variant, test numerical equivalence over a large deterministic input batch and enforce a maximum logit error tolerance.

### 6.3 Anti-cherry-picking controls

- Generate checkpoint IDs randomly and hide semantic names from the editor.
- Freeze prompts, budgets, metrics, and exclusion rules before hidden evaluation.
- Separate development and hidden test generators/seeds.
- Allow the editor to submit only once per run.
- Evaluate every preregistered run, including failures.
- Publish all valid run manifests, not only successful examples.
- Use at least three independent editor runs per checkpoint to measure stochastic variance.
- Freeze the diagnostic answer vocabulary and prompt parser before any hidden checkpoint is scored.
- Strip optimizer state, semantic filenames, training configuration, and shortcut labels from every editor-visible checkpoint package.

## 7. Statistical Plan

### 7.1 Primary endpoint

Paired change in hidden worst-group accuracy between the submitted WSP patch and the unedited checkpoint.

### 7.2 Secondary endpoints

- average and clean accuracy;
- edit success, generalization, and locality;
- expected calibration error;
- regression rate on previously correct examples;
- edit sparsity and relative delta norm;
- performance under symmetry transforms;
- causal feature evidence: change in predictions and internal responses on matched counterfactual image pairs;
- tokens, latency, wall-clock compute, and dollar cost.

### 7.3 Sample size

Do not assume that five seeds are sufficient. Run a pilot with no hidden-test access, estimate the variance across checkpoint seeds and editor replicates, then perform a simulation-based power analysis. Choose the number of checkpoints required for at least 80% power to detect the preregistered minimum practical effect.

If resource limits prevent the powered sample size, label the study exploratory and report uncertainty rather than overstating conclusions.

### 7.4 Practical-success criterion

A headline repair result requires all of the following on the primary benchmark:

- mean worst-group improvement of at least 10 percentage points;
- a paired 95% confidence interval excluding zero;
- clean-accuracy loss no worse than 2 percentage points;
- superiority to matched random edits and the no-LLM ablation baseline;
- replication across independently trained checkpoints;
- no protocol violations.

Failure to meet this criterion does not make the repository a failure. A rigorous negative result with diagnosed failure modes is still the intended research output.

## 8. Engineering Architecture

```text
Dataset generator
    → checkpoint trainer
    → checkpoint registry and hashes
    → shortcut/failure evaluator
    → audited WSP editor sandbox
    → immutable patch artifact
    → blind evaluation runner
    → statistical report and figures
```

### 8.1 Repository target

```text
weight-space-programming-for-vision/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── uv.lock
├── Makefile
├── configs/
│   ├── data/
│   ├── models/
│   └── experiments/
├── src/wsp_vision/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── editor/
│   ├── symmetries/
│   ├── evaluation/
│   └── reporting/
├── prompts/
├── tests/
├── runs/                 # machine-readable manifests; large blobs excluded
├── results/
├── figures/
├── demo/
├── docs/
│   ├── protocol.md
│   ├── dataset-card.md
│   ├── model-card.md
│   ├── reproducibility.md
│   └── limitations.md
└── report/
```

### 8.2 Quality gates

- typed Python and focused docstrings;
- unit tests for tensor serialization, patch application, budget enforcement, and edit auditing;
- property tests for function-preserving transformations;
- deterministic smoke experiment in CI;
- one-command local reproduction of a small result;
- locked dependencies and recorded hardware/software metadata;
- checkpoint, prompt, data-config, patch, and result hashes;
- no API keys, hidden data, or proprietary reasoning traces committed.

## 9. Twelve-Week Execution Plan

### Week 1 — Protocol and literature

- Create the public repository and project board.
- Write the related-work matrix and explicit novelty statement.
- Re-run a dated novelty/name search and record potentially overlapping repositories and papers.
- Freeze Tier A dataset schema, model, metrics, and failure definitions.
- Implement configuration validation and experiment manifests.
- Establish a written resource envelope: available hardware, storage, API quota, maximum spend, and maximum number of hidden runs.

**Gate:** A reviewer can explain how the project differs from WSP, WASP, Git Re-Basin, and RefineViT.

### Week 2 — Data and training foundation

- Implement deterministic SynthShapes-16 generation and counterfactual pairs.
- Train initial clean and shortcut-biased CNNs.
- Build checkpoint registry, hashes, and baseline evaluation.

**Gate:** Repeated generation and training with the same seed reproduce the documented outputs within tolerance.

### Week 3 — Editor sandbox

- Implement audited tensor access, edit saving, forward-pass budgets, final submission, and patch manifests.
- Implement the declarative tensor-edit language and containerized Python fallback.
- Block gradients, optimizers, network access, unrestricted file access, and hidden-test access.
- Add adversarial tests that attempt prohibited operations.

**Gate:** Protocol-violation tests fail closed and produce an audit record.

### Week 4 — Pilot study

- Run diagnosis and repair pilots only on development checkpoints.
- Stabilize prompts and output schemas.
- Measure across-checkpoint and across-editor variance.
- Conduct simulation-based power analysis and freeze the final protocol.

**Gate:** Preregistration document committed and tagged before hidden testing.

### Week 5 — Tier A study

- Execute the primary Direct-W0 diagnosis and Interactive-F20+A repair conditions across the powered checkpoint set.
- Execute Direct-F1, Interactive-F5, Interactive-F20, and editor portability only on the frozen ablation subset.
- Run at least three editor replicates per checkpoint.
- Preserve all valid failures and costs.

**Gate:** Results regenerate from immutable manifests without manual spreadsheet edits.

### Week 6 — Symmetry study

- Implement and verify channel permutation and ReLU-rescaling transforms.
- Repeat the primary editor conditions on equivalent networks.
- Analyze coordinate dependence and optional canonicalization.

**Gate:** Every transformed checkpoint passes numerical equivalence tests before editor evaluation.

### Week 7 — Tier A2 implementation

- Parameterize the CNN widths and checkpoint architecture manifests without changing the 6K default.
- Implement the exact 23,844-parameter model, dynamic diagnostics, prompt accounting, and generalized symmetry tests.
- Train and qualify five base checkpoints using only base-model metrics.
- Freeze and tag the Tier A2 preregistration before any real editor call.

**Gate:** All entry requirements in `SCALE_BRIDGE_PLAN.md` pass, and the 6K regression suite remains green.

### Week 8 — Tier A2 study

- Run WSP-64 and matched random controls across five checkpoints.
- Run the two frozen symmetry conditions and two WSP-256 budget ablations.
- Preserve every result and write the scale conclusion before changing datasets.

**Gate:** The two-size result table, token/cost accounting, limitations, and all declared runs are complete.

### Week 9 — Tier B implementation

- Build CIFAR-10 shortcut injection and counterfactual test splits.
- Train or reproduce compact residual checkpoints.
- Adapt the editor to tool-mediated weight inspection.

**Gate:** The shortcut produces a stable, quantified worst-group failure across seeds without destroying ordinary accuracy.

### Week 10 — Tier B study

- Run the powered subset of editor and baseline conditions.
- Test whether Tier A findings transfer.
- Do not add new favorable conditions after seeing hidden results.

**Gate:** Complete blind evaluation with signed/hash-identified result artifacts.

### Week 11 — Analysis and report

- Produce paired plots, confidence intervals, cost/quality curves, locality analysis, and edit visualizations.
- Write a six-to-eight-page technical report with limitations and negative results.
- Conduct a leakage, licensing, and reproducibility audit.
- Re-run the novelty review and narrow any claim invalidated by intervening work.

**Gate:** Every number in the report traces to a script-generated table or figure.

### Week 12 — Portfolio polish

- Build a two-minute demo showing diagnosis, submitted patch, changed tensors, and before/after predictions.
- Add an architecture diagram, dataset card, model card, reproducibility guide, and interview FAQ.
- Ask at least two technically strong reviewers to reproduce the smoke experiment and critique the claims.
- Pin the repository on GitHub and publish a concise technical article.

**Gate:** A fresh machine can run the documented smoke experiment, and external reviewers find no unsupported claim.

## 10. Scope Control and Fallbacks

### Core deliverable

Tier A, symmetry tests, rigorous baselines, the audited harness, complete results, and the report. Tier A2 becomes part of the core path only if its Stage 1 signal gate passes. A gate-triggered stop with a complete null-result report is a finished project, not an unfinished attempt.

### Required deliverable for a positive practical claim

Core deliverable plus the gated Tier A2 scale bridge, a preregistered Tier B controlled real-image validation, and the editor-portability comparison. Without Tier B, the repository may still be presented as a completed controlled study, but it must not claim practical transfer to real image models.

### Stretch target

Waterbirds or ViT validation with a relevant published editor baseline.

### Stop rules

- Do not start Tier C before the core report is reproducible.
- Do not start Tier B before the Tier A2 scale conclusion is frozen.
- Do not publicly headline a synthetic-only result as evidence of practical vision-model repair.
- If Direct-W0 cannot beat chance, continue with the preregistered direct and interactive conditions; do not rewrite the task after each failure.
- If repair fails, analyze whether the bottleneck is diagnosis, edit construction, forward budget, symmetry, or model scale.
- If proprietary-model reproducibility is weak, add an open-weight editor comparison rather than hiding the limitation.
- Do not expand into detection, segmentation, medical imaging, or large multimodal models during the ten-week core project.

## 11. Portfolio and Hiring Signal

### 11.1 What the repository should demonstrate

- **Research judgment:** accurate novelty claims, falsifiable hypotheses, appropriate baselines, and honest uncertainty;
- **ML engineering:** deterministic training/evaluation, checkpoint handling, inference instrumentation, and numerical testing;
- **Agent systems:** sandbox design, tool budgets, structured outputs, auditability, and failure recovery;
- **Statistics:** paired experiments, power analysis, confidence intervals, and stochastic-run variance;
- **Communication:** a polished README, technical report, demo, diagrams, and concise conclusions.

### 11.2 README opening structure

1. One-sentence question;
2. one honest headline result;
3. before/after visual with uncertainty;
4. method diagram;
5. exact reproduction command;
6. limitations visible above the fold;
7. links to report, protocol, data card, and results.

### 11.3 Résumé bullets

Do not write numerical claims until results exist. Use this structure:

> Designed and open-sourced **WSP-VisionBench**, an audited benchmark for LLM-guided, gradient-free editing of CNN weights across **N** independently trained checkpoints, with blind evaluation, symmetry stress tests, and reproducible experiment manifests.

If the repair result succeeds:

> Improved worst-group image-classification accuracy by **X percentage points** while modifying **Y%** of weights and limiting clean-accuracy regression to **Z points**, outperforming matched random and no-LLM editing controls.

If the result is negative:

> Demonstrated that apparent weight-space repair gains collapsed under function-preserving channel permutations, isolating coordinate dependence through **N** controlled runs and publishing the complete negative benchmark.

The negative-result bullet can be as impressive as the positive one if the evidence and engineering are strong.

## 12. Critical Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| The editor simply exploits labels or prompt metadata | False appearance of weight understanding | blinded IDs, metadata ablations, weights-only condition |
| Results depend on one lucky checkpoint or editor run | Non-reproducible headline | powered seed count and editor replicates |
| Random edits perform similarly | Claimed reasoning advantage disappears | matched random controls; report honestly |
| Equivalent parameterization breaks the method | Coordinate-dependent behavior | make symmetry testing a primary contribution |
| Fine-tuning dominates | WSP lacks practical value | report quality/cost/locality tradeoff, not only accuracy |
| Proprietary model changes | Reproduction becomes unstable | log model IDs/dates; include an open-weight comparison if feasible |
| Synthetic benchmark looks toy-like | Weak hiring signal | Tier B real-image validation and production-grade harness |
| Wider-model failure is caused by a smaller editable fraction | Ambiguous scale conclusion | fixed 64-scalar primary condition plus preregistered 256-scalar ablation |
| Scope expands uncontrollably | No polished deliverable | enforce core/strong/stretch gates and stop rules |
| Prior art overlaps the claim | Credibility damage | explicit positioning against WASP, RefineViT, and weight-space learning |
| The original authors release a general benchmark first | “Benchmark” contribution loses novelty | differentiate on vision counterfactuals, symmetry invariance, and locality; revalidate claims twice |
| GitHub is polished but science is weak | Superficial portfolio | external technical review and trace every claim to evidence |

## 13. Definition of Done

The project is complete only when all core items below are verified:

- [ ] Tier A dataset and counterfactual generator are deterministic and documented.
- [ ] Tier A2 evaluates the exact 23,844-parameter model under fixed and proportional patch budgets if its entry gate passes; otherwise the gate-triggered stop is documented.
- [ ] Tier B controlled real-image validation is complete before any practical-transfer claim; otherwise it is explicitly deferred and no such claim is made.
- [ ] Multiple independently trained checkpoints are included according to the power plan.
- [ ] The editor sandbox enforces and audits the no-gradient protocol.
- [ ] Required baselines and all preregistered editor conditions are executed.
- [ ] At least one separately versioned editor has run the frozen portability subset.
- [ ] Function-preserving symmetry transforms pass numerical equivalence tests.
- [ ] Hidden evaluation is run once under the frozen protocol.
- [ ] Every result is reproducible from machine-readable manifests.
- [ ] Statistics include uncertainty and editor stochasticity.
- [ ] All valid runs, including failures, are published or accounted for.
- [ ] The report makes no unsupported novelty, causality, understanding, or state-of-the-art claims.
- [ ] A fresh-machine smoke reproduction succeeds.
- [ ] At least two external technical reviewers have provided feedback and material issues are resolved or documented.
- [ ] README, demo, report, data/model cards, licensing, and profile presentation are complete.

## 14. Decisions Explicitly Rejected

- A four-network reproduction presented as original research;
- a synthetic-shapes-only demo with no real-image validation target;
- one successful seed or hand-selected edit;
- claiming that the LLM “understands” weights without symmetry and metadata controls;
- comparing accuracy without locality, regression, or cost;
- treating a proprietary-model result as perfectly reproducible;
- starting with a large ViT before the benchmark and harness work;
- jumping from the 6K synthetic model directly to CIFAR-10 without isolating parameter scale;
- promising employment outcomes from a project title alone.

This plan is designed to produce an artifact that is credible in research, ML engineering, and agent-systems interviews—not merely visually impressive on GitHub.
