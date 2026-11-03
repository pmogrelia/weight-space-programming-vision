# Stage 2 Plan: Weight-Space Programming Scale Bridge

## Decision

After the 6,164-parameter MVP is complete, the next stage will repeat the frozen vision experiment with a **23,844-parameter CNN** on the same dataset and shortcut.

This stage is a controlled scale test, not a second MVP and not yet a real-image generalization claim. Holding the task, data generator, layer count, metrics, and editor objective fixed lets the project ask one interpretable question:

> Does an LLM-directed sparse weight repair remain targeted when the editable network is approximately 3.87 times larger?

The original Weight-Space Programming experiments span roughly 20K-parameter game networks, a 171,664-weight control policy, and a 1.48M-weight Go network. The 23.8K bridge therefore moves this project just beyond the paper's smallest game-writing architecture while remaining small enough for full, auditable weight serialization. See the [original experiment report](https://rikiyat.github.io/blog/weight-space-programming/).

## Position in the project

| Stage | Network | Purpose | Status requirement |
|---|---:|---|---|
| Stage 1 | 6,164 parameters | Prove the harness and establish the first controlled result | Must finish first |
| Stage 2 | 23,844 parameters | Isolate parameter-scale sensitivity | This plan |
| Stage 3 | Compact CNN/ResNet on real images | Test external validity and tool-mediated inspection | Gated on Stage 2 |

Stage 2 does not replace Stage 1. Stage 1 remains the cheapest debugging surface and the clearest mechanistic result.

## Architecture

Use the same three-convolution topology and change only channel width:

    3×16×16
    → Conv(16, 3×3, padding=1) + ReLU
    → Conv(32, 3×3, padding=1) + ReLU
    → Conv(64, 3×3, padding=1) + ReLU
    → global average pooling
    → Linear(64 → 4)

Exact trainable parameter count:

| Layer | Parameters |
|---|---:|
| Conv 3→16 | 448 |
| Conv 16→32 | 4,640 |
| Conv 32→64 | 18,496 |
| Linear 64→4 | 260 |
| **Total** | **23,844** |

The implementation should parameterize channel widths rather than duplicate the model class. Checkpoint manifests must include the exact architecture configuration so a 6K checkpoint cannot be loaded into the 23.8K model accidentally.

## Measured prompt feasibility

A local preflight using the current five-decimal serializer produced:

- 23,844 parameters;
- 202,177 prompt characters;
- 202,177 UTF-8 bytes;
- approximately 50,544 tokens using a rough four-characters-per-token conversion.

The estimate is not an accounting result. The first valid API response must record the provider's actual input tokens, output tokens, cached tokens, latency, request ID, model identifier, and cost calculation. If the provider rejects the request or the prompt exceeds the declared context budget, Stage 2 stops; it does not silently switch to a different tool-mediated protocol.

## Research hypotheses

### S2-H1 — Targeted repair

At the 23.8K scale, the WSP patch improves hidden counterfactual accuracy more than a patch matched by tensor, sparsity, and delta norm.

### S2-H2 — Locality

The WSP patch preserves correlated-distribution accuracy and does not create a materially larger regression rate than the matched random control.

### S2-H3 — Scale retention

The direction of the WSP-minus-random effect is consistent between the 6K and 23.8K models. Effect magnitude will be reported, but no ratio is interpreted when the 6K effect is near zero.

### S2-H4 — Coordinate robustness

The editor's result remains useful on function-preserving channel permutations of the wider model.

### S2-H5 — Budget sensitivity

Any loss of effectiveness under a fixed 64-scalar budget is distinguishable from a simple reduction in editable parameter fraction by a preregistered 256-scalar ablation.

## Entry gates

Do not spend API credits on Stage 2 until all of these are true:

1. Stage 1 has completed its five checkpoint runs and matched-random controls.
2. Stage 1 shows enough targeted signal to justify scaling: WSP improves counterfactual accuracy on at least three of five seeds and its median paired advantage over matched random is at least five percentage points.
3. Stage 1's prompt, hidden seed, patch validation, and result schema are frozen.
4. The 23.8K architecture passes parameter-count, checkpoint round-trip, patch, and symmetry tests.
5. Five wider base checkpoints are trained with seeds 0, 1, 2, 3, and 4.
6. Their median correlated accuracy is at least 90%.
7. Their median correlated-minus-counterfactual gap is at least 40 percentage points.
8. At least four of five checkpoints have a gap of at least 30 percentage points.
9. No hidden Stage 2 metric has been inspected while choosing architecture or training settings.
10. A maximum API spend and maximum call count are written into the preregistration.

If Stage 1 fails gate 2, stop after the Stage 1 report; do not spend more calls trying to rescue the claim with scale. If the wider base checkpoints do not meet gates 6–8, adjust the training/data design using base-model metrics only, regenerate all five checkpoints, and document the discarded configuration. Do not inspect editor outcomes while tuning the failure mode.

## Frozen information partitions

| Partition | Seed | Editor visibility | Purpose |
|---|---:|---:|---|
| Training | model seed 0–4 | Weights only | Produce independent checkpoints |
| Development diagnostics | 20,000 | Yes | Paired activation summaries and visible metrics |
| Hidden evaluation | 40,000 | Never | Primary result |
| Optional final confirmation | 50,000 | Never until report draft | One-time robustness check |

The same object generator and shortcut definition are used at both model sizes. Only the channel widths change.

## Experiment matrix

### Core conditions: all five checkpoints

| Condition | Calls per checkpoint | Patch budget | Role |
|---|---:|---:|---|
| Unedited | 0 | 0 | Baseline |
| WSP-64 | 1 | 64 scalars | Primary treatment and fixed operational budget |
| Matched random-64 | 0 | Matched to WSP-64 | Causal control for perturbation size |
| Random-border fine-tuning | 0 | Unrestricted gradients | Conventional repair comparison |

Core API total: **five calls**.

### Symmetry subset

Before seeing editor results, select model seeds 1 and 4. Apply a verified function-preserving channel permutation, then repeat WSP-64 with the identical prompt template and diagnostics protocol.

Symmetry API total: **two calls**.

### Proportional-budget subset

Before seeing editor results, select model seeds 2 and 4. Repeat the editor with a 256-scalar limit. This is approximately the same editable fraction as 64 scalars in the 6,164-parameter model:

- 6K model, 64 scalars: approximately 1.04%;
- 23.8K model, 64 scalars: approximately 0.27%;
- 23.8K model, 256 scalars: approximately 1.07%.

Every WSP-256 patch gets its own matched random-256 control.

Budget-ablation API total: **two calls**.

### Optional stochasticity subset

Only if the written spend cap allows it, run two additional editor replicates per seed on seeds 0 and 3, using the same WSP-64 prompt and no prompt modifications.

Optional API total: **four calls**.

The core plus symmetry plus budget design requires **nine API calls**. The strong version with stochasticity requires **thirteen**. These are hard maximums, not targets to exceed.

## Editor protocol

The Stage 2 editor must use the Stage 1 semantic prompt unchanged except for automatically generated architecture shapes, tensor values, channel diagnostics, and the declared patch limit.

Allowed information:

- architecture and tensor names/shapes;
- full serialized weights;
- correlated, counterfactual, and neutral development metrics;
- paired neutral/counterfactual feature-shift summaries;
- classifier-weight norms by feature channel;
- the natural-language repair objective.

Forbidden information:

- hidden seed, hidden predictions, or hidden metrics;
- gradients, optimizer state, training history, or loss derivatives;
- semantic checkpoint filenames;
- prior editor answers from another seed;
- results from a candidate patch before final submission;
- prompt changes based on hidden performance.

Each request produces one irrevocable patch proposal. The patch validator enforces known floating-point tensors, in-range unique indices, finite deltas, the scalar budget, and the existing absolute delta bound. Reports also include global and per-tensor relative delta norms so absolute-bound behavior can be audited across scales.

All declared WSP and matched-random patches must be generated and hash-locked before the hidden evaluator runs. The evaluator then scores the complete batch and reveals the results together. A Stage 2 hidden result may never inform a later Stage 2 prompt or patch.

## Metrics

### Primary endpoint

For each seed:

    counterfactual_accuracy(WSP-64) - counterfactual_accuracy(unedited)

The primary comparison is the paired difference between that value and the matched-random-64 accuracy change.

### Required secondary endpoints

- correlated, counterfactual, and neutral accuracy;
- worst-class accuracy on every split;
- regression rate among examples correctly classified by the base model;
- changed scalar count and fraction;
- per-tensor and global delta L2 norms;
- delta norm relative to the edited tensor and full checkpoint;
- editor input/output tokens, latency, and cost;
- patch layer distribution;
- symmetry-transformed performance;
- WSP-64 versus WSP-256 on the frozen subset.

### Descriptive success threshold

Call Stage 2 a successful scale replication only if all are true:

1. WSP-64 improves counterfactual accuracy on at least four of five seeds.
2. Median WSP-64 counterfactual gain is at least 10 percentage points.
3. Median WSP-minus-random paired advantage is at least 5 percentage points.
4. Median correlated-accuracy loss is no worse than 5 percentage points.
5. No patch violates the frozen validator or information policy.

With five seeds, this remains an exploratory result. Report paired values and bootstrap intervals; do not present a small-sample p-value as definitive proof.

If WSP-64 fails but WSP-256 succeeds, the conclusion is budget sensitivity, not scale invariance. If both fail, report a scale failure. If random matches WSP, report no evidence of targeted editing.

## Statistical and reporting rules

- Analyze by paired checkpoint seed.
- Publish every valid run, including API refusals and harmful patches.
- Report the median, each individual seed, and a bootstrap confidence interval.
- Do not pool Stage 1 and Stage 2 observations as if they were identically distributed.
- Do not compute a scaling-retention ratio when the Stage 1 denominator is below five percentage points.
- Treat editor replicates as nested within checkpoints, not as independent checkpoints.
- Label the study exploratory unless a later power analysis justifies more seeds.

## Implementation work

1. Replace hard-coded feature width 32 in diagnostics with the model's declared feature dimension.
2. Generate the editor's architecture description from the checkpoint manifest instead of the current hard-coded 8→16→32 text.
3. Add architecture configuration to checkpoint manifests and strict loading.
4. Parameterize SmallCNN channel widths while preserving the 6K default.
5. Add an exact 23,844-parameter test.
6. Generalize channel permutation tests to each adjacent convolution pair.
7. Add prompt byte count and provider-token accounting to run metadata.
8. Add relative patch-norm reporting without changing Stage 1 validation semantics.
9. Add a machine-readable experiment matrix and a command that refuses undeclared conditions.
10. Add a batch hidden evaluator that requires a patch hash or a terminal refusal/error record for every declared condition.
11. Add aggregate reporting across both sizes.

No implementation change may alter the already recorded 6K default behavior without an explicit checkpoint-format version.

## Two-week execution schedule

### Days 1–2 — Generalize safely

- Parameterize model widths and manifests.
- Remove hard-coded diagnostic dimensions.
- Add scale, serialization, checkpoint, patch, and symmetry tests.
- Re-run all Stage 1 tests and the seed-0 baseline.

Gate: the default 6K hashes/metrics remain reproducible within the documented tolerance.

### Days 3–4 — Qualify the wider failure mode

- Train five 23.8K checkpoints.
- Evaluate only base distributions.
- Apply the entry gates.
- Record training time and resource use.

Gate: all base-model qualification rules pass before any editor call.

### Day 5 — Freeze and preflight

- Commit the Stage 2 preregistration.
- Freeze prompt hash, seeds, conditions, spend cap, and model identifier.
- Run one schema-only mocked editor test.
- Calculate prompt bytes and verify the request fits the declared context budget.

Gate: tag the preregistration commit before the first real request.

### Days 6–7 — Core editor runs

- Execute WSP-64 across all five seeds.
- Generate matched random controls.
- Hash-lock every declared patch, then apply and evaluate the complete batch on the hidden seed exactly once.
- Preserve request and patch metadata.

### Day 8 — Scale diagnostics

- Run the two symmetry conditions.
- Run the two WSP-256 budget conditions.
- Run optional replicates only if preregistered and within the spend cap.

### Days 9–10 — Analysis and portfolio packaging

- Build the two-size comparison table and plots.
- Write the scale conclusion and strongest counterexample.
- Update the technical report, model card, and limitations.
- Record a short demo that switches between 6K and 23.8K manifests.

## Stop rules

- Do not tune the wider architecture after seeing any Stage 2 editor result.
- Do not begin the full scale study if Stage 1 lacks the preregistered targeted signal.
- Do not change the prompt between 6K and 23.8K except for generated shapes, weights, diagnostics, and budget.
- Do not add a fourth model size; two sizes answer the bridge question.
- Do not move to CIFAR-10 merely because the synthetic result is negative.
- Do not claim practical image-model repair from either synthetic stage.
- Do not switch to an interactive tool loop mid-study if full serialization fails.
- Do not exceed thirteen Stage 2 API calls.
- Do not discard a seed because the patch is unhelpful.
- Do not promote WSP-256 to the primary condition after seeing WSP-64 fail.

## Iterative critique record

### Critique 1 — “Scale directly to CIFAR-10”

**Problem:** This changes parameter count, architecture, image distribution, shortcut salience, and editor interface simultaneously. A failure would be uninterpretable.

**Revision:** Insert the 23.8K same-task bridge before real images.

### Critique 2 — “One wider checkpoint is enough for a portfolio demo”

**Problem:** One seed invites cherry-picking and provides no evidence that the editor targets a learned mechanism rather than a lucky coordinate pattern.

**Revision:** Require five independent checkpoints, paired random controls, and publish all runs.

### Critique 3 — “Reuse the 64-scalar limit and call it a fair scale test”

**Problem:** The editable fraction falls from about 1.04% to 0.27%, so a negative result could reflect budget shrinkage rather than network scale.

**Revision:** Keep WSP-64 as the primary operational-budget condition and preregister WSP-256 on two seeds as a proportional-budget diagnostic.

### Critique 4 — “Run many prompts and keep the best”

**Problem:** This converts the LLM into an unreported hyperparameter search and invalidates the hidden evaluation.

**Revision:** Freeze one semantic prompt, one submission per declared call, fixed subsets, and a thirteen-call hard ceiling.

### Critique 5 — “A wider synthetic model solves the toy-dataset criticism”

**Problem:** Width does not create external validity. The same generator can still reward task-specific reasoning.

**Revision:** Describe Stage 2 only as scale evidence. Keep real-image validation as a separately gated Stage 3.

### Critique 6 — “Matching the original paper's smallest parameter count proves parity”

**Problem:** Parameter count does not equal task difficulty, architecture, information access, or experimental quality.

**Revision:** Use the original paper only to motivate scale. Make claims solely from this repository's controlled comparisons.

### Critique 7 — “Scale the study even if the 6K editor is indistinguishable from random”

**Problem:** Scaling a null treatment adds cost without testing retention of an established effect and encourages a post-hoc rescue narrative.

**Revision:** Require a modest Stage 1 WSP-over-random signal before the full Stage 2 API study. A Stage 1 null result is a legitimate stopping point.

### Critique 8 — “Evaluate each hidden result immediately”

**Problem:** Even with a frozen prompt, seeing early outcomes can influence error handling, optional calls, and interpretation of later submissions.

**Revision:** Generate and hash-lock the complete declared patch batch before a one-time hidden reveal.

### Remaining limitations after revision

- Full weight serialization may encourage superficial numeric heuristics.
- Five checkpoints give limited statistical power.
- A proprietary editor can change over time.
- The synthetic shortcut remains simpler than natural spurious correlations.
- Channel permutations test one symmetry family, not all equivalent parameterizations.
- WSP and gradient fine-tuning do not have equal information budgets.

These are reportable limitations, not reasons to expand scope before completing the stage.

## Definition of done

- [ ] The default 6K model remains backward-compatible.
- [ ] The wider model has exactly 23,844 trainable parameters.
- [ ] Five wider base checkpoints satisfy the frozen qualification gates.
- [ ] The Stage 2 preregistration predates every real editor request.
- [ ] Every declared patch was hash-locked before the batched hidden reveal.
- [ ] Five WSP-64 and five matched-random-64 results are complete.
- [ ] Two verified symmetry runs are complete.
- [ ] Two WSP-256 budget-ablation runs and controls are complete.
- [ ] Every valid run, failure, request ID, token count, latency, and cost is recorded.
- [ ] Aggregate results show individual paired seeds and uncertainty.
- [ ] The report distinguishes fixed-budget, proportional-budget, and symmetry conclusions.
- [ ] No claim treats synthetic scale evidence as real-image external validity.
- [ ] The repository remains reproducible from a fresh Python 3.11 environment.

Only after every applicable item is complete should Stage 3 real-image work begin.

