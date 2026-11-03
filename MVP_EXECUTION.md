# MVP Execution Plan

## Definition of done

The project is complete when the public repository contains:

1. Five independently trained shortcut-biased checkpoints.
2. A frozen prompt, development seed, hidden seed, and 64-scalar patch limit.
3. One valid LLM patch and one matched random patch per checkpoint.
4. An unedited, LLM, random, and fine-tuning result for every seed.
5. The same editor run on at least two function-preserving channel permutations.
6. A single aggregate table with uncertainty, API cost, runtime, and all failures.
7. A short report that states whether the hypothesis was supported.

Anything beyond that list is optional.

## Eight-week schedule

### Week 1 — Verify the failure mode

- Install Python 3.11 and project dependencies.
- Run the tests and local-study for seed 0.
- Confirm high correlated accuracy and a clear counterfactual gap.
- Inspect generated tensors and catch label leakage or broken counterfactual pairing.

Stop rule: do not call the API until the base model reliably exhibits the intended failure.

### Week 2 — Freeze the protocol

- Choose five checkpoint seeds.
- Freeze development and hidden seeds.
- Freeze the prompt, editor model, reasoning effort, patch size, and delta bound.
- Write the expected outcome and minimum useful counterfactual improvement before seeing editor results.

### Week 3 — Run base checkpoints

- Train and evaluate all five checkpoints.
- Run the fine-tuning comparison.
- Record wall time, machine, package versions, and checkpoint hashes.

### Week 4 — Pilot one editor run

- Request one patch on seed 0.
- Validate and apply it.
- Confirm that no hidden metrics entered the request.
- Check that a malformed or oversized patch fails closed.

Stop rule: one prompt-format repair is acceptable; repeated prompt tuning against hidden performance is not.

### Week 5 — Run the frozen editor study

- Run the remaining checkpoints without changing the prompt.
- Generate one matched random patch per LLM patch.
- Record API request IDs, model identifiers, token usage, latency, and cost.
- Preserve unsuccessful and refused runs.

### Week 6 — Symmetry test

- Verify channel-permuted checkpoints are numerically equivalent.
- Run the frozen editor on two permuted checkpoints.
- Compare patch quality with the original coordinates.

### Week 7 — Analyze

- Aggregate seed-level accuracy deltas and regression rates.
- Add bootstrap confidence intervals or clearly label the study exploratory.
- Categorize which tensors and channels were changed.
- Write a failure analysis even if the main result is positive.

### Week 8 — Package for hiring

- Add one architecture figure and one results plot.
- Record a 60–90 second terminal demo.
- Write a two-page report and concise README result summary.
- Publish and pin the original GitHub repository.
- Practice explaining why the controls matter in under two minutes.

## Scope locks

Do not add these until the definition of done is met:

- CIFAR-10 or Fashion-MNIST;
- a ResNet or vision transformer;
- a second LLM provider;
- an interactive 20-step agent loop;
- a web dashboard;
- hyperparameter sweeps;
- claims of a new state of the art.

The first acceptable stretch goal is one additional shortcut type on the same dataset and architecture.

## Decision rubric

| Outcome | Interpretation | Good portfolio response |
|---|---|---|
| LLM beats random and preserves clean accuracy | Evidence worth replicating | Add editor repeats and confidence intervals |
| LLM helps but trails fine-tuning | Plausible but inefficient method | Analyze information and compute tradeoffs |
| LLM matches random | No evidence of targeted repair | Publish the null result and patch analysis |
| LLM breaks under channel permutation | Coordinate-sensitive editing | Lead with the symmetry finding |
| LLM damages clean accuracy | Repair lacks locality | Show regression analysis and safer constraints |

A well-controlled negative result is a finished project. Endless prompt tuning is not.

