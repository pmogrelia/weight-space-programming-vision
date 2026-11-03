# Local Verification Record

This record is a runtime check of the MVP, not the final research result.

## Environment

- Date: November 3, 2026
- Hardware: Apple Silicon CPU execution
- Python used for final verification: 3.11.17
- PyTorch: 2.15.0
- NumPy: 2.4.6
- OpenAI SDK: 3.43.0

The repository pins Python 3.11 as its reproduction target.

## Checks completed

- Eight unit/integration tests passed, including a mocked structured editor request.
- Ruff formatting and lint checks passed.
- OpenAI Responses structured parsing was confirmed available in the installed SDK.
- No OpenAI request was made because no API key was supplied.
- A full default local-study completed on CPU in approximately 22 seconds.

## Seed-0 baseline smoke result

| Model | Correlated accuracy | Counterfactual accuracy | Neutral accuracy |
|---|---:|---:|---:|
| Shortcut-biased base | 96.78% | 25.68% | 54.20% |
| Random-border fine-tune, 100 steps | 75.20% | 50.10% | 58.50% |

The base checkpoint exhibits the intended failure mode: a 71.09 percentage-point gap between correlated and counterfactual accuracy. The gradient control improves counterfactual accuracy but sacrifices correlated performance, which makes preservation metrics important.

The function-preserving conv1 channel permutation produced a maximum logit difference of 2.98e-6 over 128 deterministic examples.

## What remains unverified

- The LLM editor result, matched random patch, and multi-seed aggregate.
- API cost, token usage, and repeatability.
- Editor behavior on the symmetry-transformed checkpoint.

Those items require an API key and the frozen five-seed study described in MVP_EXECUTION.md.

