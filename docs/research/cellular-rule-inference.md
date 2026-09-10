# Unseen Cellular Rule Inference

This is a bounded experiment in learning from observations at inference time.
The [noise and regime-change follow-up](cellular-rule-stress.md) extends this
protocol with training-seed replication and contradictory or stale evidence.
It follows the single-rule Life result in
[the previous experiment](../../issues/closed/grid-next-observation-emergence.md).
It is an experimental problem, not a new general-purpose world/model framework.

## Question

After training across cellular rules, can a fixed model predict a new rule's
next board using only examples of that rule's before/after observations?

Single-rule learning and in-context inference are different claims. The old
Life checkpoint learned a rule in its weights. Here the evaluation rule never
appears in training, and evaluation never updates weights.

## Declared Structure

- The source world is the existing binary outer-totalistic cellular family,
  with dead borders and birth on zero neighbors excluded.
- A demonstration is a before/after pair of full boards. Demonstrations are
  independent initial states under one rule, not necessarily a trajectory.
- Each token contains the two values of an aligned cell pair, a learned spatial
  position, an example index, and a demonstration/query role.
- Query tokens have only the current cell value; the second channel is zero.
- The unchanged `SharedTransformerCore` reads the combined tokens. A linear
  head predicts each query cell. The standard model is d256/h1024/heads8/l6.
- Rule IDs, birth/survival sets, neighbor counts, locality constraints, and
  symbolic lookup results never enter the neural model.

The paired cells, board geometry and example grouping are injected structure.
This experiment does not claim to discover the input format or an arbitrary
unstructured physical law.

## Data And Splits

- Training: 256 distinct rules; rule seed 1701. New episodes are sampled during
  training using data seed 3101. The number of demonstrations is drawn from
  `[0, 1, 1, 4, 4, 8, 8, 8]`.
- Default boards: 6x6. Each board independently draws initial alive probability
  from 0.2, 0.5 or 0.8. Demonstration inputs identical to the query are rejected.
- Validation: 32 unseen rules, four query boards each; seeds 9101/12001.
- Final test: 64 unseen rules, eight query boards each; seeds 19101/23001.
  Both training and validation rules are excluded by actual birth/survival
  identity, not just by random seed. Test results do not choose checkpoints.
- All context counts use the same query boards and nested demonstration prefixes.
- Checkpoints record explicit training rules, configuration and RNG/optimizer
  state. Evaluation records explicit test rules, exclusions and checkpoint hash.

There is no canonical downloaded dataset. Episodes are generated from the
declared configuration. Run outputs are disposable; retain a needed checkpoint
outside `runs/` before cleaning runs.

## Controls And Metrics

The main curve compares 0, 1, 4 and 8 demonstration pairs on held-out rules.

- Correct context: demonstrations and query target use the same unseen rule.
- Wrong context: keep all initial boards fixed, replace demonstration outputs
  with outputs from a different unseen rule, and score against the original target.
- Donor target: score those same wrong-context predictions against the donor
  rule's query target. This measures adaptation, not just sensitivity to corruption.
- Frequency baseline: estimate next-cell frequency conditional only on the
  current cell's value, ignoring spatial arrangement and neighbor counts.
- Rule-family lookup: an analysis-only baseline uses known neighbor counts to
  recover observed rule-table entries. Unobserved entries default to zero.

Report changed/unchanged cell accuracy so copying or flipping cannot masquerade
as success. Also report accuracy separately on query conditions observed and
unobserved in the demonstrations. A condition is `(current cell, live neighbor
count)`: this oracle assumes the known rule family and is not a model input.
An unobserved condition is not always ambiguous (birth on zero is fixed by the
family), so coverage is explicitly evidence coverage, not a universal uncertainty
estimate. The lookup is exact on covered cells by construction.

Confidence intervals resample whole rules, not correlated cells. Context gains
and correct-minus-wrong differences use paired rule-level bootstrap intervals.
These intervals cover test-rule sampling variation, not training-seed variation.

Improvement over zero context is useful but insufficient: global output
frequency alone can help. Compare against the geometry-free frequency baseline
and require both changed and unchanged cells to improve before making stronger
rule-inference claims.

## Reproduction

```sh
RUN_DIR=runs/cellular-rule-inference-seed31 DEVICE=cuda \
  bash scripts/run_cellular_rule_inference_experiment.sh

uv run --no-project --with matplotlib python scripts/plot_cellular_rule_inference.py \
  --evaluation runs/cellular-rule-inference-seed31/test.json \
  --output-prefix runs/cellular-rule-inference-seed31/context-curve
```

The shell script trains for 6000 steps, evaluates validation, then the separate
final test. Override `MODEL_SEED`, `TRAINING_STEPS`, `DEVICE` and `PYTHON` when
needed, recording any deviations. RunPod transport uses the existing
`scripts/runpod/run_command.sh`, with this script in `SYNC_PATHS` and the actual
workload in `REMOTE_COMMAND`.

For a continued training run use the training CLI's `--resume` with the same
configuration and a larger `--max-steps`. Checkpoints retain optimizer and RNG
state; the learning-rate schedule is linear warmup followed by a constant rate.

## Results

Measured 2026-09-09 with the protocol above, model seed 31, learning rate
0.0003, batch 16 and 6000 steps. No checkpoint or hyperparameter was selected
using final-test results. Training used CUDA bfloat16 autocast; final evaluation
used float32. The final test contains 512 query boards / 18,432 cells.

| Demonstrations | Correct context | Wrong context / original target | Frequency baseline | Family-aware lookup | Evidence coverage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 54.09% | 54.09% | 54.09% | 54.09% | 0.00% |
| 1 | 83.37% | 55.16% | 61.45% | 84.66% | 64.24% |
| 4 | 97.07% | 55.26% | 64.16% | 98.25% | 95.56% |
| 8 | 98.86% | 55.47% | 64.92% | 99.70% | 99.34% |

At eight demonstrations:

- Correct-context accuracy: 98.86%, rule-bootstrap 95% CI 98.62–99.08%.
- Paired gain over zero context: +44.77 percentage points, CI +41.10–48.32.
- Correct-minus-wrong context: +43.39 points, CI +39.61–47.05.
- Donor-target accuracy after replacing demonstration outputs: 98.71%.
  Predictions follow the demonstrated alternate rule, not simply the original
  query or the original rule.
- Changed-cell accuracy 98.77%, unchanged-cell accuracy 98.94%.
- Evidence-covered cell accuracy 99.13%; uncovered cell accuracy 58.20%.
  Only 122 cells remain uncovered at eight demonstrations, so that last
  percentage has very limited support.

Interpretation: within this cellular family, the model learned to use observed
transitions to predict under unseen rules without weight updates. Its advantage
over the frequency baseline supports spatially specific inference. More
demonstrations mainly expose more of the local conditions needed for the query;
already-covered conditions are predicted well even with one demonstration
(98.07%). This is not evidence of predicting unconstrained, unobserved rule bits.

Limits: one training seed, one rule family, 6x6 boards, fully observed states,
and explicitly aligned before/after cell pairs. Cross-family inference, new
board sizes, noisy or partial observations, and training-seed robustness remain
unmeasured. The oracle's family knowledge must not be attributed to the model.

The checkpoint, configuration, training log, validation/test JSON, timing
records and PNG/PDF/SVG/CSV figure are preserved locally under
`models/cellular-rule-inference-20260909-seed31/`; generated artifacts are not
versioned. Checkpoint SHA256:
`76b03efd956697b470623c8aac19c638f10e2c26d8ac5ae04fea18bdeb37ed29`.

Implementation checks: 466 unit tests passed, including eight new tests for
world parity, rule-identity splits, query leakage, coverage, counterfactual
contexts, family lookup, context gradients and exact CPU resume. Ruff and shell
syntax checks passed. The figure was rendered and visually inspected.
