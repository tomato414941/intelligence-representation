# GridWorld Held-Out Generalization

## Issue

The GridWorld step predictor can fit the generated transition table, but it
still fails to generalize next-cell prediction to held-out current agent cells.

This is relevant because action-conditioned future prediction is one of the
project's stronger evidence targets. A model that only fits seen agent cells is
useful as a training-path smoke check, but it is not evidence of robust
world-modeling generalization.

## Local Check

Checked locally on 2026-05-03 with CPU runs.

| Condition | Train Cases | Eval Cases | Train Next-Cell Accuracy | Eval Next-Cell Accuracy |
| --- | ---: | ---: | ---: | ---: |
| no held-out | 25 | 0 | 0.8000 | none |
| held-out `(0, 2)` | 20 | 5 | 0.9000 | 0.0000 |
| held-out `(0, 2)`, 1000 steps | 20 | 5 | 0.9000 | 0.0000 |
| held-out `(0, 0)` | 20 | 5 | 0.9000 | 0.0000 |
| held-out `(1, 0)` | 20 | 5 | 0.9500 | 0.2000 |

Metrics were written under `runs/local-checks/` and are treated as local
generated artifacts.

## Current Interpretation

The current evidence document is not stale for this result. The held-out
`(0, 2)` result still reproduces with the current code.

Increasing training from 200 to 1000 steps did not improve held-out `(0, 2)`
next-cell accuracy, so the failure is probably not just a short-training issue.
Likely causes include:

| Area | Question |
| --- | --- |
| input representation | Does the model receive position information in a form that supports rule-like extrapolation? |
| capacity or optimization | Can a slightly different model or schedule learn the transition rule? |
| split design | Is holding out a full agent cell too strict for the current tiny table? |
| evaluation target | Should next-cell prediction be complemented with action-sensitive ranking diagnostics? |

## Candidate Direction

Keep the existing result as a limitation. Before changing architecture broadly,
run a narrow diagnostic that separates memorization from transition-rule
learning.

Possible next checks:

| Check | Purpose |
| --- | --- |
| per-action predictions for held-out cells | See whether errors are systematic or arbitrary. |
| simpler non-Transformer baseline | Test whether the task construction itself supports generalization. |
| explicit coordinate features | Test whether position encoding is the bottleneck. |
| multiple held-out cells and seeds | Avoid over-reading one split or initialization. |

## Non-Goal

Do not treat this as a reason to add a broad world-model abstraction. The next
step should be a small diagnostic or baseline tied directly to the current
GridWorld transition task.
