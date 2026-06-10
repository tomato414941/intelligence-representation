# GridWorld Held-Out Generalization

Status: open.

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

## Per-Action Diagnostic (2026-06-09)

Ran a held-out sweep diagnostic over every valid agent cell with seeds
31/32/33, dumping per-action next-cell predictions for the held-out cell.
Metrics were written under `runs/local-checks/`. The diagnostic tool was
temporary and has been removed after this check; it can be restored from
commit `7db9e1d` (`intrep.problems.grid_step_prediction.diagnose_heldout`).

Two model regimes were checked:

| Model | Train Next-Cell Accuracy | Eval Next-Cell Accuracy |
| --- | ---: | ---: |
| `d256-h1024-heads8-l6` (current CLI default) | 0.20-0.25 (all 15 runs) | 0.00-0.20 |
| `d32-h64-heads2-l1` (CLI default at the 2026-05-03 check) | 0.90-1.00 | 0.00-0.40 |

The 2026-05-03 table above reproduces only with the small model. The current
default `d256-h1024-heads8-l6` (changed in `0e9b794`, 2026-05-05) does not
even fit the 20 train cases with the CLI's default optimization settings
(lr 0.01, 200 steps, batch 5, no warmup), so its eval numbers say nothing
about generalization.

Per-action predictions in the small-model regime (75 held-out predictions)
are systematic, not arbitrary:

| Slice | Accuracy |
| --- | ---: |
| all held-out predictions | 10/75 (0.13) |
| true next cell is a training-visible agent cell | 7/24 (0.29) |
| true next cell is the held-out cell itself (blocked or stay) | 3/51 (0.06) |

17 of 25 held-out (cell, action) cases have "stay at the current cell" as the
true answer (edge/wall blocks plus `stay`), but the model predicted the
current cell in only 4 of 75 predictions. The model avoids emitting the
held-out cell id even though that cell still appears as a next-cell target
for inbound training transitions. Errors often keep the action's general
direction but mislocate the agent (for example held-out `(1, 2)` with
`right` predicts `(0, 2)` in all three seeds).

This points at the output formulation: next-cell prediction as a
classification over absolute cell ids requires the model to associate "agent
channel active at X" with "output class X" separately per cell, so the
blocked/stay copy rule cannot transfer to an agent cell never seen as the
current cell during training. Optimization is not the bottleneck in the
small-model regime (train accuracy 0.90-1.00).

## Current Interpretation

The current evidence document is not stale for this result. The held-out
`(0, 2)` result still reproduces with the current code in the small-model
regime.

Increasing training from 200 to 1000 steps did not improve held-out `(0, 2)`
next-cell accuracy, so the failure is probably not just a short-training issue.
The 2026-06-09 per-action diagnostic narrows the likely causes: in the regime
where training fits, errors concentrate on emitting the held-out cell id as
output, which implicates the absolute-cell-id classification target rather
than capacity or optimization.

Earlier candidate causes for reference:

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

| Check | Status |
| --- | --- |
| per-action predictions for held-out cells | Done 2026-06-09: errors are systematic; the model avoids emitting the held-out cell id. |
| multiple held-out cells and seeds | Done 2026-06-09: all 5 cells x 3 seeds; eval accuracy 0.00-0.40, mean near chance. |
| simpler non-Transformer baseline | Open. |
| explicit coordinate features | Open. The diagnostic suggests testing a relative-move or coordinate-delta output target before input-side changes. |

## Non-Goal

Do not treat this as a reason to add a broad world-model abstraction. The next
step should be a small diagnostic or baseline tied directly to the current
GridWorld transition task.
