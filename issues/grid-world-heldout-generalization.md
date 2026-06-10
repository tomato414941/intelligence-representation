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

## Representation And Data-Support Experiment (2026-06-10)

Crossed two factors in a 2x2 design with the small model (`d32-h64-heads2-l1`),
holding out one agent cell at a time across seeds 31/32/33:

- input: learned absolute position embeddings (current) vs explicit
  normalized row/col coordinate channels with no learned position embedding
- output: absolute next-cell-id classification (current) vs relative-move
  classification (5 delta classes, decoded back to a next cell for the same
  next-cell accuracy metric)

The experiment tool was temporary and has been removed after this check; it
can be restored from commit `5674974`
(`intrep.problems.grid_step_prediction.heldout_representation_experiment`).
Metrics were written under `runs/local-checks/`.

On the original 2x3 grid (20 train cases, all 5 cells held out in turn):

| Input | Output | Train | Eval | Eval stay cases | Eval move cases |
| --- | --- | ---: | ---: | ---: | ---: |
| absolute | absolute | 0.85 | 0.11 | 0.06 | 0.21 |
| absolute | relative | 0.93 | 0.45 | 0.63 | 0.08 |
| coordinates | absolute | 0.73 | 0.17 | 0.14 | 0.25 |
| coordinates | relative | 0.85 | 0.52 | 0.65 | 0.25 |

Caution: 17 of 25 held-out answers are "stay", so a trivial always-stay
predictor scores 0.68 eval here. At 20 train cases the relative-output gain
comes mostly from defaulting to "stay" at unseen cells, not from learning the
move/block rule (move-case accuracy 0.08-0.25).

On a 4x5 grid (90 train cases; held-out cells (0,0), (0,2), (1,2), (2,2),
(3,0); trivial baselines: always-stay 0.44, always-apply-action 0.56):

| Input | Output | Steps | Train | Eval | Eval stay | Eval move |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| absolute | absolute | 600 | 0.29 | 0.01 | 0.03 | 0.00 |
| absolute | relative | 600 | 0.75 | 0.67 | 0.52 | 0.79 |
| coordinates | absolute | 600 | 0.36 | 0.13 | 0.21 | 0.07 |
| coordinates | relative | 600 | 0.76 | 0.75 | 0.55 | 0.90 |
| absolute | absolute | 1500 | 0.41 | 0.03 | 0.06 | 0.00 |
| absolute | relative | 1500 | 0.81 | 0.71 | 0.48 | 0.88 |
| coordinates | absolute | 1500 | 0.68 | 0.33 | 0.24 | 0.40 |
| coordinates | relative | 1500 | 0.81 | 0.80 | 0.64 | 0.93 |

Findings:

1. The absolute next-cell-id output is the primary structural blocker. It
   cannot transfer the stay/copy rule to unseen agent cells, and on the
   larger table it stops fitting even the train split (0.29-0.68 train
   accuracy) because every (cell, action) pair must be memorized separately.
2. Relative-move output plus enough data support produces genuine rule
   learning: with 90 train cases the move rule generalizes to held-out cells
   (move-case accuracy 0.88-0.93, interior cell (2,2) near-perfect), well
   above both trivial baselines.
3. Explicit coordinate input channels help consistently but secondarily
   (eval 0.71 -> 0.80 at 1500 steps).
4. Data support matters independently: on the 2x3 table no representation
   choice reaches rule learning; 20 transitions are too few to induce the
   block rule.
5. The residual weakness is blocked/stay detection at unseen edge and corner
   cells (stay-case accuracy 0.64 at 1500 steps), and it shrinks with more
   training, so it looks like an optimization residual rather than another
   structural blocker.

A model-size check on the original setup: `d256-h1024-heads8-l6` reaches
train 0.95 with lr 0.001, warmup 100, and 1000 steps (the current CLI default
optimization does not fit it; see the 2026-06-09 note) and still scores eval
0.0 on held-out `(0, 2)`, so the failure is not model-size-specific.

## Current Interpretation

The held-out generalization failure is now explained by two compounding
causes, confirmed by the 2026-06-10 experiment:

1. Output formulation: absolute next-cell-id classification forces per-cell
   memorization and blocks rule transfer. A relative-move target removes
   this blocker.
2. Data support: the 2x3 transition table (20 train cases) is too small to
   induce the block rule under any representation tested; the 4x5 table
   (90 train cases) is enough for the relative-move formulation to
   generalize.

Explicit coordinate input features help secondarily. Capacity and schedule
were not the cause (the failure reproduces from `d32-l1` to `d256-l6` once
each regime actually fits training).

Earlier candidate causes for reference:

| Area | Question |
| --- | --- |
| input representation | Does the model receive position information in a form that supports rule-like extrapolation? |
| capacity or optimization | Can a slightly different model or schedule learn the transition rule? |
| split design | Is holding out a full agent cell too strict for the current tiny table? |
| evaluation target | Should next-cell prediction be complemented with action-sensitive ranking diagnostics? |

## Candidate Direction

The diagnostics are done. The remaining decision is whether to promote the
findings into the main grid step prediction path:

- Switch the next-cell training target to relative-move classification
  (decode to a next cell for evaluation), or add it alongside the absolute
  target.
- Use a larger grid (for example 4x5) for the held-out generalization
  evidence run, since the 2x3 table cannot support rule learning.
- Optionally add coordinate input channels for the secondary gain.
- Fix or document the grid CLI default mismatch: `d256` defaults need
  lr 0.001, warmup, and more steps to fit this task.

Diagnostic check log:

| Check | Status |
| --- | --- |
| per-action predictions for held-out cells | Done 2026-06-09: errors are systematic; the model avoids emitting the held-out cell id. |
| multiple held-out cells and seeds | Done 2026-06-09: all 5 cells x 3 seeds; eval accuracy 0.00-0.40, mean near chance. |
| explicit coordinate features | Done 2026-06-10: helps, but secondary to the output formulation. |
| relative-move output target | Done 2026-06-10: removes the structural blocker; generalizes with enough data support. |
| larger transition table | Done 2026-06-10: 4x5 grid with relative output reaches eval 0.80 (move cases 0.93). |
| simpler non-Transformer baseline | Open. Less urgent now that the Transformer generalizes under the relative formulation. |

## Non-Goal

Do not treat this as a reason to add a broad world-model abstraction. The next
step should be a small diagnostic or baseline tied directly to the current
GridWorld transition task.
