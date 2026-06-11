# Grid Next-Observation Generalization Emergence

Status: open.

Implementation note (2026-06-10): a first implementation of the Design below
(commits `c2cb8a8`, `f2fe516`, `ac7415b`) was reverted in `2597fed`. The
evaluation design needs rework before reimplementation: headline metrics
should make trivial strategies score zero by construction (for example,
score only what changes), and the world's dynamics distribution (the current
tiny grid makes "nothing moves" the answer in 17 of 25 transitions) is a
design variable, not a given. Standing baseline comparisons that reinterpret
an inflatable score treat the symptom; the measurement instrument itself
should not award free points. Baselines remain useful as a design-time check
that this holds.

## Question

Does held-out generalization of transition rules emerge under a generic
next-observation prediction objective as experience diversity and quantity
scale, without injecting world-specific structure into the formulation?

This replaces the question of the closed issue
[grid-world-heldout-generalization](closed/grid-world-heldout-generalization.md),
which explained the original failure (absolute next-cell-id output plus
insufficient data support) but whose candidate fix (relative-move output)
would have encoded the world's movement rule into the output head and
weakened the evaluation pressure this surface exists to provide.

## Why This Formulation

Per [Bitter Lesson Correction](../docs/bitter-lesson.md), the project centers
source data, learnable representation, predictive computation, and evaluation
pressure. Structure injection is not forbidden, but it is a measured variable,
not a default: any structure built into the formulation must be declared, and
its effect on what the evaluation can claim must be stated.

Predicting the next observation itself is the domain-general predictive
objective (the next-token / next-frame analog). It preserves the observation's
spatial format without asserting any movement, blocking, or symmetry rule.

## Design

Training objective:

- Input: current observation as one token per cell (3 channels), plus an
  optional action token. Action-less worlds (for example cellular automata)
  must fit the same head, so nothing may assume the action token exists.
- Output: per-cell classification of the next observation cell content
  (empty / agent / goal / wall), read from each cell's own token position.
- Reward and terminated stay as thin auxiliary heads.

Derived evaluation metrics (not training targets):

- next-agent-cell accuracy, read as the argmax over predicted agent-class
  probabilities across cells
- per-cell accuracy and whole-grid exact match

Mandatory trivial baselines, reported alongside every run:

- copy: predict no change
- naive-action-apply: move the agent by the action delta when in bounds,
  ignoring walls
- per-cell majority class

The copy baseline matters most: under next-observation prediction most cells
are static, so per-cell accuracy is inflated and any claim must beat copy on
the cells that change.

Data as a world family:

- Sample wall and goal layouts on a fixed 4x5 grid first, with recorded
  generation provenance (seed, parameters). Mazes are this family with denser
  walls, not a separate world.
- Held-out hierarchy: held-out agent cells within layouts, then held-out
  layouts, later held-out sizes and held-out rules (cellular automaton rule
  family; Life is one rule).

## Declared Injected Structure

Injected: 2D lattice tokenization, input/output cell alignment (cell i is
predicted from token i), action tokenization, the cell class vocabulary.

Not injected: movement rules, blocking rules, translation symmetry, locality.
These are what the model must acquire under training and evaluation pressure.

Reference points from the closed issue (small model, 4x5 grid, held-out
cells): absolute next-cell-id classification (structure-destroying lower
reference, eval 0.03 at 1500 steps) and relative-move classification
(structure-injected upper bound, eval 0.80, move cases 0.93).

## Cellular Measurement (2026-06-11)

The question was answered first on the cellular world instead of gridworld:
no action, no reward, no goal, so nothing but the update rule remains. The
measurement instrument is the score pair changed-cell accuracy (a copy
strategy scores 0) and unchanged-cell accuracy (an all-flip strategy scores
0), evaluated on initial states never used in training (seed-range overlap is
rejected mechanically).

Life (B3/S23), 6x6, dead borders, `d256-h1024-heads8-l6`, 1000 steps,
evaluated on 64 unseen states (seed 100000). Training fits perfectly at every
N, so train scores carry no information; only the unseen-state scores
separate memorization from rule acquisition:

| Train states N | Eval changed-cell | Eval unchanged-cell |
| ---: | ---: | ---: |
| 16 | 0.485 | 0.716 |
| 64 | 0.547 | 0.727 |
| 256 | 0.898 | 0.926 |
| 1024 | 1.000 | 0.999 |

N=1024 confirmed across model seeds 31/32/33 (changed 1.000 in all three,
unchanged 0.999-1.000). Commands: `intrep.train_cellular_step_prediction`
and `intrep.problems.cellular_step_prediction.evaluate`; run artifacts under
`runs/local-checks/cellular-life-*`.

Conclusion: with no rule injected into the formulation, rule acquisition
emerges from prediction practice alone as experience grows, and is complete
within this setting by N=1024. Replicating across random rules of the family
was considered and skipped as near-certain given this result; the genuinely
open follow-up is held-out-rule inference (training across rules and
predicting under a rule never trained on, which requires conditioning on
example transitions).

## Plan

| Step | Status |
| --- | --- |
| cellular automaton rule-family world as a data generator | Done 2026-06-11. |
| cheat-resistant score pair + train/eval overlap rejection | Done 2026-06-11 (replaces the baseline-battery step; the instrument awards no free points by construction). |
| separate train and evaluate commands, data declared by seeds | Done 2026-06-11 (cellular). |
| emergence sweep over experience quantity | Done 2026-06-11 (table above). |
| layout sampler with provenance | Done 2026-06-10 (`worlds/gridworld/layouts.py`); held-out layout split for gridworld remains open. |
| gridworld wired onto the same prediction head (action token) | Open. |
| held-out-rule inference across the rule family | Open. The genuinely uncertain question. |

## Non-Goal

No shared world abstraction until at least two wired problems demonstrate
duplication. The relative-move head is not promoted into the main path; it
remains a recorded reference point.
