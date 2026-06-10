# Grid Next-Observation Generalization Emergence

Status: open.

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

## Plan

| Step | Status |
| --- | --- |
| trivial baseline battery with shared metrics | Open. |
| next-observation objective (dataset, per-cell head, derived metrics) | Open. |
| layout sampler with provenance and held-out layout split | Open. |
| train-fit smoke anchor for CLI defaults (drift protection) | Open. |
| emergence sweep over layout count and data quantity, baselines alongside | Open. |
| cellular automaton rule-family world as a data generator | Open. |

## Non-Goal

No shared world abstraction until at least two wired problems demonstrate
duplication. The relative-move head is not promoted into the main path; it
remains a recorded reference point.
