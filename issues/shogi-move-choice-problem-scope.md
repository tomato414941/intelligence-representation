# Shogi Move Choice Problem Scope

Status: open.

## Issue

`ShogiMoveChoice` sounds like a policy problem: given a shogi position and legal
move candidates, choose or score the next move.

The current `shogi_move_choice` package is broader. It also contains value
prediction, policy/value joint training, target generation, data selection
plumbing, PyTorch sample materialization, model building, training, evaluation,
and checkpointing.

## Why It Matters

This makes it easy to keep adding unrelated responsibilities under
`ShogiMoveChoice` just because the current implementation already lives there.

The main mixed responsibilities are:

- Move choice / policy: chosen move, candidate scores, MultiPV policy targets
- Position value: winner-derived returns or engine score targets
- Data and target construction: which game records are included and how their
  policy/value targets are derived
- Training execution: training configuration, optimization, evaluation, and
  checkpoint handling

Policy and value can be trained jointly, but that is a different claim from
`MoveChoice` being the right name and boundary for every related component.

## Direction

Before expanding source-local target policy, decide which problem boundary is
intended:

- keep `ShogiMoveChoice` as the policy/candidate-scoring problem and move value
  to a separate problem later
- rename or replace the boundary with a policy/value problem such as
  `ShogiPolicyValue`
- keep the current package name only as a historical implementation boundary
  while moving data selection and target construction out of the problem scope

Avoid compatibility aliases if this boundary is renamed or split.

Names should stay easy to change. Do not make problem or package names the
long-term identity of artifacts. Prefer explicit manifests and configurations
that describe Sample Schema, model shape, objectives, target policy, and
training settings.

Use names as readable hints, not strict ontology rules. Avoid names that are so
broad that they absorb unrelated responsibilities, but do not block useful work
on naming precision alone.

## Acceptance Criteria

- Decide whether `ShogiMoveChoiceExample` should remain policy/value or become
  policy-only.
- Decide whether `value_target` belongs in this problem boundary or in a
  separate position-value problem.
- Decide where source-local target policy should live before adding more target
  generation behavior to `shogi_move_choice`.
- Decide which artifact fields describe meaning independently from package
  names.
- Update related issue names or references if this boundary changes.

## Non-Goals

- immediate package-wide rename
- implement a generic problem framework
- change shogi model architecture in this issue
