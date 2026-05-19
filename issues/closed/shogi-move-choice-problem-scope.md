# Shogi Move Choice Problem Scope

Status: closed.

## Issue

`ShogiMoveChoice` sounds like a policy problem: given a shogi position and legal
move candidates, choose or score the next move.

The old `shogi_move_choice` package was too broad. It contained policy-only
move-choice examples, position-value examples, policy/value joint examples,
target generation, data selection plumbing, PyTorch sample materialization,
model building, training, evaluation, and checkpointing.

## Why It Matters

That made it easy to keep adding unrelated responsibilities under
`ShogiMoveChoice` just because the implementation already lived there.

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

The boundary has been renamed and split enough for the current implementation:

- `ShogiMoveChoiceExample` is policy-only.
- `ShogiPositionValueExample` is value-only.
- `ShogiMovePolicyValueExample` is the joint policy/value example used by the
  current training path.
- The package, training, model, checkpoint, and CLI boundary is now
  `shogi_policy_value`.

Avoid compatibility aliases if this boundary is renamed or split.

Names should stay easy to change. Do not make problem or package names the
long-term identity of artifacts. Prefer explicit manifests and configurations
that describe Sample Schema, model shape, objectives, target policy, and
training settings.

Use names as readable hints, not strict ontology rules. Avoid names that are so
broad that they absorb unrelated responsibilities, but do not block useful work
on naming precision alone.

## Acceptance Criteria

- [x] Decide whether `ShogiMoveChoiceExample` should remain policy/value or become
  policy-only.
- [x] Decide whether `value_target` belongs in this problem boundary or in a
  separate position-value problem.
- [x] Decide where source-local target policy should live before adding more target
  generation behavior to `shogi_policy_value`.
- [x] Decide which artifact fields describe meaning independently from package
  names.
- [x] Update related issue names or references if this boundary changes.

## Non-Goals

- implement a generic problem framework
- change shogi model architecture in this issue
