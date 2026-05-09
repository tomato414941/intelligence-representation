# Shogi Target Policy Boundary

Status: closed.

## Issue

Experience records should be independent of problem definitions. They should
store what happened in the world, while problems decide how selected records and
evidence become inputs and targets.

`ShogiPolicyValueDataSelection` used to include both source selection and
target-policy fields:

- `policy_target_source`
- `value_target_source`
- `policy_temperature_cp`
- `policy_mate_cp`
- `score_cp_scale`

Conceptually, Data Selection should decide what records are included for a
declared use. Target policy should decide how policy/value targets are derived
from selected records.

## Resolution

`ShogiPolicyValueDataSelection` now keeps included records under train/eval
sources and moves target derivation into a separate `target_construction`
object.

The current implementation still stores Data Selection and Target Construction
in one JSON file, but they are separate fields with separate meanings.

The old ambiguous names were removed without compatibility aliases:

- `policy_target_source`
- `value_target_source`

The target construction names now say where the signal comes from when needed:

- `chosen_move`
- `decision_usi_multipv`
- `winner`
- `decision_usi_score`
