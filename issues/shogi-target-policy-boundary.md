# Shogi Target Policy Boundary

Status: open. Priority: low.

## Issue

Experience records should be independent of problem definitions. They should
store what happened in the world, while problems decide how selected records and
evidence become inputs and targets.

`ShogiPolicyValueDataSelection` currently includes both source selection and
target-policy fields:

- `policy_target_source`
- `value_target_source`
- `policy_temperature_cp`
- `policy_mate_cp`
- `score_cp_scale`

Conceptually, Data Selection should decide what records are included for a
declared use. Target policy should decide how policy/value targets are derived
from selected records.

## Current Policy

Keep target policy in `ShogiPolicyValueDataSelection` for now.

This is an intentional KISS compromise. Current shogi runs need one compact file
that explains both selected sources and policy/value target derivation. Splitting
it now would add another concept and another file without a concrete reuse need.

## Trigger

Revisit when a concrete run needs to reuse the same selected source set with
multiple target policies, or when target derivation policy becomes shared across
multiple shogi data selections.

## Non-Goals

- split the implementation now
- introduce a generic target-policy framework
- add compatibility aliases
