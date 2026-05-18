# Shogi Policy Move Index Audit

Status: closed
Priority: medium

## Problem

The current shogi policy head scores candidate legal moves using compact move
features:

- from square or no-from-square marker
- to square
- promotion flag
- drop piece type

This avoids a fixed global move-index head, but the project should verify that
the candidate-move representation, legal-move mask, label index, and optional
policy-target distributions stay aligned across all shogi training paths.

This matters more once generated MCTS visit-count distributions are used as
policy targets, because target mass must map to exactly the same candidate moves
that the model scores.

## Desired Shape

Policy target construction and tensorization should make one responsibility
clear:

```text
legal candidate moves
-> candidate move features and mask
-> label or policy-target distribution over those candidates
```

The model should not depend on accidental ordering or ambiguous move encoding.

## Close Condition

- Candidate ordering, label indices, and policy-target distribution alignment
  are documented and tested.
- Tests cover drops, promotions, illegal target moves, and missing target moves.
- Any architectural change, such as a fixed global move-index head, is split
  into a separate issue if still warranted.

## Resolution

Closed on 2026-05-18.

The current shogi policy head is a candidate-list policy head, not a fixed
global move-index head. The candidate list is `legal_moves`; the same list owns
the coordinate system for:

- candidate move features
- candidate mask
- chosen-move label index
- optional policy-target distributions, including MCTS visit counts

The audit found no evidence that the current training paths map target mass to
the wrong candidate. The remaining risk was that the invariant was implicit.
Tests now pin the invariant:

- `tests.test_shogi_policy_value` covers direct candidate-order alignment with
  drop and promotion moves.
- `tests.test_shogi_policy_value_data` covers MCTS visit-count targets through
  `ShogiGameRecord -> ShogiPolicyValueExample -> ShogiPolicyValueDataset`.
- existing validation tests reject missing non-`chosen_move` targets and target
  moves outside `legal_moves`.

No fixed global move-index head is needed for the current design.
