# Shogi Policy Move Index Audit

Status: open
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
