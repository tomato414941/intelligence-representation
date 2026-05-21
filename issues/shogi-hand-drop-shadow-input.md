# Shogi Hand Drop-Shadow Input

Status: open.
Priority: medium.

## Issue

The current shogi position entries represent hand pieces as global counts. That
is a clean ownership representation, but it does not directly expose where hand
pieces can materialize on the board.

For shogi, hand pieces are not only material counts. They create square-level
drop potential:

- which squares each held piece can be dropped on
- which squares are unavailable because of drop constraints
- which drops affect king safety, checking potential, or defensive coverage
- where the opponent's hand pieces create immediate pressure

The current `dlshogi-like-no-entering-king` entry should stay focused on
current-position piece, hand, check, and attack features. Adding drop-shadow
there would mix another axis into the baseline comparison.

## Desired Shape

Keep hand counts as global elements. Add drop-shadow only as a separate shogi
model entry when the project is ready to compare hand representation variants.

Candidate entry:

```text
shogi-action-plane-policy-output-dlshogi-like-no-entering-king-drop-shadow
```

The intended comparison is:

```text
global hand counts only
vs
global hand counts + square-level drop-shadow
```

The output head, core, training data, and training recipe should remain fixed
for that comparison.

## Non-Goals

- Do not replace global hand counts with drop-shadow.
- Do not add hand piece tokens as part of this issue.
- Do not add entering-king features as part of this issue.
- Do not make this a dlshogi compatibility effort.

## Acceptance Criteria

- Decide the exact square-level drop-shadow fields.
- Add a separate position input entry instead of changing the existing
  `dlshogi-like-no-entering-king` entry.
- Ensure the input manifest makes the added drop-shadow fields explicit.
- Verify tensor cache creation and one-step training for the new entry.
