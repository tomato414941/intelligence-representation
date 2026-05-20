# Shogi History And Repetition Input Features

Status: closed
Priority: medium

## Problem

The current shogi position input representation describes the current position
well, but it intentionally does not encode position history, repetition count,
or no-progress rule context.

That is a different responsibility from basic position description. History and
repetition depend on game trajectory and source record context, not only the
current SFEN position.

## Desired Shape

Decide whether the shogi policy/value model should receive compact
history/repetition features, such as:

- previous move or recent move tokens
- repeated-position count bucket
- no-progress or draw-rule context when available
- explicit unknown buckets when the source does not carry enough history

The design should say how training examples, tensor caches, inference, and
runtime evaluation handle positions where the history context is unavailable.

## Non-Goals

- Do not add history by pretending it is recoverable from the current SFEN.
- Do not force all source records to carry full game history before there is a
  concrete training or evaluation need.
- Do not combine this with deeper tactical threat features.

## Close Condition

- Decide whether history/repetition features belong in the shogi model input now.
- If included, define the feature schema and unknown-context behavior.
- If deferred, document why current-position input remains acceptable for the
  next training run.

## Resolution

Do not add history/repetition input features for the next shogi training run.

History and repetition context cannot be recovered from a standalone current
SFEN. Adding it now would require unknown-context handling across training
examples, tensor caches, inference, and runtime evaluation. That complexity is
not justified before the current-position model has completed a real training
run.

The current-position input remains acceptable for the next run. Repetition,
illegal moves, terminal state, and draw-rule enforcement remain game/runtime
responsibilities. This can be revisited later if value errors around repetition
or no-progress positions become a measured weakness.
