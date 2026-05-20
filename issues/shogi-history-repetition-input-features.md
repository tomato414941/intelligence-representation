# Shogi History And Repetition Input Features

Status: open
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
