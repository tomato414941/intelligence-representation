# Shogi Input Feature Audit

Status: open
Priority: medium

## Problem

The current shogi policy/value model builds position inputs from SFEN into a
compact token sequence:

- side to move
- 81 board-square piece tokens
- capped hand-piece count tokens

Before increasing model size or drawing conclusions from more training data, the
project should verify whether this representation preserves the shogi state
information needed for policy/value learning.

Potential audit points include:

- promoted piece representation
- black/white orientation and side-to-move handling
- hand-piece count capping
- move count, repetition, no-progress, and other rule-context fields
- whether the representation is intended to be absolute-board or player-relative

## Desired Shape

The project should have a written, test-backed statement of what shogi state
information is represented, what is intentionally omitted, and why the omission
is acceptable for the current learning target.

This is an audit issue first. Do not change the model architecture until the
missing or ambiguous features are identified.

## Close Condition

- Current input features are documented against shogi policy/value requirements.
- Tests cover important encoding invariants.
- Any missing features that should be added are split into concrete follow-up
  issues.
