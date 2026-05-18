# Shogi Input Feature Audit

Status: open
Priority: medium

## Problem

The current shogi policy/value model builds position inputs from SFEN into a
compact token sequence:

- side to move
- 81 board-square piece tokens
- hand-piece count tokens

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

## Progress

2026-05-18:

- Qhapaq full records contained 2,460,781 positions.
- 66,326 positions, about 2.7%, had at least one hand-piece count greater than
  six.
- Over-cap positions were pawn-only in this corpus; the maximum observed pawn
  hand count was 14.
- The hand-count token range was expanded from 0..6 to 0..18 so legal pawn hand
  counts no longer collapse at six.
- The model input coordinate system was changed from absolute-board to
  side-to-move-relative:
  - board squares are rotated for white to move
  - pieces are encoded as own/opponent instead of black/white
  - hands are encoded as own/opponent instead of black/white
  - candidate move from/to squares use the same relative coordinate system
- Checkpoints now record `input_encoding`. Older checkpoints without that input
  identity are rejected instead of being compatibility-loaded into the new
  representation.

Remaining audit areas include rule/history context, attack or check features,
and whether the current compact token representation is strong enough after the
relative-coordinate change.
