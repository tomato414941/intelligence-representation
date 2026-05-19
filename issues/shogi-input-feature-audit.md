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
- Tensor caches now record `input_encoding`. Older tensor caches without that
  identity are rejected instead of being silently reused after representation
  changes.

2026-05-19:

External shogi neural-network inputs are richer than the current compact token
sequence:

- AlphaZero shogi uses side-to-move-relative inputs with position history,
  pieces in hand, move-count, and repetition-state features.
- dlshogi/PGX-style inputs include own/opponent piece planes, hand planes,
  attack planes, attack-count planes, and an in-check feature.

Current project differences:

- Represented now:
  - side to move
  - whether the side to move is in check
  - side-to-move-relative board squares
  - own/opponent piece identity, including promoted piece types through
    `python-shogi` piece types
  - own/opponent hand counts capped at 18
  - side-to-move-relative candidate move from/to squares
  - promotion and drop-piece candidate move fields
- Not represented now:
  - move count or game phase
  - position history
  - repetition or no-progress rule context
  - attack maps or attack-count features

Expected value and cost:

- `in_check` is a low-cost feature compared with neural-network inference and
  may help tactical policy/value learning.
- move count is also low-cost, but its isolated strength impact is less clear.
- history and repetition features are rule-correctness features. They may matter
  for draw/repetition handling, but they add data-shape complexity.
- attack maps and attack counts are plausible strength features because strong
  shogi input designs use them. They also add CPU-side feature construction
  cost, especially for MCTS leaf evaluation, so they should be evaluated as a
  separate performance-sensitive change.

Remaining audit areas include rule/history context, attack features,
and whether the current compact token representation is strong enough after the
relative-coordinate change.

Recommended follow-up split:

- Consider a move-count feature separately.
- Keep attack-map and attack-count features as a separate issue because their
  strength upside and CPU cost are both larger.
- Keep history/repetition features separate from basic position features.

2026-05-19:

- Added an `in_check` token to the shogi position input sequence.
- The position token layout is now side-to-move token, in-check token, board
  square tokens, then hand-count tokens.
- The input identity was changed to `shogi_side_to_move_relative_in_check`, so
  older checkpoints and tensor caches are rejected.
- Tests cover both safe and checked side-to-move positions.
