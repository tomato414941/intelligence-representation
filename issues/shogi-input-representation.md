# Shogi Input Representation

Status: open
Priority: medium

## Problem

The project needs a strong shogi position input representation before spending
substantial compute on full policy/value training.

The goal is not merely to audit missing features. The goal is to design an input
representation that gives the model useful shogi structure:

- the position should be side-to-move-relative
- board-square information should be easy for a Transformer to attend over
- global state such as hands, check, and move count should be explicit
- expensive or rule-context features should have clear follow-up paths
- checkpoints and tensor caches must identify the exact input encoding

## Current Shape

The current implementation still uses a compact token sequence rather than a
full `9x9xN` input plane tensor.

Current represented information:

- side to move
- whether the side to move is in check
- coarse move-count bucket with an explicit unknown bucket
- side-to-move-relative board squares
- own/opponent piece identity, including promoted piece types through
  `python-shogi` piece types
- own/opponent attack-count buckets per relative square, capped at 3
- own/opponent hand counts capped at 18
- side-to-move-relative candidate move from/to squares
- promotion and drop-piece candidate move fields

Current missing or intentionally deferred information:

- position history
- repetition or no-progress rule context
- piece-type-specific attack maps

## Desired Shape

The preferred direction is a Transformer-native shogi feature sequence:

- global tokens:
  - state/value token
  - side-to-move token
  - in-check token
  - move-count bucket token
  - own hand tokens for 7 piece types
  - opponent hand tokens for 7 piece types
- square tokens x81:
  - side-to-move-relative square coordinate embedding
  - own/opponent piece identity
  - own attack-count bucket
  - opponent attack-count bucket
  - optional future piece-type attack features

This keeps the shogi board as 81 square subjects while still allowing global
state to be represented without broadcasting it over every square.

## Close Condition

- The intended shogi position input representation is documented.
- The implemented input encoding matches that representation.
- Tests cover important encoding invariants.
- Any deliberately omitted features are split into concrete follow-up issues.

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

- Reviewed external shogi neural-network input designs:
  - AlphaZero shogi uses side-to-move-relative inputs with position history,
    pieces in hand, move-count, and repetition-state features.
  - dlshogi/PGX-style inputs include own/opponent piece planes, hand planes,
    attack planes, attack-count planes, and an in-check feature.
- Added an `in_check` token to the shogi position input sequence.
- Added own/opponent attack-count buckets per relative square.
- Added a global move-count bucket token to the shogi position input sequence.
- Move counts use coarse buckets with an explicit unknown bucket:
  unknown, 1-30, 31-60, 61-90, 91-120, 121-160, 161-220, and 221+.
- The current input identity is
  `shogi_side_to_move_relative_in_check_move_count_bucket_attack_counts`, so
  older checkpoints and tensor caches are rejected.
- Replaced the shared Transformer position input layer with a
  Transformer-native feature sequence:
  - 18 global tokens: state/value token, side-to-move, in-check,
    move-count bucket, and 14 hand tokens
  - 81 square tokens, each combining side-to-move-relative square identity,
    piece identity, own attack-count bucket, and opponent attack-count bucket
- The current input identity is `shogi_global_square_feature_sequence_v1`.

## Follow-Ups

- Keep piece-type-specific attack-map features as a separate issue because their
  strength upside and CPU cost are both larger.
- Keep history/repetition features separate from basic position features because
  they interact with rule context and data generation.
