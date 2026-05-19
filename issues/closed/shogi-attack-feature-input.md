# Shogi Attack Feature Input

Status: closed
Priority: high

## Problem

The current shogi policy/value input represents pieces, hands, side to move,
and whether the side to move is in check. It does not explicitly represent
attack maps or attack counts.

Strong shogi neural-network input designs such as dlshogi/PGX include attack
features. These may help the model learn tactical concepts such as defended
pieces, hanging pieces, king danger, forcing moves, and local control without
requiring the network to infer all attacks from piece placement alone.

## Desired Shape

Add shogi-specific attack input features before changing the policy output
representation:

- side-to-move-relative coordinates
- own and opponent attack presence
- own and opponent attack-count buckets, likely `>=1`, `>=2`, and `>=3`
- explicit tests for rotation and own/opponent perspective
- updated input identity so older checkpoints and tensor caches are rejected

Keep this concrete to shogi policy/value. Do not introduce a generic attack-map
framework before another game needs one.

## Cost And Risk

Attack features add CPU-side feature construction. This is likely acceptable for
fixed tensor-cache training, but it may matter for MCTS leaf evaluation and
self-play generation, where features are built repeatedly.

Close condition should include a small timing check or at least a documented
assessment of feature-construction cost.

## Close Condition

- Shogi position inputs include attack features.
- Tests cover attack presence/count semantics and side-to-move-relative
  orientation.
- Checkpoint and tensor-cache identity rejects pre-attack-feature artifacts.
- The CPU cost impact is measured or explicitly documented.

Closed because all close conditions are met by the 2026-05-19 implementation.

## Progress

2026-05-19:

- Added side-to-move-relative attack count tokens to the shogi position input.
- The first attack block stores own attack counts per relative square; the
  second stores opponent attack counts per relative square.
- Counts are capped into buckets `0..3`, so attack presence is represented by
  nonzero buckets and heavy overlap is represented by the `3` bucket.
- The input identity changed to
  `shogi_side_to_move_relative_in_check_attack_counts`, so older checkpoints and
  tensor caches are rejected.
- Tests cover own/opponent attack counts and white-to-move relative orientation.
- Local CPU timing on 2,180 tokenizations was about 754 positions/sec. This is
  acceptable for tensor-cache construction, but MCTS leaf-evaluation cost should
  still be watched in throughput measurements.
