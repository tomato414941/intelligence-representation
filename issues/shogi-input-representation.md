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
- checkpoints and tensor caches must identify the exact input schema

## Current Shape

The current implementation uses explicit feature groups rather than treating a
single flattened token-id sequence as the canonical model input.

Current represented information:

- side to move
- whether the side to move is in check
- coarse move-count bucket with an explicit unknown bucket
- side-to-move-relative board squares
- own/opponent piece identity, including promoted piece types through
  `python-shogi` piece types
- own/opponent attack-count buckets per relative square, capped at 3
- own/opponent square piece-type attack features
- own/opponent king-relative square features
- own/opponent drop-shadow features for legal drops from hand by piece type
- counterfactual removal features per square and board-piece token:
  whether removing the occupying piece exposes own king check, exposes opponent
  king check, or removes a sliding-line blocker
- capture-to-hand flow features per square and board-piece token:
  whether giving that piece to the opponent creates a near-king drop danger, and
  whether capturing that opponent piece creates a near-king drop opportunity
- fixed 40 piece tokens for board and hand pieces, with board pieces ordered by
  side-to-move-relative square and remaining incomplete-position slots padded
  as empty
- fixed 52 line tokens for files, ranks, and both diagonal families, with
  line-kind, king-on-line, slider-on-line, and occupancy features
- dynamic pair relation ids over the full token sequence, currently covering
  piece-on-square, piece-attacks-square, hand-piece-drops-to-square,
  piece-attacks-piece, piece-defends-piece, and same-side piece relations
- own/opponent hand counts capped at 18
- side-to-move-relative candidate move from/to squares
- promotion and drop-piece candidate move fields
- shared-core candidate policy scoring uses legal move tokens that cross-attend
  to the encoded position

The canonical position input object is `ShogiPositionFeatures`:

- `global_feature_ids`
- `square_feature_ids`
- `piece_feature_ids`
- `line_feature_ids`
- `pair_relation_ids`

There is no canonical flat position vector. Boundaries that consume positions
must carry the grouped feature object.

Current missing or intentionally deferred information:

- position history
- repetition or no-progress rule context
- pin and discovered-attack context
- king-safety aggregate features

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
  - own/opponent square piece-type attack features
  - own/opponent king-relative square features
  - own/opponent drop-shadow features by hand piece type
  - counterfactual removal features
  - capture-to-hand flow features
- piece tokens x40:
  - location kind: board, hand, or empty
  - own/opponent piece identity
  - side-to-move-relative square for board pieces, or unknown for hand/empty
  - own king-relative square feature for board pieces, or unknown
  - opponent king-relative square feature for board pieces, or unknown
  - counterfactual removal features for board pieces, or false for hand/empty
  - capture-to-hand flow features for board pieces, or false for hand/empty
- line tokens x52:
  - 9 file tokens
  - 9 rank tokens
  - 17 rising diagonal tokens
  - 17 falling diagonal tokens
  - line kind
  - whether own/opponent king is on the line
  - whether own/opponent sliding piece is on the line
  - occupancy count

This keeps the shogi board as 81 square subjects while still allowing global
state to be represented without broadcasting it over every square. The piece
tokens add a piece-subject view of the same board rather than dedicated
relation tokens. Piece tokens are treated as a set-like sequence and do not use
slot-position embeddings. Square tokens retain square identity and the shared
shogi Transformer adds static position geometry attention bias over the 81
board-square tokens and between line tokens and their member squares. Dynamic
pair relation ids are added as learned attention bias so token-to-token shogi
relations are represented on the attention pair, not only as token-local
features.

## Close Condition

- The intended shogi position input representation is documented.
- The implemented input schema matches that representation.
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
- Checkpoints now record `input_schema_id`. Older checkpoints without that input
  identity are rejected instead of being compatibility-loaded into the new
  representation.
- Tensor caches now record `input_schema_id`. Older tensor caches without that
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
- The input identity at this step was
  `shogi_side_to_move_relative_in_check_move_count_bucket_attack_counts`, so
  older checkpoints and tensor caches are rejected.
- Replaced the shared Transformer position input layer with a
  Transformer-native feature sequence:
  - 18 global tokens: state/value token, side-to-move, in-check,
    move-count bucket, and 14 hand tokens
  - 81 square tokens, each combining side-to-move-relative square identity,
    piece identity, own attack-count bucket, and opponent attack-count bucket
- The input identity at this step was
  `shogi_global_square_feature_sequence_v1`.
- Added own/opponent square piece-type attack features.
  These preserve the existing square-token sequence length while giving the
  model the quality of square control, not only the attack count.
- Added own/opponent king-relative square features so each square can express
  its location relative to both kings. This is not a full NNUE HalfKP-style
  feature; it is a square-subject feature that lets an occupied square expose
  the occupying piece together with its king-relative location.
- The input identity at this step was
  `shogi_global_square_feature_sequence_v2`.
- Added fixed 40-slot piece tokens. Occupied board pieces are ordered by
  side-to-move-relative square, and remaining slots are padded as empty. Each
  piece token includes slot state, piece identity, relative square, and
  own/opponent king-relative square features.
- The current input identity is
  `shogi_global_square_piece_feature_sequence_v1`.
- Extended fixed 40-slot piece tokens to include hand pieces after board
  pieces. Global hand-count tokens remain as direct aggregate count features,
  while hand piece tokens expose held pieces through the same piece-subject
  sequence as board pieces.
- The current input identity is
  `shogi_global_square_all_piece_feature_sequence`.
- Removed piece slot embeddings so same-type pieces and hand pieces are not
  given artificial slot identity beyond their actual features.
- Added a learnable static square-square geometry attention bias over board
  square tokens. The relation ids are based on side-to-move-relative `dx,dy`
  offsets, giving attention direct access to board geometry without adding
  position-dependent tactical features.
- Added own/opponent drop-shadow features so each square can see which held
  piece types can legally be dropped there by either side.
- Added 52 line tokens for files, ranks, and both diagonal families. Each line
  token carries line-kind, king-on-line, slider-on-line, and occupancy features.
  Line-square attention bias marks which squares belong to each line, giving the
  Transformer explicit long-range board subjects without replacing square
  tokens.
- Removed the flattened `position_token_ids` compatibility path. Tensor
  samples, batches, inference, model input, and tensor cache payloads now carry
  global/square/piece/line feature groups.
- Added counterfactual removal features to square and board-piece tokens.
  These expose cheap causal roles such as "removing this piece exposes own
  king check" and "this piece is a sliding-line blocker."
- Added capture-to-hand flow features to square and board-piece tokens. These
  expose a cheap approximation of whether a captured piece would become a
  dangerous or useful near-king drop resource.
- Added dynamic `pair_relation_ids` to `ShogiPositionFeatures` and tensor
  caches. The shared Transformer uses them as learned pair-relation attention
  bias.
- Replaced shared-core candidate policy scoring with legal move tokens that
  cross-attend to the encoded position before producing candidate logits.
- The current input identity is
  `shogi_global_square_piece_line_pair_drop_counterfactual_flow_feature_sequence`.

## Follow-Ups

- Keep history/repetition features separate from basic position features because
  they interact with rule context and data generation.
- Consider deeper pin/discovered-attack and threat-response features only after
  the current representation has been exercised; the current schema now carries
  a cheap counterfactual/blocker approximation but not full tactical threat
  search.
