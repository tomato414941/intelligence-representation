from __future__ import annotations

import hashlib
import json

import shogi


HAND_PIECE_TYPES = (
    shogi.PAWN,
    shogi.LANCE,
    shogi.KNIGHT,
    shogi.SILVER,
    shogi.GOLD,
    shogi.BISHOP,
    shogi.ROOK,
)
BOARD_TOKEN_COUNT = 81
SQUARE_ATTACK_PIECE_TYPES = tuple(shogi.PIECE_TYPES)
SQUARE_ATTACK_PIECE_TYPE_COUNT = len(SQUARE_ATTACK_PIECE_TYPES)
PIECE_SLOT_COUNT = 40
COUNTERFACTUAL_FEATURE_COUNT = 3
DROP_POTENTIAL_FEATURE_COUNT = 2
PIECE_FEATURE_COUNT = 5 + COUNTERFACTUAL_FEATURE_COUNT + DROP_POTENTIAL_FEATURE_COUNT
GLOBAL_TOKEN_COUNT = 18
LINE_TOKEN_COUNT = 9 + 9 + 17 + 17
LINE_FEATURE_COUNT = 6
SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT = GLOBAL_TOKEN_COUNT + BOARD_TOKEN_COUNT + PIECE_SLOT_COUNT + LINE_TOKEN_COUNT
SHOGI_POSITION_INPUT_SCHEMA_ID = (
    "shogi_global_square_piece_line_pair_edge_drop_shadow_coarse_counterfactual_drop_potential_feature_sequence"
)

STATE_TOKEN_INDEX = 0
GLOBAL_SIDE_TO_MOVE_TOKEN_INDEX = 1
GLOBAL_IN_CHECK_TOKEN_INDEX = 2
GLOBAL_MOVE_COUNT_TOKEN_INDEX = 3
GLOBAL_HAND_TOKEN_OFFSET = 4
SQUARE_TOKEN_OFFSET = GLOBAL_TOKEN_COUNT
PIECE_TOKEN_OFFSET = SQUARE_TOKEN_OFFSET + BOARD_TOKEN_COUNT
LINE_TOKEN_OFFSET = PIECE_TOKEN_OFFSET + PIECE_SLOT_COUNT
PAIR_RELATION_NONE = 0
PAIR_RELATION_PIECE_ON_SQUARE = 1
PAIR_RELATION_PIECE_ATTACKS_SQUARE = 2
PAIR_RELATION_HAND_PIECE_DROPS_TO_SQUARE = 3
PAIR_RELATION_PIECE_ATTACKS_PIECE = 4
PAIR_RELATION_PIECE_DEFENDS_PIECE = 5
PAIR_RELATION_PIECE_SAME_SIDE = 6
PAIR_RELATION_COUNT = 7

EMPTY_SQUARE_TOKEN_ID = 0
OWN_PIECE_OFFSET = 1
OPPONENT_PIECE_OFFSET = 15
SIDE_TO_MOVE_BLACK_TOKEN_ID = 29
SIDE_TO_MOVE_WHITE_TOKEN_ID = 30
NOT_IN_CHECK_TOKEN_ID = 31
IN_CHECK_TOKEN_ID = 32
ATTACK_COUNT_TOKEN_MAX = 3
OWN_ATTACK_OFFSET = 33
OPPONENT_ATTACK_OFFSET = OWN_ATTACK_OFFSET + ATTACK_COUNT_TOKEN_MAX + 1
HAND_COUNT_TOKEN_MAX = 18
OWN_HAND_OFFSET = OPPONENT_ATTACK_OFFSET + ATTACK_COUNT_TOKEN_MAX + 1
OPPONENT_HAND_OFFSET = OWN_HAND_OFFSET + HAND_COUNT_TOKEN_MAX + 1
MOVE_COUNT_BUCKET_UNKNOWN = 0
MOVE_COUNT_BUCKETS = (
    (1, 30),
    (31, 60),
    (61, 90),
    (91, 120),
    (121, 160),
    (161, 220),
)
MOVE_COUNT_BUCKET_OFFSET = OPPONENT_HAND_OFFSET + HAND_COUNT_TOKEN_MAX + 1
MOVE_COUNT_BUCKET_OVERFLOW = len(MOVE_COUNT_BUCKETS) + 1
MOVE_COUNT_BUCKET_VOCAB_SIZE = MOVE_COUNT_BUCKET_OVERFLOW + 1
OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET = MOVE_COUNT_BUCKET_OFFSET + MOVE_COUNT_BUCKET_VOCAB_SIZE
OPPONENT_SQUARE_PIECE_TYPE_ATTACK_OFFSET = OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET + SQUARE_ATTACK_PIECE_TYPE_COUNT * 2
KING_RELATIVE_SQUARE_OFFSET_BUCKET_COUNT = 17 * 17
KING_RELATIVE_SQUARE_BUCKET_UNKNOWN = 0
OWN_KING_RELATIVE_SQUARE_OFFSET = OPPONENT_SQUARE_PIECE_TYPE_ATTACK_OFFSET + SQUARE_ATTACK_PIECE_TYPE_COUNT * 2
OPPONENT_KING_RELATIVE_SQUARE_OFFSET = OWN_KING_RELATIVE_SQUARE_OFFSET + KING_RELATIVE_SQUARE_OFFSET_BUCKET_COUNT + 1
OWN_DROP_SHADOW_OFFSET = OPPONENT_KING_RELATIVE_SQUARE_OFFSET + KING_RELATIVE_SQUARE_OFFSET_BUCKET_COUNT + 1
OPPONENT_DROP_SHADOW_OFFSET = OWN_DROP_SHADOW_OFFSET + len(HAND_PIECE_TYPES) * 2
LINE_KIND_OFFSET = OPPONENT_DROP_SHADOW_OFFSET + len(HAND_PIECE_TYPES) * 2
LINE_OWN_KING_ON_LINE_OFFSET = LINE_KIND_OFFSET + 4
LINE_OPPONENT_KING_ON_LINE_OFFSET = LINE_OWN_KING_ON_LINE_OFFSET + 2
LINE_OWN_SLIDER_ON_LINE_OFFSET = LINE_OPPONENT_KING_ON_LINE_OFFSET + 2
LINE_OPPONENT_SLIDER_ON_LINE_OFFSET = LINE_OWN_SLIDER_ON_LINE_OFFSET + 2
LINE_OCCUPANCY_COUNT_MAX = 9
LINE_OCCUPANCY_COUNT_OFFSET = LINE_OPPONENT_SLIDER_ON_LINE_OFFSET + 2
COUNTERFACTUAL_REMOVAL_SELF_CHECK_OFFSET = LINE_OCCUPANCY_COUNT_OFFSET + LINE_OCCUPANCY_COUNT_MAX + 1
COUNTERFACTUAL_REMOVAL_OPPONENT_CHECK_OFFSET = COUNTERFACTUAL_REMOVAL_SELF_CHECK_OFFSET + 2
COUNTERFACTUAL_REMOVAL_COARSE_SLIDER_BLOCKER_OFFSET = COUNTERFACTUAL_REMOVAL_OPPONENT_CHECK_OFFSET + 2
OPPONENT_DROP_POTENTIAL_AFTER_LOSING_PIECE_OFFSET = COUNTERFACTUAL_REMOVAL_COARSE_SLIDER_BLOCKER_OFFSET + 2
OWN_DROP_POTENTIAL_AFTER_CAPTURING_PIECE_OFFSET = OPPONENT_DROP_POTENTIAL_AFTER_LOSING_PIECE_OFFSET + 2
PIECE_LOCATION_EMPTY_TOKEN_ID = OWN_DROP_POTENTIAL_AFTER_CAPTURING_PIECE_OFFSET + 2
PIECE_LOCATION_BOARD_TOKEN_ID = PIECE_LOCATION_EMPTY_TOKEN_ID + 1
PIECE_LOCATION_HAND_TOKEN_ID = PIECE_LOCATION_BOARD_TOKEN_ID + 1
PIECE_SQUARE_UNKNOWN_TOKEN_ID = PIECE_LOCATION_HAND_TOKEN_ID + 1
PIECE_SQUARE_OFFSET = PIECE_SQUARE_UNKNOWN_TOKEN_ID + 1
SHOGI_POSITION_VOCAB_SIZE = PIECE_SQUARE_OFFSET + BOARD_TOKEN_COUNT
SHOGI_POSITION_GLOBAL_SLOT_COUNT = GLOBAL_TOKEN_COUNT
SHOGI_POSITION_SQUARE_COUNT = BOARD_TOKEN_COUNT
SHOGI_POSITION_SQUARE_FEATURE_COUNT = (
    3
    + SQUARE_ATTACK_PIECE_TYPE_COUNT * 2
    + 2
    + len(HAND_PIECE_TYPES) * 2
    + COUNTERFACTUAL_FEATURE_COUNT
    + DROP_POTENTIAL_FEATURE_COUNT
)
SHOGI_POSITION_SQUARE_SLOT_COUNT = BOARD_TOKEN_COUNT
SHOGI_POSITION_PIECE_SLOT_COUNT = PIECE_SLOT_COUNT
SHOGI_POSITION_PIECE_FEATURE_COUNT = PIECE_FEATURE_COUNT
SHOGI_POSITION_LINE_SLOT_COUNT = LINE_TOKEN_COUNT
SHOGI_POSITION_LINE_FEATURE_COUNT = LINE_FEATURE_COUNT
SHOGI_POSITION_STATE_TOKEN_ID = SHOGI_POSITION_VOCAB_SIZE
SHOGI_POSITION_FEATURE_VOCAB_SIZE = SHOGI_POSITION_STATE_TOKEN_ID + 1


def shogi_position_feature_manifest() -> dict[str, object]:
    return {
        "input_schema_id": SHOGI_POSITION_INPUT_SCHEMA_ID,
        "coordinate_system": "side_to_move_relative_180_rotation",
        "global_token_count": GLOBAL_TOKEN_COUNT,
        "square_token_count": BOARD_TOKEN_COUNT,
        "piece_slot_count": PIECE_SLOT_COUNT,
        "line_token_count": LINE_TOKEN_COUNT,
        "feature_sequence_token_count": SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
        "global_feature_count": 1,
        "square_feature_count": SHOGI_POSITION_SQUARE_FEATURE_COUNT,
        "piece_feature_count": SHOGI_POSITION_PIECE_FEATURE_COUNT,
        "line_feature_count": SHOGI_POSITION_LINE_FEATURE_COUNT,
        "feature_vocab_size": SHOGI_POSITION_FEATURE_VOCAB_SIZE,
        "feature_groups": ["global", "square", "piece", "line", "pair_relation_edges"],
        "global_features": [
            "state",
            "side_to_move",
            "in_check",
            "move_count_bucket",
            "own_hand_counts",
            "opponent_hand_counts",
        ],
        "square_features": [
            "piece_identity",
            "own_attack_count",
            "opponent_attack_count",
            "own_piece_type_attacks",
            "opponent_piece_type_attacks",
            "own_king_relative_square",
            "opponent_king_relative_square",
            "own_drop_shadow",
            "opponent_drop_shadow",
            "counterfactual_removal_self_check",
            "counterfactual_removal_opponent_check",
            "counterfactual_removal_coarse_slider_blocker",
            "opponent_drop_potential_after_losing_piece",
            "own_drop_potential_after_capturing_piece",
        ],
        "piece_features": [
            "location_kind",
            "piece_identity",
            "relative_square",
            "own_king_relative_square",
            "opponent_king_relative_square",
            "counterfactual_removal_self_check",
            "counterfactual_removal_opponent_check",
            "counterfactual_removal_coarse_slider_blocker",
            "opponent_drop_potential_after_losing_piece",
            "own_drop_potential_after_capturing_piece",
        ],
        "line_features": [
            "line_kind",
            "own_king_on_line",
            "opponent_king_on_line",
            "own_slider_on_line",
            "opponent_slider_on_line",
            "occupancy_count",
        ],
        "pair_relations": {
            "none": PAIR_RELATION_NONE,
            "piece_on_square": PAIR_RELATION_PIECE_ON_SQUARE,
            "piece_attacks_square": PAIR_RELATION_PIECE_ATTACKS_SQUARE,
            "hand_piece_drops_to_square": PAIR_RELATION_HAND_PIECE_DROPS_TO_SQUARE,
            "piece_attacks_piece": PAIR_RELATION_PIECE_ATTACKS_PIECE,
            "piece_defends_piece": PAIR_RELATION_PIECE_DEFENDS_PIECE,
            "piece_same_side": PAIR_RELATION_PIECE_SAME_SIDE,
        },
        "pair_relation_representation": "edge_list",
        "hand_piece_types": list(HAND_PIECE_TYPES),
        "square_attack_piece_types": list(SQUARE_ATTACK_PIECE_TYPES),
        "attack_count_token_max": ATTACK_COUNT_TOKEN_MAX,
        "hand_count_token_max": HAND_COUNT_TOKEN_MAX,
        "move_count_buckets": [list(bucket) for bucket in MOVE_COUNT_BUCKETS],
        "move_count_bucket_overflow": MOVE_COUNT_BUCKET_OVERFLOW,
        "king_relative_square_bucket_count": KING_RELATIVE_SQUARE_OFFSET_BUCKET_COUNT,
        "line_token_layout": {
            "files": 9,
            "ranks": 9,
            "rising_diagonals": 17,
            "falling_diagonals": 17,
        },
        "token_offsets": {
            "state": STATE_TOKEN_INDEX,
            "global_side_to_move": GLOBAL_SIDE_TO_MOVE_TOKEN_INDEX,
            "global_in_check": GLOBAL_IN_CHECK_TOKEN_INDEX,
            "global_move_count": GLOBAL_MOVE_COUNT_TOKEN_INDEX,
            "global_hand": GLOBAL_HAND_TOKEN_OFFSET,
            "square": SQUARE_TOKEN_OFFSET,
            "piece": PIECE_TOKEN_OFFSET,
            "line": LINE_TOKEN_OFFSET,
        },
        "vocab_offsets": {
            "empty_square": EMPTY_SQUARE_TOKEN_ID,
            "own_piece": OWN_PIECE_OFFSET,
            "opponent_piece": OPPONENT_PIECE_OFFSET,
            "side_to_move_black": SIDE_TO_MOVE_BLACK_TOKEN_ID,
            "side_to_move_white": SIDE_TO_MOVE_WHITE_TOKEN_ID,
            "not_in_check": NOT_IN_CHECK_TOKEN_ID,
            "in_check": IN_CHECK_TOKEN_ID,
            "own_attack": OWN_ATTACK_OFFSET,
            "opponent_attack": OPPONENT_ATTACK_OFFSET,
            "own_hand": OWN_HAND_OFFSET,
            "opponent_hand": OPPONENT_HAND_OFFSET,
            "move_count_bucket": MOVE_COUNT_BUCKET_OFFSET,
            "own_square_piece_type_attack": OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
            "opponent_square_piece_type_attack": OPPONENT_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
            "own_king_relative_square": OWN_KING_RELATIVE_SQUARE_OFFSET,
            "opponent_king_relative_square": OPPONENT_KING_RELATIVE_SQUARE_OFFSET,
            "own_drop_shadow": OWN_DROP_SHADOW_OFFSET,
            "opponent_drop_shadow": OPPONENT_DROP_SHADOW_OFFSET,
            "line_kind": LINE_KIND_OFFSET,
            "line_own_king_on_line": LINE_OWN_KING_ON_LINE_OFFSET,
            "line_opponent_king_on_line": LINE_OPPONENT_KING_ON_LINE_OFFSET,
            "line_own_slider_on_line": LINE_OWN_SLIDER_ON_LINE_OFFSET,
            "line_opponent_slider_on_line": LINE_OPPONENT_SLIDER_ON_LINE_OFFSET,
            "line_occupancy_count": LINE_OCCUPANCY_COUNT_OFFSET,
            "counterfactual_removal_self_check": COUNTERFACTUAL_REMOVAL_SELF_CHECK_OFFSET,
            "counterfactual_removal_opponent_check": COUNTERFACTUAL_REMOVAL_OPPONENT_CHECK_OFFSET,
            "counterfactual_removal_coarse_slider_blocker": COUNTERFACTUAL_REMOVAL_COARSE_SLIDER_BLOCKER_OFFSET,
            "opponent_drop_potential_after_losing_piece": OPPONENT_DROP_POTENTIAL_AFTER_LOSING_PIECE_OFFSET,
            "own_drop_potential_after_capturing_piece": OWN_DROP_POTENTIAL_AFTER_CAPTURING_PIECE_OFFSET,
            "piece_location_empty": PIECE_LOCATION_EMPTY_TOKEN_ID,
            "piece_location_board": PIECE_LOCATION_BOARD_TOKEN_ID,
            "piece_location_hand": PIECE_LOCATION_HAND_TOKEN_ID,
            "piece_square_unknown": PIECE_SQUARE_UNKNOWN_TOKEN_ID,
            "piece_square": PIECE_SQUARE_OFFSET,
            "position_state": SHOGI_POSITION_STATE_TOKEN_ID,
        },
    }


def shogi_position_feature_manifest_hash() -> str:
    payload = json.dumps(shogi_position_feature_manifest(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SHOGI_POSITION_FEATURE_MANIFEST = shogi_position_feature_manifest()
SHOGI_POSITION_FEATURE_MANIFEST_HASH = shogi_position_feature_manifest_hash()
