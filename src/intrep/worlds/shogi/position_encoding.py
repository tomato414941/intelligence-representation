from __future__ import annotations

from intrep.worlds.shogi.position_schema import *
from intrep.worlds.shogi.position_features import (
    ShogiPairRelationEdges,
    ShogiPositionFeatures,
    stack_shogi_pair_relation_edges,
    stack_shogi_position_features,
    validate_shogi_position_feature_structure,
)
from intrep.worlds.shogi.position_building import shogi_position_features_from_sfen
from intrep.worlds.shogi.position_geometry import (
    absolute_to_relative_square,
    king_relative_offset_bucket,
    opponent_color,
    relative_to_absolute_square,
)
from intrep.worlds.shogi.position_square_features import (
    attack_count_token_id,
    attack_token_ids,
    drop_shadow_token_id_rows,
    drop_shadow_token_ids,
    hand_token_ids,
    in_check_token_id,
    king_relative_square_token_id,
    king_relative_square_token_ids,
    legal_drop_targets_by_piece_type,
    move_count_bucket_feature_id,
    piece_token_id,
    relative_square_token_id,
    side_to_move_token_id,
    square_feature_id_rows,
    square_piece_type_attack_token_id_rows,
    square_piece_type_attack_token_ids,
)
from intrep.worlds.shogi.position_tactical_heuristics import (
    counterfactual_removal_token_id_rows,
    drop_potential_token_id_rows,
    hand_piece_type_after_capture,
    has_pseudo_drop_target,
    king_is_attacked,
    king_zone_absolute_squares,
    line_side_has_piece,
    line_side_has_slider,
    piece_type_can_drop_on_square,
    coarse_slider_line_blocker,
)
from intrep.worlds.shogi.position_line_features import (
    king_on_absolute_squares,
    line_feature_id_rows,
    line_feature_token_ids,
    line_kind_index,
    line_slot_feature_token_ids,
    piece_slides_on_line,
    slider_on_absolute_squares,
    squares_for_line_index,
)
from intrep.worlds.shogi.position_piece_features import (
    PieceSlotRelationInfo,
    board_piece_slot_token_ids,
    empty_piece_slot_token_ids,
    hand_piece_slot_token_ids,
    hand_piece_token_ids,
    piece_feature_id_rows,
    piece_feature_token_ids,
    piece_slot_relation_infos,
)
from intrep.worlds.shogi.position_pair_relations import pair_relation_edges_from_board
