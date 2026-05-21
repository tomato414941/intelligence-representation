from __future__ import annotations

from intrep.representation.inputs.shogi_position_features.position_schema import *
from intrep.representation.inputs.shogi_position_features.position_features import (
    ShogiPairRelationEdges,
    ShogiPositionFeatures,
    stack_shogi_pair_relation_edges,
    stack_shogi_position_features,
    validate_integer_tensor_shape,
    validate_pair_relation_edge_structure,
)
from intrep.representation.inputs.shogi_position_features.position_rich_building import shogi_rich_position_features_from_sfen
from intrep.domains.shogi.coordinates import (
    absolute_to_relative_square,
    king_relative_offset_bucket,
    opponent_color,
    relative_to_absolute_square,
)
from intrep.representation.inputs.shogi_position_features.position_square_features import (
    attack_count_feature_id,
    attack_feature_ids,
    drop_shadow_feature_id_rows,
    drop_shadow_feature_ids,
    hand_feature_ids,
    in_check_feature_id,
    king_relative_square_feature_id,
    king_relative_square_feature_ids,
    legal_drop_targets_by_piece_type,
    move_count_bucket_feature_id,
    piece_feature_id,
    relative_square_feature_id,
    side_to_move_feature_id,
    square_feature_id_rows,
    square_piece_type_attack_feature_id_rows,
    square_piece_type_attack_feature_ids,
)
from intrep.representation.inputs.shogi_position_features.position_tactical_heuristics import (
    counterfactual_removal_feature_id_rows,
    drop_potential_feature_id_rows,
    hand_piece_type_after_capture,
    has_pseudo_drop_target,
    king_is_attacked,
    king_zone_absolute_squares,
    line_side_has_piece,
    line_side_has_slider,
    piece_type_can_drop_on_square,
    coarse_slider_line_blocker,
)
from intrep.representation.inputs.shogi_position_features.position_line_features import (
    king_on_absolute_squares,
    line_element_feature_ids,
    line_feature_id_rows,
    line_feature_ids,
    line_kind_index,
    piece_slides_on_line,
    slider_on_absolute_squares,
    squares_for_line_index,
)
from intrep.representation.inputs.shogi_position_features.position_piece_features import (
    PieceElementRelationInfo,
    board_piece_element_feature_ids,
    empty_piece_element_feature_ids,
    hand_piece_element_feature_ids,
    hand_piece_feature_ids,
    piece_feature_id_rows,
    piece_feature_ids,
    piece_element_relation_infos,
)
from intrep.representation.inputs.shogi_position_features.position_pair_relations import pair_relation_edges_from_board


def validate_shogi_rich_position_feature_structure(features: ShogiPositionFeatures) -> None:
    validate_integer_tensor_shape(
        "global_feature_ids",
        features.global_feature_ids,
        (SHOGI_RICH_POSITION_GLOBAL_ELEMENT_COUNT,),
    )
    validate_integer_tensor_shape(
        "square_feature_ids",
        features.square_feature_ids,
        (SHOGI_POSITION_SQUARE_COUNT, SHOGI_RICH_POSITION_SQUARE_FEATURE_COUNT),
    )
    validate_integer_tensor_shape(
        "piece_feature_ids",
        features.piece_feature_ids,
        (SHOGI_RICH_POSITION_PIECE_SLOT_COUNT, SHOGI_RICH_POSITION_PIECE_FEATURE_COUNT),
    )
    validate_integer_tensor_shape(
        "line_feature_ids",
        features.line_feature_ids,
        (SHOGI_RICH_POSITION_LINE_ELEMENT_COUNT, SHOGI_RICH_POSITION_LINE_FEATURE_COUNT),
    )
    validate_pair_relation_edge_structure(
        features.pair_relation_edges,
        element_count=SHOGI_RICH_POSITION_ELEMENT_COUNT,
        relation_count=PAIR_RELATION_COUNT,
    )
