from __future__ import annotations

import shogi
import torch

from intrep.representation.inputs.shogi_position_features.position_derived import _ShogiPositionDerivedRelations
from intrep.representation.inputs.shogi_position_features.position_features import ShogiPositionFeatures
from intrep.representation.inputs.shogi_position_features.position_pair_relations import pair_relation_edges_from_board
from intrep.representation.inputs.shogi_position_features.position_piece_features import piece_feature_id_rows, piece_slot_relation_infos
from intrep.representation.inputs.shogi_position_features.position_line_features import line_feature_id_rows
from intrep.representation.inputs.shogi_position_features.position_schema import SHOGI_POSITION_STATE_FEATURE_ID
from intrep.representation.inputs.shogi_position_features.position_square_features import (
    drop_shadow_feature_id_rows,
    hand_feature_ids,
    in_check_feature_id,
    legal_drop_targets_by_piece_type,
    move_count_bucket_feature_id,
    side_to_move_feature_id,
    square_feature_id_rows,
)
from intrep.representation.inputs.shogi_position_features.position_tactical_heuristics import (
    counterfactual_removal_feature_id_rows,
    drop_potential_feature_id_rows,
)
from intrep.domains.shogi.coordinates import opponent_color


def shogi_rich_position_features_from_sfen(position_sfen: str) -> ShogiPositionFeatures:
    board = shogi.Board(position_sfen)
    derived = _shogi_position_derived_relations(board)
    global_feature_ids = torch.tensor(
        [
            SHOGI_POSITION_STATE_FEATURE_ID,
            side_to_move_feature_id(board.turn),
            in_check_feature_id(board.is_check()),
            move_count_bucket_feature_id(board.move_number),
            *hand_feature_ids(board),
        ],
        dtype=torch.long,
    )
    square_feature_ids = torch.tensor(square_feature_id_rows(board, derived=derived), dtype=torch.long)
    piece_feature_ids = torch.tensor(piece_feature_id_rows(board, derived=derived), dtype=torch.long)
    line_feature_ids = torch.tensor(line_feature_id_rows(board), dtype=torch.long)
    pair_relation_edges = pair_relation_edges_from_board(board, derived=derived)
    return ShogiPositionFeatures(
        global_feature_ids=global_feature_ids,
        square_feature_ids=square_feature_ids,
        piece_feature_ids=piece_feature_ids,
        line_feature_ids=line_feature_ids,
        pair_relation_edges=pair_relation_edges,
    )


def _shogi_position_derived_relations(board: shogi.Board) -> _ShogiPositionDerivedRelations:
    legal_drop_targets_by_color = {
        board.turn: legal_drop_targets_by_piece_type(board, board.turn),
        opponent_color(board.turn): legal_drop_targets_by_piece_type(board, opponent_color(board.turn)),
    }
    return _ShogiPositionDerivedRelations(
        counterfactual_removal_feature_rows=counterfactual_removal_feature_id_rows(board),
        drop_potential_feature_rows=drop_potential_feature_id_rows(board),
        drop_shadow_feature_rows=drop_shadow_feature_id_rows(board, legal_drop_targets_by_color=legal_drop_targets_by_color),
        legal_drop_targets_by_color=legal_drop_targets_by_color,
        piece_slot_relation_infos=piece_slot_relation_infos(board),
    )
