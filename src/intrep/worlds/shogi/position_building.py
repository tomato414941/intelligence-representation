from __future__ import annotations

import shogi
import torch

from intrep.worlds.shogi.position_derived import _ShogiPositionDerivedRelations
from intrep.worlds.shogi.position_features import ShogiPositionFeatures
from intrep.worlds.shogi.position_pair_relations import pair_relation_edges_from_board
from intrep.worlds.shogi.position_piece_features import piece_feature_id_rows, piece_slot_relation_infos
from intrep.worlds.shogi.position_line_features import line_feature_id_rows
from intrep.worlds.shogi.position_schema import SHOGI_POSITION_STATE_FEATURE_ID
from intrep.worlds.shogi.position_square_features import (
    drop_shadow_token_id_rows,
    hand_token_ids,
    in_check_token_id,
    legal_drop_targets_by_piece_type,
    move_count_bucket_feature_id,
    side_to_move_token_id,
    square_feature_id_rows,
)
from intrep.worlds.shogi.position_tactical_heuristics import (
    counterfactual_removal_token_id_rows,
    drop_potential_token_id_rows,
)
from intrep.worlds.shogi.position_geometry import opponent_color


def shogi_position_features_from_sfen(position_sfen: str) -> ShogiPositionFeatures:
    board = shogi.Board(position_sfen)
    derived = _shogi_position_derived_relations(board)
    global_feature_ids = torch.tensor(
        [
            SHOGI_POSITION_STATE_FEATURE_ID,
            side_to_move_token_id(board.turn),
            in_check_token_id(board.is_check()),
            move_count_bucket_feature_id(board.move_number),
            *hand_token_ids(board),
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
        counterfactual_removal_token_rows=counterfactual_removal_token_id_rows(board),
        drop_potential_token_rows=drop_potential_token_id_rows(board),
        drop_shadow_token_rows=drop_shadow_token_id_rows(board, legal_drop_targets_by_color=legal_drop_targets_by_color),
        legal_drop_targets_by_color=legal_drop_targets_by_color,
        piece_slot_relation_infos=piece_slot_relation_infos(board),
    )
