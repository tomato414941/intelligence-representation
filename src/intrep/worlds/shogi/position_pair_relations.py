from __future__ import annotations

import torch
import shogi

from intrep.worlds.shogi.position_derived import _ShogiPositionDerivedRelations
from intrep.worlds.shogi.position_features import ShogiPairRelationEdges
from intrep.worlds.shogi.position_geometry import absolute_to_relative_square, relative_to_absolute_square
from intrep.worlds.shogi.position_schema import *


def pair_relation_edges_from_board(
    board: shogi.Board,
    *,
    derived: _ShogiPositionDerivedRelations | None = None,
) -> ShogiPairRelationEdges:
    source_token_indices: list[int] = []
    target_token_indices: list[int] = []
    relation_ids: list[int] = []

    def add_edge(source: int, target: int, relation_id: int) -> None:
        source_token_indices.append(source)
        target_token_indices.append(target)
        relation_ids.append(relation_id)

    def add_bidirectional_edge(source: int, target: int, relation_id: int) -> None:
        add_edge(source, target, relation_id)
        add_edge(target, source, relation_id)

    if derived is None:
        from intrep.worlds.shogi.position_building import _shogi_position_derived_relations

        derived = _shogi_position_derived_relations(board)
    slot_infos = derived.piece_slot_relation_infos
    legal_drop_targets = derived.legal_drop_targets_by_color
    for piece_slot, info in enumerate(slot_infos):
        if info.piece is None:
            continue
        piece_token_index = PIECE_TOKEN_OFFSET + piece_slot
        if info.location_kind == "board" and info.relative_square is not None:
            from_absolute_square = relative_to_absolute_square(info.relative_square, board.turn)
            square_token_index = SQUARE_TOKEN_OFFSET + info.relative_square
            add_bidirectional_edge(piece_token_index, square_token_index, PAIR_RELATION_PIECE_ON_SQUARE)
            for relative_square in range(BOARD_TOKEN_COUNT):
                absolute_square = relative_to_absolute_square(relative_square, board.turn)
                if from_absolute_square in board.attackers(info.piece.color, absolute_square):
                    target_token_index = SQUARE_TOKEN_OFFSET + relative_square
                    add_bidirectional_edge(piece_token_index, target_token_index, PAIR_RELATION_PIECE_ATTACKS_SQUARE)
        elif info.location_kind == "hand":
            targets = legal_drop_targets[info.piece.color].get(info.piece.piece_type, set())
            for absolute_square in targets:
                relative_square = absolute_to_relative_square(absolute_square, board.turn)
                square_token_index = SQUARE_TOKEN_OFFSET + relative_square
                add_bidirectional_edge(piece_token_index, square_token_index, PAIR_RELATION_HAND_PIECE_DROPS_TO_SQUARE)

    for source_slot, source_info in enumerate(slot_infos):
        if source_info.piece is None:
            continue
        source_token_index = PIECE_TOKEN_OFFSET + source_slot
        for target_slot, target_info in enumerate(slot_infos):
            if source_slot == target_slot or target_info.piece is None:
                continue
            target_token_index = PIECE_TOKEN_OFFSET + target_slot
            if source_info.piece.color == target_info.piece.color:
                add_edge(source_token_index, target_token_index, PAIR_RELATION_PIECE_SAME_SIDE)
            if source_info.location_kind != "board" or target_info.location_kind != "board":
                continue
            if source_info.relative_square is None or target_info.relative_square is None:
                continue
            source_absolute_square = relative_to_absolute_square(source_info.relative_square, board.turn)
            target_absolute_square = relative_to_absolute_square(target_info.relative_square, board.turn)
            if source_absolute_square in board.attackers(source_info.piece.color, target_absolute_square):
                add_edge(
                    source_token_index,
                    target_token_index,
                    PAIR_RELATION_PIECE_DEFENDS_PIECE
                    if source_info.piece.color == target_info.piece.color
                    else PAIR_RELATION_PIECE_ATTACKS_PIECE
                )
    return ShogiPairRelationEdges(
        source_token_indices=torch.tensor(source_token_indices, dtype=torch.long),
        target_token_indices=torch.tensor(target_token_indices, dtype=torch.long),
        relation_ids=torch.tensor(relation_ids, dtype=torch.long),
    )
