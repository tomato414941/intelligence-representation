from __future__ import annotations

import torch
import shogi

from intrep.representation.inputs.shogi_position_features.position_derived import _ShogiPositionDerivedRelations
from intrep.representation.inputs.shogi_position_features.position_features import ShogiPairRelationEdges
from intrep.worlds.shogi.coordinates import absolute_to_relative_square, relative_to_absolute_square
from intrep.representation.inputs.shogi_position_features.position_schema import *


def pair_relation_edges_from_board(
    board: shogi.Board,
    *,
    derived: _ShogiPositionDerivedRelations | None = None,
) -> ShogiPairRelationEdges:
    source_element_indices: list[int] = []
    target_element_indices: list[int] = []
    relation_ids: list[int] = []

    def add_edge(source: int, target: int, relation_id: int) -> None:
        source_element_indices.append(source)
        target_element_indices.append(target)
        relation_ids.append(relation_id)

    def add_bidirectional_edge(source: int, target: int, relation_id: int) -> None:
        add_edge(source, target, relation_id)
        add_edge(target, source, relation_id)

    if derived is None:
        from intrep.representation.inputs.shogi_position_features.position_rich_building import _shogi_position_derived_relations

        derived = _shogi_position_derived_relations(board)
    piece_infos = derived.piece_element_relation_infos
    legal_drop_targets = derived.legal_drop_targets_by_color
    for piece_element_offset, info in enumerate(piece_infos):
        if info.piece is None:
            continue
        piece_element_index = RICH_PIECE_ELEMENT_OFFSET + piece_element_offset
        if info.location_kind == "board" and info.relative_square is not None:
            from_absolute_square = relative_to_absolute_square(info.relative_square, board.turn)
            square_element_index = RICH_SQUARE_ELEMENT_OFFSET + info.relative_square
            add_bidirectional_edge(piece_element_index, square_element_index, PAIR_RELATION_PIECE_ON_SQUARE)
            for relative_square in range(SQUARE_ELEMENT_COUNT):
                absolute_square = relative_to_absolute_square(relative_square, board.turn)
                if from_absolute_square in board.attackers(info.piece.color, absolute_square):
                    target_element_index = RICH_SQUARE_ELEMENT_OFFSET + relative_square
                    add_bidirectional_edge(piece_element_index, target_element_index, PAIR_RELATION_PIECE_ATTACKS_SQUARE)
        elif info.location_kind == "hand":
            targets = legal_drop_targets[info.piece.color].get(info.piece.piece_type, set())
            for absolute_square in targets:
                relative_square = absolute_to_relative_square(absolute_square, board.turn)
                square_element_index = RICH_SQUARE_ELEMENT_OFFSET + relative_square
                add_bidirectional_edge(piece_element_index, square_element_index, PAIR_RELATION_HAND_PIECE_DROPS_TO_SQUARE)

    for source_element_offset, source_info in enumerate(piece_infos):
        if source_info.piece is None:
            continue
        source_element_index = RICH_PIECE_ELEMENT_OFFSET + source_element_offset
        for target_element_offset, target_info in enumerate(piece_infos):
            if source_element_offset == target_element_offset or target_info.piece is None:
                continue
            target_element_index = RICH_PIECE_ELEMENT_OFFSET + target_element_offset
            if source_info.piece.color == target_info.piece.color:
                add_edge(source_element_index, target_element_index, PAIR_RELATION_PIECE_SAME_SIDE)
            if source_info.location_kind != "board" or target_info.location_kind != "board":
                continue
            if source_info.relative_square is None or target_info.relative_square is None:
                continue
            source_absolute_square = relative_to_absolute_square(source_info.relative_square, board.turn)
            target_absolute_square = relative_to_absolute_square(target_info.relative_square, board.turn)
            if source_absolute_square in board.attackers(source_info.piece.color, target_absolute_square):
                add_edge(
                    source_element_index,
                    target_element_index,
                    PAIR_RELATION_PIECE_DEFENDS_PIECE
                    if source_info.piece.color == target_info.piece.color
                    else PAIR_RELATION_PIECE_ATTACKS_PIECE
                )
    return ShogiPairRelationEdges(
        source_element_indices=torch.tensor(source_element_indices, dtype=torch.long),
        target_element_indices=torch.tensor(target_element_indices, dtype=torch.long),
        relation_ids=torch.tensor(relation_ids, dtype=torch.long),
    )
