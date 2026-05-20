from __future__ import annotations

import shogi

from intrep.worlds.shogi.position_derived import _ShogiPositionDerivedRelations
from intrep.worlds.shogi.position_geometry import (
    absolute_to_relative_square,
    king_relative_offset_bucket,
    opponent_color,
    relative_to_absolute_square,
)
from intrep.worlds.shogi.position_schema import *


def side_to_move_token_id(color: int) -> int:
    if color == shogi.BLACK:
        return SIDE_TO_MOVE_BLACK_TOKEN_ID
    if color == shogi.WHITE:
        return SIDE_TO_MOVE_WHITE_TOKEN_ID
    raise ValueError(f"unsupported shogi color: {color}")


def in_check_token_id(in_check: bool) -> int:
    return IN_CHECK_TOKEN_ID if in_check else NOT_IN_CHECK_TOKEN_ID


def move_count_bucket_token_id(move_number: int | None) -> int:
    if move_number is None or move_number <= 0:
        return MOVE_COUNT_BUCKET_OFFSET + MOVE_COUNT_BUCKET_UNKNOWN
    for bucket_index, (start, end) in enumerate(MOVE_COUNT_BUCKETS, start=1):
        if start <= move_number <= end:
            return MOVE_COUNT_BUCKET_OFFSET + bucket_index
    return MOVE_COUNT_BUCKET_OFFSET + MOVE_COUNT_BUCKET_OVERFLOW


def relative_square_token_id(board: shogi.Board, relative_square: int) -> int:
    absolute_square = relative_to_absolute_square(relative_square, board.turn)
    return piece_token_id(board.piece_at(absolute_square), own_color=board.turn)


def piece_token_id(piece: shogi.Piece | None, *, own_color: int) -> int:
    if piece is None:
        return EMPTY_SQUARE_TOKEN_ID
    if piece.color == own_color:
        return OWN_PIECE_OFFSET + int(piece.piece_type) - 1
    if piece.color == opponent_color(own_color):
        return OPPONENT_PIECE_OFFSET + int(piece.piece_type) - 1
    raise ValueError(f"unsupported shogi piece color: {piece.color}")


def hand_token_ids(board: shogi.Board) -> list[int]:
    token_ids: list[int] = []
    for color, offset in ((board.turn, OWN_HAND_OFFSET), (opponent_color(board.turn), OPPONENT_HAND_OFFSET)):
        pieces_in_hand = board.pieces_in_hand[color]
        for piece_type in HAND_PIECE_TYPES:
            count = pieces_in_hand[piece_type]
            token_ids.append(offset + min(count, HAND_COUNT_TOKEN_MAX))
    return token_ids


def attack_token_ids(board: shogi.Board) -> list[int]:
    token_ids: list[int] = []
    for relative_square in range(BOARD_TOKEN_COUNT):
        absolute_square = relative_to_absolute_square(relative_square, board.turn)
        token_ids.append(attack_count_token_id(board, board.turn, absolute_square, offset=OWN_ATTACK_OFFSET))
    opponent = opponent_color(board.turn)
    for relative_square in range(BOARD_TOKEN_COUNT):
        absolute_square = relative_to_absolute_square(relative_square, board.turn)
        token_ids.append(attack_count_token_id(board, opponent, absolute_square, offset=OPPONENT_ATTACK_OFFSET))
    return token_ids


def attack_count_token_id(board: shogi.Board, color: int, square: int, *, offset: int) -> int:
    count = len(board.attackers(color, square))
    return offset + min(count, ATTACK_COUNT_TOKEN_MAX)


def square_feature_id_rows(
    board: shogi.Board,
    *,
    derived: _ShogiPositionDerivedRelations | None = None,
) -> list[list[int]]:
    pieces = [relative_square_token_id(board, square) for square in range(BOARD_TOKEN_COUNT)]
    attacks = attack_token_ids(board)
    own_square_piece_type_attacks, opponent_square_piece_type_attacks = square_piece_type_attack_token_id_rows(board)
    king_relative_squares = king_relative_square_token_ids(board)
    if derived is None:
        from intrep.worlds.shogi.position_building import _shogi_position_derived_relations

        derived = _shogi_position_derived_relations(board)
    rows: list[list[int]] = []
    for relative_square in range(BOARD_TOKEN_COUNT):
        rows.append(
            [
                pieces[relative_square],
                attacks[relative_square],
                attacks[BOARD_TOKEN_COUNT + relative_square],
                *own_square_piece_type_attacks[relative_square],
                *opponent_square_piece_type_attacks[relative_square],
                king_relative_squares[relative_square],
                king_relative_squares[BOARD_TOKEN_COUNT + relative_square],
                *derived.drop_shadow_token_rows[relative_square],
                *derived.counterfactual_removal_token_rows[relative_square],
                *derived.gift_flow_token_rows[relative_square],
            ]
        )
    return rows


def square_piece_type_attack_token_ids(board: shogi.Board) -> list[int]:
    token_ids: list[int] = []
    own_rows, opponent_rows = square_piece_type_attack_token_id_rows(board)
    for row in own_rows:
        token_ids.extend(row)
    for row in opponent_rows:
        token_ids.extend(row)
    return token_ids


def square_piece_type_attack_token_id_rows(board: shogi.Board) -> tuple[list[list[int]], list[list[int]]]:
    return (
        _square_piece_type_attack_token_id_rows_for_color(
            board,
            board.turn,
            offset=OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
        ),
        _square_piece_type_attack_token_id_rows_for_color(
            board,
            opponent_color(board.turn),
            offset=OPPONENT_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
        ),
    )


def _square_piece_type_attack_token_id_rows_for_color(board: shogi.Board, color: int, *, offset: int) -> list[list[int]]:
    rows: list[list[int]] = []
    piece_type_to_index = {piece_type: index for index, piece_type in enumerate(SQUARE_ATTACK_PIECE_TYPES)}
    for relative_square in range(BOARD_TOKEN_COUNT):
        absolute_square = relative_to_absolute_square(relative_square, board.turn)
        attacked_piece_types: set[int] = set()
        for attacker_square in board.attackers(color, absolute_square):
            piece = board.piece_at(attacker_square)
            if piece is not None:
                attacked_piece_types.add(int(piece.piece_type))
        row: list[int] = []
        for piece_type in SQUARE_ATTACK_PIECE_TYPES:
            feature_index = piece_type_to_index[piece_type]
            row.append(offset + feature_index * 2 + int(piece_type in attacked_piece_types))
        rows.append(row)
    return rows


def king_relative_square_token_ids(board: shogi.Board) -> list[int]:
    token_ids: list[int] = []
    token_ids.extend(
        _king_relative_square_token_ids_for_color(
            board,
            board.turn,
            offset=OWN_KING_RELATIVE_SQUARE_OFFSET,
        )
    )
    token_ids.extend(
        _king_relative_square_token_ids_for_color(
            board,
            opponent_color(board.turn),
            offset=OPPONENT_KING_RELATIVE_SQUARE_OFFSET,
        )
    )
    return token_ids


def _king_relative_square_token_ids_for_color(board: shogi.Board, color: int, *, offset: int) -> list[int]:
    return [
        king_relative_square_token_id(board, color, relative_square, offset=offset)
        for relative_square in range(BOARD_TOKEN_COUNT)
    ]


def king_relative_square_token_id(board: shogi.Board, color: int, relative_square: int, *, offset: int) -> int:
    king_square = board.king_squares[color]
    if king_square is None:
        return offset + KING_RELATIVE_SQUARE_BUCKET_UNKNOWN
    relative_king_square = absolute_to_relative_square(int(king_square), board.turn)
    return offset + 1 + king_relative_offset_bucket(relative_square, relative_king_square)


def drop_shadow_token_ids(board: shogi.Board) -> list[int]:
    token_ids: list[int] = []
    rows = drop_shadow_token_id_rows(board)
    for row in rows:
        token_ids.extend(row[: len(HAND_PIECE_TYPES)])
    for row in rows:
        token_ids.extend(row[len(HAND_PIECE_TYPES) :])
    return token_ids


def drop_shadow_token_id_rows(
    board: shogi.Board,
    *,
    legal_drop_targets_by_color: dict[int, dict[int, set[int]]] | None = None,
) -> list[list[int]]:
    if legal_drop_targets_by_color is None:
        legal_drop_targets_by_color = {
            board.turn: legal_drop_targets_by_piece_type(board, board.turn),
            opponent_color(board.turn): legal_drop_targets_by_piece_type(board, opponent_color(board.turn)),
        }
    own_legal_drop_targets = legal_drop_targets_by_color[board.turn]
    opponent_legal_drop_targets = legal_drop_targets_by_color[opponent_color(board.turn)]
    rows: list[list[int]] = []
    for relative_square in range(BOARD_TOKEN_COUNT):
        absolute_square = relative_to_absolute_square(relative_square, board.turn)
        row: list[int] = []
        for piece_index, piece_type in enumerate(HAND_PIECE_TYPES):
            row.append(OWN_DROP_SHADOW_OFFSET + piece_index * 2 + int(absolute_square in own_legal_drop_targets[piece_type]))
        for piece_index, piece_type in enumerate(HAND_PIECE_TYPES):
            row.append(
                OPPONENT_DROP_SHADOW_OFFSET
                + piece_index * 2
                + int(absolute_square in opponent_legal_drop_targets[piece_type])
            )
        rows.append(row)
    return rows


def legal_drop_targets_by_piece_type(board: shogi.Board, color: int) -> dict[int, set[int]]:
    perspective_board = shogi.Board(board.sfen())
    perspective_board.turn = color
    targets = {piece_type: set() for piece_type in HAND_PIECE_TYPES}
    for move in perspective_board.legal_moves:
        if move.drop_piece_type in targets:
            targets[int(move.drop_piece_type)].add(int(move.to_square))
    return targets
