from __future__ import annotations

import shogi
import torch


BOARD_TOKEN_COUNT = 81
ATTACK_TOKEN_COUNT = BOARD_TOKEN_COUNT * 2
HAND_TOKEN_COUNT = 14
GLOBAL_TOKEN_COUNT = 18
SIDE_TO_MOVE_TOKEN_INDEX = 0
IN_CHECK_TOKEN_INDEX = 1
MOVE_COUNT_TOKEN_INDEX = 2
BOARD_TOKEN_OFFSET = 3
ATTACK_TOKEN_OFFSET = BOARD_TOKEN_OFFSET + BOARD_TOKEN_COUNT
HAND_TOKEN_OFFSET = ATTACK_TOKEN_OFFSET + ATTACK_TOKEN_COUNT
SHOGI_POSITION_TOKEN_COUNT = HAND_TOKEN_OFFSET + HAND_TOKEN_COUNT
SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT = GLOBAL_TOKEN_COUNT + BOARD_TOKEN_COUNT
SHOGI_POSITION_INPUT_ENCODING = "shogi_global_square_feature_sequence_v1"

STATE_TOKEN_INDEX = 0
GLOBAL_SIDE_TO_MOVE_TOKEN_INDEX = 1
GLOBAL_IN_CHECK_TOKEN_INDEX = 2
GLOBAL_MOVE_COUNT_TOKEN_INDEX = 3
GLOBAL_HAND_TOKEN_OFFSET = 4
SQUARE_TOKEN_OFFSET = GLOBAL_TOKEN_COUNT

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
SHOGI_POSITION_VOCAB_SIZE = MOVE_COUNT_BUCKET_OFFSET + MOVE_COUNT_BUCKET_VOCAB_SIZE
SHOGI_POSITION_GLOBAL_SLOT_COUNT = GLOBAL_TOKEN_COUNT
SHOGI_POSITION_SQUARE_COUNT = BOARD_TOKEN_COUNT
SHOGI_POSITION_SQUARE_FEATURE_COUNT = 3
SHOGI_POSITION_SQUARE_SLOT_COUNT = BOARD_TOKEN_COUNT
SHOGI_POSITION_STATE_TOKEN_ID = SHOGI_POSITION_VOCAB_SIZE
SHOGI_POSITION_FEATURE_VOCAB_SIZE = SHOGI_POSITION_STATE_TOKEN_ID + 1

HAND_PIECE_TYPES = (
    shogi.PAWN,
    shogi.LANCE,
    shogi.KNIGHT,
    shogi.SILVER,
    shogi.GOLD,
    shogi.BISHOP,
    shogi.ROOK,
)


def shogi_position_token_ids_from_sfen(position_sfen: str) -> torch.Tensor:
    board = shogi.Board(position_sfen)
    token_ids = [
        side_to_move_token_id(board.turn),
        in_check_token_id(board.is_check()),
        move_count_bucket_token_id(board.move_number),
    ]
    token_ids.extend(relative_square_token_id(board, square) for square in range(BOARD_TOKEN_COUNT))
    token_ids.extend(attack_token_ids(board))
    token_ids.extend(hand_token_ids(board))
    return torch.tensor(token_ids, dtype=torch.long)


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


def absolute_to_relative_square(square: int, turn: int) -> int:
    if turn == shogi.BLACK:
        return square
    if turn == shogi.WHITE:
        return BOARD_TOKEN_COUNT - 1 - square
    raise ValueError(f"unsupported shogi color: {turn}")


def relative_to_absolute_square(square: int, turn: int) -> int:
    return absolute_to_relative_square(square, turn)


def opponent_color(color: int) -> int:
    if color == shogi.BLACK:
        return shogi.WHITE
    if color == shogi.WHITE:
        return shogi.BLACK
    raise ValueError(f"unsupported shogi color: {color}")
