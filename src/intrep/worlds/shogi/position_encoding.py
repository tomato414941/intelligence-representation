from __future__ import annotations

from dataclasses import dataclass

import shogi
import torch


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
PIECE_FEATURE_COUNT = 5
GLOBAL_TOKEN_COUNT = 18
LINE_TOKEN_COUNT = 9 + 9 + 17 + 17
LINE_FEATURE_COUNT = 6
SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT = GLOBAL_TOKEN_COUNT + BOARD_TOKEN_COUNT + PIECE_SLOT_COUNT + LINE_TOKEN_COUNT
SHOGI_POSITION_INPUT_SCHEMA_ID = "shogi_global_square_drop_shadow_all_piece_line_state_feature_sequence"

STATE_TOKEN_INDEX = 0
GLOBAL_SIDE_TO_MOVE_TOKEN_INDEX = 1
GLOBAL_IN_CHECK_TOKEN_INDEX = 2
GLOBAL_MOVE_COUNT_TOKEN_INDEX = 3
GLOBAL_HAND_TOKEN_OFFSET = 4
SQUARE_TOKEN_OFFSET = GLOBAL_TOKEN_COUNT
PIECE_TOKEN_OFFSET = SQUARE_TOKEN_OFFSET + BOARD_TOKEN_COUNT
LINE_TOKEN_OFFSET = PIECE_TOKEN_OFFSET + PIECE_SLOT_COUNT

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
PIECE_LOCATION_EMPTY_TOKEN_ID = LINE_OCCUPANCY_COUNT_OFFSET + LINE_OCCUPANCY_COUNT_MAX + 1
PIECE_LOCATION_BOARD_TOKEN_ID = PIECE_LOCATION_EMPTY_TOKEN_ID + 1
PIECE_LOCATION_HAND_TOKEN_ID = PIECE_LOCATION_BOARD_TOKEN_ID + 1
PIECE_SQUARE_UNKNOWN_TOKEN_ID = PIECE_LOCATION_HAND_TOKEN_ID + 1
PIECE_SQUARE_OFFSET = PIECE_SQUARE_UNKNOWN_TOKEN_ID + 1
SHOGI_POSITION_VOCAB_SIZE = PIECE_SQUARE_OFFSET + BOARD_TOKEN_COUNT
SHOGI_POSITION_GLOBAL_SLOT_COUNT = GLOBAL_TOKEN_COUNT
SHOGI_POSITION_SQUARE_COUNT = BOARD_TOKEN_COUNT
SHOGI_POSITION_SQUARE_FEATURE_COUNT = 3 + SQUARE_ATTACK_PIECE_TYPE_COUNT * 2 + 2 + len(HAND_PIECE_TYPES) * 2
SHOGI_POSITION_SQUARE_SLOT_COUNT = BOARD_TOKEN_COUNT
SHOGI_POSITION_PIECE_SLOT_COUNT = PIECE_SLOT_COUNT
SHOGI_POSITION_PIECE_FEATURE_COUNT = PIECE_FEATURE_COUNT
SHOGI_POSITION_LINE_SLOT_COUNT = LINE_TOKEN_COUNT
SHOGI_POSITION_LINE_FEATURE_COUNT = LINE_FEATURE_COUNT
SHOGI_POSITION_STATE_TOKEN_ID = SHOGI_POSITION_VOCAB_SIZE
SHOGI_POSITION_FEATURE_VOCAB_SIZE = SHOGI_POSITION_STATE_TOKEN_ID + 1


@dataclass(frozen=True)
class ShogiPositionFeatures:
    global_feature_ids: torch.Tensor
    square_feature_ids: torch.Tensor
    piece_feature_ids: torch.Tensor
    line_feature_ids: torch.Tensor

    def to(self, device: torch.device) -> "ShogiPositionFeatures":
        return ShogiPositionFeatures(
            global_feature_ids=self.global_feature_ids.to(device),
            square_feature_ids=self.square_feature_ids.to(device),
            piece_feature_ids=self.piece_feature_ids.to(device),
            line_feature_ids=self.line_feature_ids.to(device),
        )


def stack_shogi_position_features(features: list[ShogiPositionFeatures]) -> ShogiPositionFeatures:
    return ShogiPositionFeatures(
        global_feature_ids=torch.stack([feature.global_feature_ids for feature in features]),
        square_feature_ids=torch.stack([feature.square_feature_ids for feature in features]),
        piece_feature_ids=torch.stack([feature.piece_feature_ids for feature in features]),
        line_feature_ids=torch.stack([feature.line_feature_ids for feature in features]),
    )


def shogi_position_features_from_sfen(position_sfen: str) -> ShogiPositionFeatures:
    board = shogi.Board(position_sfen)
    global_feature_ids = torch.tensor(
        [
            SHOGI_POSITION_STATE_TOKEN_ID,
            side_to_move_token_id(board.turn),
            in_check_token_id(board.is_check()),
            move_count_bucket_token_id(board.move_number),
            *hand_token_ids(board),
        ],
        dtype=torch.long,
    )
    square_feature_ids = torch.tensor(square_feature_id_rows(board), dtype=torch.long)
    piece_feature_ids = torch.tensor(piece_feature_id_rows(board), dtype=torch.long)
    line_feature_ids = torch.tensor(line_feature_id_rows(board), dtype=torch.long)
    return ShogiPositionFeatures(
        global_feature_ids=global_feature_ids,
        square_feature_ids=square_feature_ids,
        piece_feature_ids=piece_feature_ids,
        line_feature_ids=line_feature_ids,
    )


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


def square_feature_id_rows(board: shogi.Board) -> list[list[int]]:
    pieces = [relative_square_token_id(board, square) for square in range(BOARD_TOKEN_COUNT)]
    attacks = attack_token_ids(board)
    own_square_piece_type_attacks, opponent_square_piece_type_attacks = square_piece_type_attack_token_id_rows(board)
    king_relative_squares = king_relative_square_token_ids(board)
    drop_shadows = drop_shadow_token_id_rows(board)
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
                *drop_shadows[relative_square],
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


def drop_shadow_token_id_rows(board: shogi.Board) -> list[list[int]]:
    own_legal_drop_targets = legal_drop_targets_by_piece_type(board, board.turn)
    opponent_legal_drop_targets = legal_drop_targets_by_piece_type(board, opponent_color(board.turn))
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


def line_feature_token_ids(board: shogi.Board) -> list[int]:
    token_ids: list[int] = []
    for row in line_feature_id_rows(board):
        token_ids.extend(row)
    return token_ids


def line_feature_id_rows(board: shogi.Board) -> list[list[int]]:
    return [line_slot_feature_token_ids(board, line_index) for line_index in range(LINE_TOKEN_COUNT)]


def line_slot_feature_token_ids(board: shogi.Board, line_index: int) -> list[int]:
    line_kind = line_kind_index(line_index)
    relative_squares = squares_for_line_index(line_index)
    absolute_squares = {relative_to_absolute_square(square, board.turn) for square in relative_squares}
    own_king_on_line = king_on_absolute_squares(board, board.turn, absolute_squares)
    opponent_king_on_line = king_on_absolute_squares(board, opponent_color(board.turn), absolute_squares)
    own_slider_on_line = slider_on_absolute_squares(board, board.turn, absolute_squares, line_kind=line_kind)
    opponent_slider_on_line = slider_on_absolute_squares(
        board,
        opponent_color(board.turn),
        absolute_squares,
        line_kind=line_kind,
    )
    occupancy_count = sum(1 for square in absolute_squares if board.piece_at(square) is not None)
    return [
        LINE_KIND_OFFSET + line_kind,
        LINE_OWN_KING_ON_LINE_OFFSET + int(own_king_on_line),
        LINE_OPPONENT_KING_ON_LINE_OFFSET + int(opponent_king_on_line),
        LINE_OWN_SLIDER_ON_LINE_OFFSET + int(own_slider_on_line),
        LINE_OPPONENT_SLIDER_ON_LINE_OFFSET + int(opponent_slider_on_line),
        LINE_OCCUPANCY_COUNT_OFFSET + min(occupancy_count, LINE_OCCUPANCY_COUNT_MAX),
    ]


def line_kind_index(line_index: int) -> int:
    if line_index < 9:
        return 0
    if line_index < 18:
        return 1
    if line_index < 35:
        return 2
    return 3


def squares_for_line_index(line_index: int) -> tuple[int, ...]:
    if line_index < 9:
        file_index = line_index
        return tuple(rank * 9 + file_index for rank in range(9))
    if line_index < 18:
        rank = line_index - 9
        return tuple(rank * 9 + file_index for file_index in range(9))
    if line_index < 35:
        diagonal = line_index - 18
        return tuple(square for square in range(BOARD_TOKEN_COUNT) if square // 9 + square % 9 == diagonal)
    diagonal = line_index - 35
    return tuple(square for square in range(BOARD_TOKEN_COUNT) if square // 9 - square % 9 + 8 == diagonal)


def king_on_absolute_squares(board: shogi.Board, color: int, absolute_squares: set[int]) -> bool:
    king_square = board.king_squares[color]
    return king_square is not None and int(king_square) in absolute_squares


def slider_on_absolute_squares(
    board: shogi.Board,
    color: int,
    absolute_squares: set[int],
    *,
    line_kind: int,
) -> bool:
    for square in absolute_squares:
        piece = board.piece_at(square)
        if piece is not None and piece.color == color and piece_slides_on_line(piece.piece_type, line_kind):
            return True
    return False


def piece_slides_on_line(piece_type: int, line_kind: int) -> bool:
    if line_kind == 0:
        return piece_type in (shogi.LANCE, shogi.ROOK, shogi.PROM_ROOK)
    if line_kind == 1:
        return piece_type in (shogi.ROOK, shogi.PROM_ROOK)
    return piece_type in (shogi.BISHOP, shogi.PROM_BISHOP)


def piece_feature_token_ids(board: shogi.Board) -> list[int]:
    return [token_id for row in piece_feature_id_rows(board) for token_id in row]


def piece_feature_id_rows(board: shogi.Board) -> list[list[int]]:
    piece_features: list[int] = []
    for relative_square in range(BOARD_TOKEN_COUNT):
        absolute_square = relative_to_absolute_square(relative_square, board.turn)
        piece = board.piece_at(absolute_square)
        if piece is not None:
            piece_features.extend(board_piece_slot_token_ids(board, piece, relative_square))
    piece_features.extend(hand_piece_slot_token_ids(board))
    empty_slot_count = PIECE_SLOT_COUNT - len(piece_features) // PIECE_FEATURE_COUNT
    if empty_slot_count < 0:
        raise ValueError("shogi board contains more pieces than supported piece slots")
    for _ in range(empty_slot_count):
        piece_features.extend(empty_piece_slot_token_ids())
    return [
        piece_features[index : index + PIECE_FEATURE_COUNT]
        for index in range(0, len(piece_features), PIECE_FEATURE_COUNT)
    ]


def board_piece_slot_token_ids(board: shogi.Board, piece: shogi.Piece, relative_square: int) -> list[int]:
    return [
        PIECE_LOCATION_BOARD_TOKEN_ID,
        piece_token_id(piece, own_color=board.turn),
        PIECE_SQUARE_OFFSET + relative_square,
        king_relative_square_token_id(
            board,
            board.turn,
            relative_square,
            offset=OWN_KING_RELATIVE_SQUARE_OFFSET,
        ),
        king_relative_square_token_id(
            board,
            opponent_color(board.turn),
            relative_square,
            offset=OPPONENT_KING_RELATIVE_SQUARE_OFFSET,
        ),
    ]


def hand_piece_slot_token_ids(board: shogi.Board) -> list[int]:
    token_ids: list[int] = []
    for color in (board.turn, opponent_color(board.turn)):
        for piece_type in HAND_PIECE_TYPES:
            for _ in range(board.pieces_in_hand[color][piece_type]):
                token_ids.extend(
                    hand_piece_token_ids(
                        shogi.Piece(piece_type, color),
                        own_color=board.turn,
                    )
                )
    return token_ids


def hand_piece_token_ids(piece: shogi.Piece, *, own_color: int) -> list[int]:
    return [
        PIECE_LOCATION_HAND_TOKEN_ID,
        piece_token_id(piece, own_color=own_color),
        PIECE_SQUARE_UNKNOWN_TOKEN_ID,
        OWN_KING_RELATIVE_SQUARE_OFFSET + KING_RELATIVE_SQUARE_BUCKET_UNKNOWN,
        OPPONENT_KING_RELATIVE_SQUARE_OFFSET + KING_RELATIVE_SQUARE_BUCKET_UNKNOWN,
    ]


def empty_piece_slot_token_ids() -> list[int]:
    return [
        PIECE_LOCATION_EMPTY_TOKEN_ID,
        EMPTY_SQUARE_TOKEN_ID,
        PIECE_SQUARE_UNKNOWN_TOKEN_ID,
        OWN_KING_RELATIVE_SQUARE_OFFSET + KING_RELATIVE_SQUARE_BUCKET_UNKNOWN,
        OPPONENT_KING_RELATIVE_SQUARE_OFFSET + KING_RELATIVE_SQUARE_BUCKET_UNKNOWN,
    ]


def king_relative_offset_bucket(relative_square: int, relative_king_square: int) -> int:
    square_file = relative_square % 9
    square_rank = relative_square // 9
    king_file = relative_king_square % 9
    king_rank = relative_king_square // 9
    file_delta = square_file - king_file
    rank_delta = square_rank - king_rank
    return (rank_delta + 8) * 17 + file_delta + 8


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
