import unittest

import shogi
import torch

from intrep.worlds.shogi.position_encoding import (
    ATTACK_TOKEN_OFFSET,
    BOARD_TOKEN_OFFSET,
    ATTACK_COUNT_TOKEN_MAX,
    HAND_COUNT_TOKEN_MAX,
    HAND_TOKEN_OFFSET,
    HAND_PIECE_TYPES,
    IN_CHECK_TOKEN_ID,
    IN_CHECK_TOKEN_INDEX,
    MOVE_COUNT_BUCKET_OFFSET,
    MOVE_COUNT_BUCKET_OVERFLOW,
    MOVE_COUNT_TOKEN_INDEX,
    NOT_IN_CHECK_TOKEN_ID,
    OPPONENT_ATTACK_OFFSET,
    OPPONENT_HAND_OFFSET,
    OPPONENT_PIECE_OFFSET,
    OWN_ATTACK_OFFSET,
    OWN_HAND_OFFSET,
    OWN_PIECE_OFFSET,
    SHOGI_POSITION_TOKEN_COUNT,
    SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
    SHOGI_POSITION_VOCAB_SIZE,
    SIDE_TO_MOVE_BLACK_TOKEN_ID,
    SIDE_TO_MOVE_WHITE_TOKEN_ID,
    absolute_to_relative_square,
    move_count_bucket_token_id,
    shogi_position_token_ids_from_sfen,
)


class ShogiPositionEncodingTest(unittest.TestCase):
    def test_encodes_start_position_as_fixed_length_token_ids(self) -> None:
        token_ids = shogi_position_token_ids_from_sfen(shogi.Board().sfen())

        self.assertEqual(token_ids.dtype, torch.long)
        self.assertEqual(tuple(token_ids.shape), (SHOGI_POSITION_TOKEN_COUNT,))
        self.assertEqual(SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT, 99)
        self.assertEqual(int(token_ids[0].item()), SIDE_TO_MOVE_BLACK_TOKEN_ID)
        self.assertEqual(int(token_ids[IN_CHECK_TOKEN_INDEX].item()), NOT_IN_CHECK_TOKEN_ID)
        self.assertEqual(int(token_ids[MOVE_COUNT_TOKEN_INDEX].item()), MOVE_COUNT_BUCKET_OFFSET + 1)
        self.assertGreaterEqual(int(token_ids.min().item()), 0)
        self.assertLess(int(token_ids.max().item()), SHOGI_POSITION_VOCAB_SIZE)

    def test_side_to_move_changes_after_one_move(self) -> None:
        board = shogi.Board()
        board.push_usi("7g7f")

        token_ids = shogi_position_token_ids_from_sfen(board.sfen())

        self.assertEqual(int(token_ids[0].item()), SIDE_TO_MOVE_WHITE_TOKEN_ID)

    def test_position_is_side_to_move_relative(self) -> None:
        board = shogi.Board()
        board.set_piece_at(shogi.SQUARE_NAMES.index("5e"), shogi.Piece(shogi.PAWN, shogi.WHITE))
        black_tokens = shogi_position_token_ids_from_sfen(board.sfen())
        board.turn = shogi.WHITE
        white_tokens = shogi_position_token_ids_from_sfen(board.sfen())

        black_relative_square = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        white_relative_square = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.WHITE)

        self.assertEqual(
            int(black_tokens[BOARD_TOKEN_OFFSET + black_relative_square].item()),
            OPPONENT_PIECE_OFFSET + shogi.PAWN - 1,
        )
        self.assertEqual(
            int(white_tokens[BOARD_TOKEN_OFFSET + white_relative_square].item()),
            OWN_PIECE_OFFSET + shogi.PAWN - 1,
        )

    def test_encodes_whether_side_to_move_is_in_check(self) -> None:
        board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        checked_board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 w - 1")

        safe_tokens = shogi_position_token_ids_from_sfen(board.sfen())
        checked_tokens = shogi_position_token_ids_from_sfen(checked_board.sfen())

        self.assertEqual(int(safe_tokens[IN_CHECK_TOKEN_INDEX].item()), NOT_IN_CHECK_TOKEN_ID)
        self.assertEqual(int(checked_tokens[IN_CHECK_TOKEN_INDEX].item()), IN_CHECK_TOKEN_ID)

    def test_encodes_move_count_bucket(self) -> None:
        early_board = shogi.Board()
        late_board = shogi.Board()
        late_board.move_number = 95

        early_tokens = shogi_position_token_ids_from_sfen(early_board.sfen())
        late_tokens = shogi_position_token_ids_from_sfen(late_board.sfen())

        self.assertEqual(int(early_tokens[MOVE_COUNT_TOKEN_INDEX].item()), MOVE_COUNT_BUCKET_OFFSET + 1)
        self.assertEqual(int(late_tokens[MOVE_COUNT_TOKEN_INDEX].item()), MOVE_COUNT_BUCKET_OFFSET + 4)

    def test_move_count_bucket_supports_unknown_and_overflow(self) -> None:
        self.assertEqual(move_count_bucket_token_id(None), MOVE_COUNT_BUCKET_OFFSET)
        self.assertEqual(move_count_bucket_token_id(0), MOVE_COUNT_BUCKET_OFFSET)
        self.assertEqual(move_count_bucket_token_id(221), MOVE_COUNT_BUCKET_OFFSET + MOVE_COUNT_BUCKET_OVERFLOW)

    def test_encodes_attack_counts_relative_to_side_to_move(self) -> None:
        board = shogi.Board()
        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        relative_7f = absolute_to_relative_square(shogi.SQUARE_NAMES.index("7f"), shogi.BLACK)
        relative_3d = absolute_to_relative_square(shogi.SQUARE_NAMES.index("3d"), shogi.BLACK)
        own_attack_index = ATTACK_TOKEN_OFFSET + relative_7f
        opponent_attack_index = ATTACK_TOKEN_OFFSET + 81 + relative_3d

        self.assertEqual(ATTACK_COUNT_TOKEN_MAX, 3)
        self.assertEqual(int(token_ids[own_attack_index].item()), OWN_ATTACK_OFFSET + 1)
        self.assertEqual(int(token_ids[opponent_attack_index].item()), OPPONENT_ATTACK_OFFSET + 1)

    def test_white_to_move_attack_counts_are_side_to_move_relative(self) -> None:
        black_board = shogi.Board()
        white_board = shogi.Board()
        white_board.turn = shogi.WHITE
        black_relative_7f = absolute_to_relative_square(shogi.SQUARE_NAMES.index("7f"), shogi.BLACK)
        white_relative_7f = absolute_to_relative_square(shogi.SQUARE_NAMES.index("7f"), shogi.WHITE)

        black_tokens = shogi_position_token_ids_from_sfen(black_board.sfen())
        white_tokens = shogi_position_token_ids_from_sfen(white_board.sfen())

        self.assertEqual(
            int(black_tokens[ATTACK_TOKEN_OFFSET + black_relative_7f].item()),
            OWN_ATTACK_OFFSET + 1,
        )
        self.assertEqual(
            int(white_tokens[ATTACK_TOKEN_OFFSET + 81 + white_relative_7f].item()),
            OPPONENT_ATTACK_OFFSET + 1,
        )

    def test_large_own_pawn_hand_counts_are_not_collapsed_to_six(self) -> None:
        board = shogi.Board()
        for _ in range(14):
            board.add_piece_into_hand(shogi.PAWN, shogi.BLACK)

        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        black_pawn_hand_index = HAND_TOKEN_OFFSET + HAND_PIECE_TYPES.index(shogi.PAWN)

        self.assertEqual(HAND_COUNT_TOKEN_MAX, 18)
        self.assertEqual(int(token_ids[black_pawn_hand_index].item()), OWN_HAND_OFFSET + 14)
        self.assertLess(int(token_ids[black_pawn_hand_index].item()), SHOGI_POSITION_VOCAB_SIZE)

    def test_white_to_move_hands_are_side_to_move_relative(self) -> None:
        board = shogi.Board()
        board.add_piece_into_hand(shogi.PAWN, shogi.BLACK)
        board.add_piece_into_hand(shogi.PAWN, shogi.WHITE)
        board.add_piece_into_hand(shogi.PAWN, shogi.WHITE)
        board.turn = shogi.WHITE

        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        own_pawn_hand_index = HAND_TOKEN_OFFSET + HAND_PIECE_TYPES.index(shogi.PAWN)
        opponent_pawn_hand_index = HAND_TOKEN_OFFSET + len(HAND_PIECE_TYPES) + HAND_PIECE_TYPES.index(shogi.PAWN)

        self.assertEqual(int(token_ids[own_pawn_hand_index].item()), OWN_HAND_OFFSET + 2)
        self.assertEqual(int(token_ids[opponent_pawn_hand_index].item()), OPPONENT_HAND_OFFSET + 1)


if __name__ == "__main__":
    unittest.main()
