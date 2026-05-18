import unittest

import shogi
import torch

from intrep.worlds.shogi.position_encoding import (
    HAND_COUNT_TOKEN_MAX,
    HAND_PIECE_TYPES,
    OPPONENT_HAND_OFFSET,
    OPPONENT_PIECE_OFFSET,
    OWN_HAND_OFFSET,
    OWN_PIECE_OFFSET,
    SHOGI_POSITION_TOKEN_COUNT,
    SHOGI_POSITION_VOCAB_SIZE,
    SIDE_TO_MOVE_BLACK_TOKEN_ID,
    SIDE_TO_MOVE_WHITE_TOKEN_ID,
    absolute_to_relative_square,
    shogi_position_token_ids_from_sfen,
)


class ShogiPositionEncodingTest(unittest.TestCase):
    def test_encodes_start_position_as_fixed_length_token_ids(self) -> None:
        token_ids = shogi_position_token_ids_from_sfen(shogi.Board().sfen())

        self.assertEqual(token_ids.dtype, torch.long)
        self.assertEqual(tuple(token_ids.shape), (SHOGI_POSITION_TOKEN_COUNT,))
        self.assertEqual(int(token_ids[0].item()), SIDE_TO_MOVE_BLACK_TOKEN_ID)
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

        self.assertEqual(int(black_tokens[1 + black_relative_square].item()), OPPONENT_PIECE_OFFSET + shogi.PAWN - 1)
        self.assertEqual(int(white_tokens[1 + white_relative_square].item()), OWN_PIECE_OFFSET + shogi.PAWN - 1)

    def test_large_own_pawn_hand_counts_are_not_collapsed_to_six(self) -> None:
        board = shogi.Board()
        for _ in range(14):
            board.add_piece_into_hand(shogi.PAWN, shogi.BLACK)

        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        black_pawn_hand_index = 1 + 81 + HAND_PIECE_TYPES.index(shogi.PAWN)

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
        own_pawn_hand_index = 1 + 81 + HAND_PIECE_TYPES.index(shogi.PAWN)
        opponent_pawn_hand_index = 1 + 81 + len(HAND_PIECE_TYPES) + HAND_PIECE_TYPES.index(shogi.PAWN)

        self.assertEqual(int(token_ids[own_pawn_hand_index].item()), OWN_HAND_OFFSET + 2)
        self.assertEqual(int(token_ids[opponent_pawn_hand_index].item()), OPPONENT_HAND_OFFSET + 1)


if __name__ == "__main__":
    unittest.main()
