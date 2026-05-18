import unittest

import shogi
import torch

from intrep.worlds.shogi.position_encoding import (
    HAND_BLACK_OFFSET,
    HAND_COUNT_TOKEN_MAX,
    HAND_PIECE_TYPES,
    SHOGI_POSITION_TOKEN_COUNT,
    SHOGI_POSITION_VOCAB_SIZE,
    SIDE_TO_MOVE_BLACK_TOKEN_ID,
    SIDE_TO_MOVE_WHITE_TOKEN_ID,
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

    def test_large_pawn_hand_counts_are_not_collapsed_to_six(self) -> None:
        board = shogi.Board()
        for _ in range(14):
            board.add_piece_into_hand(shogi.PAWN, shogi.BLACK)

        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        black_pawn_hand_index = 1 + 81 + HAND_PIECE_TYPES.index(shogi.PAWN)

        self.assertEqual(HAND_COUNT_TOKEN_MAX, 18)
        self.assertEqual(int(token_ids[black_pawn_hand_index].item()), HAND_BLACK_OFFSET + 14)
        self.assertLess(int(token_ids[black_pawn_hand_index].item()), SHOGI_POSITION_VOCAB_SIZE)


if __name__ == "__main__":
    unittest.main()
