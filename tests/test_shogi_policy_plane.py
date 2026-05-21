import unittest

import shogi

from intrep.representation.outputs.shogi_policy_plane_encoding import (
    SHOGI_POLICY_PLANE_ACTION_COUNT,
    SHOGI_POLICY_PLANE_MOVE_TYPE_COUNT,
    shogi_policy_plane_action_index,
    shogi_policy_plane_legal_mask,
    shogi_policy_plane_move_from_action_index,
)


class ShogiPolicyPlaneTest(unittest.TestCase):
    def test_action_index_is_fixed_board_square_times_move_type_space(self) -> None:
        self.assertEqual(SHOGI_POLICY_PLANE_MOVE_TYPE_COUNT, 43)
        self.assertEqual(SHOGI_POLICY_PLANE_ACTION_COUNT, 81 * 43)

    def test_encodes_normal_promotion_drop_long_and_knight_moves(self) -> None:
        moves = ("7g7f", "8h2b+", "P*5e", "8h5e", "2i3g")

        indices = {shogi_policy_plane_action_index(move, turn=shogi.BLACK) for move in moves}

        self.assertEqual(len(indices), len(moves))

    def test_action_index_is_side_to_move_relative(self) -> None:
        black_index = shogi_policy_plane_action_index("7g7f", turn=shogi.BLACK)
        white_index = shogi_policy_plane_action_index("3c3d", turn=shogi.WHITE)

        self.assertEqual(black_index, white_index)

    def test_legal_mask_matches_initial_legal_moves(self) -> None:
        board = shogi.Board()

        mask = shogi_policy_plane_legal_mask(board)

        self.assertEqual(tuple(mask.shape), (SHOGI_POLICY_PLANE_ACTION_COUNT,))
        self.assertEqual(int(mask.sum().item()), len(list(board.legal_moves)))

    def test_round_trips_legal_initial_moves_through_action_index(self) -> None:
        board = shogi.Board()

        for move in board.legal_moves:
            action_index = shogi_policy_plane_action_index(move.usi(), turn=board.turn)

            self.assertEqual(shogi_policy_plane_move_from_action_index(action_index, board), move.usi())

    def test_rejects_action_index_that_is_not_legal_in_board(self) -> None:
        board = shogi.Board()
        illegal_index = shogi_policy_plane_action_index("P*5e", turn=board.turn)

        with self.assertRaisesRegex(ValueError, "legal move"):
            shogi_policy_plane_move_from_action_index(illegal_index, board)


if __name__ == "__main__":
    unittest.main()
