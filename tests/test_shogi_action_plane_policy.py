import unittest

import shogi

from intrep.representation.outputs.shogi_action_plane_policy_encoding import (
    SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT,
    SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT,
    shogi_action_plane_policy_action_index,
    shogi_action_plane_policy_legal_mask,
    shogi_action_plane_policy_move_from_action_index,
)


class ShogiActionPlanePolicyTest(unittest.TestCase):
    def test_action_index_is_fixed_board_square_times_move_type_space(self) -> None:
        self.assertEqual(SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT, 27)
        self.assertEqual(SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT, 81 * 27)

    def test_encodes_normal_promotion_drop_long_and_knight_moves(self) -> None:
        moves = ("7g7f", "8h2b+", "P*5e", "8h5e", "2i3g")

        indices = {shogi_action_plane_policy_action_index(move, turn=shogi.BLACK) for move in moves}

        self.assertEqual(len(indices), len(moves))

    def test_action_index_is_side_to_move_relative(self) -> None:
        black_index = shogi_action_plane_policy_action_index("7g7f", turn=shogi.BLACK)
        white_index = shogi_action_plane_policy_action_index("3c3d", turn=shogi.WHITE)

        self.assertEqual(black_index, white_index)

    def test_legal_mask_matches_initial_legal_moves(self) -> None:
        board = shogi.Board()

        mask = shogi_action_plane_policy_legal_mask(board)

        self.assertEqual(tuple(mask.shape), (SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT,))
        self.assertEqual(int(mask.sum().item()), len(list(board.legal_moves)))

    def test_round_trips_legal_initial_moves_through_action_index(self) -> None:
        board = shogi.Board()

        for move in board.legal_moves:
            action_index = shogi_action_plane_policy_action_index(move.usi(), turn=board.turn)

            self.assertEqual(shogi_action_plane_policy_move_from_action_index(action_index, board), move.usi())

    def test_round_trips_legal_moves_in_played_positions_through_action_index(self) -> None:
        board = shogi.Board()

        for ply in range(80):
            legal_moves = tuple(board.legal_moves)
            indices = {
                shogi_action_plane_policy_action_index(move.usi(), turn=board.turn)
                for move in legal_moves
            }
            self.assertEqual(len(indices), len(legal_moves))
            for move in legal_moves:
                action_index = shogi_action_plane_policy_action_index(move.usi(), turn=board.turn)
                self.assertEqual(shogi_action_plane_policy_move_from_action_index(action_index, board), move.usi())

            board.push(legal_moves[(ply * 17 + 3) % len(legal_moves)])

    def test_rejects_action_index_that_is_not_legal_in_board(self) -> None:
        board = shogi.Board()
        illegal_index = shogi_action_plane_policy_action_index("P*5e", turn=board.turn)

        with self.assertRaisesRegex(ValueError, "legal move"):
            shogi_action_plane_policy_move_from_action_index(illegal_index, board)


if __name__ == "__main__":
    unittest.main()
