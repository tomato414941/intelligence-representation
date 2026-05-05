import unittest

from intrep.worlds.shogi.game_record import PlayerSpec, ShogiGameRecord, shogi_game_ply_records_from_usi_moves
from intrep.worlds.shogi.game_replay import replay_shogi_game_record, validate_shogi_game_record


BLACK_PLAYER = PlayerSpec(kind="checkpoint", name="black-model", settings={})
WHITE_PLAYER = PlayerSpec(kind="yaneuraou", name="white-engine", settings={"go_command": "go nodes 1"})


class ShogiGameReplayTest(unittest.TestCase):
    def test_replays_legal_game_and_recovers_sources(self) -> None:
        record = ShogiGameRecord(
            black_player=BLACK_PLAYER,
            white_player=WHITE_PLAYER,
            plies=shogi_game_ply_records_from_usi_moves(("7g7f", "3c3d")),
            winner="black",
            end_reason="resign",
        )

        plies = replay_shogi_game_record(record)

        self.assertEqual([ply.ply_index for ply in plies], [0, 1])
        self.assertEqual([ply.side_to_move for ply in plies], ["black", "white"])
        self.assertEqual([ply.move for ply in plies], ["7g7f", "3c3d"])
        self.assertEqual(plies[0].source_player, BLACK_PLAYER)
        self.assertEqual(plies[1].source_player, WHITE_PLAYER)

    def test_rejects_illegal_move(self) -> None:
        record = ShogiGameRecord(
            black_player=BLACK_PLAYER,
            white_player=WHITE_PLAYER,
            plies=shogi_game_ply_records_from_usi_moves(("7g7f", "7g7f")),
        )

        with self.assertRaisesRegex(ValueError, "illegal move at ply 1"):
            replay_shogi_game_record(record)

    def test_rejects_max_plies_with_winner(self) -> None:
        record = ShogiGameRecord(
            black_player=BLACK_PLAYER,
            white_player=WHITE_PLAYER,
            plies=shogi_game_ply_records_from_usi_moves(("7g7f",)),
            winner="black",
            end_reason="max_plies",
        )

        with self.assertRaisesRegex(ValueError, "max_plies"):
            validate_shogi_game_record(record)


if __name__ == "__main__":
    unittest.main()
