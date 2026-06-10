import unittest

from intrep.worlds.shogi.game_record import ShogiActorSpec, ShogiGameRecord, shogi_game_record_from_usi_moves
from intrep.worlds.shogi.game_replay import replay_shogi_game_record, validate_shogi_game_record


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={})
WHITE_ACTOR = ShogiActorSpec(kind="usi_engine", name="white-engine", settings={"go_command": "go nodes 1"})


def _record(moves: tuple[str, ...], winner: str | None = None, end_reason: str | None = None) -> ShogiGameRecord:
    return shogi_game_record_from_usi_moves(
        moves,
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        winner=winner,
        end_reason=end_reason,
    )


class ShogiGameReplayTest(unittest.TestCase):
    def test_replays_legal_game_and_recovers_sources(self) -> None:
        record = _record(("7g7f", "3c3d"), winner="black", end_reason="resign")

        plies = replay_shogi_game_record(record)

        self.assertEqual([ply.ply_index for ply in plies], [0, 1])
        self.assertEqual([ply.side_to_move for ply in plies], ["black", "white"])
        self.assertEqual([ply.move for ply in plies], ["7g7f", "3c3d"])
        self.assertEqual(plies[0].source_actor, BLACK_ACTOR)
        self.assertEqual(plies[1].source_actor, WHITE_ACTOR)

    def test_rejects_illegal_move(self) -> None:
        corrupt = _record(("7g7f", "7g7f"))

        with self.assertRaisesRegex(ValueError, "illegal move"):
            replay_shogi_game_record(corrupt)

    def test_rejects_max_plies_with_winner(self) -> None:
        record = _record(("7g7f",), winner="black", end_reason="max_plies")

        with self.assertRaisesRegex(ValueError, "max_plies"):
            validate_shogi_game_record(record)


if __name__ == "__main__":
    unittest.main()
