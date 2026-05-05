import tempfile
import unittest
from pathlib import Path

from intrep.worlds.shogi.game_record import (
    PlayerSpec,
    ShogiGameRecord,
    ShogiGamePlyRecord,
    load_shogi_game_records_jsonl,
    shogi_game_ply_records_from_usi_moves,
    write_shogi_game_records_jsonl,
)


BLACK_PLAYER = PlayerSpec(kind="checkpoint", name="black-model", settings={"checkpoint": "black.pt"})
WHITE_PLAYER = PlayerSpec(kind="yaneuraou", name="white-engine", settings={"go_command": "go nodes 1"})


class ShogiGameRecordTest(unittest.TestCase):
    def test_round_trips_shogi_game_records_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"

            write_shogi_game_records_jsonl(
                path,
                [
                    ShogiGameRecord(
                        black_player=BLACK_PLAYER,
                        white_player=WHITE_PLAYER,
                        plies=shogi_game_ply_records_from_usi_moves(("7g7f", "3c3d")),
                        winner="white",
                        end_reason="resign",
                    ),
                    ShogiGameRecord(
                        black_player=BLACK_PLAYER,
                        white_player=WHITE_PLAYER,
                        plies=shogi_game_ply_records_from_usi_moves(("2g2f",)),
                        winner=None,
                        end_reason="max_plies",
                    ),
                ],
            )
            records = load_shogi_game_records_jsonl(path)

        self.assertEqual(
            records,
            [
                ShogiGameRecord(
                    black_player=BLACK_PLAYER,
                    white_player=WHITE_PLAYER,
                    plies=shogi_game_ply_records_from_usi_moves(("7g7f", "3c3d")),
                    winner="white",
                    end_reason="resign",
                ),
                ShogiGameRecord(
                    black_player=BLACK_PLAYER,
                    white_player=WHITE_PLAYER,
                    plies=shogi_game_ply_records_from_usi_moves(("2g2f",)),
                    winner=None,
                    end_reason="max_plies",
                ),
            ],
        )

    def test_loads_arena_shogi_game_record_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            path.write_text(
                (
                    '{"black_player":{"kind":"checkpoint","name":"direct","settings":{"checkpoint":"model.pt"}},'
                    '"white_player":{"kind":"yaneuraou","name":"yaneuraou","settings":{"go_command":"go nodes 1"}},'
                    '"plies":[{"side":"black","position":"position startpos","bestmove":"2g2f",'
                    '"ponder":null,"usi_info_lines":["info depth 1 nodes 1 pv 2g2f"]}],'
                    '"end_reason":"max_plies","winner":null}\n'
                ),
                encoding="utf-8",
            )

            records = load_shogi_game_records_jsonl(path)

        self.assertEqual(
            records,
            [
                ShogiGameRecord(
                    black_player=PlayerSpec(kind="checkpoint", name="direct", settings={"checkpoint": "model.pt"}),
                    white_player=PlayerSpec(kind="yaneuraou", name="yaneuraou", settings={"go_command": "go nodes 1"}),
                    plies=(
                        ShogiGamePlyRecord(
                            side="black",
                            position="position startpos",
                            bestmove="2g2f",
                            usi_info_lines=("info depth 1 nodes 1 pv 2g2f",),
                        ),
                    ),
                    winner=None,
                    end_reason="max_plies",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
