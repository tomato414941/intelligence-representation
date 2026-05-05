import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intrep.worlds.shogi.game_record import PlayerSpec, ShogiGameRecord, shogi_game_ply_records_from_usi_moves, write_shogi_game_records_jsonl
from intrep.tasks.shogi_move_choice.examples import load_shogi_move_choice_examples_jsonl
from intrep.tasks.shogi_move_choice.prepare_examples import main


BLACK_PLAYER = PlayerSpec(kind="checkpoint", name="black-model", settings={})
WHITE_PLAYER = PlayerSpec(kind="checkpoint", name="white-model", settings={})
YANEURAOU_PLAYER = PlayerSpec(kind="yaneuraou", name="yaneuraou", settings={"go_command": "go nodes 1"})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return ShogiGameRecord(BLACK_PLAYER, WHITE_PLAYER, shogi_game_ply_records_from_usi_moves(moves), winner)


class PrepareShogiMoveChoiceExamplesCliTest(unittest.TestCase):
    def test_writes_examples_from_game_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            examples_path = root / "examples.jsonl"
            write_shogi_game_records_jsonl(games_path, [_record(("7g7f", "3c3d"), "white")])

            with patch(
                "sys.argv",
                [
                    "prepare_examples",
                    "--games-jsonl",
                    str(games_path),
                    "--examples-jsonl",
                    str(examples_path),
                ],
            ):
                main()

            examples = load_shogi_move_choice_examples_jsonl(examples_path)
            self.assertEqual(len(examples), 2)
            self.assertEqual(examples[0].chosen_move, "7g7f")
            self.assertEqual(examples[0].game_index, 0)
            self.assertEqual(examples[0].ply_index, 0)
            self.assertEqual(examples[1].ply_index, 1)

    def test_writes_one_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            examples_path = root / "examples.jsonl"
            write_shogi_game_records_jsonl(
                games_path,
                [
                    _record(("7g7f", "3c3d"), "white"),
                    _record(("2g2f", "8c8d"), "black"),
                    _record(("7g7f", "8c8d"), "white"),
                ],
            )

            with patch(
                "sys.argv",
                [
                    "prepare_examples",
                    "--games-jsonl",
                    str(games_path),
                    "--examples-jsonl",
                    str(examples_path),
                    "--shard-index",
                    "1",
                    "--shard-count",
                    "2",
                ],
            ):
                main()

            examples = load_shogi_move_choice_examples_jsonl(examples_path)
            self.assertEqual(len(examples), 2)
            self.assertEqual(examples[0].chosen_move, "2g2f")
            self.assertEqual(examples[0].game_index, 1)
            self.assertEqual(examples[0].ply_index, 0)

    def test_filters_examples_by_source_player_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            examples_path = root / "examples.jsonl"
            write_shogi_game_records_jsonl(
                games_path,
                [
                    ShogiGameRecord(
                        BLACK_PLAYER,
                        YANEURAOU_PLAYER,
                        shogi_game_ply_records_from_usi_moves(("7g7f", "3c3d", "2g2f", "8c8d")),
                        "white",
                    )
                ],
            )

            with patch(
                "sys.argv",
                [
                    "prepare_examples",
                    "--games-jsonl",
                    str(games_path),
                    "--examples-jsonl",
                    str(examples_path),
                    "--include-player-kind",
                    "yaneuraou",
                ],
            ):
                main()

            examples = load_shogi_move_choice_examples_jsonl(examples_path)
            self.assertEqual([example.chosen_move for example in examples], ["3c3d", "8c8d"])
            self.assertEqual([example.value_target for example in examples], [1.0, 1.0])
            self.assertEqual([example.game_index for example in examples], [0, 0])
            self.assertEqual([example.ply_index for example in examples], [1, 3])


if __name__ == "__main__":
    unittest.main()
