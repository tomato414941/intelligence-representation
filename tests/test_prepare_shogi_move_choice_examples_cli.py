import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intrep.prepare_shogi_move_choice_examples import main
from intrep.shogi.game_record import PlayerSpec, ShogiGameRecord, write_shogi_game_records_jsonl
from intrep.shogi.move_choice import load_shogi_move_choice_examples_jsonl


BLACK_PLAYER = PlayerSpec(kind="checkpoint", name="black-model", settings={})
WHITE_PLAYER = PlayerSpec(kind="checkpoint", name="white-model", settings={})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return ShogiGameRecord(BLACK_PLAYER, WHITE_PLAYER, moves, winner)


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
                    "prepare_shogi_move_choice_examples",
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
                    "prepare_shogi_move_choice_examples",
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


if __name__ == "__main__":
    unittest.main()
