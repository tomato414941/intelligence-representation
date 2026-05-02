import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intrep.prepare_shogi_move_choice_examples import main
from intrep.shogi_game_record import ShogiGameRecord, write_shogi_game_records_jsonl
from intrep.shogi_move_choice import load_shogi_move_choice_examples_jsonl


class PrepareShogiMoveChoiceExamplesCliTest(unittest.TestCase):
    def test_writes_examples_from_game_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            examples_path = root / "examples.jsonl"
            write_shogi_game_records_jsonl(games_path, [ShogiGameRecord(("7g7f", "3c3d"), "w")])

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

    def test_writes_one_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            examples_path = root / "examples.jsonl"
            write_shogi_game_records_jsonl(
                games_path,
                [
                    ShogiGameRecord(("7g7f", "3c3d"), "w"),
                    ShogiGameRecord(("2g2f", "8c8d"), "b"),
                    ShogiGameRecord(("7g7f", "8c8d"), "w"),
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


if __name__ == "__main__":
    unittest.main()
