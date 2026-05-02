import tempfile
import unittest
from pathlib import Path

from intrep.shogi_game_record import ShogiGameRecord, load_shogi_game_records_jsonl, write_shogi_game_records_jsonl
from intrep.split_shogi_game_records import split_shogi_game_records_jsonl


class SplitShogiGameRecordsTest(unittest.TestCase):
    def test_splits_at_game_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            records = [
                ShogiGameRecord(("7g7f",), "b"),
                ShogiGameRecord(("3c3d",), "w"),
                ShogiGameRecord(("2g2f",), "b"),
                ShogiGameRecord(("8c8d",), "w"),
            ]
            write_shogi_game_records_jsonl(games_path, records)

            train_count, eval_count = split_shogi_game_records_jsonl(
                games_jsonl=games_path,
                train_jsonl=train_path,
                eval_jsonl=eval_path,
                eval_ratio=0.25,
            )

            self.assertEqual(train_count, 3)
            self.assertEqual(eval_count, 1)
            self.assertEqual(load_shogi_game_records_jsonl(train_path), records[:3])
            self.assertEqual(load_shogi_game_records_jsonl(eval_path), records[3:])

    def test_rejects_invalid_eval_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            write_shogi_game_records_jsonl(
                games_path,
                [ShogiGameRecord(("7g7f",), "b"), ShogiGameRecord(("3c3d",), "w")],
            )

            with self.assertRaisesRegex(ValueError, "eval-ratio"):
                split_shogi_game_records_jsonl(
                    games_jsonl=games_path,
                    train_jsonl=root / "train.jsonl",
                    eval_jsonl=root / "eval.jsonl",
                    eval_ratio=0.0,
                )


if __name__ == "__main__":
    unittest.main()
