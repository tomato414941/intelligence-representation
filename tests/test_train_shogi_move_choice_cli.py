import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intrep.shogi_game_record import ShogiGameRecord, write_shogi_game_records_jsonl
from intrep.train_shogi_move_choice import main


class TrainShogiMoveChoiceCliTest(unittest.TestCase):
    def test_trains_from_game_records_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            train_examples_path = root / "train-examples.jsonl"
            checkpoint_path = root / "shogi.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [ShogiGameRecord(("7g7f", "3c3d"), "w")])
            write_shogi_game_records_jsonl(eval_games_path, [ShogiGameRecord(("2g2f", "8c8d"), "b")])

            with patch(
                "sys.argv",
                [
                    "train_shogi_move_choice",
                    "--train-games-jsonl",
                    str(train_games_path),
                    "--eval-games-jsonl",
                    str(eval_games_path),
                    "--write-train-examples-jsonl",
                    str(train_examples_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--max-steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--embedding-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--num-heads",
                    "2",
                    "--max-train-eval-examples",
                    "2",
                    "--max-eval-examples",
                    "2",
                ],
            ):
                main()

            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertTrue(train_examples_path.exists())


if __name__ == "__main__":
    unittest.main()
