import tempfile
import unittest
import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from intrep.worlds.shogi.game_record import PlayerSpec, ShogiGameRecord, shogi_game_ply_records_from_usi_moves, write_shogi_game_records_jsonl
from intrep.train_shogi_move_choice import main


BLACK_PLAYER = PlayerSpec(kind="checkpoint", name="black-model", settings={})
WHITE_PLAYER = PlayerSpec(kind="checkpoint", name="white-model", settings={})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return ShogiGameRecord(BLACK_PLAYER, WHITE_PLAYER, shogi_game_ply_records_from_usi_moves(moves), winner)


class TrainShogiMoveChoiceCliTest(unittest.TestCase):
    def test_trains_from_game_records_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            train_examples_path = root / "train-examples.jsonl"
            checkpoint_path = root / "shogi.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])

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
                    "--log-every",
                    "1",
                    "--num-workers",
                    "0",
                ],
            ), patch("sys.stdout", new_callable=StringIO) as stdout:
                main()

            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertTrue(train_examples_path.exists())
            self.assertIn("step=1/1", stdout.getvalue())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["config"]["num_workers"], 0)
            self.assertEqual(metrics["config"]["policy_loss_weight"], 1.0)
            self.assertEqual(metrics["config"]["value_loss_weight"], 0.0)
            self.assertEqual(metrics["raw_train_case_count"], 2)
            self.assertEqual(metrics["raw_eval_case_count"], 2)
            self.assertEqual(metrics["used_eval_case_count"], 2)
            self.assertEqual(metrics["metrics"]["eval_case_count"], 2)

    def test_writes_periodic_checkpoints_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            checkpoint_path = root / "shogi.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])

            with patch(
                "sys.argv",
                [
                    "train_shogi_move_choice",
                    "--train-games-jsonl",
                    str(train_games_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--max-steps",
                    "3",
                    "--batch-size",
                    "2",
                    "--embedding-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--num-heads",
                    "2",
                    "--checkpoint-every",
                    "1",
                    "--metrics-every",
                    "2",
                    "--keep-last-n-checkpoints",
                    "2",
                ],
            ), patch("sys.stdout", new_callable=StringIO):
                main()

            self.assertFalse((root / "checkpoint_step_1.pt").exists())
            self.assertTrue((root / "checkpoint_step_2.pt").exists())
            self.assertTrue((root / "checkpoint_step_3.pt").exists())
            step_metrics_path = root / "metrics_step_2.json"
            self.assertTrue(step_metrics_path.exists())
            step_metrics = json.loads(step_metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(step_metrics["step"], 2)
            self.assertEqual(step_metrics["max_steps"], 3)
            self.assertIn("loss", step_metrics)


if __name__ == "__main__":
    unittest.main()
