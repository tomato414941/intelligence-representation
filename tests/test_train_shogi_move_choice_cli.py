import tempfile
import unittest
import json
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import shogi

from intrep.tasks.shogi_move_choice.dataset_definition import load_shogi_move_choice_dataset_definition
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    shogi_game_transitions_from_usi_moves,
    write_shogi_game_records_jsonl,
)
from intrep.train_shogi_move_choice import main


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={})
WHITE_ACTOR = ShogiActorSpec(kind="checkpoint", name="white-model", settings={})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        initial_position_sfen=shogi.Board().sfen(),
        transitions=shogi_game_transitions_from_usi_moves(moves, winner=winner),
        winner=winner,
    )


def _record_with_policy_targets(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    record = _record(moves, winner)
    transitions = tuple(
        replace(transition, policy_targets={transition.action_usi: 1.0})
        for transition in record.transitions
    )
    return ShogiGameRecord(
        black_actor=record.black_actor,
        white_actor=record.white_actor,
        initial_position_sfen=record.initial_position_sfen,
        transitions=transitions,
        winner=record.winner,
        end_reason=record.end_reason,
    )


class TrainShogiMoveChoiceCliTest(unittest.TestCase):
    def test_trains_from_game_records_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            dataset_definition_path = root / "dataset.json"
            checkpoint_path = root / "shogi.pt"
            best_checkpoint_path = root / "shogi-best.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            dataset_definition_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-move-choice",
                        "objective": "shogi move-choice policy",
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "train_shogi_move_choice",
                    "--dataset-definition",
                    str(dataset_definition_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--best-checkpoint-path",
                    str(best_checkpoint_path),
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
                    "--eval-every",
                    "1",
                    "--num-workers",
                    "0",
                ],
            ), patch("sys.stdout", new_callable=StringIO) as stdout:
                main()

            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(best_checkpoint_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertIn("step=1/1", stdout.getvalue())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["dataset_definition"]["name"], "test-shogi-move-choice")
            self.assertEqual(metrics["best_checkpoint_path"], str(best_checkpoint_path))
            self.assertIn(metrics["metrics"]["best_eval_step"], {0, 1})
            self.assertIsNotNone(metrics["metrics"]["best_eval_loss"])
            self.assertEqual(metrics["config"]["num_workers"], 0)
            self.assertEqual(metrics["config"]["policy_loss_weight"], 1.0)
            self.assertEqual(metrics["config"]["value_loss_weight"], 0.0)
            self.assertEqual(metrics["raw_train_case_count"], 2)
            self.assertEqual(metrics["raw_eval_case_count"], 2)
            self.assertEqual(metrics["used_eval_case_count"], 2)
            self.assertEqual(metrics["metrics"]["eval_case_count"], 2)
            self.assertEqual(metrics["train_policy_target_summary"]["available_count"], 0)
            self.assertEqual(metrics["train_policy_target_summary"]["missing_count"], 2)
            self.assertEqual(metrics["eval_policy_target_summary"]["available_ratio"], 0.0)

    def test_writes_policy_target_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            dataset_definition_path = root / "dataset.json"
            checkpoint_path = root / "shogi.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record_with_policy_targets(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            dataset_definition_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-move-choice",
                        "objective": "shogi move-choice policy",
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "train_shogi_move_choice",
                    "--dataset-definition",
                    str(dataset_definition_path),
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
                    "--num-workers",
                    "0",
                ],
            ), patch("sys.stdout", new_callable=StringIO):
                main()

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["train_policy_target_summary"]["available_count"], 2)
            self.assertEqual(metrics["train_policy_target_summary"]["missing_count"], 0)
            self.assertEqual(metrics["train_policy_target_summary"]["available_ratio"], 1.0)
            self.assertEqual(metrics["train_policy_target_summary"]["mean_nonzero_count"], 1.0)
            self.assertEqual(metrics["eval_policy_target_summary"]["available_count"], 0)
            self.assertEqual(metrics["eval_policy_target_summary"]["missing_count"], 2)

    def test_writes_periodic_checkpoints_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            dataset_definition_path = root / "dataset.json"
            checkpoint_path = root / "shogi.pt"
            metrics_path = root / "metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "white")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "black")])
            dataset_definition_path.write_text(
                json.dumps(
                    {
                        "name": "test-shogi-move-choice",
                        "objective": "shogi move-choice policy",
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "train_shogi_move_choice",
                    "--dataset-definition",
                    str(dataset_definition_path),
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

    def test_rejects_unsplit_dataset_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.jsonl"
            dataset_definition_path = root / "dataset.json"
            write_shogi_game_records_jsonl(games_path, [_record(("7g7f",), None)])
            dataset_definition_path.write_text(
                json.dumps(
                    {
                        "name": "bad-unsplit",
                        "objective": "shogi move-choice policy",
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "split"):
                load_shogi_move_choice_dataset_definition(dataset_definition_path)

    def test_rejects_example_jsonl_dataset_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_definition_path = root / "dataset.json"
            dataset_definition_path.write_text(
                json.dumps(
                    {
                        "name": "bad-source-kind",
                        "objective": "shogi move-choice policy",
                        "train_sources": [{"kind": "examples_jsonl", "path": "train-examples.jsonl"}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": "eval-games.jsonl"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "game_records_jsonl"):
                load_shogi_move_choice_dataset_definition(dataset_definition_path)


if __name__ == "__main__":
    unittest.main()
