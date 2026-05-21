from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from intrep.problems.shogi_policy_value.evaluate import main
from intrep.problems.shogi_policy_value.checkpoint import save_shogi_policy_value_model_checkpoint
from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, build_shogi_policy_value_model
from intrep.domains.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    shogi_game_record_from_usi_moves,
    write_shogi_game_records_jsonl,
)


BLACK_ACTOR = ShogiActorSpec(kind="checkpoint", name="black-model", settings={})
WHITE_ACTOR = ShogiActorSpec(kind="checkpoint", name="white-model", settings={})


def _record(moves: tuple[str, ...], winner: str | None) -> ShogiGameRecord:
    return shogi_game_record_from_usi_moves(
        moves,
        black_actor=BLACK_ACTOR,
        white_actor=WHITE_ACTOR,
        winner=winner,
    )


class EvaluateShogiPolicyValueCliTest(unittest.TestCase):
    def test_evaluates_checkpoint_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_games_path = root / "train-games.jsonl"
            eval_games_path = root / "eval-games.jsonl"
            data_selection_path = root / "data-selection.json"
            checkpoint_path = root / "checkpoint.pt"
            metrics_path = root / "eval-metrics.json"
            write_shogi_game_records_jsonl(train_games_path, [_record(("7g7f", "3c3d"), "black")])
            write_shogi_game_records_jsonl(eval_games_path, [_record(("2g2f", "8c8d"), "white")])
            data_selection_path.write_text(
                json.dumps(
                    {
                        "name": "eval-only-test",
                        "objective": "shogi move-choice policy/value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_games_path)}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_games_path)}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = ShogiPolicyValueTrainingConfig(
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                num_layers=1,
            )
            save_shogi_policy_value_model_checkpoint(checkpoint_path, build_shogi_policy_value_model(config), config)

            with patch(
                "sys.argv",
                [
                    "intrep.problems.shogi_policy_value.evaluate",
                    "--data-selection",
                    str(data_selection_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--batch-size",
                    "2",
                    "--max-train-examples",
                    "2",
                    "--max-eval-examples",
                    "2",
                ],
            ), patch("sys.stdout", new_callable=StringIO):
                main()

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["data_selection"]["name"], "eval-only-test")
        self.assertEqual(payload["raw_train_case_count"], 2)
        self.assertEqual(payload["raw_eval_case_count"], 2)
        self.assertEqual(payload["used_train_case_count"], 2)
        self.assertEqual(payload["used_eval_case_count"], 2)
        self.assertEqual(payload["checkpoint_path"], str(checkpoint_path))
        self.assertIn("loss", payload["train_metrics"])
        self.assertIn("loss", payload["eval_metrics"])


if __name__ == "__main__":
    unittest.main()
