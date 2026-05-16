import json
import tempfile
import unittest
from pathlib import Path

from intrep.problems.shogi_policy_value.training_inputs import load_shogi_policy_value_training_inputs


class ShogiPolicyValueTrainingInputsTest(unittest.TestCase):
    def test_lists_data_selection_sources_and_tensor_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train-games.jsonl"
            eval_ = root / "eval-games.jsonl"
            cache = root / "cache" / "shogi-policy-value-tensors"
            selection = root / "data-selection.json"
            train.write_text("", encoding="utf-8")
            eval_.write_text("", encoding="utf-8")
            cache.mkdir(parents=True)
            selection.write_text(
                json.dumps(
                    {
                        "name": "test",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": "train-games.jsonl"}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": "eval-games.jsonl"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            inputs = load_shogi_policy_value_training_inputs(
                data_selection_path=selection,
                tensor_cache_path=cache,
            )

            self.assertEqual(
                set(inputs.artifact_paths()),
                {selection, train, eval_, cache},
            )

    def test_rejects_missing_tensor_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train-games.jsonl"
            eval_ = root / "eval-games.jsonl"
            selection = root / "data-selection.json"
            train.write_text("", encoding="utf-8")
            eval_.write_text("", encoding="utf-8")
            selection.write_text(
                json.dumps(
                    {
                        "name": "test",
                        "objective": "shogi policy-value",
                        "target_construction": {
                            "policy": "chosen_move",
                            "policy_temperature_cp": 100.0,
                            "policy_mate_cp": 100000.0,
                            "value": "winner",
                            "score_cp_scale": 600.0,
                        },
                        "train_sources": [{"kind": "game_records_jsonl", "path": "train-games.jsonl"}],
                        "eval_sources": [{"kind": "game_records_jsonl", "path": "eval-games.jsonl"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "tensor cache not found"):
                load_shogi_policy_value_training_inputs(
                    data_selection_path=selection,
                    tensor_cache_path=root / "missing-cache",
                )


if __name__ == "__main__":
    unittest.main()
