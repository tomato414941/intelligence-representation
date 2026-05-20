import json
import tempfile
import unittest
from pathlib import Path

from intrep.problems.shogi_policy_value.training_inputs import load_shogi_policy_value_training_inputs


class ShogiPolicyValueTrainingInputsTest(unittest.TestCase):
    def test_lists_data_selection_sources_and_tensor_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train-examples.jsonl"
            eval_ = root / "eval-examples.jsonl"
            cache = root / "cache" / "legal-move"
            selection = root / "data-selection.json"
            train.write_text("", encoding="utf-8")
            eval_.write_text("", encoding="utf-8")
            cache.mkdir(parents=True)
            selection.write_text(
                json.dumps(
                    {
                        "name": "test",
                        "objective": "shogi policy-value",
                        "train_sources": [{"kind": "shogi_policy_value_examples_jsonl", "path": "train-examples.jsonl"}],
                        "eval_sources": [{"kind": "shogi_policy_value_examples_jsonl", "path": "eval-examples.jsonl"}],
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
            train = root / "train-examples.jsonl"
            eval_ = root / "eval-examples.jsonl"
            selection = root / "data-selection.json"
            train.write_text("", encoding="utf-8")
            eval_.write_text("", encoding="utf-8")
            selection.write_text(
                json.dumps(
                    {
                        "name": "test",
                        "objective": "shogi policy-value",
                        "train_sources": [{"kind": "shogi_policy_value_examples_jsonl", "path": "train-examples.jsonl"}],
                        "eval_sources": [{"kind": "shogi_policy_value_examples_jsonl", "path": "eval-examples.jsonl"}],
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
