from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from intrep.generated_environment_corpus import (
    environment_pair_documents_from_examples,
    generated_environment_corpus_selection,
    generated_environment_pair_documents,
    generated_environment_train_eval_documents,
    main,
    write_generated_environment_pair_jsonl,
)
from intrep.mixed_corpus import load_mixed_documents_jsonl
from intrep.transition_data import generated_find_examples, split_generated_examples


class GeneratedEnvironmentCorpusTest(unittest.TestCase):
    def test_generates_all_generated_find_examples_as_symbolic_natural_pairs(self) -> None:
        documents = generated_environment_pair_documents()

        self.assertEqual(len(documents), 144)
        self.assertEqual(documents[0].id, "env_pair_symbolic_generated_鍵_箱_棚")
        self.assertEqual(documents[1].id, "env_pair_natural_generated_鍵_箱_棚")
        self.assertEqual(documents[0].modality, "environment_symbolic")
        self.assertEqual(documents[1].modality, "environment_natural")
        self.assertIn("<obs>", documents[0].content)
        self.assertIn("<action>", documents[0].content)
        self.assertIn("<next_obs>", documents[0].content)
        self.assertIn("next observation", documents[1].content)

    def test_train_eval_documents_use_transition_data_split(self) -> None:
        train_documents, eval_documents = generated_environment_train_eval_documents(
            "generated_held_out_object"
        )

        self.assertEqual(len(train_documents), 24)
        self.assertEqual(len(eval_documents), 24)
        self.assertTrue(all(document.id.startswith("env_pair_") for document in train_documents))
        self.assertTrue(any("時計" in document.id for document in eval_documents))

        _, slices = split_generated_examples(generated_find_examples())
        held_out_location_documents = environment_pair_documents_from_examples(
            slices["generated_held_out_location"]
        )
        self.assertEqual(len(held_out_location_documents), 12)

    def test_strict_train_eval_documents_include_metadata_and_negative_cases(self) -> None:
        train_documents, eval_documents = generated_environment_train_eval_documents(
            "generated_strict_same_entity_negative"
        )

        self.assertEqual(len(train_documents), 8)
        self.assertEqual(len(eval_documents), 2)
        self.assertIn('group_id="generated_strict_train"', train_documents[0].content)
        self.assertIn(
            'group_id="generated_strict_same_entity_negative"',
            eval_documents[0].content,
        )
        self.assertIn("<next_obs> none", eval_documents[0].content)
        self.assertIn('hard_negative_nexts="箱 at 棚"', eval_documents[0].content)

    def test_strict_noisy_documents_render_stable_hard_negative_markers(self) -> None:
        _, eval_documents = generated_environment_train_eval_documents("generated_strict_noisy")

        self.assertEqual(len(eval_documents), 2)
        self.assertIn(
            'hard_negative_nexts="ポーチ at 机 | 時計 at ポーチ | 箱 at 棚 | 鍵 at 箱"',
            eval_documents[0].content,
        )

    def test_generated_environment_corpus_selection_names_eval_slice(self) -> None:
        selection = generated_environment_corpus_selection("generated_held_out_container")

        self.assertEqual(selection.eval_label, "generated_held_out_container")
        self.assertEqual(len(selection.train_documents), 24)
        self.assertEqual(len(selection.eval_documents), 24)
        self.assertTrue(
            all(document.modality.startswith("environment_") for document in selection.eval_documents)
        )

    def test_write_generated_environment_pair_jsonl_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "generated.jsonl"

            documents = write_generated_environment_pair_jsonl(output_path)

            self.assertEqual(load_mixed_documents_jsonl(output_path), documents)

    def test_cli_writes_all_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "generated.jsonl"
            output = io.StringIO()

            with redirect_stdout(output):
                main(["--output", str(output_path)])

            documents = load_mixed_documents_jsonl(output_path)

        self.assertEqual(len(documents), 144)
        self.assertIn("wrote documents=144 pairs=72", output.getvalue())

    def test_cli_writes_train_eval_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            output = io.StringIO()

            with redirect_stdout(output):
                main(
                    [
                        "--train-output",
                        str(train_path),
                        "--eval-output",
                        str(eval_path),
                        "--eval-slice",
                        "generated_held_out_container",
                    ]
                )

            train_documents = load_mixed_documents_jsonl(train_path)
            eval_documents = load_mixed_documents_jsonl(eval_path)

        self.assertEqual(len(train_documents), 24)
        self.assertEqual(len(eval_documents), 24)
        self.assertIn("eval_slice=generated_held_out_container", output.getvalue())

    def test_cli_writes_strict_train_eval_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            output = io.StringIO()

            with redirect_stdout(output):
                main(
                    [
                        "--train-output",
                        str(train_path),
                        "--eval-output",
                        str(eval_path),
                        "--eval-slice",
                        "generated_strict_partial",
                    ]
                )

            train_documents = load_mixed_documents_jsonl(train_path)
            eval_documents = load_mixed_documents_jsonl(eval_path)

        self.assertEqual(len(train_documents), 8)
        self.assertEqual(len(eval_documents), 2)
        self.assertIn("eval_slice=generated_strict_partial", output.getvalue())

    def test_cli_requires_complete_output_arguments(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit):
                main(["--train-output", "train.jsonl"])

        self.assertIn("--train-output and --eval-output", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
