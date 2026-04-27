from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from intrep.corpus_split import main, split_mixed_documents, write_split_jsonl
from intrep.mixed_corpus import (
    MixedDocument,
    load_mixed_documents_jsonl,
    write_mixed_documents_jsonl,
)


class CorpusSplitTest(unittest.TestCase):
    def test_split_is_stable_and_preserves_all_documents(self) -> None:
        documents = [
            MixedDocument(id=f"doc_{index}", modality="external_text", content=f"text {index}")
            for index in range(20)
        ]

        first = split_mixed_documents(documents, eval_ratio=0.25, seed="same")
        second = split_mixed_documents(documents, eval_ratio=0.25, seed="same")

        self.assertEqual(first, second)
        combined_ids = {document.id for document in first.train_documents + first.eval_documents}
        self.assertEqual(combined_ids, {document.id for document in documents})
        self.assertGreater(len(first.train_documents), 0)
        self.assertGreater(len(first.eval_documents), 0)

    def test_split_validates_ratio_and_key(self) -> None:
        documents = [MixedDocument(id="a", modality="text", content="alpha")]

        with self.assertRaisesRegex(ValueError, "eval_ratio"):
            split_mixed_documents(documents, eval_ratio=0.0)
        with self.assertRaisesRegex(ValueError, "key"):
            split_mixed_documents(documents, key="source")  # type: ignore[arg-type]

    def test_write_split_jsonl_round_trips_outputs(self) -> None:
        documents = [
            MixedDocument(id=f"doc_{index}", modality="external_text", content=f"text {index}")
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "corpus.jsonl"
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            write_mixed_documents_jsonl(input_path, documents)

            split = write_split_jsonl(
                input_path,
                train_path,
                eval_path,
                eval_ratio=0.3,
                seed="split-test",
            )

            self.assertEqual(load_mixed_documents_jsonl(train_path), split.train_documents)
            self.assertEqual(load_mixed_documents_jsonl(eval_path), split.eval_documents)

    def test_cli_writes_split_files(self) -> None:
        documents = [
            MixedDocument(id=f"doc_{index}", modality="external_text", content=f"text {index}")
            for index in range(8)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "corpus.jsonl"
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            write_mixed_documents_jsonl(input_path, documents)

            output = io.StringIO()
            with redirect_stdout(output):
                main(
                    [
                        "--input",
                        str(input_path),
                        "--train-output",
                        str(train_path),
                        "--eval-output",
                        str(eval_path),
                        "--eval-ratio",
                        "0.25",
                    ]
                )

            self.assertTrue(train_path.exists())
            self.assertTrue(eval_path.exists())
            self.assertIn("wrote train=", output.getvalue())


if __name__ == "__main__":
    unittest.main()
