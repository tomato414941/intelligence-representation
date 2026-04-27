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

    def test_category_split_preserves_multi_document_modalities(self) -> None:
        documents = [
            MixedDocument(id="fiction_1", modality="external_book_fiction", content="a"),
            MixedDocument(id="technical_1", modality="external_technical_text", content="b"),
            MixedDocument(id="memoir_1", modality="external_book_memoir", content="c"),
            MixedDocument(id="fiction_2", modality="external_book_fiction", content="d"),
            MixedDocument(id="technical_2", modality="external_technical_text", content="e"),
            MixedDocument(id="fiction_3", modality="external_book_fiction", content="f"),
            MixedDocument(id="technical_3", modality="external_technical_text", content="g"),
        ]

        split = split_mixed_documents(
            documents,
            eval_ratio=0.33,
            strategy="category",
            seed="category-test",
        )

        train_modalities = {document.modality for document in split.train_documents}
        eval_modalities = {document.modality for document in split.eval_documents}
        self.assertIn("external_book_fiction", train_modalities)
        self.assertIn("external_book_fiction", eval_modalities)
        self.assertIn("external_technical_text", train_modalities)
        self.assertIn("external_technical_text", eval_modalities)
        self.assertIn("external_book_memoir", train_modalities)
        self.assertNotIn("external_book_memoir", eval_modalities)

    def test_category_split_is_stable_and_preserves_input_order(self) -> None:
        documents = [
            MixedDocument(id=f"doc_{index}", modality="text" if index % 2 else "code", content="x")
            for index in range(8)
        ]

        first = split_mixed_documents(documents, strategy="category", seed="same")
        second = split_mixed_documents(documents, strategy="category", seed="same")

        self.assertEqual(first, second)
        document_order = {document.id: index for index, document in enumerate(documents)}
        self.assertEqual(
            [document_order[document.id] for document in first.train_documents],
            sorted(document_order[document.id] for document in first.train_documents),
        )
        self.assertEqual(
            [document_order[document.id] for document in first.eval_documents],
            sorted(document_order[document.id] for document in first.eval_documents),
        )

    def test_category_split_validates_strategy_and_category_key(self) -> None:
        documents = [MixedDocument(id="a", modality="text", content="alpha")]

        with self.assertRaisesRegex(ValueError, "strategy"):
            split_mixed_documents(documents, strategy="unknown")
        with self.assertRaisesRegex(ValueError, "category_key"):
            split_mixed_documents(documents, strategy="category", category_key="content")

    def test_environment_pair_split_keeps_pairs_together(self) -> None:
        documents = [
            MixedDocument(id="env_pair_symbolic_001", modality="environment_symbolic", content="<obs> a"),
            MixedDocument(id="env_pair_natural_001", modality="environment_natural", content="A."),
            MixedDocument(id="env_pair_symbolic_002", modality="environment_symbolic", content="<obs> b"),
            MixedDocument(id="env_pair_natural_002", modality="environment_natural", content="B."),
            MixedDocument(id="env_pair_symbolic_003", modality="environment_symbolic", content="<obs> c"),
            MixedDocument(id="env_pair_natural_003", modality="environment_natural", content="C."),
            MixedDocument(id="text_001", modality="text", content="background"),
        ]

        split = split_mixed_documents(
            documents,
            eval_ratio=0.34,
            strategy="environment-pair",
            seed="pair-test",
        )

        train_episode_ids = {_environment_pair_suffix(document.id) for document in split.train_documents}
        eval_episode_ids = {_environment_pair_suffix(document.id) for document in split.eval_documents}
        train_episode_ids.discard(None)
        eval_episode_ids.discard(None)
        self.assertTrue(eval_episode_ids)
        self.assertTrue(train_episode_ids)
        self.assertTrue(train_episode_ids.isdisjoint(eval_episode_ids))
        self.assertEqual(
            {document.modality for document in split.eval_documents if document.modality.startswith("environment_")},
            {"environment_symbolic", "environment_natural"},
        )

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
                strategy="stable-hash",
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

    def test_cli_category_strategy_writes_split_files(self) -> None:
        documents = [
            MixedDocument(id=f"text_{index}", modality="text", content=f"text {index}")
            for index in range(4)
        ] + [
            MixedDocument(id=f"code_{index}", modality="code", content=f"code {index}")
            for index in range(4)
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
                        "--strategy",
                        "category",
                        "--category-key",
                        "modality",
                    ]
                )

            train_modalities = {document.modality for document in load_mixed_documents_jsonl(train_path)}
            eval_modalities = {document.modality for document in load_mixed_documents_jsonl(eval_path)}

        self.assertEqual(train_modalities, {"text", "code"})
        self.assertEqual(eval_modalities, {"text", "code"})
        self.assertIn("wrote train=", output.getvalue())

    def test_cli_environment_pair_strategy_writes_split_files(self) -> None:
        documents = [
            MixedDocument(id="env_pair_symbolic_001", modality="environment_symbolic", content="<obs> a"),
            MixedDocument(id="env_pair_natural_001", modality="environment_natural", content="A."),
            MixedDocument(id="env_pair_symbolic_002", modality="environment_symbolic", content="<obs> b"),
            MixedDocument(id="env_pair_natural_002", modality="environment_natural", content="B."),
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
                        "--strategy",
                        "environment-pair",
                    ]
                )

            train_ids = {_environment_pair_suffix(document.id) for document in load_mixed_documents_jsonl(train_path)}
            eval_ids = {_environment_pair_suffix(document.id) for document in load_mixed_documents_jsonl(eval_path)}

        self.assertEqual(len(train_ids), 1)
        self.assertEqual(len(eval_ids), 1)
        self.assertTrue(train_ids.isdisjoint(eval_ids))
        self.assertIn("wrote train=", output.getvalue())


def _environment_pair_suffix(document_id: str) -> str | None:
    for prefix in ("env_pair_symbolic_", "env_pair_natural_"):
        if document_id.startswith(prefix):
            return document_id.removeprefix(prefix)
    return None


if __name__ == "__main__":
    unittest.main()
