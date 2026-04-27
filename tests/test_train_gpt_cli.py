from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from intrep import train_gpt
from intrep.mixed_corpus import MixedDocument


@dataclass(frozen=True)
class FakeTrainingResult:
    initial_loss: float = 4.0
    final_loss: float = 2.5
    steps: int = 3
    token_count: int = 123
    best_loss: float = 2.25
    loss_reduction: float = 1.5
    loss_reduction_ratio: float = 0.375


class TrainGPTCLITest(unittest.TestCase):
    def test_default_uses_builtin_corpus(self) -> None:
        captured_documents = object()

        def fake_train_mixed_gpt(*, documents=None, training_config):
            nonlocal captured_documents
            captured_documents = documents
            self.assertEqual(training_config.max_steps, 3)
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(["--max-steps", "3"])

        self.assertIsNone(captured_documents)
        self.assertIn("corpus=builtin tokens=123 steps=3", output.getvalue())

    def test_file_corpus_uses_document_loader(self) -> None:
        loaded_documents = [MixedDocument(id="custom", modality="text", content="hello")]
        captured_documents: list[MixedDocument] | None = None
        captured_path: Path | None = None

        def fake_loader(path: str | Path) -> list[MixedDocument]:
            nonlocal captured_path
            captured_path = Path(path)
            return loaded_documents

        def fake_train_mixed_gpt(*, documents=None, training_config):
            nonlocal captured_documents
            captured_documents = documents
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(
                    ["--corpus", "file", "--corpus-path", "corpus.jsonl"],
                    document_loader=fake_loader,
                )

        self.assertEqual(captured_path, Path("corpus.jsonl"))
        self.assertEqual(captured_documents, loaded_documents)
        self.assertIn("corpus=corpus.jsonl", output.getvalue())

    def test_file_corpus_requires_path(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                train_gpt.main(["--corpus", "file"])

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("--corpus-path is required", error_output.getvalue())

    def test_loss_summary_prints_compact_line(self) -> None:
        def fake_train_mixed_gpt(*, documents=None, training_config):
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(["--loss-summary"])

        self.assertIn(
            "loss initial=4.0000 final=2.5000 best=2.2500 delta=1.5000 ratio=37.50%",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
