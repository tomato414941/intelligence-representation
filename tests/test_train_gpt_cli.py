from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from intrep import train_gpt
from intrep.mixed_corpus import MixedDocument, write_mixed_documents_jsonl


@dataclass(frozen=True)
class FakeTrainingResult:
    initial_loss: float = 4.0
    final_loss: float = 2.5
    steps: int = 3
    token_count: int = 123
    best_loss: float = 2.25
    loss_reduction: float = 1.5
    loss_reduction_ratio: float = 0.375
    loss_history: tuple[float, ...] = (4.0, 3.0, 2.5)
    initial_train_loss: float | None = 4.25
    final_train_loss: float | None = 2.75
    initial_eval_loss: float | None = 4.5
    final_eval_loss: float | None = 3.5


class TrainGPTCLITest(unittest.TestCase):
    def test_default_uses_builtin_corpus(self) -> None:
        captured_documents = object()

        def fake_train_mixed_gpt(*, documents=None, eval_documents=None, training_config):
            nonlocal captured_documents
            captured_documents = documents
            self.assertIsNone(eval_documents)
            self.assertEqual(training_config.max_steps, 3)
            self.assertIsNone(training_config.batch_stride)
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(["--max-steps", "3"])

        self.assertIsNone(captured_documents)
        self.assertIn("corpus=builtin eval_corpus=none tokens=123 steps=3", output.getvalue())
        self.assertIn("train_avg_initial=4.2500 train_avg_final=2.7500", output.getvalue())

    def test_batch_stride_passes_to_training_config(self) -> None:
        def fake_train_mixed_gpt(*, documents=None, eval_documents=None, training_config):
            self.assertEqual(training_config.batch_stride, 5)
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(["--batch-stride", "5"])

        self.assertIn("corpus=builtin eval_corpus=none tokens=123 steps=3", output.getvalue())

    def test_file_corpus_uses_document_loader(self) -> None:
        loaded_documents = [MixedDocument(id="custom", modality="text", content="hello")]
        captured_documents: list[MixedDocument] | None = None
        captured_path: Path | None = None

        def fake_loader(path: str | Path) -> list[MixedDocument]:
            nonlocal captured_path
            captured_path = Path(path)
            return loaded_documents

        def fake_train_mixed_gpt(*, documents=None, eval_documents=None, training_config):
            nonlocal captured_documents
            captured_documents = documents
            self.assertIsNone(eval_documents)
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

    def test_eval_corpus_uses_document_loader(self) -> None:
        train_documents = [MixedDocument(id="train", modality="text", content="train")]
        eval_documents = [MixedDocument(id="eval", modality="text", content="eval")]
        captured_eval_documents: list[MixedDocument] | None = None

        def fake_loader(path: str | Path) -> list[MixedDocument]:
            if Path(path) == Path("train.jsonl"):
                return train_documents
            if Path(path) == Path("eval.jsonl"):
                return eval_documents
            raise AssertionError(path)

        def fake_train_mixed_gpt(*, documents=None, eval_documents=None, training_config):
            nonlocal captured_eval_documents
            self.assertEqual(documents, train_documents)
            captured_eval_documents = eval_documents
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(
                    [
                        "--corpus",
                        "file",
                        "--corpus-path",
                        "train.jsonl",
                        "--eval-corpus-path",
                        "eval.jsonl",
                    ],
                    document_loader=fake_loader,
                )

        self.assertEqual(captured_eval_documents, eval_documents)
        self.assertIn("eval_corpus=eval.jsonl", output.getvalue())

    def test_loss_summary_prints_compact_line(self) -> None:
        def fake_train_mixed_gpt(*, documents=None, eval_documents=None, training_config):
            return FakeTrainingResult()

        output = io.StringIO()
        with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
            with redirect_stdout(output):
                train_gpt.main(["--loss-summary"])

        self.assertIn(
            "loss initial=4.0000 final=2.5000 best=2.2500 delta=1.5000 ratio=37.50%"
            " eval_initial=4.5000 eval_final=3.5000"
            " train_avg_initial=4.2500 train_avg_final=2.7500",
            output.getvalue(),
        )

    def test_loss_history_path_writes_json(self) -> None:
        def fake_train_mixed_gpt(*, documents=None, eval_documents=None, training_config):
            return FakeTrainingResult()

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            loss_history_path = Path(temp_dir) / "loss-history.json"
            with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
                with redirect_stdout(output):
                    train_gpt.main(["--loss-history-path", str(loss_history_path)])

            payload = json.loads(loss_history_path.read_text(encoding="utf-8"))

        self.assertIn("corpus=builtin eval_corpus=none tokens=123 steps=3", output.getvalue())
        self.assertEqual(
            payload,
            {
                "steps": 3,
                "token_count": 123,
                "batch_stride": None,
                "initial_loss": 4.0,
                "final_loss": 2.5,
                "best_loss": 2.25,
                "loss_history": [4.0, 3.0, 2.5],
                "initial_train_loss": 4.25,
                "final_train_loss": 2.75,
                "initial_eval_loss": 4.5,
                "final_eval_loss": 3.5,
            },
        )

    def test_file_corpus_eval_corpus_and_loss_history_use_real_jsonl_loader(self) -> None:
        train_documents = [
            MixedDocument(id="train_text_001", modality="text", content="train observation"),
            MixedDocument(
                id="train_env_001",
                modality="environment_symbolic",
                content="<obs> box closed <action> open box <next_obs> key visible",
            ),
        ]
        eval_documents = [
            MixedDocument(id="eval_text_001", modality="text", content="eval observation"),
        ]
        captured_documents: list[MixedDocument] | None = None
        captured_eval_documents: list[MixedDocument] | None = None

        def fake_train_mixed_gpt(*, documents=None, eval_documents=None, training_config):
            nonlocal captured_documents, captured_eval_documents
            captured_documents = documents
            captured_eval_documents = eval_documents
            self.assertEqual(training_config.max_steps, 2)
            return FakeTrainingResult(
                steps=2,
                token_count=456,
                initial_eval_loss=5.5,
                final_eval_loss=4.25,
            )

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corpus_path = temp_path / "train.jsonl"
            eval_corpus_path = temp_path / "eval.jsonl"
            loss_history_path = temp_path / "loss-history.json"
            write_mixed_documents_jsonl(corpus_path, train_documents)
            write_mixed_documents_jsonl(eval_corpus_path, eval_documents)

            with patch.object(train_gpt, "train_mixed_gpt", fake_train_mixed_gpt):
                with redirect_stdout(output):
                    train_gpt.main(
                        [
                            "--corpus",
                            "file",
                            "--corpus-path",
                            str(corpus_path),
                            "--eval-corpus-path",
                            str(eval_corpus_path),
                            "--max-steps",
                            "2",
                            "--batch-stride",
                            "7",
                            "--loss-summary",
                            "--loss-history-path",
                            str(loss_history_path),
                        ]
                    )

            payload = json.loads(loss_history_path.read_text(encoding="utf-8"))

        self.assertEqual(captured_documents, train_documents)
        self.assertEqual(captured_eval_documents, eval_documents)
        stdout = output.getvalue()
        self.assertIn(f"corpus={corpus_path}", stdout)
        self.assertIn(f"eval_corpus={eval_corpus_path}", stdout)
        self.assertIn("tokens=456 steps=2 initial_loss=4.0000 final_loss=2.5000", stdout)
        self.assertIn("train_avg_initial=4.2500 train_avg_final=2.7500", stdout)
        self.assertIn("eval_initial=5.5000 eval_final=4.2500", stdout)
        self.assertEqual(payload["batch_stride"], 7)
        self.assertEqual(payload["initial_eval_loss"], 5.5)
        self.assertEqual(payload["final_eval_loss"], 4.25)


if __name__ == "__main__":
    unittest.main()
