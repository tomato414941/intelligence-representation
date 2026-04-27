from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from intrep import evaluate_next_observation
from intrep.mixed_corpus import MixedDocument


@dataclass(frozen=True)
class FakeTrainingResult:
    steps: int = 3
    token_count: int = 123


@dataclass(frozen=True)
class FakeRankingMetrics:
    top1_accuracy: float = 0.75
    mean_positive_loss: float = 1.25
    mean_best_distractor_loss: float = 2.5
    mean_margin: float = 1.25


@dataclass(frozen=True)
class FakeEvaluationResult:
    train_case_count: int = 5
    eval_case_count: int = 5
    training_result: FakeTrainingResult = FakeTrainingResult()
    before_metrics: FakeRankingMetrics = FakeRankingMetrics(top1_accuracy=0.25)
    after_metrics: FakeRankingMetrics = FakeRankingMetrics()

    @property
    def train_cases(self) -> list[object]:
        return [object()] * self.train_case_count

    @property
    def eval_cases(self) -> list[object]:
        return [object()] * self.eval_case_count


class EvaluateNextObservationCLITest(unittest.TestCase):
    def test_default_evaluates_builtin_corpus(self) -> None:
        captured_documents: list[MixedDocument] | None = None

        def fake_evaluate_next_observation_learning(
            documents,
            *,
            eval_documents=None,
            training_config,
        ):
            nonlocal captured_documents
            captured_documents = documents
            self.assertIsNone(eval_documents)
            self.assertEqual(training_config.max_steps, 3)
            self.assertIsNone(training_config.batch_stride)
            return FakeEvaluationResult()

        output = io.StringIO()
        with patch.object(
            evaluate_next_observation,
            "evaluate_next_observation_learning",
            fake_evaluate_next_observation_learning,
        ):
            with redirect_stdout(output):
                evaluate_next_observation.main(["--max-steps", "3"])

        self.assertIsNotNone(captured_documents)
        assert captured_documents is not None
        self.assertTrue(captured_documents)
        stdout = output.getvalue()
        self.assertIn("intrep next-observation evaluation", stdout)
        self.assertIn("corpus=builtin eval_corpus=train train_cases=5 eval_cases=5", stdout)
        self.assertIn("tokens=123 steps=3", stdout)
        self.assertIn(
            "before_top1_accuracy=0.2500 after_top1_accuracy=0.7500"
            " before_margin=1.2500 after_margin=1.2500",
            stdout,
        )

    def test_file_corpus_uses_document_loader_and_passes_training_args(self) -> None:
        documents = [
            MixedDocument(
                id="case_001",
                modality="environment_symbolic",
                content="<obs> a <action> move <next_obs> b",
            ),
            MixedDocument(
                id="case_002",
                modality="environment_symbolic",
                content="<obs> c <action> move <next_obs> d",
            ),
        ]
        captured_path: Path | None = None
        captured_documents: list[MixedDocument] | None = None

        def fake_loader(path: str | Path) -> list[MixedDocument]:
            nonlocal captured_path
            captured_path = Path(path)
            return documents

        def fake_evaluate_next_observation_learning(
            documents_arg,
            *,
            eval_documents=None,
            training_config,
        ):
            nonlocal captured_documents
            captured_documents = documents_arg
            self.assertIsNone(eval_documents)
            self.assertEqual(training_config.max_steps, 9)
            self.assertEqual(training_config.context_length, 16)
            self.assertEqual(training_config.batch_size, 2)
            self.assertEqual(training_config.batch_stride, 4)
            return FakeEvaluationResult(
                train_case_count=2,
                eval_case_count=2,
                training_result=FakeTrainingResult(steps=9, token_count=45),
                after_metrics=FakeRankingMetrics(top1_accuracy=1.0),
            )

        output = io.StringIO()
        with patch.object(
            evaluate_next_observation,
            "evaluate_next_observation_learning",
            fake_evaluate_next_observation_learning,
        ):
            with redirect_stdout(output):
                evaluate_next_observation.main(
                    [
                        "--corpus",
                        "file",
                        "--corpus-path",
                        "corpus.jsonl",
                        "--max-steps",
                        "9",
                        "--context-length",
                        "16",
                        "--batch-size",
                        "2",
                        "--batch-stride",
                        "4",
                    ],
                    document_loader=fake_loader,
                )

        self.assertEqual(captured_path, Path("corpus.jsonl"))
        self.assertEqual(captured_documents, documents)
        self.assertIn(
            "corpus=corpus.jsonl eval_corpus=train train_cases=2 eval_cases=2 tokens=45 steps=9",
            output.getvalue(),
        )
        self.assertIn("after_top1_accuracy=1.0000", output.getvalue())

    def test_eval_corpus_path_is_used_for_held_out_evaluation(self) -> None:
        train_documents = [
            MixedDocument(id="train_001", modality="environment_symbolic", content="train"),
        ]
        eval_documents = [
            MixedDocument(id="eval_001", modality="environment_symbolic", content="eval"),
        ]
        captured_eval_documents: list[MixedDocument] | None = None

        def fake_loader(path: str | Path) -> list[MixedDocument]:
            if Path(path) == Path("train.jsonl"):
                return train_documents
            if Path(path) == Path("eval.jsonl"):
                return eval_documents
            raise AssertionError(path)

        def fake_evaluate_next_observation_learning(
            documents_arg,
            *,
            eval_documents=None,
            training_config,
        ):
            nonlocal captured_eval_documents
            self.assertEqual(documents_arg, train_documents)
            captured_eval_documents = eval_documents
            return FakeEvaluationResult(train_case_count=1, eval_case_count=1)

        output = io.StringIO()
        with patch.object(
            evaluate_next_observation,
            "evaluate_next_observation_learning",
            fake_evaluate_next_observation_learning,
        ):
            with redirect_stdout(output):
                evaluate_next_observation.main(
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

    def test_builtin_grid_corpus_extracts_grid_cases(self) -> None:
        captured_documents: list[MixedDocument] | None = None

        def fake_evaluate_next_observation_learning(
            documents,
            *,
            eval_documents=None,
            training_config,
        ):
            nonlocal captured_documents
            captured_documents = documents
            self.assertIsNone(eval_documents)
            return FakeEvaluationResult(train_case_count=7, eval_case_count=7)

        output = io.StringIO()
        with patch.object(
            evaluate_next_observation,
            "evaluate_next_observation_learning",
            fake_evaluate_next_observation_learning,
        ):
            with redirect_stdout(output):
                evaluate_next_observation.main(
                    ["--corpus", "builtin-grid"],
                )

        self.assertIsNotNone(captured_documents)
        assert captured_documents is not None
        self.assertIn("grid", {document.modality for document in captured_documents})
        self.assertIn(
            "corpus=builtin-grid eval_corpus=train train_cases=7 eval_cases=7",
            output.getvalue(),
        )

    def test_file_corpus_requires_path(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                evaluate_next_observation.main(["--corpus", "file"])

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("--corpus-path is required", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
