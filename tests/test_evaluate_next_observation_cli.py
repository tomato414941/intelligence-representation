from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from intrep import evaluate_next_observation
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
    initial_train_loss: float | None = 4.25
    final_train_loss: float | None = 2.75
    initial_eval_loss: float | None = 4.5
    final_eval_loss: float | None = 3.5


@dataclass(frozen=True)
class FakeRankingMetrics:
    top1_accuracy: float = 0.75
    mean_positive_loss: float = 1.25
    mean_best_distractor_loss: float = 2.5
    mean_margin: float = 1.25


@dataclass(frozen=True)
class FakeRankingSummary:
    overall: FakeRankingMetrics
    per_modality: dict[str, FakeRankingMetrics]
    modality_counts: dict[str, int]


@dataclass(frozen=True)
class FakeEvaluationResult:
    train_case_count: int = 5
    eval_case_count: int = 5
    training_result: FakeTrainingResult = FakeTrainingResult()
    before_metrics: FakeRankingMetrics = FakeRankingMetrics(top1_accuracy=0.25)
    after_metrics: FakeRankingMetrics = FakeRankingMetrics()
    before_summary: FakeRankingSummary | None = None
    after_summary: FakeRankingSummary | None = None

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
            distractor_policy="hard",
        ):
            nonlocal captured_documents
            captured_documents = documents
            self.assertIsNone(eval_documents)
            self.assertEqual(training_config.max_steps, 3)
            self.assertIsNone(training_config.batch_stride)
            self.assertEqual(distractor_policy, "hard")
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
        self.assertIn(
            "corpus=builtin eval_corpus=train train_cases=5 eval_cases=5 tokens=123 steps=3 "
            "eval_split=train generalization_eval=false distractor_policy=hard",
            stdout,
        )
        self.assertIn("tokens=123 steps=3", stdout)
        self.assertIn(
            "before_top1_accuracy=0.2500 after_top1_accuracy=0.7500"
            " top1_delta=0.5000 before_margin=1.2500 after_margin=1.2500"
            " margin_delta=0.0000",
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
            distractor_policy="hard",
        ):
            nonlocal captured_documents
            captured_documents = documents_arg
            self.assertIsNone(eval_documents)
            self.assertEqual(training_config.max_steps, 9)
            self.assertEqual(training_config.context_length, 16)
            self.assertEqual(training_config.batch_size, 2)
            self.assertEqual(training_config.batch_stride, 4)
            self.assertEqual(distractor_policy, "all_other")
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
                        "--distractor-policy",
                        "all_other",
                    ],
                    document_loader=fake_loader,
                )

        self.assertEqual(captured_path, Path("corpus.jsonl"))
        self.assertEqual(captured_documents, documents)
        self.assertIn(
            "corpus=corpus.jsonl eval_corpus=train train_cases=2 eval_cases=2 tokens=45 steps=9 "
            "eval_split=train generalization_eval=false distractor_policy=all_other",
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
            distractor_policy="hard",
        ):
            nonlocal captured_eval_documents
            self.assertEqual(documents_arg, train_documents)
            captured_eval_documents = eval_documents
            self.assertEqual(distractor_policy, "hard")
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

    def test_metrics_path_writes_json_payload(self) -> None:
        before_summary = FakeRankingSummary(
            overall=FakeRankingMetrics(top1_accuracy=0.25, mean_margin=0.5),
            per_modality={
                "environment_symbolic": FakeRankingMetrics(
                    top1_accuracy=0.25,
                    mean_margin=0.5,
                )
            },
            modality_counts={"environment_symbolic": 5},
        )
        after_summary = FakeRankingSummary(
            overall=FakeRankingMetrics(top1_accuracy=0.75, mean_margin=1.25),
            per_modality={
                "environment_symbolic": FakeRankingMetrics(
                    top1_accuracy=0.75,
                    mean_margin=1.25,
                )
            },
            modality_counts={"environment_symbolic": 5},
        )

        def fake_evaluate_next_observation_learning(
            documents,
            *,
            eval_documents=None,
            training_config,
            distractor_policy="hard",
        ):
            self.assertEqual(training_config.max_steps, 3)
            self.assertEqual(distractor_policy, "hard")
            return FakeEvaluationResult(
                before_metrics=before_summary.overall,
                after_metrics=after_summary.overall,
                before_summary=before_summary,
                after_summary=after_summary,
            )

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            with patch.object(
                evaluate_next_observation,
                "evaluate_next_observation_learning",
                fake_evaluate_next_observation_learning,
            ):
                with redirect_stdout(output):
                    evaluate_next_observation.main(
                        ["--max-steps", "3", "--metrics-path", str(metrics_path)]
                    )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn("top1_delta=0.5000", output.getvalue())
        self.assertEqual(payload["corpus"], "builtin")
        self.assertEqual(payload["eval_corpus"], "train")
        self.assertEqual(payload["eval_split"], "train")
        self.assertFalse(payload["generalization_eval"])
        self.assertTrue(payload["warnings"])
        self.assertEqual(payload["distractor_policy"], "hard")
        self.assertEqual(payload["train_case_count"], 5)
        self.assertEqual(payload["eval_case_count"], 5)
        self.assertEqual(payload["training"]["steps"], 3)
        self.assertEqual(payload["training"]["token_count"], 123)
        self.assertEqual(payload["training"]["final_train_loss"], 2.75)
        self.assertEqual(
            payload["ranking"]["before"]["modality_counts"],
            {"environment_symbolic": 5},
        )
        self.assertEqual(
            payload["ranking"]["after"]["per_modality"]["environment_symbolic"]["top1_accuracy"],
            0.75,
        )
        self.assertEqual(payload["deltas"], {"top1_accuracy": 0.5, "mean_margin": 0.75})

    def test_builtin_grid_corpus_extracts_grid_cases(self) -> None:
        captured_documents: list[MixedDocument] | None = None

        def fake_evaluate_next_observation_learning(
            documents,
            *,
            eval_documents=None,
            training_config,
            distractor_policy="hard",
        ):
            nonlocal captured_documents
            captured_documents = documents
            self.assertIsNone(eval_documents)
            self.assertEqual(distractor_policy, "hard")
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
            "corpus=builtin-grid eval_corpus=train train_cases=7 eval_cases=7 tokens=123 steps=3 "
            "eval_split=train generalization_eval=false distractor_policy=hard",
            output.getvalue(),
        )

    def test_generated_environment_corpus_uses_generated_train_eval_split(self) -> None:
        captured_train_documents: list[MixedDocument] | None = None
        captured_eval_documents: list[MixedDocument] | None = None

        def fake_evaluate_next_observation_learning(
            documents,
            *,
            eval_documents=None,
            training_config,
            distractor_policy="hard",
        ):
            nonlocal captured_train_documents, captured_eval_documents
            captured_train_documents = documents
            captured_eval_documents = eval_documents
            self.assertEqual(distractor_policy, "hard")
            return FakeEvaluationResult(train_case_count=12, eval_case_count=12)

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
                        "generated-environment",
                        "--generated-eval-slice",
                        "generated_held_out_container",
                    ],
                )

        self.assertIsNotNone(captured_train_documents)
        self.assertIsNotNone(captured_eval_documents)
        assert captured_train_documents is not None
        assert captured_eval_documents is not None
        self.assertEqual(len(captured_train_documents), 24)
        self.assertEqual(len(captured_eval_documents), 24)
        self.assertIn(
            "corpus=generated-environment eval_corpus=generated_held_out_container "
            "train_cases=12 eval_cases=12 tokens=123 steps=3 "
            "eval_split=held_out generalization_eval=true distractor_policy=hard",
            output.getvalue(),
        )

    def test_generated_seen_reports_train_split_not_generalization(self) -> None:
        def fake_evaluate_next_observation_learning(
            documents,
            *,
            eval_documents=None,
            training_config,
            distractor_policy="hard",
        ):
            self.assertIsNotNone(eval_documents)
            return FakeEvaluationResult(train_case_count=12, eval_case_count=12)

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            with patch.object(
                evaluate_next_observation,
                "evaluate_next_observation_learning",
                fake_evaluate_next_observation_learning,
            ):
                with redirect_stdout(output):
                    evaluate_next_observation.main(
                        [
                            "--corpus",
                            "generated-environment",
                            "--generated-eval-slice",
                            "generated_seen",
                            "--metrics-path",
                            str(metrics_path),
                        ],
                    )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn(
            "eval_corpus=generated_seen train_cases=12 eval_cases=12 tokens=123 steps=3 "
            "eval_split=train generalization_eval=false distractor_policy=hard",
            output.getvalue(),
        )
        self.assertEqual(payload["eval_split"], "train")
        self.assertFalse(payload["generalization_eval"])
        self.assertEqual(payload["training"]["eval_split"], "train")
        self.assertFalse(payload["training"]["generalization_eval"])
        self.assertTrue(payload["warnings"])

    def test_same_entity_distractor_policy_is_passed_through(self) -> None:
        captured_policy: str | None = None

        def fake_evaluate_next_observation_learning(
            documents,
            *,
            eval_documents=None,
            training_config,
            distractor_policy="hard",
        ):
            nonlocal captured_policy
            captured_policy = distractor_policy
            return FakeEvaluationResult()

        with patch.object(
            evaluate_next_observation,
            "evaluate_next_observation_learning",
            fake_evaluate_next_observation_learning,
        ):
            with redirect_stdout(io.StringIO()):
                evaluate_next_observation.main(["--distractor-policy", "same_entity"])

        self.assertEqual(captured_policy, "same_entity")

    def test_file_corpus_requires_path(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                evaluate_next_observation.main(["--corpus", "file"])

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("--corpus-path is required", error_output.getvalue())

    def test_file_corpus_can_evaluate_signal_jsonl_with_signal_rendering(self) -> None:
        captured_documents: list[MixedDocument] | None = None

        def fake_evaluate_next_observation_learning(
            documents,
            *,
            eval_documents=None,
            training_config,
            distractor_policy="hard",
        ):
            nonlocal captured_documents
            captured_documents = documents
            self.assertIsNone(eval_documents)
            return FakeEvaluationResult(train_case_count=2, eval_case_count=2)

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "signal.jsonl"
            corpus_path.write_text(
                (
                    '{"id":"case_1","role":"observation","modality":"environment_symbolic",'
                    '"content":"<obs> box closed <action> open box <next_obs> key visible"}\n'
                    '{"id":"case_2","role":"observation","modality":"environment_symbolic",'
                    '"content":"<obs> box closed <action> open box <next_obs> coin visible"}\n'
                ),
                encoding="utf-8",
            )

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
                            str(corpus_path),
                            "--corpus-format",
                            "signal",
                            "--render-format",
                            "signal-tags",
                        ]
                    )

        assert captured_documents is not None
        self.assertEqual(captured_documents[0].modality, "observation")
        self.assertIn('channel="observation"', captured_documents[0].content)
        self.assertIn("corpus_format=signal render_format=signal-tags", output.getvalue())

    def test_file_corpus_rejects_payload_ref_signal_jsonl(self) -> None:
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "signal.jsonl"
            corpus_path.write_text(
                (
                    '{"channel":"image",'
                    '"payload_ref":{"uri":"dataset://images/frame-1.png","media_type":"image/png"}}\n'
                ),
                encoding="utf-8",
            )

            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    evaluate_next_observation.main(
                        [
                            "--corpus",
                            "file",
                            "--corpus-path",
                            str(corpus_path),
                            "--corpus-format",
                            "signal",
                            "--render-format",
                            "signal-tags",
                        ]
                    )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires a channel-specific loader or encoder", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
