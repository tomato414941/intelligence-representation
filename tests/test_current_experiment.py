from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from intrep import current_experiment
from intrep.byte_tokenizer import ByteTokenizer
from intrep.gpt_model import GPTConfig
from intrep.gpt_training import GPTTrainingConfig
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
    training_result: FakeTrainingResult = FakeTrainingResult()
    before_metrics: FakeRankingMetrics = FakeRankingMetrics(top1_accuracy=0.25)
    after_metrics: FakeRankingMetrics = FakeRankingMetrics()
    before_summary: FakeRankingSummary = FakeRankingSummary(
        overall=FakeRankingMetrics(top1_accuracy=0.25),
        per_modality={
            "environment_symbolic": FakeRankingMetrics(top1_accuracy=0.25),
        },
        modality_counts={"environment_symbolic": 2},
    )
    after_summary: FakeRankingSummary = FakeRankingSummary(
        overall=FakeRankingMetrics(),
        per_modality={
            "environment_symbolic": FakeRankingMetrics(),
        },
        modality_counts={"environment_symbolic": 2},
    )

    @property
    def train_cases(self) -> list[object]:
        return [object(), object()]

    @property
    def eval_cases(self) -> list[object]:
        return [object(), object()]


class CurrentExperimentTest(unittest.TestCase):
    def test_existing_builtin_corpora_use_existing_corpus_selection(self) -> None:
        for corpus_name in ("builtin", "builtin-grid"):
            with self.subTest(corpus=corpus_name):
                corpus = current_experiment.select_corpus(corpus_name)

                summary = current_experiment.run_current_experiment(
                    corpus.documents,
                    corpus_label=corpus.label,
                    training_config=GPTTrainingConfig(
                        context_length=16,
                        batch_size=2,
                        max_steps=1,
                        seed=47,
                    ),
                    evaluation_runner=lambda *args, **kwargs: FakeEvaluationResult(),
                )

                self.assertEqual(summary["corpus"]["label"], corpus_name)
                self.assertEqual(summary["corpus"]["eval_label"], "train")
                self.assertEqual(summary["training_loss"]["steps"], 3)
                self.assertIn("initial_train_perplexity", summary["language_modeling"])
                self.assertIn("final_train_perplexity", summary["language_modeling"])
                self.assertEqual(summary["next_observation"]["status"], "evaluated")
                self.assertGreaterEqual(summary["next_observation"]["eval_case_count"], 2)
                self.assertEqual(
                    summary["coverage"]["train"]["document_count"],
                    len(corpus.documents),
                )

    def test_external_local_file_paths_run_real_learning_and_held_out_evaluation(self) -> None:
        eval_documents = [
            MixedDocument(
                id="eval_a",
                modality="environment_symbolic",
                content="<obs> eval a <action> act <next_obs> eval done",
            ),
            MixedDocument(
                id="eval_b",
                modality="environment_symbolic",
                content="<obs> eval b <action> act <next_obs> eval other",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "train.jsonl"
            eval_path = Path(temp_dir) / "eval.jsonl"
            write_mixed_documents_jsonl(train_path, _case_documents())
            write_mixed_documents_jsonl(eval_path, eval_documents)

            train_corpus = current_experiment.select_corpus("file", train_path)
            eval_corpus = current_experiment.select_corpus("file", eval_path)

            summary = current_experiment.run_current_experiment(
                train_corpus.documents,
                eval_documents=eval_corpus.documents,
                corpus_label=train_corpus.label,
                eval_corpus_label=eval_corpus.label,
                training_config=GPTTrainingConfig(
                    context_length=16,
                    batch_size=2,
                    max_steps=1,
                    seed=53,
                ),
                evaluation_runner=lambda *args, **kwargs: FakeEvaluationResult(),
            )

        self.assertEqual(summary["corpus"]["label"], str(train_path))
        self.assertEqual(summary["corpus"]["eval_label"], str(eval_path))
        self.assertEqual(summary["next_observation"]["status"], "evaluated")
        self.assertEqual(summary["next_observation"]["eval_case_count"], 2)
        self.assertEqual(summary["coverage"]["eval"]["document_count"], 2)
        self.assertIsNotNone(summary["training_loss"]["initial_eval_loss"])
        self.assertIsNotNone(summary["training_loss"]["final_eval_loss"])
        self.assertIsNotNone(summary["language_modeling"]["initial_eval_perplexity"])
        self.assertIsNotNone(summary["language_modeling"]["final_eval_perplexity"])

    def test_run_current_experiment_uses_injected_documents_config_and_runner(self) -> None:
        documents = _case_documents()
        eval_documents = [
            MixedDocument(
                id="eval_a",
                modality="environment_symbolic",
                content="<obs> eval a <action> act <next_obs> eval done",
            ),
            MixedDocument(
                id="eval_b",
                modality="environment_symbolic",
                content="<obs> eval b <action> act <next_obs> eval other",
            ),
        ]
        captured_documents: list[MixedDocument] | None = None
        captured_eval_documents: list[MixedDocument] | None = None

        def fake_runner(documents_arg, *, eval_documents=None, training_config, model_config=None):
            nonlocal captured_documents, captured_eval_documents
            captured_documents = documents_arg
            captured_eval_documents = eval_documents
            self.assertIsNone(model_config)
            self.assertEqual(training_config.max_steps, 3)
            self.assertEqual(training_config.context_length, 16)
            return FakeEvaluationResult()

        summary = current_experiment.run_current_experiment(
            documents,
            eval_documents=eval_documents,
            corpus_label="injected-train",
            eval_corpus_label="injected-eval",
            training_config=GPTTrainingConfig(
                max_steps=3,
                context_length=16,
                batch_size=2,
                batch_stride=4,
            ),
            evaluation_runner=fake_runner,
        )

        self.assertEqual(captured_documents, documents)
        self.assertEqual(captured_eval_documents, eval_documents)
        self.assertEqual(summary["corpus"]["label"], "injected-train")
        self.assertEqual(summary["corpus"]["eval_label"], "injected-eval")
        self.assertEqual(summary["training_config"]["batch_stride"], 4)
        self.assertEqual(summary["training_loss"]["initial_loss"], 4.0)
        self.assertEqual(summary["training_loss"]["final_train_loss"], 2.75)
        self.assertEqual(summary["language_modeling"]["train_loss_delta"], 1.5)
        self.assertEqual(summary["language_modeling"]["eval_loss_delta"], 1.0)
        self.assertEqual(summary["next_observation"]["before"]["top1_accuracy"], 0.25)
        self.assertEqual(summary["next_observation"]["after"]["top1_accuracy"], 0.75)
        self.assertEqual(
            summary["next_observation"]["modality_counts"],
            {"environment_symbolic": 2},
        )

    def test_cli_loads_file_corpora_and_emits_json_summary(self) -> None:
        train_documents = _case_documents()
        eval_documents = [
            MixedDocument(
                id="eval_a",
                modality="environment_symbolic",
                content="<obs> eval a <action> act <next_obs> eval done",
            ),
            MixedDocument(
                id="eval_b",
                modality="environment_symbolic",
                content="<obs> eval b <action> act <next_obs> eval other",
            ),
        ]
        captured_documents: list[MixedDocument] | None = None
        captured_eval_documents: list[MixedDocument] | None = None

        def fake_runner(documents_arg, *, eval_documents=None, training_config, model_config=None):
            nonlocal captured_documents, captured_eval_documents
            captured_documents = documents_arg
            captured_eval_documents = eval_documents
            self.assertEqual(training_config.max_steps, 9)
            self.assertEqual(training_config.learning_rate, 0.01)
            self.assertIsNone(model_config)
            return FakeEvaluationResult()

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_path = temp_path / "train.jsonl"
            eval_path = temp_path / "eval.jsonl"
            summary_path = temp_path / "summary.json"
            write_mixed_documents_jsonl(train_path, train_documents)
            write_mixed_documents_jsonl(eval_path, eval_documents)

            with redirect_stdout(output):
                current_experiment.main(
                    [
                        "--corpus",
                        "file",
                        "--corpus-path",
                        str(train_path),
                        "--eval-corpus-path",
                        str(eval_path),
                        "--max-steps",
                        "9",
                        "--learning-rate",
                        "0.01",
                        "--output",
                        str(summary_path),
                    ],
                    evaluation_runner=fake_runner,
                )

            stdout_payload = json.loads(output.getvalue())
            file_payload = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(captured_documents, train_documents)
        self.assertEqual(captured_eval_documents, eval_documents)
        self.assertEqual(stdout_payload, file_payload)
        self.assertEqual(stdout_payload["corpus"]["label"], str(train_path))
        self.assertEqual(stdout_payload["corpus"]["eval_label"], str(eval_path))
        self.assertEqual(stdout_payload["training_loss"]["loss_history"], [4.0, 3.0, 2.5])
        self.assertEqual(stdout_payload["language_modeling"]["train_loss_delta"], 1.5)

    def test_natural_language_only_corpus_reports_language_modeling_and_skips_action_eval(self) -> None:
        train_documents = [
            MixedDocument(
                id="train_text",
                modality="external_text",
                content="alpha beta gamma alpha beta gamma alpha beta gamma",
            )
        ]
        eval_documents = [
            MixedDocument(
                id="eval_text",
                modality="external_text",
                content="delta epsilon zeta delta epsilon zeta delta epsilon zeta",
            )
        ]

        summary = current_experiment.run_current_experiment(
            train_documents,
            eval_documents=eval_documents,
            corpus_label="natural-train",
            eval_corpus_label="natural-eval",
            training_config=GPTTrainingConfig(
                context_length=8,
                batch_size=1,
                max_steps=1,
                seed=59,
            ),
            model_config=_small_model_config(context_length=8),
        )

        self.assertEqual(summary["corpus"]["label"], "natural-train")
        self.assertEqual(summary["next_observation"]["status"], "skipped")
        self.assertEqual(summary["next_observation"]["eval_case_count"], 0)
        self.assertIsNotNone(summary["language_modeling"]["initial_train_perplexity"])
        self.assertIsNotNone(summary["language_modeling"]["final_train_perplexity"])
        self.assertIsNotNone(summary["language_modeling"]["initial_eval_perplexity"])
        self.assertIsNotNone(summary["language_modeling"]["final_eval_perplexity"])

    def test_cli_builtin_grid_uses_existing_grid_corpus(self) -> None:
        captured_documents: list[MixedDocument] | None = None

        def fake_runner(documents_arg, *, eval_documents=None, training_config, model_config=None):
            nonlocal captured_documents
            captured_documents = documents_arg
            self.assertIsNone(eval_documents)
            return FakeEvaluationResult()

        output = io.StringIO()
        with redirect_stdout(output):
            current_experiment.main(["--corpus", "builtin-grid"], evaluation_runner=fake_runner)

        self.assertIsNotNone(captured_documents)
        assert captured_documents is not None
        self.assertIn("grid", {document.modality for document in captured_documents})
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["corpus"]["label"], "builtin-grid")
        self.assertEqual(payload["corpus"]["eval_label"], "train")

    def test_file_corpus_requires_path(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                current_experiment.main(["--corpus", "file"])

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("--corpus-path is required", error_output.getvalue())


def _case_documents() -> list[MixedDocument]:
    return [
        MixedDocument(
            id="case_a",
            modality="environment_symbolic",
            content="<obs> train a <action> act <next_obs> train done",
        ),
        MixedDocument(
            id="case_b",
            modality="environment_symbolic",
            content="<obs> train b <action> act <next_obs> train other",
        ),
    ]


def _small_model_config(*, context_length: int) -> GPTConfig:
    return GPTConfig(
        vocab_size=ByteTokenizer.vocab_size,
        context_length=context_length,
        embedding_dim=8,
        num_heads=2,
        hidden_dim=16,
    )


if __name__ == "__main__":
    unittest.main()
