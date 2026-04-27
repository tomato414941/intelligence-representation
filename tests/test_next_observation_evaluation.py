import unittest

from intrep.byte_tokenizer import ByteTokenizer
from intrep.gpt_model import DecoderOnlyGPT, GPTConfig
from intrep.gpt_training import GPTTrainingConfig
from intrep.mixed_corpus import MixedDocument
from intrep.next_observation_evaluation import (
    NextObservationEvaluationResult,
    evaluate_next_observation_learning,
)


class NextObservationEvaluationTest(unittest.TestCase):
    def test_evaluates_cases_before_and_after_training(self) -> None:
        documents = _documents()
        scored: list[tuple[str, str, str]] = []

        def score(
            model: DecoderOnlyGPT,
            _tokenizer: ByteTokenizer,
            prefix: str,
            continuation: str,
        ) -> float:
            scored.append((type(model).__name__, prefix, continuation))
            if "box" in prefix and "key visible" in continuation:
                return 0.1
            if "desk" in prefix and "cup on shelf" in continuation:
                return 0.2
            return 0.9

        result = evaluate_next_observation_learning(
            documents,
            distractor_policy="all_other",
            training_config=GPTTrainingConfig(
                context_length=16,
                batch_size=2,
                max_steps=1,
                learning_rate=0.005,
                seed=23,
            ),
            model_config=GPTConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=16,
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
            ),
            score_continuation_loss=score,
        )

        self.assertIsInstance(result, NextObservationEvaluationResult)
        self.assertEqual(result.case_count, 2)
        self.assertEqual([case.id for case in result.train_cases], ["case_key", "case_cup"])
        self.assertEqual([case.id for case in result.eval_cases], ["case_key", "case_cup"])
        self.assertEqual(result.training_result.steps, 1)
        self.assertEqual(result.before_metrics.top1_accuracy, 1.0)
        self.assertEqual(result.after_metrics.top1_accuracy, 1.0)
        self.assertEqual(result.before_summary.modality_counts, {"environment_symbolic": 2})
        self.assertEqual(result.after_summary.modality_counts, {"environment_symbolic": 2})
        self.assertEqual(len(scored), 8)
        self.assertEqual({model_name for model_name, _, _ in scored}, {"DecoderOnlyGPT"})

    def test_passes_distractor_policy_to_before_and_after_ranking(self) -> None:
        documents = [
            MixedDocument(
                id="case_key",
                modality="environment_symbolic",
                content="<obs> key in box <action> find key <next_obs> key at shelf",
            ),
            MixedDocument(
                id="case_grid",
                modality="grid",
                content="ignored grid case",
            ),
            MixedDocument(
                id="case_cup",
                modality="environment_symbolic",
                content="<obs> cup on desk <action> find cup <next_obs> cup at shelf",
            ),
        ]
        scored: list[tuple[str, str]] = []

        def score(
            _model: DecoderOnlyGPT,
            _tokenizer: ByteTokenizer,
            prefix: str,
            continuation: str,
        ) -> float:
            scored.append((prefix, continuation))
            return 0.1 if continuation in prefix else 0.9

        evaluate_next_observation_learning(
            documents,
            training_config=GPTTrainingConfig(
                context_length=24,
                batch_size=2,
                max_steps=1,
                seed=41,
            ),
            model_config=GPTConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=24,
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
            ),
            distractor_policy="hard",
            score_continuation_loss=score,
        )

        self.assertEqual(len(scored), 8)

    def test_can_evaluate_held_out_documents_separately_from_training_documents(self) -> None:
        scored: list[str] = []

        def score(
            _model: DecoderOnlyGPT,
            _tokenizer: ByteTokenizer,
            prefix: str,
            continuation: str,
        ) -> float:
            scored.append(prefix + continuation)
            if "eval" in prefix and "done" in continuation:
                return 0.1
            return 0.9

        result = evaluate_next_observation_learning(
            [
                MixedDocument(
                    id="train_a",
                    modality="environment_symbolic",
                    content="<obs> train a <action> act <next_obs> train done",
                ),
                MixedDocument(
                    id="train_b",
                    modality="environment_symbolic",
                    content="<obs> train b <action> act <next_obs> train other",
                ),
            ],
            eval_documents=[
                MixedDocument(
                    id="eval_a",
                    modality="environment_symbolic",
                    content="<obs> eval a <action> act <next_obs> done",
                ),
                MixedDocument(
                    id="eval_b",
                    modality="environment_symbolic",
                    content="<obs> eval b <action> act <next_obs> other",
                ),
            ],
            training_config=GPTTrainingConfig(
                context_length=16,
                batch_size=2,
                max_steps=1,
                seed=29,
            ),
            model_config=GPTConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=16,
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
            ),
            score_continuation_loss=score,
        )

        self.assertEqual([case.id for case in result.train_cases], ["train_a", "train_b"])
        self.assertEqual([case.id for case in result.eval_cases], ["eval_a", "eval_b"])
        self.assertEqual(result.case_count, 2)
        self.assertTrue(scored)
        self.assertTrue(all("eval" in item for item in scored))

    def test_runs_short_real_training_and_ranking_with_small_gpt(self) -> None:
        result = evaluate_next_observation_learning(
            _documents(),
            training_config=GPTTrainingConfig(
                context_length=24,
                batch_size=2,
                max_steps=2,
                learning_rate=0.005,
                seed=31,
            ),
            model_config=GPTConfig(
                vocab_size=ByteTokenizer.vocab_size,
                context_length=24,
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
            ),
        )

        self.assertEqual(result.case_count, 2)
        self.assertEqual(result.training_result.steps, 2)
        self.assertEqual(len(result.training_result.loss_history), 2)
        self.assertGreaterEqual(result.before_metrics.top1_accuracy, 0.0)
        self.assertLessEqual(result.before_metrics.top1_accuracy, 1.0)
        self.assertGreaterEqual(result.after_metrics.top1_accuracy, 0.0)
        self.assertLessEqual(result.after_metrics.top1_accuracy, 1.0)
        self.assertGreater(result.before_metrics.mean_positive_loss, 0.0)
        self.assertGreater(result.after_metrics.mean_positive_loss, 0.0)

    def test_requires_enough_extracted_cases_for_ranking(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            evaluate_next_observation_learning(
                [
                    MixedDocument(
                        id="single_case",
                        modality="environment_symbolic",
                        content="<obs> key in box <action> open box <next_obs> key visible",
                    )
                ],
                training_config=GPTTrainingConfig(
                    context_length=16,
                    batch_size=1,
                    max_steps=1,
                ),
                model_config=GPTConfig(
                    vocab_size=ByteTokenizer.vocab_size,
                    context_length=16,
                    embedding_dim=8,
                    num_heads=2,
                    hidden_dim=16,
                ),
            )

    def test_rejects_model_config_context_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_length must match"):
            evaluate_next_observation_learning(
                _documents(),
                training_config=GPTTrainingConfig(context_length=16, max_steps=1),
                model_config=GPTConfig(
                    vocab_size=ByteTokenizer.vocab_size,
                    context_length=8,
                    embedding_dim=8,
                    num_heads=2,
                    hidden_dim=16,
                ),
            )


def _documents() -> list[MixedDocument]:
    return [
        MixedDocument(
            id="case_key",
            modality="environment_symbolic",
            content="<obs> key in box ; box closed <action> open box <next_obs> key visible",
        ),
        MixedDocument(
            id="case_cup",
            modality="environment_symbolic",
            content="<obs> cup on desk ; shelf empty <action> move cup shelf <next_obs> cup on shelf",
        ),
    ]


if __name__ == "__main__":
    unittest.main()
