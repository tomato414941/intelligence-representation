import unittest

from intrep.gpt_training import GPTTrainingConfig
from intrep.mixed_corpus import default_mixed_documents
from intrep.mixed_corpus_evaluation import build_train_eval_document_split
from intrep.symbolic_to_natural_evaluation import evaluate_symbolic_to_natural_learning


class SymbolicToNaturalEvaluationTest(unittest.TestCase):
    def test_evaluates_symbolic_to_natural_pairs_before_and_after_training(self) -> None:
        documents = default_mixed_documents()

        result = evaluate_symbolic_to_natural_learning(
            documents,
            training_config=GPTTrainingConfig(max_steps=2, batch_size=2, seed=3),
        )

        self.assertEqual(result.pair_count, 5)
        self.assertEqual(len(result.train_pairs), 5)
        self.assertEqual(result.training_result.steps, 2)
        self.assertGreaterEqual(result.before_metrics.top1_accuracy, 0.0)
        self.assertLessEqual(result.before_metrics.top1_accuracy, 1.0)
        self.assertGreaterEqual(result.after_metrics.top1_accuracy, 0.0)
        self.assertLessEqual(result.after_metrics.top1_accuracy, 1.0)

    def test_supports_held_out_environment_pair_documents(self) -> None:
        split = build_train_eval_document_split(default_mixed_documents(), eval_episode_count=2)

        result = evaluate_symbolic_to_natural_learning(
            split.train_documents,
            eval_documents=split.eval_documents,
            training_config=GPTTrainingConfig(max_steps=2, batch_size=2, seed=3),
        )

        self.assertEqual(result.pair_count, 2)
        self.assertEqual([pair.episode_id for pair in result.eval_pairs], ["001", "002"])
        self.assertNotIn("001", [pair.episode_id for pair in result.train_pairs])

    def test_requires_two_eval_pairs_for_distractors(self) -> None:
        split = build_train_eval_document_split(default_mixed_documents(), eval_episode_count=1)

        with self.assertRaisesRegex(ValueError, "at least two"):
            evaluate_symbolic_to_natural_learning(
                split.train_documents,
                eval_documents=split.eval_documents,
                training_config=GPTTrainingConfig(max_steps=1),
            )


if __name__ == "__main__":
    unittest.main()
