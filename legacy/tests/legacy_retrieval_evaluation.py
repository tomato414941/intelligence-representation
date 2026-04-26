import unittest

from experiments.retrieval_evaluation import (
    EvaluationCase,
    EvaluationObservation,
    NgramRetrieverAdapter,
    evaluate_case,
    evaluate_cases,
    smoke_cases,
)


class RetrievalEvaluationTest(unittest.TestCase):
    def test_evaluate_case_computes_precision_and_recall(self) -> None:
        case = EvaluationCase(
            name="simple",
            observations=[
                EvaluationObservation(id="obs_1", content="alpha beta", tags=[]),
                EvaluationObservation(id="obs_2", content="beta gamma", tags=[]),
            ],
            query="beta",
            expected_relevant_ids=["obs_1"],
            k=2,
        )

        result = evaluate_case(case, NgramRetrieverAdapter())

        self.assertEqual(result.retrieved_ids, ["obs_1", "obs_2"])
        self.assertEqual(result.precision_at_k, 0.5)
        self.assertEqual(result.recall_at_k, 1.0)

    def test_empty_expected_relevance_has_recall_one(self) -> None:
        case = EvaluationCase(
            name="no_expected",
            observations=[EvaluationObservation(id="obs_1", content="alpha", tags=[])],
            query="alpha",
            expected_relevant_ids=[],
            k=1,
        )

        result = evaluate_case(case, NgramRetrieverAdapter())

        self.assertEqual(result.recall_at_k, 1.0)

    def test_ngram_adapter_retrieves_japanese_observation(self) -> None:
        case = EvaluationCase(
            name="japanese_ngram",
            observations=[
                EvaluationObservation(
                    id="obs_1",
                    content="Transformerは系列モデルというより関係計算器に近い",
                    tags=[],
                ),
                EvaluationObservation(
                    id="obs_2",
                    content="State Memoryは真理DBではなくキャッシュとして扱う",
                    tags=[],
                ),
            ],
            query="関係計算",
            expected_relevant_ids=["obs_1"],
            k=1,
        )

        result = evaluate_case(case, NgramRetrieverAdapter())

        self.assertEqual(result.retrieved_ids, ["obs_1"])
        self.assertEqual(result.precision_at_k, 1.0)
        self.assertEqual(result.recall_at_k, 1.0)

    def test_evaluate_cases_computes_mean_scores(self) -> None:
        summary = evaluate_cases(smoke_cases(), NgramRetrieverAdapter())

        self.assertEqual(len(summary.results), 3)
        self.assertGreater(summary.mean_precision_at_k, 0.0)
        self.assertGreater(summary.mean_recall_at_k, 0.0)


if __name__ == "__main__":
    unittest.main()
