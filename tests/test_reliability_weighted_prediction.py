import unittest

from experiments.predictor_interface import Action, Fact
from experiments.reliability_weighted_prediction import (
    ReliabilityWeightedPredictor,
    _build_context,
    evaluate_reliability_cases,
    smoke_cases,
)
from experiments.observation_assisted_prediction import MemoryInput


class ReliabilityWeightedPredictionTest(unittest.TestCase):
    def test_high_reliability_source_resolves_same_time_conflict(self) -> None:
        context = _build_context(
            [
                MemoryInput(content="located_at(鍵, 箱)", timestamp="t1"),
                MemoryInput(content="located_at(箱, 棚)", timestamp="t3"),
                MemoryInput(content="located_at(箱, 机)", timestamp="t3"),
            ]
        )
        predictor = ReliabilityWeightedPredictor(
            context,
            reliability_by_source={"fixture_1": 0.9, "fixture_2": 0.95, "fixture_3": 0.4},
        )

        prediction = predictor.predict(Action(type="find", actor="太郎", object="鍵", target="unknown"))

        self.assertEqual(prediction.state, "resolved_with_uncertainty")
        self.assertEqual(prediction.fact.key(), ("鍵", "located_at", "棚"))
        self.assertEqual(prediction.confidence, 0.95)
        self.assertEqual([fact.key() for fact in prediction.counter_candidates], [("鍵", "located_at", "机")])
        self.assertEqual(prediction.provenance_observation_ids, ["obs_1", "obs_2"])
        self.assertEqual(prediction.counterevidence_observation_ids, ["obs_3"])

    def test_similar_reliability_stays_conflict(self) -> None:
        context = _build_context(
            [
                MemoryInput(content="located_at(鍵, 箱)", timestamp="t1"),
                MemoryInput(content="located_at(箱, 棚)", timestamp="t3"),
                MemoryInput(content="located_at(箱, 机)", timestamp="t3"),
            ]
        )
        predictor = ReliabilityWeightedPredictor(
            context,
            reliability_by_source={"fixture_1": 0.9, "fixture_2": 0.75, "fixture_3": 0.7},
        )

        prediction = predictor.predict(Action(type="find", actor="太郎", object="鍵", target="unknown"))

        self.assertEqual(prediction.state, "conflict")
        self.assertIsNone(prediction.fact)
        self.assertEqual(
            {fact.key() for fact in prediction.candidates},
            {("鍵", "located_at", "棚"), ("鍵", "located_at", "机")},
        )
        self.assertEqual(prediction.conflict_observation_ids, ["obs_2", "obs_3"])

    def test_evaluation_reports_resolution_and_conflict_rates(self) -> None:
        summary = evaluate_reliability_cases(smoke_cases())

        self.assertEqual(summary.accuracy, 0.5)
        self.assertEqual(summary.resolved_with_uncertainty_rate, 0.5)
        self.assertEqual(summary.conflict_rate, 0.5)
        self.assertTrue(summary.results[0].correct)
        self.assertEqual(summary.results[1].prediction_state, "conflict")


if __name__ == "__main__":
    unittest.main()
