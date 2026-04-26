import unittest

from experiments.observation_assisted_prediction import (
    ContextFactPredictor,
    evaluate_observation_assisted_cases,
    smoke_cases,
)
from experiments.observation_memory import Observation
from experiments.predictor_interface import Action, Fact


class ObservationAssistedPredictionTest(unittest.TestCase):
    def test_context_fact_predictor_uses_context_for_find_action(self) -> None:
        predictor = ContextFactPredictor(
            [
                Observation(
                    id="obs_1",
                    content="located_at(鍵, 棚)",
                    modality="text",
                    source="fixture",
                    timestamp="t0",
                    tags=["鍵"],
                )
            ]
        )

        prediction = predictor.predict([], Action(type="find", actor="太郎", object="鍵", target="unknown"))

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.key(), ("鍵", "located_at", "棚"))

    def test_context_fact_predictor_falls_back_to_rule_based_predictor(self) -> None:
        predictor = ContextFactPredictor([])

        prediction = predictor.predict([], Action(type="place", actor="佐藤", object="本", target="図書館"))

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.key(), ("本", "located_at", "図書館"))

    def test_retrieved_memory_beats_no_memory_on_smoke_cases(self) -> None:
        summary = evaluate_observation_assisted_cases(smoke_cases(), context_limit=2)
        accuracies = {item.condition: item.accuracy for item in summary.condition_summaries}

        self.assertLess(accuracies["no_memory"], accuracies["retrieved_memory"])
        self.assertEqual(accuracies["retrieved_memory"], 1.0)

    def test_retrieved_memory_selects_relevant_observation(self) -> None:
        summary = evaluate_observation_assisted_cases(smoke_cases(), context_limit=2)

        result = next(
            item
            for item in summary.results
            if item.case_name == "find_key_location" and item.condition == "retrieved_memory"
        )

        self.assertTrue(result.correct)
        self.assertEqual(result.context_size, 1)
        self.assertEqual(result.retrieved_observation_ids, ["obs_1"])

    def test_recent_memory_can_miss_relevant_older_observation(self) -> None:
        summary = evaluate_observation_assisted_cases(smoke_cases(), context_limit=2)

        result = next(
            item
            for item in summary.results
            if item.case_name == "find_key_location" and item.condition == "recent_memory"
        )

        self.assertFalse(result.correct)
        self.assertEqual(result.retrieved_observation_ids, ["obs_2", "obs_3"])


if __name__ == "__main__":
    unittest.main()
