import unittest

from experiments.multihop_observation_prediction import (
    MultiHopContextPredictor,
    evaluate_multihop_cases,
    smoke_cases,
)
from experiments.observation_memory import Observation
from experiments.predictor_interface import Action


class MultiHopObservationPredictionTest(unittest.TestCase):
    def test_multihop_predictor_follows_location_chain(self) -> None:
        predictor = MultiHopContextPredictor(
            [
                Observation(
                    id="obs_1",
                    content="located_at(鍵, 箱)",
                    modality="text",
                    source="fixture",
                    timestamp="t0",
                ),
                Observation(
                    id="obs_2",
                    content="located_at(箱, 棚)",
                    modality="text",
                    source="fixture",
                    timestamp="t0",
                ),
            ]
        )

        prediction = predictor.predict([], Action(type="find", actor="太郎", object="鍵", target="unknown"))

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.key(), ("鍵", "located_at", "棚"))

    def test_direct_memory_is_not_enough_for_multihop_case(self) -> None:
        summary = evaluate_multihop_cases(smoke_cases(), context_limit=2)

        result = next(item for item in summary.results if item.condition == "direct_memory")

        self.assertFalse(result.correct)
        self.assertEqual(result.predicted_fact.key(), ("鍵", "located_at", "箱"))
        self.assertEqual(result.retrieved_observation_ids, ["obs_1"])

    def test_multihop_memory_solves_multihop_case(self) -> None:
        summary = evaluate_multihop_cases(smoke_cases(), context_limit=2)

        result = next(item for item in summary.results if item.condition == "multi_hop_memory")

        self.assertTrue(result.correct)
        self.assertEqual(result.predicted_fact.key(), ("鍵", "located_at", "棚"))
        self.assertEqual(result.retrieved_observation_ids, ["obs_1", "obs_3"])

    def test_summary_shows_multihop_improvement(self) -> None:
        summary = evaluate_multihop_cases(smoke_cases(), context_limit=2)
        accuracies = {item.condition: item.accuracy for item in summary.condition_summaries}

        self.assertEqual(accuracies["no_memory"], 0.0)
        self.assertEqual(accuracies["direct_memory"], 0.0)
        self.assertEqual(accuracies["multi_hop_memory"], 1.0)


if __name__ == "__main__":
    unittest.main()
