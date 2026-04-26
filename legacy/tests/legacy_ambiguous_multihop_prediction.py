import unittest

from experiments.ambiguous_multihop_prediction import (
    AmbiguousMultiHopPredictor,
    evaluate_ambiguous_multihop_cases,
    smoke_cases,
)
from experiments.observation_memory import Observation
from experiments.predictor_interface import Action


class AmbiguousMultiHopPredictionTest(unittest.TestCase):
    def test_predictor_returns_ambiguous_candidates(self) -> None:
        predictor = AmbiguousMultiHopPredictor(
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
                Observation(
                    id="obs_3",
                    content="located_at(箱, 机)",
                    modality="text",
                    source="fixture",
                    timestamp="t0",
                ),
            ]
        )

        prediction = predictor.predict(Action(type="find", actor="太郎", object="鍵", target="unknown"))

        self.assertEqual(prediction.state, "ambiguous")
        self.assertEqual(
            {candidate.key() for candidate in prediction.candidates},
            {("鍵", "located_at", "棚"), ("鍵", "located_at", "机")},
        )

    def test_predictor_returns_resolved_for_single_candidate(self) -> None:
        predictor = AmbiguousMultiHopPredictor(
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

        prediction = predictor.predict(Action(type="find", actor="太郎", object="鍵", target="unknown"))

        self.assertEqual(prediction.state, "resolved")
        self.assertEqual(prediction.candidates[0].key(), ("鍵", "located_at", "棚"))

    def test_evaluation_accepts_candidate_set_without_collapsing_to_one_answer(self) -> None:
        summary = evaluate_ambiguous_multihop_cases(smoke_cases())

        self.assertEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.ambiguous_rate, 1.0)
        self.assertEqual(summary.results[0].prediction_state, "ambiguous")
        self.assertTrue(summary.results[0].correct)


if __name__ == "__main__":
    unittest.main()
