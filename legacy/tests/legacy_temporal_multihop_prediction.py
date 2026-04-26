import unittest

from experiments.observation_memory import Observation
from experiments.predictor_interface import Action
from experiments.temporal_multihop_prediction import (
    TemporalMultiHopPredictor,
    evaluate_temporal_multihop_cases,
    smoke_cases,
)


class TemporalMultiHopPredictionTest(unittest.TestCase):
    def test_predictor_uses_latest_observation_for_same_subject_predicate(self) -> None:
        predictor = TemporalMultiHopPredictor(
            [
                Observation(
                    id="obs_1",
                    content="located_at(鍵, 箱)",
                    modality="text",
                    source="fixture",
                    timestamp="t1",
                ),
                Observation(
                    id="obs_2",
                    content="located_at(箱, 棚)",
                    modality="text",
                    source="fixture",
                    timestamp="t2",
                ),
                Observation(
                    id="obs_3",
                    content="located_at(箱, 机)",
                    modality="text",
                    source="fixture",
                    timestamp="t3",
                ),
            ]
        )

        prediction = predictor.predict(Action(type="find", actor="太郎", object="鍵", target="unknown"))

        self.assertIsNotNone(prediction.fact)
        self.assertEqual(prediction.fact.key(), ("鍵", "located_at", "机"))
        self.assertEqual(prediction.provenance_observation_ids, ["obs_1", "obs_3"])
        self.assertEqual(prediction.superseded_observation_ids, ["obs_2"])

    def test_predictor_keeps_old_observation_as_superseded(self) -> None:
        summary = evaluate_temporal_multihop_cases(smoke_cases())
        result = summary.results[0]

        self.assertTrue(result.correct)
        self.assertEqual(result.provenance_observation_ids, ["obs_1", "obs_3"])
        self.assertEqual(result.superseded_observation_ids, ["obs_2"])

    def test_summary_reports_accuracy(self) -> None:
        summary = evaluate_temporal_multihop_cases(smoke_cases())

        self.assertEqual(summary.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
