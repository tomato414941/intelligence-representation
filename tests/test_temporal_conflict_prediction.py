import unittest

from experiments.observation_memory import Observation
from experiments.predictor_interface import Action
from experiments.temporal_conflict_prediction import (
    TemporalConflictPredictor,
    evaluate_temporal_conflict_cases,
    smoke_cases,
)


class TemporalConflictPredictionTest(unittest.TestCase):
    def test_predictor_reports_same_timestamp_conflict(self) -> None:
        predictor = TemporalConflictPredictor(
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
                    timestamp="t3",
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

        self.assertEqual(prediction.state, "conflict")
        self.assertIsNone(prediction.fact)
        self.assertEqual(
            {candidate.key() for candidate in prediction.candidates},
            {("鍵", "located_at", "棚"), ("鍵", "located_at", "机")},
        )
        self.assertEqual(prediction.provenance_observation_ids, ["obs_1"])
        self.assertEqual(prediction.conflict_observation_ids, ["obs_2", "obs_3"])

    def test_predictor_resolves_different_timestamp_observations_by_recency(self) -> None:
        predictor = TemporalConflictPredictor(
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

        self.assertEqual(prediction.state, "resolved")
        self.assertEqual(prediction.fact.key(), ("鍵", "located_at", "机"))
        self.assertEqual(prediction.superseded_observation_ids, ["obs_2"])
        self.assertEqual(prediction.conflict_observation_ids, [])

    def test_evaluation_accepts_conflict_candidate_set(self) -> None:
        summary = evaluate_temporal_conflict_cases(smoke_cases())

        self.assertEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.conflict_rate, 1.0)
        self.assertEqual(summary.results[0].prediction_state, "conflict")
        self.assertTrue(summary.results[0].correct)


if __name__ == "__main__":
    unittest.main()
