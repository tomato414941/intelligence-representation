import unittest

from experiments.predictive_loop import Action, Fact, PredictiveWorld


class PredictiveLoopTest(unittest.TestCase):
    def test_matching_observation_confirms_prediction(self) -> None:
        world = PredictiveWorld()
        world.add_fact(Fact(subject="佐藤", predicate="has", object="本"))

        prediction = world.predict(Action(type="place", actor="佐藤", object="本", target="図書館"))
        observation = world.observe(Fact(subject="本", predicate="located_at", object="図書館"))
        error = world.compare(prediction, observation)

        self.assertEqual(prediction.expected.key(), ("本", "located_at", "図書館"))
        self.assertEqual(prediction.status, "confirmed")
        self.assertEqual(error.type, "none")
        self.assertIn(("本", "located_at", "図書館"), {fact.key() for fact in world.state})

    def test_mismatching_observation_records_prediction_error(self) -> None:
        world = PredictiveWorld()

        prediction = world.predict(Action(type="place", actor="佐藤", object="本", target="図書館"))
        observation = world.observe(Fact(subject="本", predicate="located_at", object="机"))
        error = world.compare(prediction, observation)

        self.assertEqual(prediction.status, "mismatch")
        self.assertEqual(error.type, "mismatch")
        self.assertEqual(error.expected.key(), ("本", "located_at", "図書館"))
        self.assertEqual(error.observed.key(), ("本", "located_at", "机"))

    def test_update_log_records_prediction_flow(self) -> None:
        world = PredictiveWorld()

        prediction = world.predict(Action(type="place", actor="佐藤", object="本", target="図書館"))
        observation = world.observe(Fact(subject="本", predicate="located_at", object="図書館"))
        world.compare(prediction, observation)

        self.assertEqual(
            [entry.type for entry in world.update_log],
            ["predict", "add_fact", "prediction_error"],
        )


if __name__ == "__main__":
    unittest.main()
