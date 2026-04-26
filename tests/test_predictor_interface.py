import unittest

from experiments.predictor_interface import Action, Fact, PredictiveWorld, RuleBasedPredictor


class PredictorInterfaceTest(unittest.TestCase):
    def test_rule_based_predictor_predicts_place_action(self) -> None:
        predictor = RuleBasedPredictor()

        prediction = predictor.predict([], Action(type="place", actor="佐藤", object="本", target="図書館"))

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.key(), ("本", "located_at", "図書館"))

    def test_predictive_world_uses_injected_predictor(self) -> None:
        class ConstantPredictor:
            def predict(self, state: list[Fact], action: Action) -> Fact | None:
                return Fact(subject="x", predicate="predicted", object="y")

        world = PredictiveWorld(ConstantPredictor())

        prediction = world.predict(Action(type="anything", actor="a", object="b", target="c"))

        self.assertIsNotNone(prediction.expected)
        self.assertEqual(prediction.expected.key(), ("x", "predicted", "y"))

    def test_unsupported_action_returns_unsupported_prediction(self) -> None:
        world = PredictiveWorld(RuleBasedPredictor())

        prediction = world.predict(Action(type="throw", actor="佐藤", object="本", target="床"))
        error = world.compare(prediction, None)

        self.assertIsNone(prediction.expected)
        self.assertEqual(prediction.status, "unsupported")
        self.assertEqual(error.type, "unsupported")

    def test_prediction_mismatch_records_error(self) -> None:
        world = PredictiveWorld(RuleBasedPredictor())

        prediction = world.predict(Action(type="place", actor="佐藤", object="本", target="図書館"))
        observation = world.observe(Fact(subject="本", predicate="located_at", object="机"))
        error = world.compare(prediction, observation)

        self.assertEqual(prediction.status, "mismatch")
        self.assertEqual(error.type, "mismatch")
        self.assertEqual(error.expected.key(), ("本", "located_at", "図書館"))
        self.assertEqual(error.observed.key(), ("本", "located_at", "机"))


if __name__ == "__main__":
    unittest.main()
