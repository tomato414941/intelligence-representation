import unittest

from intrep.dataset import ActionConditionedExample
from intrep.predictors import StateAwarePredictor
from intrep.transition_data import generate_examples, held_out_object_examples, split_examples
from intrep.types import Action, Fact


class StateAwarePredictorTest(unittest.TestCase):
    def test_predicts_held_out_object_from_state_relations(self) -> None:
        train, _ = split_examples(generate_examples())
        predictor = StateAwarePredictor()
        predictor.fit(train)
        case = held_out_object_examples()[0]

        prediction = predictor.predict(case.state_before, case.action)

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.key(), case.expected_observation.key())

    def test_predicts_all_held_out_object_cases(self) -> None:
        train, _ = split_examples(generate_examples())
        predictor = StateAwarePredictor()
        predictor.fit(train)

        predictions = [
            predictor.predict(case.state_before, case.action)
            for case in held_out_object_examples()
        ]

        self.assertTrue(all(prediction is not None for prediction in predictions))

    def test_predicts_direct_location_from_state(self) -> None:
        predictor = StateAwarePredictor()
        action = Action(type="find", actor="太郎", object="財布", target="unknown")

        prediction = predictor.predict(
            [Fact(subject="財布", predicate="located_at", object="机")],
            action,
        )

        self.assertEqual(
            prediction,
            Fact(subject="財布", predicate="located_at", object="机"),
        )

    def test_find_state_location_overrides_frequency_prediction(self) -> None:
        action = Action(type="find", actor="太郎", object="財布", target="unknown")
        predictor = StateAwarePredictor()
        predictor.fit(
            [
                ActionConditionedExample(
                    id="wallet_seen_elsewhere",
                    state_before=[Fact(subject="財布", predicate="located_at", object="棚")],
                    action=action,
                    expected_observation=Fact(subject="財布", predicate="located_at", object="棚"),
                )
            ]
        )

        prediction = predictor.predict(
            [Fact(subject="財布", predicate="located_at", object="机")],
            action,
        )

        self.assertEqual(
            prediction,
            Fact(subject="財布", predicate="located_at", object="机"),
        )


if __name__ == "__main__":
    unittest.main()
