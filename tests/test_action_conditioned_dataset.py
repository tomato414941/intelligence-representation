import unittest

from experiments.action_conditioned_dataset import (
    ActionConditionedExample,
    dumps_jsonl,
    example_from_dict,
    example_to_dict,
    loads_jsonl,
    smoke_examples,
)
from experiments.predictor_evaluation import evaluate_prediction_cases
from experiments.predictor_interface import Action, Fact, RuleBasedPredictor


class ActionConditionedDatasetTest(unittest.TestCase):
    def test_example_round_trips_through_dict(self) -> None:
        example = ActionConditionedExample(
            id="place",
            state_before=[Fact(subject="佐藤", predicate="has", object="本")],
            action=Action(type="place", actor="佐藤", object="本", target="図書館"),
            expected_observation=Fact(subject="本", predicate="located_at", object="図書館"),
            expected_state_after=[Fact(subject="本", predicate="located_at", object="図書館")],
        )

        loaded = example_from_dict(example_to_dict(example))

        self.assertEqual(loaded.id, "place")
        self.assertEqual(loaded.state_before[0].key(), ("佐藤", "has", "本"))
        self.assertEqual(loaded.action.type, "place")
        self.assertEqual(loaded.expected_observation.key(), ("本", "located_at", "図書館"))
        self.assertEqual(loaded.expected_state_after[0].key(), ("本", "located_at", "図書館"))

    def test_examples_round_trip_through_jsonl(self) -> None:
        content = dumps_jsonl(smoke_examples())

        loaded = loads_jsonl(content)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].id, "place_book_library")
        self.assertIsNone(loaded[1].expected_observation)

    def test_example_converts_to_prediction_case(self) -> None:
        example = smoke_examples()[0]

        case = example.to_prediction_case()

        self.assertEqual(case.name, "place_book_library")
        self.assertEqual(case.expected_fact.key(), ("本", "located_at", "図書館"))

    def test_dataset_can_feed_predictor_evaluation(self) -> None:
        cases = [example.to_prediction_case() for example in smoke_examples()]

        summary = evaluate_prediction_cases(cases, RuleBasedPredictor())

        self.assertEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.unsupported_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
