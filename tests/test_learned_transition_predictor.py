import unittest

from intrep.environment import MiniTransitionEnvironment
from intrep.predictors import FrequencyTransitionPredictor, RuleBasedPredictor
from intrep.transition_data import (
    compare_predictors,
    generate_examples,
    split_examples,
    smoke_comparison,
)
from intrep.types import Action, Fact


class LearnedTransitionPredictorTest(unittest.TestCase):
    def test_environment_generates_container_chain_observation(self) -> None:
        environment = MiniTransitionEnvironment({"鍵": "箱", "箱": "棚"})

        observation = environment.apply(Action(type="find", actor="太郎", object="鍵", target="unknown"))

        self.assertEqual(observation.key(), ("鍵", "located_at", "棚"))

    def test_generate_examples_creates_action_conditioned_data(self) -> None:
        examples = generate_examples()

        self.assertGreater(len(examples), 5)
        self.assertEqual(examples[0].expected_observation.key(), ("鍵", "located_at", "棚"))
        self.assertEqual(examples[1].action.type, "move_container")
        self.assertEqual(examples[1].expected_observation.key(), ("箱", "located_at", "机"))

    def test_frequency_predictor_learns_from_training_examples(self) -> None:
        train, test = split_examples(generate_examples())
        predictor = FrequencyTransitionPredictor()
        predictor.fit(train)

        find_key_case = next(example for example in test if example.action.type == "find" and example.action.object == "鍵")
        prediction = predictor.predict(find_key_case.state_before, find_key_case.action)

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.key(), ("鍵", "located_at", "棚"))

    def test_frequency_predictor_returns_unsupported_for_unknown_pattern(self) -> None:
        predictor = FrequencyTransitionPredictor()
        predictor.fit([])

        prediction = predictor.predict([], Action(type="find", actor="太郎", object="財布", target="unknown"))

        self.assertIsNone(prediction)

    def test_learned_predictor_beats_rule_based_predictor_on_generated_test_set(self) -> None:
        comparison = smoke_comparison()

        self.assertEqual(comparison.train_size, 6)
        self.assertEqual(comparison.test_size, 6)
        self.assertLess(comparison.rule_summary.accuracy, comparison.learned_summary.accuracy)
        self.assertEqual(comparison.learned_summary.accuracy, 1.0)
        self.assertGreater(comparison.rule_summary.unsupported_rate, comparison.learned_summary.unsupported_rate)

    def test_comparison_uses_existing_prediction_evaluation(self) -> None:
        train, test = split_examples(generate_examples())

        comparison = compare_predictors(train, test)

        self.assertEqual(len(comparison.rule_summary.results), len(test))
        self.assertEqual(len(comparison.learned_summary.results), len(test))
        self.assertIsInstance(RuleBasedPredictor().predict([], test[0].action), (Fact, type(None)))


if __name__ == "__main__":
    unittest.main()
