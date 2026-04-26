import unittest

from intrep.dataset import ActionConditionedExample
from intrep.environment import MiniTransitionEnvironment
from intrep.evaluation import evaluate_prediction_cases
from intrep.predictors import FrequencyTransitionPredictor, RuleBasedPredictor
from intrep.transition_data import generate_examples, split_examples
from intrep.types import Action, Fact
from intrep.update_loop import PredictionErrorUpdateLoop


class IntrepImportsTest(unittest.TestCase):
    def test_prototype_surface_imports(self) -> None:
        train, test = split_examples(generate_examples())
        predictor = FrequencyTransitionPredictor()
        predictor.fit(train)

        self.assertIsNotNone(ActionConditionedExample)
        self.assertIsNotNone(MiniTransitionEnvironment)
        self.assertIsNotNone(evaluate_prediction_cases)
        self.assertIsNotNone(RuleBasedPredictor)
        self.assertIsNotNone(PredictionErrorUpdateLoop)
        self.assertIsInstance(test[0].action, Action)
        self.assertIsInstance(test[0].state_before[0], Fact)
        self.assertIsNotNone(predictor.predict(test[0].state_before, test[0].action))


if __name__ == "__main__":
    unittest.main()
