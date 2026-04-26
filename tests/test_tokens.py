import unittest

from intrep.tokens import action_token, fact_from_token, fact_token, model_input_tokens, target_token
from intrep.types import Action, Fact


class TokensTest(unittest.TestCase):
    def test_fact_and_action_tokens_are_stable(self) -> None:
        fact = Fact(subject="財布", predicate="located_at", object="ケース")
        action = Action(type="find", actor="太郎", object="財布", target="unknown")

        self.assertEqual(fact_token(fact), "FACT:財布:located_at:ケース")
        self.assertEqual(action_token(action), "ACTION:find:財布:unknown")

    def test_model_input_tokens_sort_state_and_add_predict_marker(self) -> None:
        state = [
            Fact(subject="財布", predicate="located_at", object="ケース"),
            Fact(subject="ケース", predicate="located_at", object="引き出し"),
        ]
        action = Action(type="find", actor="太郎", object="財布", target="unknown")

        tokens = model_input_tokens(state, action)

        self.assertEqual(tokens[-2:], ["ACTION:find:財布:unknown", "PREDICT"])
        self.assertEqual(tokens[0], "FACT:ケース:located_at:引き出し")

    def test_target_token_round_trips_fact_and_unsupported(self) -> None:
        fact = Fact(subject="財布", predicate="located_at", object="引き出し")

        self.assertEqual(fact_from_token(target_token(fact)), fact)
        self.assertIsNone(fact_from_token(target_token(None)))


if __name__ == "__main__":
    unittest.main()
