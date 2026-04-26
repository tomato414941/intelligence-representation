import unittest

from experiments.semantic_state import SemanticState


class SemanticStateTest(unittest.TestCase):
    def test_transfer_and_place_update_current_state(self) -> None:
        state = SemanticState()

        state.observe({"type": "claim", "subject": "田中", "predicate": "has", "object": "本"})
        state.observe({"type": "transfer", "actor": "田中", "recipient": "佐藤", "object": "本"})
        state.observe({"type": "place", "actor": "佐藤", "object": "本", "location": "図書館"})

        rendered = [claim.render() for claim in state.all_claims()]
        self.assertEqual(
            rendered,
            [
                "has(田中, 本): inactive",
                "has(佐藤, 本): inactive",
                "located_at(本, 図書館): active",
            ],
        )
        self.assertEqual(state.answer_location("本"), "本は図書館にある可能性が高い")


if __name__ == "__main__":
    unittest.main()
