import unittest

from experiments.contextual_state_update import ContextualState


class ContextualStateTest(unittest.TestCase):
    def test_transfer_deactivates_previous_owner_and_adds_new_owner(self) -> None:
        state = ContextualState()

        state.observe(
            {
                "type": "claim",
                "subject": "田中",
                "predicate": "has",
                "object": "本",
                "time": "t1",
                "context": "world",
            }
        )
        state.observe(
            {
                "type": "transfer",
                "actor": "田中",
                "recipient": "佐藤",
                "object": "本",
                "time": "t2",
                "context": "world",
            }
        )

        claims = state.all_claims()
        self.assertEqual(claims[0].status, "inactive")
        self.assertEqual(claims[0].invalidated_by, "obs_2")
        self.assertEqual(claims[1].subject, "佐藤")
        self.assertEqual(claims[1].predicate, "has")
        self.assertEqual(claims[1].object, "本")
        self.assertEqual(claims[1].time, "t2")
        self.assertEqual(claims[1].status, "active")

    def test_place_deactivates_owner_and_adds_location(self) -> None:
        state = ContextualState()

        state.observe(
            {
                "type": "claim",
                "subject": "佐藤",
                "predicate": "has",
                "object": "本",
                "time": "t1",
                "context": "world",
            }
        )
        state.observe(
            {
                "type": "place",
                "actor": "佐藤",
                "object": "本",
                "location": "図書館",
                "time": "t2",
                "context": "world",
            }
        )

        claims = state.all_claims()
        self.assertEqual(claims[0].status, "inactive")
        self.assertEqual(claims[1].subject, "本")
        self.assertEqual(claims[1].predicate, "located_at")
        self.assertEqual(claims[1].object, "図書館")
        self.assertEqual(claims[1].status, "active")

    def test_same_time_conflict_becomes_uncertain(self) -> None:
        state = ContextualState()

        state.observe(
            {
                "type": "claim",
                "subject": "田中",
                "predicate": "has",
                "object": "本",
                "time": "t1",
                "context": "world",
            }
        )
        state.observe(
            {
                "type": "claim",
                "subject": "佐藤",
                "predicate": "has",
                "object": "本",
                "time": "t1",
                "context": "world",
            }
        )

        claims = state.all_claims()
        self.assertEqual(claims[0].status, "active")
        self.assertEqual(claims[1].status, "uncertain")
        self.assertEqual(claims[1].conflicts_with, [claims[0].id])


if __name__ == "__main__":
    unittest.main()
