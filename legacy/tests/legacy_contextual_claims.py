import unittest

from experiments.contextual_claims import ContextualClaimState


class ContextualClaimStateTest(unittest.TestCase):
    def test_same_object_same_context_and_time_conflicts(self) -> None:
        state = ContextualClaimState()

        first = state.add_claim(
            subject="田中",
            predicate="has",
            object="本",
            source="obs_1",
            time="t1",
            context="world",
        )
        second = state.add_claim(
            subject="佐藤",
            predicate="has",
            object="本",
            source="obs_2",
            time="t1",
            context="world",
        )

        self.assertEqual(first.status, "active")
        self.assertEqual(second.status, "uncertain")
        self.assertEqual(second.conflicts_with, [first.id])
        self.assertEqual([claim.id for claim in state.active_claims()], [first.id])
        self.assertEqual([claim.id for claim in state.uncertain_claims()], [second.id])

    def test_different_time_does_not_conflict(self) -> None:
        state = ContextualClaimState()

        first = state.add_claim(
            subject="田中",
            predicate="has",
            object="本",
            source="obs_1",
            time="t1",
            context="world",
        )
        second = state.add_claim(
            subject="佐藤",
            predicate="has",
            object="本",
            source="obs_2",
            time="t2",
            context="world",
        )

        self.assertEqual(first.status, "active")
        self.assertEqual(second.status, "active")
        self.assertEqual(second.conflicts_with, [])

    def test_different_belief_owner_does_not_conflict(self) -> None:
        state = ContextualClaimState()

        first = state.add_claim(
            subject="田中",
            predicate="has",
            object="本",
            source="obs_1",
            time="t1",
            context="world",
            owner_of_belief="world",
        )
        second = state.add_claim(
            subject="佐藤",
            predicate="has",
            object="本",
            source="obs_2",
            time="t1",
            context="world",
            owner_of_belief="田中",
        )

        self.assertEqual(first.status, "active")
        self.assertEqual(second.status, "active")
        self.assertEqual(second.conflicts_with, [])


if __name__ == "__main__":
    unittest.main()
