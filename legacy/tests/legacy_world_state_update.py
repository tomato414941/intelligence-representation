import unittest

from experiments.world_state_update import WorldState


class WorldStateTest(unittest.TestCase):
    def test_state_transition_deactivates_before_and_adds_after(self) -> None:
        world = WorldState()
        original = world.add_fact("has", "佐藤", "本")

        transition = world.apply_transition(
            before={"has": ["佐藤", "本"]},
            after={"located_at": ["本", "図書館"]},
            source="obs_1",
        )

        self.assertEqual(original.status, "inactive")
        self.assertEqual(original.invalidated_by, transition.id)
        self.assertEqual(len(world.active_facts()), 1)
        self.assertEqual(world.active_facts()[0].key(), ("本", "located_at", "図書館"))
        self.assertEqual(world.active_facts()[0].source, transition.id)

    def test_missing_before_fact_does_not_block_after_fact(self) -> None:
        world = WorldState()

        transition = world.apply_transition(
            before={"has": ["佐藤", "本"]},
            after={"located_at": ["本", "図書館"]},
            source="obs_1",
        )

        self.assertEqual(len(world.inactive_facts()), 0)
        self.assertEqual(len(world.active_facts()), 1)
        self.assertEqual(world.transitions[0], transition)


if __name__ == "__main__":
    unittest.main()
