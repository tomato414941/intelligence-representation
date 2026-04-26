import unittest

from experiments.state_hub import StateHub


class StateHubTest(unittest.TestCase):
    def test_text_observation_updates_belief_state_and_provenance(self) -> None:
        hub = StateHub()

        observation = hub.observe(
            modality="text",
            payload={"subject": "田中", "predicate": "has", "object": "本"},
        )

        self.assertEqual(len(hub.observations), 1)
        self.assertEqual(len(hub.beliefs), 1)
        self.assertEqual(hub.beliefs[0].supporting_observations, [observation.id])
        self.assertEqual(hub.provenance[0].observation_id, observation.id)
        self.assertEqual(hub.provenance[0].target_type, "belief")

    def test_action_result_updates_world_state_and_provenance(self) -> None:
        hub = StateHub()
        original = hub.add_world_fact("has", "佐藤", "本", source_observation="seed")

        observation = hub.observe(
            modality="action_result",
            payload={
                "before": {"has": ["佐藤", "本"]},
                "after": {"located_at": ["本", "図書館"]},
            },
        )

        self.assertEqual(original.status, "inactive")
        self.assertEqual(original.invalidated_by, observation.id)
        self.assertEqual(len(hub.active_world_facts()), 1)
        self.assertEqual(hub.active_world_facts()[0].key(), ("本", "located_at", "図書館"))
        self.assertTrue(
            any(
                item.observation_id == observation.id
                and item.target_type == "fact"
                and item.relation == "invalidated"
                for item in hub.provenance
            )
        )
        self.assertTrue(
            any(
                item.observation_id == observation.id
                and item.target_type == "fact"
                and item.relation == "created"
                for item in hub.provenance
            )
        )

    def test_conflicting_text_observation_updates_conflict_state(self) -> None:
        hub = StateHub()

        hub.observe(modality="text", payload={"subject": "田中", "predicate": "has", "object": "本"})
        observation = hub.observe(
            modality="text",
            payload={"subject": "佐藤", "predicate": "has", "object": "本"},
        )

        self.assertEqual(len(hub.beliefs), 2)
        self.assertEqual(hub.beliefs[0].status, "active")
        self.assertEqual(hub.beliefs[1].status, "uncertain")
        self.assertEqual(hub.beliefs[0].counter_observations, [observation.id])
        self.assertEqual(len(hub.conflicts), 1)
        self.assertEqual(hub.conflicts[0].source_observation, observation.id)
        self.assertTrue(
            any(
                item.observation_id == observation.id
                and item.target_type == "conflict"
                and item.target_id == hub.conflicts[0].id
                for item in hub.provenance
            )
        )


if __name__ == "__main__":
    unittest.main()
