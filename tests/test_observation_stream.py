import unittest

from experiments.observation_stream import ObservationStreamMemory


class ObservationStreamMemoryTest(unittest.TestCase):
    def test_text_observation_creates_claim_and_belief(self) -> None:
        memory = ObservationStreamMemory()

        observation = memory.ingest(
            modality="text",
            payload={"subject": "田中", "predicate": "has", "object": "本"},
        )

        self.assertEqual(observation.modality, "text")
        self.assertEqual(len(memory.observations), 1)
        self.assertEqual(len(memory.claims), 1)
        self.assertEqual(len(memory.beliefs), 1)
        self.assertEqual(memory.claims[0].observation_id, observation.id)
        self.assertEqual(memory.beliefs[0].supporting_claims, [memory.claims[0].id])

    def test_action_result_creates_event_and_state_transition(self) -> None:
        memory = ObservationStreamMemory()

        observation = memory.ingest(
            modality="action_result",
            payload={
                "event_type": "place",
                "actor": "佐藤",
                "object": "本",
                "location": "図書館",
                "before": {"has": ["佐藤", "本"]},
                "after": {"located_at": ["本", "図書館"]},
                "time": "t2",
            },
        )

        self.assertEqual(observation.modality, "action_result")
        self.assertEqual(len(memory.observations), 1)
        self.assertEqual(len(memory.events), 1)
        self.assertEqual(len(memory.state_transitions), 1)
        self.assertEqual(memory.events[0].observation_id, observation.id)
        self.assertEqual(memory.events[0].type, "place")
        self.assertEqual(memory.events[0].participants["location"], "図書館")
        self.assertEqual(memory.state_transitions[0].cause, memory.events[0].id)
        self.assertEqual(memory.state_transitions[0].after, {"located_at": ["本", "図書館"]})

    def test_unsupported_modality_only_stores_observation(self) -> None:
        memory = ObservationStreamMemory()

        memory.ingest(modality="sensor", payload={"temperature": 22.0})

        self.assertEqual(len(memory.observations), 1)
        self.assertEqual(len(memory.claims), 0)
        self.assertEqual(len(memory.events), 0)
        self.assertEqual(len(memory.state_transitions), 0)


if __name__ == "__main__":
    unittest.main()
