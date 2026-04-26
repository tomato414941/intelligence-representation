import unittest

from experiments.observation_belief_conflict import SemanticMemory


class SemanticMemoryTest(unittest.TestCase):
    def test_observation_is_saved_when_belief_is_added(self) -> None:
        memory = SemanticMemory()

        belief = memory.process_claim({"subject": "田中", "predicate": "has", "object": "本"})

        self.assertEqual(len(memory.observations), 1)
        self.assertEqual(belief.evidence, ["obs_1"])
        self.assertEqual(memory.update_log[0].type, "add")

    def test_duplicate_claim_merges_evidence(self) -> None:
        memory = SemanticMemory()

        first = memory.process_claim({"subject": "田中", "predicate": "has", "object": "本"})
        second = memory.process_claim({"subject": "田中", "predicate": "has", "object": "本"})

        self.assertIs(first, second)
        self.assertEqual(first.evidence, ["obs_1", "obs_2"])
        self.assertEqual(len(memory.beliefs), 1)
        self.assertEqual(memory.update_log[-1].type, "merge")

    def test_conflicting_claim_creates_conflict_object(self) -> None:
        memory = SemanticMemory()

        first = memory.process_claim({"subject": "田中", "predicate": "has", "object": "本"})
        second = memory.process_claim({"subject": "佐藤", "predicate": "has", "object": "本"})

        self.assertEqual(first.status, "active")
        self.assertEqual(second.status, "uncertain")
        self.assertEqual(first.counterevidence, ["obs_2"])
        self.assertEqual(len(memory.conflicts), 1)
        self.assertEqual(memory.conflicts[0].belief_a, first.id)
        self.assertEqual(memory.conflicts[0].belief_b, second.id)
        self.assertEqual(memory.update_log[-1].type, "contradict")

    def test_retire_belief_records_update(self) -> None:
        memory = SemanticMemory()
        belief = memory.process_claim({"subject": "田中", "predicate": "has", "object": "本"})

        memory.retire_belief(belief.id, {"reason": "manual cleanup"})

        self.assertEqual(belief.status, "retired")
        self.assertEqual(memory.update_log[-1].type, "retire")
        self.assertEqual(memory.update_log[-1].target, belief.id)


if __name__ == "__main__":
    unittest.main()
