import unittest

from experiments.semantic_memory import SemanticMemory


class SemanticMemoryTest(unittest.TestCase):
    def test_ingest_separates_observation_claim_and_belief(self) -> None:
        memory = SemanticMemory()

        claim = memory.ingest({"subject": "田中", "predicate": "has", "object": "本"}, source="test")

        self.assertEqual(len(memory.observations), 1)
        self.assertEqual(len(memory.claims), 1)
        self.assertEqual(len(memory.beliefs), 1)
        self.assertEqual(claim.observation_id, "obs_1")
        self.assertEqual(memory.beliefs[0].supporting_claims, [claim.id])
        self.assertEqual(memory.update_log[0].type, "add_claim")

    def test_duplicate_claim_merges_into_existing_belief(self) -> None:
        memory = SemanticMemory()

        first = memory.ingest({"subject": "田中", "predicate": "has", "object": "本"})
        second = memory.ingest({"subject": "田中", "predicate": "has", "object": "本"})

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(len(memory.claims), 2)
        self.assertEqual(len(memory.beliefs), 1)
        self.assertEqual(memory.beliefs[0].supporting_claims, [first.id, second.id])
        self.assertEqual(memory.update_log[-1].type, "merge_belief")

    def test_conflicting_claim_creates_uncertain_belief_and_conflict(self) -> None:
        memory = SemanticMemory()

        first = memory.ingest({"subject": "田中", "predicate": "has", "object": "本"})
        second = memory.ingest({"subject": "佐藤", "predicate": "has", "object": "本"})

        self.assertEqual(len(memory.beliefs), 2)
        self.assertEqual(memory.beliefs[0].status, "active")
        self.assertEqual(memory.beliefs[1].status, "uncertain")
        self.assertEqual(memory.beliefs[0].counter_claims, [second.id])
        self.assertEqual(memory.beliefs[1].counter_claims, [first.id])
        self.assertEqual(len(memory.conflicts), 1)
        self.assertEqual(memory.update_log[-1].type, "create_conflict")

    def test_different_owner_of_belief_does_not_conflict(self) -> None:
        memory = SemanticMemory()

        memory.ingest(
            {"subject": "田中", "predicate": "has", "object": "本", "owner_of_belief": "world"}
        )
        memory.ingest(
            {"subject": "佐藤", "predicate": "has", "object": "本", "owner_of_belief": "田中"}
        )

        self.assertEqual(len(memory.conflicts), 0)
        self.assertEqual([belief.status for belief in memory.beliefs], ["active", "active"])


if __name__ == "__main__":
    unittest.main()
