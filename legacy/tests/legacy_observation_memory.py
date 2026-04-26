import unittest

from experiments.observation_memory import ObservationMemory


class ObservationMemoryTest(unittest.TestCase):
    def test_add_observation(self) -> None:
        memory = ObservationMemory()

        observation = memory.add("Transformerは関係計算器に近い", tags=["transformer"])

        self.assertEqual(observation.id, "obs_1")
        self.assertEqual(observation.type, "observation")
        self.assertEqual(observation.tags, ["transformer"])
        self.assertEqual(memory.observations, [observation])

    def test_retrieve_by_content_and_tags(self) -> None:
        memory = ObservationMemory()
        first = memory.add("Transformerは系列モデルというより関係計算器に近い", tags=["transformer", "関係"])
        second = memory.add("Attentionはトークン間の重み付き関係を作る", tags=["attention", "関係"])
        memory.add("State Memoryは真理DBではなくキャッシュとして扱う", tags=["state-memory", "cache"])

        results = memory.retrieve("Transformer 関係")

        self.assertEqual([observation.id for observation in results], [first.id, second.id])

    def test_build_context(self) -> None:
        memory = ObservationMemory()
        first = memory.add("A")
        second = memory.add("B")

        context = memory.build_context([first, second])

        self.assertEqual(context, "[obs_1] A\n[obs_2] B")

    def test_store_generated_summary_with_links(self) -> None:
        memory = ObservationMemory()
        first = memory.add("Transformerは系列モデルというより関係計算器に近い")
        second = memory.add("Attentionはトークン間の重み付き関係を作る")

        summary = memory.store_generated_summary(
            "Transformerはトークン間関係を計算する機構として見られる。",
            links=[first.id, second.id],
        )

        self.assertEqual(summary.type, "summary")
        self.assertEqual(summary.source, "generated")
        self.assertEqual(summary.links, [first.id, second.id])
        self.assertEqual(summary.tags, ["summary"])


if __name__ == "__main__":
    unittest.main()
