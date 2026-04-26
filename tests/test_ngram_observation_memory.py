import unittest

from experiments.ngram_observation_memory import NgramObservationMemory


class NgramObservationMemoryTest(unittest.TestCase):
    def test_retrieve_japanese_text_without_spaces(self) -> None:
        memory = NgramObservationMemory()
        first = memory.add("Transformerは系列モデルというより関係計算器に近い")
        memory.add("State Memoryは真理DBではなくキャッシュとして扱う")

        results = memory.retrieve("関係計算")

        self.assertEqual([observation.id for observation in results], [first.id])

    def test_tag_match_boosts_result(self) -> None:
        memory = NgramObservationMemory()
        first = memory.add("Attentionはトークン間の重み付き関係を作る", tags=["attention"])
        memory.add("関係という言葉だけを含む別の観測")

        results = memory.retrieve("attention")

        self.assertEqual(results[0].id, first.id)

    def test_build_context(self) -> None:
        memory = NgramObservationMemory()
        first = memory.add("A")
        second = memory.add("B")

        self.assertEqual(memory.build_context([first, second]), "[obs_1] A\n[obs_2] B")


if __name__ == "__main__":
    unittest.main()
