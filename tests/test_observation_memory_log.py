import unittest

from experiments.observation_memory_log import LoggedObservationMemory


class LoggedObservationMemoryTest(unittest.TestCase):
    def test_retrieve_logs_query_and_results(self) -> None:
        memory = LoggedObservationMemory()
        first = memory.add("Transformerは系列モデルというより関係計算器に近い", tags=["transformer", "関係"])
        second = memory.add("Attentionはトークン間の重み付き関係を作る", tags=["attention", "関係"])

        results = memory.retrieve("Transformer 関係", timestamp="t1")

        self.assertEqual([observation.id for observation in results], [first.id, second.id])
        self.assertEqual(len(memory.update_log), 1)
        self.assertEqual(memory.update_log[0].type, "retrieve")
        self.assertEqual(memory.update_log[0].query, "Transformer 関係")
        self.assertEqual(memory.update_log[0].input_observations, [first.id, second.id])
        self.assertEqual(memory.update_log[0].timestamp, "t1")

    def test_build_context_logs_inputs(self) -> None:
        memory = LoggedObservationMemory()
        first = memory.add("A")
        second = memory.add("B")

        context = memory.build_context([first, second], timestamp="t2")

        self.assertEqual(context, "[obs_1] A\n[obs_2] B")
        self.assertEqual(memory.update_log[0].type, "build_context")
        self.assertEqual(memory.update_log[0].input_observations, [first.id, second.id])
        self.assertEqual(memory.update_log[0].timestamp, "t2")

    def test_generated_summary_logs_query_inputs_and_output(self) -> None:
        memory = LoggedObservationMemory()
        first = memory.add("Transformerは系列モデルというより関係計算器に近い")
        second = memory.add("Attentionはトークン間の重み付き関係を作る")

        summary = memory.store_generated_summary(
            "Transformerはトークン間関係を計算する機構として見られる。",
            query="Transformer 関係",
            links=[first.id, second.id],
            timestamp="t3",
        )

        self.assertEqual(summary.type, "summary")
        self.assertEqual(memory.update_log[0].type, "generated_summary")
        self.assertEqual(memory.update_log[0].query, "Transformer 関係")
        self.assertEqual(memory.update_log[0].input_observations, [first.id, second.id])
        self.assertEqual(memory.update_log[0].output_observation, summary.id)
        self.assertEqual(memory.update_log[0].timestamp, "t3")


if __name__ == "__main__":
    unittest.main()
