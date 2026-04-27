import unittest

from intrep.mixed_corpus import MixedDocument, default_mixed_documents
from intrep.mixed_corpus_evaluation import evaluate_mixed_corpus_pairing


class MixedCorpusEvaluationTest(unittest.TestCase):
    def test_default_corpus_reports_environment_pairing_coverage(self) -> None:
        documents = default_mixed_documents()
        coverage = evaluate_mixed_corpus_pairing(documents)

        self.assertEqual(coverage.modality_counts["text"], 2)
        self.assertEqual(
            coverage.environment_symbolic_count,
            sum(1 for document in documents if document.modality == "environment_symbolic"),
        )
        self.assertEqual(
            coverage.environment_natural_count,
            sum(1 for document in documents if document.modality == "environment_natural"),
        )
        self.assertEqual(
            coverage.modality_counts["environment_symbolic"],
            coverage.environment_symbolic_count,
        )
        self.assertEqual(
            coverage.modality_counts["environment_natural"],
            coverage.environment_natural_count,
        )
        self.assertEqual(coverage.paired_episode_ids, ["001", "002", "pair_001", "pair_002", "pair_003"])

    def test_pairing_uses_supported_environment_document_names_only(self) -> None:
        documents = [
            MixedDocument(
                id="env_symbolic_box",
                modality="environment_symbolic",
                content="<obs> box closed",
            ),
            MixedDocument(
                id="env_natural_box",
                modality="environment_natural",
                content="The box is closed.",
            ),
            MixedDocument(
                id="custom_symbolic_box",
                modality="environment_symbolic",
                content="<obs> custom symbolic",
            ),
            MixedDocument(
                id="env_natural_unpaired",
                modality="environment_natural",
                content="An unpaired natural description.",
            ),
            MixedDocument(id="log_001", modality="log", content="status=ok"),
        ]

        coverage = evaluate_mixed_corpus_pairing(documents)

        self.assertEqual(
            coverage.modality_counts,
            {
                "environment_symbolic": 2,
                "environment_natural": 2,
                "log": 1,
            },
        )
        self.assertEqual(coverage.environment_symbolic_count, 2)
        self.assertEqual(coverage.environment_natural_count, 2)
        self.assertEqual(coverage.paired_episode_ids, ["box"])


if __name__ == "__main__":
    unittest.main()
