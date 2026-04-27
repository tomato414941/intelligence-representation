import unittest

from intrep.mixed_corpus import MixedDocument, default_mixed_documents
from intrep.mixed_corpus_evaluation import (
    build_train_eval_document_split,
    evaluate_mixed_corpus_pairing,
    extract_environment_document_pairs,
)


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

        pairs = extract_environment_document_pairs(documents)

        self.assertEqual(
            [(pair.episode_id, pair.symbolic.id, pair.natural.id) for pair in pairs],
            [
                ("001", "env_symbolic_001", "env_natural_001"),
                ("002", "env_symbolic_002", "env_natural_002"),
                ("pair_001", "env_pair_symbolic_001", "env_pair_natural_001"),
                ("pair_002", "env_pair_symbolic_002", "env_pair_natural_002"),
                ("pair_003", "env_pair_symbolic_003", "env_pair_natural_003"),
            ],
        )

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

    def test_extract_environment_document_pairs_returns_stable_supported_pairs_only(self) -> None:
        symbolic_box = MixedDocument(
            id="env_symbolic_box",
            modality="environment_symbolic",
            content="<obs> box closed",
        )
        natural_box = MixedDocument(
            id="env_natural_box",
            modality="environment_natural",
            content="The box is closed.",
        )
        pair_symbolic = MixedDocument(
            id="env_pair_symbolic_001",
            modality="environment_symbolic",
            content="<obs> coin in drawer",
        )
        pair_natural = MixedDocument(
            id="env_pair_natural_001",
            modality="environment_natural",
            content="The coin is in the drawer.",
        )
        documents = [
            pair_natural,
            MixedDocument(
                id="env_symbolic_unpaired",
                modality="environment_symbolic",
                content="<obs> unpaired",
            ),
            natural_box,
            MixedDocument(
                id="env_pair_natural_unpaired",
                modality="environment_natural",
                content="An unpaired generated environment.",
            ),
            pair_symbolic,
            MixedDocument(
                id="custom_natural_box",
                modality="environment_natural",
                content="Unsupported natural environment name.",
            ),
            symbolic_box,
        ]

        pairs = extract_environment_document_pairs(documents)

        self.assertEqual(
            [(pair.episode_id, pair.symbolic, pair.natural) for pair in pairs],
            [
                ("box", symbolic_box, natural_box),
                ("pair_001", pair_symbolic, pair_natural),
            ],
        )

    def test_split_holds_out_complete_environment_pairs_by_count(self) -> None:
        documents = default_mixed_documents()

        split = build_train_eval_document_split(documents, eval_episode_count=2)

        self.assertEqual(split.eval_episode_ids, ["001", "002"])
        self.assertEqual(
            [document.id for document in split.eval_documents],
            ["env_symbolic_001", "env_natural_001", "env_symbolic_002", "env_natural_002"],
        )
        self.assertNotIn("ja_explain_001", [document.id for document in split.eval_documents])
        self.assertIn("ja_explain_001", [document.id for document in split.train_documents])

    def test_split_holds_out_complete_environment_pairs_by_fraction(self) -> None:
        documents = default_mixed_documents()

        split = build_train_eval_document_split(documents, eval_episode_fraction=0.4)

        self.assertEqual(split.eval_episode_ids, ["001", "002"])
        self.assertEqual(
            {
                _episode_suffix(document.id)
                for document in split.eval_documents
                if document.modality.startswith("environment_")
            },
            {"001", "002"},
        )

    def test_split_keeps_unpaired_environment_documents_in_train(self) -> None:
        documents = [
            MixedDocument(id="env_symbolic_a", modality="environment_symbolic", content="<obs> a"),
            MixedDocument(id="env_natural_a", modality="environment_natural", content="A."),
            MixedDocument(id="env_symbolic_b", modality="environment_symbolic", content="<obs> b"),
            MixedDocument(id="text_001", modality="text", content="background"),
        ]

        split = build_train_eval_document_split(documents, eval_episode_count=1)

        self.assertEqual([document.id for document in split.eval_documents], ["env_symbolic_a", "env_natural_a"])
        self.assertEqual([document.id for document in split.train_documents], ["env_symbolic_b", "text_001"])

    def test_split_validates_selection_arguments(self) -> None:
        documents = default_mixed_documents()

        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_train_eval_document_split(documents)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_train_eval_document_split(documents, eval_episode_count=1, eval_episode_fraction=0.2)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            build_train_eval_document_split(documents, eval_episode_fraction=1.1)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_train_eval_document_split(documents, eval_episode_count=99)


def _episode_suffix(document_id: str) -> str:
    return document_id.replace("env_symbolic_", "").replace("env_natural_", "")


if __name__ == "__main__":
    unittest.main()
