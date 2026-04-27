import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.grid_corpus import default_grid_documents, episode_to_mixed_documents
from intrep.mixed_corpus import (
    MixedDocument,
    default_mixed_documents,
    load_mixed_documents_jsonl,
)
from intrep.next_observation_cases import (
    NextObservationCase,
    extract_next_observation_cases,
)


class NextObservationCasesTest(unittest.TestCase):
    def test_extracts_environment_symbolic_next_observation_cases(self) -> None:
        documents = default_mixed_documents()

        cases = extract_next_observation_cases(documents)
        symbolic_cases = [case for case in cases if case.modality == "environment_symbolic"]

        self.assertGreaterEqual(len(symbolic_cases), 5)
        self.assertIn(
            NextObservationCase(
                id="env_symbolic_001",
                modality="environment_symbolic",
                prefix="<obs> key in box ; box closed <action> open box <next_obs> ",
                positive_next="key visible",
            ),
            symbolic_cases,
        )
        for case in symbolic_cases:
            self.assertIn("<obs>", case.prefix)
            self.assertIn("<action>", case.prefix)
            self.assertTrue(case.prefix.endswith("<next_obs> "))
            self.assertNotIn("<next_obs>", case.positive_next)

    def test_extracts_grid_action_next_grid_cases(self) -> None:
        documents = episode_to_mixed_documents(
            {
                "id": "rank grid",
                "transitions": [
                    {
                        "text": "The agent sees a key.",
                        "grid": ("###", "#A#", "#K#"),
                        "action": "move_south",
                        "next_grid": ("###", "#.#", "#A#"),
                        "next_text": "The agent moves onto the key.",
                    }
                ],
            }
        )

        cases = extract_next_observation_cases(documents)

        self.assertEqual(
            cases,
            [
                NextObservationCase(
                    id="rank_grid_step_001",
                    modality="grid",
                    prefix="<grid>\n###\n#A#\n#K#\n</grid>\n<action> move_south\n",
                    positive_next="<next_grid>\n###\n#.#\n#A#\n</next_grid>",
                )
            ],
        )

    def test_default_grid_documents_produce_grid_cases_without_symbolic_cases(self) -> None:
        documents = default_grid_documents()

        cases = extract_next_observation_cases(documents)
        modalities = {case.modality for case in cases}

        self.assertEqual(modalities, {"grid"})
        self.assertEqual(len(cases), sum(1 for document in documents if document.modality == "grid"))
        self.assertTrue(all("<grid>" in case.prefix for case in cases))
        self.assertTrue(all("<action>" in case.prefix for case in cases))
        self.assertTrue(all("<next_grid>" in case.positive_next for case in cases))

    def test_mixed_documents_can_include_symbolic_and_grid_cases(self) -> None:
        documents = default_mixed_documents() + default_grid_documents()

        cases = extract_next_observation_cases(documents)
        modalities = {case.modality for case in cases}

        self.assertEqual(modalities, {"environment_symbolic", "grid"})

    def test_ignores_incomplete_or_non_next_observation_documents(self) -> None:
        documents = [
            MixedDocument(id="text_001", modality="text", content="background"),
            MixedDocument(
                id="env_symbolic_incomplete",
                modality="environment_symbolic",
                content="<obs> box closed <action> open box",
            ),
            MixedDocument(
                id="step_001_grid",
                modality="grid",
                content="<grid>\nA\n</grid>",
            ),
            MixedDocument(
                id="step_001_next_grid",
                modality="next_grid",
                content="<next_grid>\nA\n</next_grid>",
            ),
        ]

        self.assertEqual(extract_next_observation_cases(documents), [])

    def test_extracts_next_observation_cases_from_external_web_action_documents(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "external-corpus.jsonl"
            path.write_text(
                "\n".join(
                    [
                        (
                            '{"id":"external_web_cart","modality":"external_web",'
                            '"content":"<web url=https://example.invalid/cart> cart page shows pending order </web>"}'
                        ),
                        (
                            '{"id":"external_action_checkout","modality":"external_action",'
                            '"content":"<obs> url=https://example.invalid/cart ; order pending '
                            '<action> click checkout <next_obs> url=https://example.invalid/checkout ; checkout form visible"}'
                        ),
                        (
                            '{"id":"external_action_pay","modality":"external_action",'
                            '"content":"<obs> url=https://example.invalid/checkout ; checkout form visible '
                            '<action> submit payment <next_obs> url=https://example.invalid/receipt ; receipt visible"}'
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            documents = load_mixed_documents_jsonl(path)

        cases = extract_next_observation_cases(documents)

        self.assertEqual(
            cases,
            [
                NextObservationCase(
                    id="external_action_checkout",
                    modality="external_action",
                    prefix=(
                        "<obs> url=https://example.invalid/cart ; order pending "
                        "<action> click checkout <next_obs> "
                    ),
                    positive_next=(
                        "url=https://example.invalid/checkout ; checkout form visible"
                    ),
                ),
                NextObservationCase(
                    id="external_action_pay",
                    modality="external_action",
                    prefix=(
                        "<obs> url=https://example.invalid/checkout ; checkout form visible "
                        "<action> submit payment <next_obs> "
                    ),
                    positive_next="url=https://example.invalid/receipt ; receipt visible",
                ),
            ],
        )
        self.assertNotIn("external_web_cart", {case.id for case in cases})


if __name__ == "__main__":
    unittest.main()
