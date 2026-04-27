import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.mixed_corpus import MixedDocument
from intrep.typed_corpus import (
    infer_role_from_modality,
    load_corpus_jsonl_as_mixed_documents,
    load_typed_events_jsonl,
    load_typed_events_jsonl_v2,
    mixed_document_to_typed_event,
    typed_events_to_mixed_documents,
    write_typed_events_jsonl_v2,
)
from intrep.typed_events import TypedEvent
from intrep.next_observation_cases import extract_next_observation_cases


class TypedCorpusTest(unittest.TestCase):
    def test_infer_role_from_existing_modalities(self) -> None:
        self.assertEqual(infer_role_from_modality("grid"), "observation")
        self.assertEqual(infer_role_from_modality("action_log"), "action")
        self.assertEqual(infer_role_from_modality("next_grid"), "consequence")
        self.assertEqual(infer_role_from_modality("code"), "text")

    def test_mixed_document_to_typed_event_keeps_content(self) -> None:
        document = MixedDocument(id="case_1", modality="environment_symbolic", content="<obs> a")

        event = mixed_document_to_typed_event(document)

        self.assertEqual(event.id, "case_1")
        self.assertEqual(event.role, "observation")
        self.assertEqual(event.content, "<obs> a")

    def test_typed_events_to_mixed_documents_renders_tags(self) -> None:
        event = TypedEvent(
            id="ep1_obs0",
            role="observation",
            modality="grid",
            content="A..",
            episode_id="ep1",
            time_index=0,
        )

        documents = typed_events_to_mixed_documents([event])

        self.assertEqual(documents[0].modality, "observation:grid")
        self.assertIn('<EVENT id="ep1_obs0" role="observation"', documents[0].content)
        self.assertIn('id="ep1_obs0"', documents[0].content)

    def test_typed_jsonl_v2_helpers_round_trip_required_fields(self) -> None:
        events = [
            TypedEvent(
                id="ep1_obs0",
                role="observation",
                modality="grid",
                content="A..",
                episode_id="ep1",
                time_index=0,
                metadata={"split": "train"},
            )
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "typed-v2.jsonl"

            write_typed_events_jsonl_v2(path, events)

            loaded = load_typed_events_jsonl_v2(path)

        self.assertEqual(loaded[0].id, "ep1_obs0")
        self.assertEqual(loaded[0].role, "observation")
        self.assertEqual(loaded[0].modality, "grid")
        self.assertEqual(loaded[0].episode_id, "ep1")
        self.assertEqual(loaded[0].time_index, 0)
        self.assertEqual(loaded[0].metadata["split"], "train")

    def test_load_typed_events_jsonl_reads_v2_records(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "typed.jsonl"
            path.write_text(
                (
                    '{"id":"ep1_obs0","role":"observation","modality":"grid",'
                    '"episode_id":"ep1","time_index":0,"content":"A..","metadata":{"split":"train"}}\n'
                ),
                encoding="utf-8",
            )

            events = load_typed_events_jsonl(path)

        self.assertEqual(events[0].role, "observation")
        self.assertEqual(events[0].episode_id, "ep1")
        self.assertEqual(events[0].metadata["split"], "train")

    def test_load_corpus_auto_preserves_v1_plain_default(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            path.write_text(
                '{"id":"text_1","modality":"text","content":"hello"}\n',
                encoding="utf-8",
            )

            documents = load_corpus_jsonl_as_mixed_documents(path)

        self.assertEqual(documents, [MixedDocument(id="text_1", modality="text", content="hello")])

    def test_load_corpus_auto_renders_v2_typed_tags(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "typed.jsonl"
            path.write_text(
                '{"id":"act_1","role":"action","modality":"grid_action","content":"right"}\n',
                encoding="utf-8",
            )

            documents = load_corpus_jsonl_as_mixed_documents(path, render_format="typed-tags")

        self.assertEqual(documents[0].modality, "action:grid_action")
        self.assertIn('role="action"', documents[0].content)

    def test_load_typed_events_jsonl_validates_role(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text(
                '{"id":"bad","role":"entity","modality":"text","content":"x"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown role entity"):
                load_typed_events_jsonl(path)

    def test_typed_jsonl_v2_requires_metadata_field(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken-v2.jsonl"
            path.write_text(
                '{"id":"bad","role":"text","modality":"text","content":"x",'
                '"episode_id":null,"time_index":null}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing required fields: metadata"):
                load_typed_events_jsonl_v2(path)

    def test_typed_rendered_symbolic_documents_still_extract_next_observation_cases(self) -> None:
        documents = typed_events_to_mixed_documents(
            [
                TypedEvent(
                    id="case_1",
                    role="observation",
                    modality="environment_symbolic",
                    content="<obs> box closed <action> open box <next_obs> key visible",
                )
            ]
        )

        cases = extract_next_observation_cases(documents)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].modality, "environment_symbolic")
        self.assertEqual(cases[0].positive_next, "key visible")


if __name__ == "__main__":
    unittest.main()
