import unittest

from intrep.typed_events import TypedEvent


class TypedEventTest(unittest.TestCase):
    def test_typed_event_keeps_minimal_stream_envelope(self) -> None:
        event = TypedEvent(
            id="ep1_obs0",
            role="observation",
            modality="grid",
            content="A..",
            time_index=0,
            episode_id="ep1",
            source_id="fixture",
            metadata={"split": "train"},
        )

        self.assertEqual(event.role, "observation")
        self.assertEqual(event.id, "ep1_obs0")
        self.assertEqual(event.modality, "grid")
        self.assertEqual(event.episode_id, "ep1")
        self.assertEqual(event.time_index, 0)
        self.assertEqual(event.source_id, "fixture")
        self.assertEqual(event.metadata["split"], "train")

    def test_typed_event_accepts_legacy_metadata_envelope(self) -> None:
        event = TypedEvent(
            role="text",
            content="hello",
            metadata={
                "id": "doc1",
                "type": "markdown",
                "episode_id": "ep1",
                "time_index": 3,
                "source_id": "fixture",
                "split": "train",
            },
        )

        self.assertEqual(event.id, "doc1")
        self.assertEqual(event.modality, "markdown")
        self.assertEqual(event.episode_id, "ep1")
        self.assertEqual(event.time_index, 3)
        self.assertEqual(event.source_id, "fixture")
        self.assertEqual(event.metadata, {"split": "train"})

    def test_typed_event_rejects_unknown_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "event role must be one of"):
            TypedEvent(id="bad", role="entity", modality="text", content="x")  # type: ignore[arg-type]

    def test_typed_event_rejects_non_mapping_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "metadata must be a mapping"):
            TypedEvent(id="bad", role="text", modality="text", content="x", metadata=[])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
