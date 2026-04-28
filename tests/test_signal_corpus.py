import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.mixed_corpus import MixedDocument
from intrep.next_observation_cases import extract_next_observation_cases
from intrep.signals import PayloadRef, Signal
from intrep.signal_corpus import (
    infer_channel_from_modality,
    load_corpus_jsonl_as_mixed_documents,
    load_signals_jsonl,
    load_signals_jsonl_v2,
    mixed_document_to_signal,
    signals_to_mixed_documents,
    write_signals_jsonl_v2,
)


class SignalCorpusTest(unittest.TestCase):
    def test_infer_channel_from_existing_modalities(self) -> None:
        self.assertEqual(infer_channel_from_modality("grid"), "observation")
        self.assertEqual(infer_channel_from_modality("action_log"), "action")
        self.assertEqual(infer_channel_from_modality("next_grid"), "consequence")
        self.assertEqual(infer_channel_from_modality("code"), "text")

    def test_mixed_document_to_signal_keeps_payload(self) -> None:
        document = MixedDocument(id="case_1", modality="environment_symbolic", content="<obs> a")

        event = mixed_document_to_signal(document)

        self.assertEqual(event.channel, "observation")
        self.assertEqual(event.payload, "<obs> a")

    def test_signals_to_mixed_documents_renders_tags(self) -> None:
        event = Signal(channel="observation", payload="A..")

        documents = signals_to_mixed_documents([event])

        self.assertEqual(documents[0].modality, "observation")
        self.assertIn('<SIGNAL channel="observation"', documents[0].content)

    def test_signal_jsonl_v2_helpers_round_trip_required_fields(self) -> None:
        events = [Signal(channel="observation", payload="A..")]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "typed-v2.jsonl"

            write_signals_jsonl_v2(path, events)

            loaded = load_signals_jsonl_v2(path)

        self.assertEqual(loaded[0].channel, "observation")
        self.assertEqual(loaded[0].payload, "A..")

    def test_signal_to_mixed_document_renders_structured_text_payload(self) -> None:
        documents = signals_to_mixed_documents(
            [Signal(channel="action", payload='{"arguments":{"box":"red"},"name":"open_box"}')],
            render_format="plain",
        )

        self.assertEqual(
            documents[0].content,
            '{"arguments":{"box":"red"},"name":"open_box"}',
        )

    def test_signal_to_mixed_document_rejects_payload_ref_until_encoder_exists(self) -> None:
        event = Signal(
            channel="image",
            payload=PayloadRef(uri="dataset://images/frame-1.png", media_type="image/png"),
        )

        with self.assertRaisesRegex(ValueError, "requires a channel-specific loader or encoder"):
            signals_to_mixed_documents([event])

    def test_signal_to_mixed_document_can_render_file_image_payload_ref_as_tokens(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.pgm"
            path.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))
            event = Signal(
                channel="image",
                payload=PayloadRef(uri=path.as_uri(), media_type="image/x-portable-graymap"),
            )

            documents = signals_to_mixed_documents([event], render_format="image-tokens")

        self.assertEqual(documents[0].modality, "image")
        self.assertIn('<IMAGE_TOKENS patch_size="1" channel_bins="4">', documents[0].content)
        self.assertIn("0 63", documents[0].content)

    def test_signal_jsonl_v2_helpers_round_trip_payload_ref(self) -> None:
        events = [
            Signal(
                channel="image",
                payload=PayloadRef(
                    uri="dataset://images/frame-1.png",
                    media_type="image/png",
                    sha256="a" * 64,
                    size_bytes=123,
                ),
            )
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "signals.jsonl"

            write_signals_jsonl_v2(path, events)

            loaded = load_signals_jsonl_v2(path)

        self.assertEqual(loaded, events)

    def test_load_signals_jsonl_rejects_payload_and_payload_ref_together(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "signal.jsonl"
            path.write_text(
                (
                    '{"channel":"image","payload":"x",'
                    '"payload_ref":{"uri":"dataset://images/frame-1.png","media_type":"image/png"}}\n'
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "payload and payload_ref are mutually exclusive"):
                load_signals_jsonl_v2(path)

    def test_load_signals_jsonl_rejects_invalid_payload_ref_schema(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "signal.jsonl"
            path.write_text(
                '{"channel":"image","payload_ref":{"uri":"dataset://images/frame-1.png"}}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing payload_ref fields: media_type"):
                load_signals_jsonl_v2(path)

    def test_load_signals_jsonl_reads_legacy_content_records(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "signal.jsonl"
            path.write_text(
                '{"id":"ep1_obs0","role":"observation","content":"A.."}\n',
                encoding="utf-8",
            )

            events = load_signals_jsonl(path)

        self.assertEqual(events[0].channel, "observation")
        self.assertEqual(events[0].payload, "A..")

    def test_load_corpus_auto_preserves_v1_plain_default(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            path.write_text(
                '{"id":"text_1","modality":"text","content":"hello"}\n',
                encoding="utf-8",
            )

            documents = load_corpus_jsonl_as_mixed_documents(path)

        self.assertEqual(documents, [MixedDocument(id="text_1", modality="text", content="hello")])

    def test_signal_rendered_symbolic_documents_still_extract_next_observation_cases(self) -> None:
        documents = signals_to_mixed_documents(
            [
                Signal(
                    channel="environment_symbolic",
                    payload="<obs> box closed <action> open box <next_obs> key visible",
                )
            ]
        )

        cases = extract_next_observation_cases(documents)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].modality, "environment_symbolic")
        self.assertEqual(cases[0].positive_next, "key visible")


if __name__ == "__main__":
    unittest.main()
