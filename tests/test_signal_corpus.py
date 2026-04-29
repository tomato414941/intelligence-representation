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
    render_signals_for_training,
    reject_payload_refs,
    signals_to_mixed_documents,
    write_signals_jsonl_v2,
)


class SignalCorpusTest(unittest.TestCase):
    def test_compatibility_facade_keeps_representative_legacy_names(self) -> None:
        import intrep.signal_corpus as signal_corpus

        self.assertIs(signal_corpus.Signal, Signal)
        self.assertIs(signal_corpus.PayloadRef, PayloadRef)
        self.assertIs(signal_corpus.MixedDocument, MixedDocument)
        self.assertTrue(callable(signal_corpus.render_signal))
        self.assertTrue(callable(signal_corpus.render_signal_stream))
        self.assertTrue(callable(signal_corpus.render_payload_text))
        self.assertIs(signal_corpus.Path, Path)

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

    def test_render_signals_for_training_plain_uses_payload_text_only(self) -> None:
        corpus = render_signals_for_training(
            [
                Signal(channel="observation", payload="A.."),
                Signal(channel="action", payload="move east"),
            ],
            render_format="plain",
        )

        self.assertEqual(corpus, "A..\nmove east\n")
        self.assertNotIn("<SIGNAL", corpus)

    def test_render_signals_for_training_tags_use_signal_stream_contract(self) -> None:
        corpus = render_signals_for_training(
            [Signal(channel="observation", payload="A..")],
            render_format="signal-tags",
        )

        self.assertIn('<SIGNAL channel="observation">', corpus)
        self.assertIn("A..", corpus)

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
        self.assertIn('<IMAGE_TOKENS patch_size="1" channel_bins="4" format="flat">', documents[0].content)
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

    def test_reject_payload_refs_reports_call_site_context(self) -> None:
        event = Signal(
            channel="image",
            payload=PayloadRef(uri="dataset://images/frame-1.png", media_type="image/png"),
        )

        with self.assertRaisesRegex(ValueError, "training does not support payload_ref"):
            reject_payload_refs([event], context="training")

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

    def test_load_corpus_auto_detects_signal_records_and_renders_requested_format(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "signals.jsonl"
            path.write_text(
                '{"channel":"observation","payload":"A.."}\n',
                encoding="utf-8",
            )

            documents = load_corpus_jsonl_as_mixed_documents(
                path,
                render_format="signal-tags",
            )

        self.assertEqual(documents[0].modality, "observation")
        self.assertIn('<SIGNAL channel="observation">', documents[0].content)

    def test_load_corpus_mixed_documents_can_be_rendered_through_legacy_bridge(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            path.write_text(
                '{"id":"case_1","modality":"environment_symbolic","content":"<obs> a"}\n',
                encoding="utf-8",
            )

            documents = load_corpus_jsonl_as_mixed_documents(
                path,
                corpus_format="mixed-document",
                render_format="signal-tags",
            )

        self.assertEqual(documents[0].modality, "observation")
        self.assertIn('<SIGNAL channel="observation">', documents[0].content)

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
