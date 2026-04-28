import unittest

from intrep.signals import PayloadRef, Signal, render_payload_text, render_signal


class SignalTest(unittest.TestCase):
    def test_signal_keeps_channel_and_payload(self) -> None:
        signal = Signal(channel="observation", payload="A..")

        self.assertEqual(signal.channel, "observation")
        self.assertEqual(signal.payload, "A..")

    def test_signal_keeps_payload_ref(self) -> None:
        ref = PayloadRef(uri="dataset://images/frame-1.png", media_type="image/png")
        signal = Signal(channel="image", payload=ref)

        self.assertEqual(signal.payload, ref)

    def test_render_payload_text_keeps_text_payload(self) -> None:
        signal = Signal(channel="action", payload='{"arguments":{"box":"red"},"name":"open_box"}')

        self.assertEqual(render_payload_text(signal), '{"arguments":{"box":"red"},"name":"open_box"}')

    def test_render_signal_renders_structured_text_payload(self) -> None:
        signal = Signal(channel="action", payload='{"arguments":{"box":"red"},"name":"open_box"}')

        rendered = render_signal(signal)

        self.assertIn('channel="action"', rendered)
        self.assertIn('{"arguments":{"box":"red"},"name":"open_box"}', rendered)

    def test_render_signal_rejects_payload_ref(self) -> None:
        signal = Signal(
            channel="image",
            payload=PayloadRef(uri="dataset://images/frame-1.png", media_type="image/png"),
        )

        with self.assertRaisesRegex(ValueError, "requires a channel-specific loader or encoder"):
            render_signal(signal)

    def test_signal_rejects_unsupported_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "signal payload must be a string or PayloadRef"):
            Signal(channel="image", payload=object())

    def test_payload_ref_requires_scheme(self) -> None:
        with self.assertRaisesRegex(ValueError, "payload ref uri must include a scheme"):
            PayloadRef(uri="images/frame-1.png", media_type="image/png")

    def test_payload_ref_requires_mime_like_media_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "payload ref media_type must look like a MIME type"):
            PayloadRef(uri="dataset://images/frame-1.png", media_type="image")

    def test_payload_ref_validates_sha256(self) -> None:
        with self.assertRaisesRegex(ValueError, "payload ref sha256 must be a 64-character hex string"):
            PayloadRef(uri="dataset://images/frame-1.png", media_type="image/png", sha256="bad")

    def test_payload_ref_validates_size_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "payload ref size_bytes must be a non-negative integer"):
            PayloadRef(uri="dataset://images/frame-1.png", media_type="image/png", size_bytes=-1)

    def test_signal_rejects_legacy_content_argument(self) -> None:
        with self.assertRaises(TypeError):
            Signal(channel="text", content="hello")

    def test_signal_rejects_legacy_metadata_channel(self) -> None:
        with self.assertRaises(TypeError):
            Signal(payload="hello", metadata={"type": "text"})

    def test_signal_rejects_whitespace_channel(self) -> None:
        with self.assertRaisesRegex(ValueError, "signal channel must not contain whitespace"):
            Signal(channel="bad channel", payload="x")


if __name__ == "__main__":
    unittest.main()
