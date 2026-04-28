import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.image_rendering import render_image_token_document, render_image_token_payload
from intrep.signals import PayloadRef, Signal


class ImageRenderingTest(unittest.TestCase):
    def test_renders_image_payload_ref_as_shared_token_payload_and_document(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.pgm"
            path.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))
            event = Signal(
                channel="image",
                payload=PayloadRef(uri=path.as_uri(), media_type="image/x-portable-graymap"),
            )

            payload = render_image_token_payload(event)
            document = render_image_token_document(event)

        self.assertEqual(payload, "0 63")
        self.assertEqual(
            document,
            '<IMAGE_TOKENS patch_size="1" channel_bins="4" format="flat">\n0 63\n</IMAGE_TOKENS>\n',
        )

    def test_renders_grid_token_payload_with_patch_positions(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.pgm"
            path.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 128, 64]))
            event = Signal(
                channel="image",
                payload=PayloadRef(uri=path.as_uri(), media_type="image/x-portable-graymap"),
            )

            payload = render_image_token_payload(event, token_format="grid")
            document = render_image_token_document(event, token_format="grid")

        self.assertEqual(payload, "r0c0:0 r0c1:63\nr1c0:42 r1c1:21")
        self.assertEqual(
            document,
            (
                '<IMAGE_TOKENS patch_size="1" channel_bins="4" format="grid">\n'
                "r0c0:0 r0c1:63\n"
                "r1c0:42 r1c1:21\n"
                "</IMAGE_TOKENS>\n"
            ),
        )

    def test_non_image_payload_ref_events_fall_back_to_payload_text_error(self) -> None:
        event = Signal(
            channel="audio",
            payload=PayloadRef(uri="dataset://audio/sample.wav", media_type="audio/wav"),
        )

        with self.assertRaisesRegex(ValueError, "requires a channel-specific loader or encoder"):
            render_image_token_payload(event)

    def test_text_image_events_remain_pre_tokenized_text(self) -> None:
        event = Signal(channel="image", payload="0 63")

        self.assertEqual(render_image_token_payload(event), "0 63")
        self.assertEqual(render_image_token_document(event), "0 63")


if __name__ == "__main__":
    unittest.main()
