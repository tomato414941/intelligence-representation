import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from intrep.image_tokenizer import ImagePatchTokenizer, read_portable_image
from intrep.signals import PayloadRef


class ImagePatchTokenizerTest(unittest.TestCase):
    def test_encodes_rgb_pixels_to_patch_tokens(self) -> None:
        pixels = np.array(
            [
                [[0, 0, 0], [255, 255, 255]],
                [[255, 0, 0], [0, 255, 0]],
            ],
            dtype=np.uint8,
        )
        tokenizer = ImagePatchTokenizer(patch_size=1, channel_bins=4)

        token_ids = tokenizer.encode_pixels(pixels)

        self.assertEqual(tokenizer.vocab_size, 65)
        self.assertEqual(tokenizer.pad_id, 64)
        self.assertEqual(token_ids, [0, 63, 48, 12])

    def test_encodes_grayscale_pixels_as_rgb(self) -> None:
        pixels = np.array([[0, 255]], dtype=np.uint8)
        tokenizer = ImagePatchTokenizer(patch_size=1, channel_bins=2)

        token_ids = tokenizer.encode_pixels(pixels)

        self.assertEqual(token_ids, [0, 7])

    def test_encodes_patch_mean(self) -> None:
        pixels = np.array(
            [
                [[0, 0, 0], [255, 255, 255]],
                [[255, 0, 0], [0, 255, 0]],
            ],
            dtype=np.uint8,
        )
        tokenizer = ImagePatchTokenizer(patch_size=2, channel_bins=4)

        token_ids = tokenizer.encode_pixels(pixels)

        self.assertEqual(token_ids, [20])

    def test_rejects_non_uint8_pixels(self) -> None:
        tokenizer = ImagePatchTokenizer()

        with self.assertRaisesRegex(ValueError, "dtype uint8"):
            tokenizer.encode_pixels(np.array([[0.0, 1.0]]))

    def test_rejects_non_divisible_dimensions(self) -> None:
        tokenizer = ImagePatchTokenizer(patch_size=2)
        pixels = np.zeros((3, 2, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "divisible by patch_size"):
            tokenizer.encode_pixels(pixels)

    def test_reads_binary_ppm(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.ppm"
            path.write_bytes(b"P6\n2 1\n255\n" + bytes([0, 0, 0, 255, 255, 255]))

            pixels = read_portable_image(path)

        self.assertEqual(pixels.tolist(), [[[0, 0, 0], [255, 255, 255]]])

    def test_reads_ascii_pgm(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.pgm"
            path.write_text("P2\n2 1\n255\n0 255\n", encoding="ascii")

            pixels = read_portable_image(path)

        self.assertEqual(pixels.tolist(), [[0, 255]])

    def test_encodes_payload_ref_file_uri(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.ppm"
            path.write_bytes(b"P6\n1 1\n255\n" + bytes([255, 0, 0]))
            ref = PayloadRef(uri=path.as_uri(), media_type="image/x-portable-pixmap")
            tokenizer = ImagePatchTokenizer(patch_size=1, channel_bins=4)

            token_ids = tokenizer.encode_ref(ref)

        self.assertEqual(token_ids, [48])

    def test_rejects_non_file_payload_ref(self) -> None:
        ref = PayloadRef(uri="dataset://images/frame-1.ppm", media_type="image/x-portable-pixmap")
        tokenizer = ImagePatchTokenizer()

        with self.assertRaisesRegex(ValueError, "file:// payload refs only"):
            tokenizer.encode_ref(ref)


if __name__ == "__main__":
    unittest.main()
