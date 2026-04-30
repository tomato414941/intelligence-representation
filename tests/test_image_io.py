import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.image_io import read_portable_image


class ImageIoTest(unittest.TestCase):
    def test_reads_binary_ppm(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.ppm"
            path.write_bytes(b"P6\n2 1\n255\n" + bytes([0, 0, 0, 255, 255, 255]))

            pixels = read_portable_image(path)

        self.assertEqual(pixels.tolist(), [[[0, 0, 0], [255, 255, 255]]])

    def test_reads_binary_ppm_when_payload_starts_with_whitespace_byte(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.ppm"
            path.write_bytes(b"P6\n1 1\n255\n" + bytes([10, 20, 30]))

            pixels = read_portable_image(path)

        self.assertEqual(pixels.tolist(), [[[10, 20, 30]]])

    def test_reads_ascii_pgm(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.pgm"
            path.write_text("P2\n2 1\n255\n0 255\n", encoding="ascii")

            pixels = read_portable_image(path)

        self.assertEqual(pixels.tolist(), [[0, 255]])


if __name__ == "__main__":
    unittest.main()
