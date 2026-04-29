import gzip
import io
import struct
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.fashion_mnist_signal_corpus import (
    main,
    read_idx_images,
    read_idx_labels,
    write_fashion_mnist_signal_jsonl,
)
from intrep.image_tokenizer import ImagePatchTokenizer
from intrep.signal_io import load_signals_jsonl_v2
from intrep.signals import PayloadRef


class FashionMNISTSignalCorpusTest(unittest.TestCase):
    def test_reads_idx_images_and_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images_path = root / "images.idx3-ubyte.gz"
            labels_path = root / "labels.idx1-ubyte.gz"
            _write_idx_images(images_path, [[[0, 255], [128, 64]]])
            _write_idx_labels(labels_path, [9])

            images = read_idx_images(images_path)
            labels = read_idx_labels(labels_path)

        self.assertEqual(images.shape, (1, 2, 2))
        self.assertEqual(images[0].tolist(), [[0, 255], [128, 64]])
        self.assertEqual(labels.tolist(), [9])

    def test_writes_signal_jsonl_and_pgm_images(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images_path = root / "images.idx3-ubyte.gz"
            labels_path = root / "labels.idx1-ubyte.gz"
            output_path = root / "fashion.jsonl"
            image_output_dir = root / "images"
            _write_idx_images(
                images_path,
                [
                    [[0, 255], [128, 64]],
                    [[255, 0], [64, 128]],
                ],
            )
            _write_idx_labels(labels_path, [9, 0])

            selection = write_fashion_mnist_signal_jsonl(
                images_path=images_path,
                labels_path=labels_path,
                output_path=output_path,
                image_output_dir=image_output_dir,
                split="train",
                limit=1,
            )
            loaded = load_signals_jsonl_v2(output_path)

        self.assertEqual(selection.image_count, 1)
        self.assertEqual([event.channel for event in loaded], ["image", "label"])
        self.assertIsInstance(loaded[0].payload, PayloadRef)
        assert isinstance(loaded[0].payload, PayloadRef)
        self.assertEqual(loaded[0].payload.media_type, "image/x-portable-graymap")
        self.assertEqual(loaded[1].payload, "9:Ankle boot")

    def test_generated_payload_ref_can_be_image_tokenized(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images_path = root / "images.idx3-ubyte.gz"
            labels_path = root / "labels.idx1-ubyte.gz"
            output_path = root / "fashion.jsonl"
            image_output_dir = root / "images"
            _write_idx_images(images_path, [[[0, 255], [128, 64]]])
            _write_idx_labels(labels_path, [5])

            write_fashion_mnist_signal_jsonl(
                images_path=images_path,
                labels_path=labels_path,
                output_path=output_path,
                image_output_dir=image_output_dir,
            )
            image_signal = load_signals_jsonl_v2(output_path)[0]
            assert isinstance(image_signal.payload, PayloadRef)
            token_ids = ImagePatchTokenizer(patch_size=1, channel_bins=4).encode_ref(
                image_signal.payload
            )

        self.assertEqual(token_ids, [0, 63, 42, 21])

    def test_rejects_mismatched_image_and_label_counts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images_path = root / "images.idx3-ubyte.gz"
            labels_path = root / "labels.idx1-ubyte.gz"
            _write_idx_images(images_path, [[[0, 255], [128, 64]]])
            _write_idx_labels(labels_path, [1, 2])

            with self.assertRaisesRegex(ValueError, "counts must match"):
                write_fashion_mnist_signal_jsonl(
                    images_path=images_path,
                    labels_path=labels_path,
                    output_path=root / "fashion.jsonl",
                    image_output_dir=root / "images",
                )

    def test_cli_writes_signal_jsonl(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images_path = root / "images.idx3-ubyte.gz"
            labels_path = root / "labels.idx1-ubyte.gz"
            output_path = root / "fashion.jsonl"
            image_output_dir = root / "images"
            _write_idx_images(images_path, [[[0, 255], [128, 64]]])
            _write_idx_labels(labels_path, [3])

            with redirect_stdout(output):
                main(
                    [
                        "--images-path",
                        str(images_path),
                        "--labels-path",
                        str(labels_path),
                        "--output-path",
                        str(output_path),
                        "--image-output-dir",
                        str(image_output_dir),
                        "--split",
                        "test",
                        "--limit",
                        "1",
                    ]
                )

            loaded = load_signals_jsonl_v2(output_path)

        self.assertEqual([event.channel for event in loaded], ["image", "label"])
        self.assertIn("intrep fashion-mnist signal corpus", output.getvalue())
        self.assertIn("images=1", output.getvalue())


def _write_idx_images(path: Path, images: list[list[list[int]]]) -> None:
    count = len(images)
    rows = len(images[0])
    cols = len(images[0][0])
    payload = bytes(value for image in images for row in image for value in row)
    data = struct.pack(">IIII", 2051, count, rows, cols) + payload
    with gzip.open(path, "wb") as handle:
        handle.write(data)


def _write_idx_labels(path: Path, labels: list[int]) -> None:
    data = struct.pack(">II", 2049, len(labels)) + bytes(labels)
    with gzip.open(path, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    unittest.main()
