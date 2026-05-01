import gzip
import io
import json
import struct
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.idx_image_choice_corpus import (
    main,
    read_idx_images,
    read_idx_labels,
    write_idx_image_classification_jsonl,
    write_idx_image_choice_jsonl,
)
from intrep.image_io import read_portable_image


class IDXImageChoiceCorpusTest(unittest.TestCase):
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

    def test_writes_image_choice_jsonl_and_pgm_images(self) -> None:
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

            selection = write_idx_image_choice_jsonl(
                images_path=images_path,
                labels_path=labels_path,
                output_path=output_path,
                image_output_dir=image_output_dir,
                split="train",
                limit=1,
            )
            loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(selection.image_count, 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["choices"][9], "Ankle boot")
        self.assertEqual(loaded[0]["answer_index"], 9)

    def test_writes_image_classification_jsonl_and_pgm_images(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images_path = root / "images.idx3-ubyte.gz"
            labels_path = root / "labels.idx1-ubyte.gz"
            output_path = root / "fashion-classification.jsonl"
            image_output_dir = root / "images"
            _write_idx_images(images_path, [[[0, 255], [128, 64]]])
            _write_idx_labels(labels_path, [9])

            selection = write_idx_image_classification_jsonl(
                images_path=images_path,
                labels_path=labels_path,
                output_path=output_path,
                image_output_dir=image_output_dir,
            )
            loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(selection.image_count, 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["label_names"][9], "Ankle boot")
        self.assertEqual(loaded[0]["label_index"], 9)

    def test_writes_mnist_digit_choices(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images_path = root / "images.idx3-ubyte.gz"
            labels_path = root / "labels.idx1-ubyte.gz"
            output_path = root / "mnist.jsonl"
            image_output_dir = root / "images"
            _write_idx_images(images_path, [[[0, 255], [128, 64]]])
            _write_idx_labels(labels_path, [7])

            write_idx_image_choice_jsonl(
                images_path=images_path,
                labels_path=labels_path,
                output_path=output_path,
                image_output_dir=image_output_dir,
                label_names=tuple(str(index) for index in range(10)),
            )
            loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(loaded[0]["choices"], [str(index) for index in range(10)])
        self.assertEqual(loaded[0]["answer_index"], 7)

    def test_generated_image_path_can_be_loaded_as_image(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images_path = root / "images.idx3-ubyte.gz"
            labels_path = root / "labels.idx1-ubyte.gz"
            output_path = root / "fashion.jsonl"
            image_output_dir = root / "images"
            _write_idx_images(images_path, [[[0, 255], [128, 64]]])
            _write_idx_labels(labels_path, [5])

            write_idx_image_choice_jsonl(
                images_path=images_path,
                labels_path=labels_path,
                output_path=output_path,
                image_output_dir=image_output_dir,
            )
            record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            image_path = Path(record["image_path"])
            pixels = read_portable_image(image_path)

        self.assertEqual(pixels.tolist(), [[0, 255], [128, 64]])

    def test_rejects_mismatched_image_and_label_counts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images_path = root / "images.idx3-ubyte.gz"
            labels_path = root / "labels.idx1-ubyte.gz"
            _write_idx_images(images_path, [[[0, 255], [128, 64]]])
            _write_idx_labels(labels_path, [1, 2])

            with self.assertRaisesRegex(ValueError, "counts must match"):
                write_idx_image_choice_jsonl(
                    images_path=images_path,
                    labels_path=labels_path,
                    output_path=root / "fashion.jsonl",
                    image_output_dir=root / "images",
                )

    def test_cli_writes_image_choice_jsonl(self) -> None:
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
                        "--label-set",
                        "mnist",
                    ]
                )

            loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["answer_index"], 3)
        self.assertEqual(loaded[0]["choices"], [str(index) for index in range(10)])
        self.assertIn("intrep image-choice corpus", output.getvalue())
        self.assertIn("label_set=mnist", output.getvalue())
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
