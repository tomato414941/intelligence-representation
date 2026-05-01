import io
import json
import pickle
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from intrep.cifar10_image_corpus import (
    main,
    read_cifar10_batch,
    write_cifar10_image_classification_jsonl,
    write_cifar10_image_text_choice_jsonl,
)
from intrep.image_io import read_portable_image


class CIFAR10ImageCorpusTest(unittest.TestCase):
    def test_reads_cifar10_batch(self) -> None:
        with TemporaryDirectory() as directory:
            batch_path = Path(directory) / "data_batch_1"
            _write_cifar_batch(batch_path, labels=[3])

            images, labels = read_cifar10_batch(batch_path)

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (32, 32, 3))
        self.assertEqual(labels, [3])

    def test_writes_image_text_choice_jsonl_and_ppm_images(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            batch_path = root / "data_batch_1"
            output_path = root / "cifar10.jsonl"
            image_output_dir = root / "images"
            _write_cifar_batch(batch_path, labels=[3, 9])

            selection = write_cifar10_image_text_choice_jsonl(
                batch_paths=[batch_path],
                output_path=output_path,
                image_output_dir=image_output_dir,
                split="train",
                limit=1,
            )
            loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            pixels = read_portable_image(Path(loaded[0]["image_path"]))

        self.assertEqual(selection.image_count, 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["choices"][3], "cat")
        self.assertEqual(loaded[0]["answer_index"], 3)
        self.assertEqual(pixels.shape, (32, 32, 3))

    def test_writes_image_classification_jsonl_and_ppm_images(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            batch_path = root / "data_batch_1"
            output_path = root / "cifar10-classification.jsonl"
            image_output_dir = root / "images"
            _write_cifar_batch(batch_path, labels=[3, 9])

            selection = write_cifar10_image_classification_jsonl(
                batch_paths=[batch_path],
                output_path=output_path,
                image_output_dir=image_output_dir,
                split="train",
                limit=1,
            )
            loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            pixels = read_portable_image(Path(loaded[0]["image_path"]))

        self.assertEqual(selection.image_count, 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["label_names"][3], "cat")
        self.assertEqual(loaded[0]["label_index"], 3)
        self.assertEqual(pixels.shape, (32, 32, 3))

    def test_rejects_empty_batch_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            with self.assertRaisesRegex(ValueError, "batch_paths must not be empty"):
                write_cifar10_image_text_choice_jsonl(
                    batch_paths=[],
                    output_path=root / "cifar10.jsonl",
                    image_output_dir=root / "images",
                )

    def test_cli_writes_image_classification_jsonl(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            batch_path = root / "data_batch_1"
            output_path = root / "cifar10.jsonl"
            image_output_dir = root / "images"
            _write_cifar_batch(batch_path, labels=[8])

            with redirect_stdout(output):
                main(
                    [
                        "--batch-path",
                        str(batch_path),
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

            loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["label_index"], 8)
        self.assertIn("intrep cifar10 image corpus", output.getvalue())
        self.assertIn("images=1", output.getvalue())


def _write_cifar_batch(path: Path, *, labels: list[int]) -> None:
    rows = []
    for label in labels:
        image = np.full((3, 32, 32), label, dtype=np.uint8)
        rows.append(image.reshape(-1))
    batch = {b"data": np.stack(rows), b"labels": labels}
    with path.open("wb") as handle:
        pickle.dump(batch, handle)


if __name__ == "__main__":
    unittest.main()
