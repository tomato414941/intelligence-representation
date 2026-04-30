import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import CausalTextModel, build_causal_text_config
from intrep.fashion_mnist_image_choice_corpus import write_fashion_mnist_image_choice_jsonl
from intrep.image_classification import ImageChoiceExample, ImagePatchInputLayer, load_image_choice_examples_jsonl
from intrep.image_text_evaluation import evaluate_image_text_choices


class ImageTextEvaluationTest(unittest.TestCase):
    def test_evaluates_image_text_choice_examples(self) -> None:
        tokenizer = ByteTokenizer()
        image_input = ImagePatchInputLayer(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
        )
        text_model = CausalTextModel(
            build_causal_text_config(
                preset="tiny",
                vocab_size=tokenizer.vocab_size,
                context_length=8,
                embedding_dim=8,
            )
        )
        preferred_id = tokenizer.encode("b")[0]
        with torch.no_grad():
            text_model.token_output.output.weight.zero_()
            text_model.token_output.output.bias.zero_()
            text_model.token_output.output.bias[preferred_id] = 10.0

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.pgm"
            image_path.write_text("P2\n4 4\n255\n" + " ".join(["0"] * 16) + "\n", encoding="ascii")
            metrics = evaluate_image_text_choices(
                examples=(
                    ImageChoiceExample(
                        image_path=image_path,
                        choices=("a", "b"),
                        answer_index=1,
                    ),
                ),
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                prompt="?",
            )

        self.assertEqual(metrics.case_count, 1)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.predicted_indices, (1,))
        self.assertEqual(len(metrics.losses), 1)
        self.assertEqual(len(metrics.losses[0]), 2)
        self.assertEqual(metrics.to_dict()["predicted_indices"], [1])

    def test_evaluates_generated_fashion_mnist_image_choice_jsonl(self) -> None:
        tokenizer = ByteTokenizer()
        image_input = ImagePatchInputLayer(
            image_size=(2, 2),
            patch_size=1,
            embedding_dim=8,
        )
        text_model = CausalTextModel(
            build_causal_text_config(
                preset="tiny",
                vocab_size=tokenizer.vocab_size,
                context_length=64,
                embedding_dim=8,
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_path = root / "images.idx3-ubyte.gz"
            labels_path = root / "labels.idx1-ubyte.gz"
            examples_path = root / "fashion.jsonl"
            _write_idx_images(images_path, [[[0, 255], [128, 64]]])
            _write_idx_labels(labels_path, [9])
            write_fashion_mnist_image_choice_jsonl(
                images_path=images_path,
                labels_path=labels_path,
                output_path=examples_path,
                image_output_dir=root / "images",
                limit=1,
            )
            examples = load_image_choice_examples_jsonl(examples_path)
            metrics = evaluate_image_text_choices(
                examples=examples,
                image_input_layer=image_input,
                text_model=text_model,
                tokenizer=tokenizer,
                prompt="label: ",
            )

        self.assertEqual(metrics.case_count, 1)
        self.assertEqual(len(metrics.predicted_indices), 1)
        self.assertEqual(len(metrics.losses), 1)
        self.assertEqual(len(metrics.losses[0]), 10)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)


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
