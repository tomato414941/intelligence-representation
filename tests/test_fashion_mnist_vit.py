import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from intrep.fashion_mnist_vit import (
    ClassificationHead,
    FASHION_MNIST_LABELS,
    ImageChoiceExample,
    ImageClassificationConfig,
    ImagePatchAdapter,
    PatchTransformerClassifier,
    SharedTransformerCore,
    load_image_choice_examples_jsonl,
    image_label_tensors_from_examples,
    train_fashion_mnist_classifier,
)


class FashionMNISTViTTest(unittest.TestCase):
    def test_loads_image_choice_examples_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fashion.jsonl"
            _write_image_choice_examples(path, Path(directory) / "images")

            examples = load_image_choice_examples_jsonl(path)
            images, labels = image_label_tensors_from_examples(examples)

        self.assertEqual(len(examples), 2)
        self.assertIsInstance(examples[0], ImageChoiceExample)
        self.assertEqual(examples[0].answer_text, "Ankle boot")
        self.assertEqual(images.shape, torch.Size([2, 2, 2]))
        self.assertEqual(labels.tolist(), [9, 0])
        self.assertEqual(float(images[0, 0, 0]), 0.0)
        self.assertEqual(float(images[0, 0, 1]), 1.0)

    def test_image_label_tensors_from_examples_uses_shared_example_core(self) -> None:
        with TemporaryDirectory() as directory:
            image_a = Path(directory) / "a.pgm"
            image_b = Path(directory) / "b.pgm"
            image_a.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))
            image_b.write_bytes(b"P5\n2 1\n255\n" + bytes([128, 64]))
            examples = [
                ImageChoiceExample(image_path=image_a, choices=FASHION_MNIST_LABELS, answer_index=1),
                ImageChoiceExample(image_path=image_b, choices=FASHION_MNIST_LABELS, answer_index=2),
            ]

            images, labels = image_label_tensors_from_examples(examples)

        self.assertEqual(images.shape, torch.Size([2, 1, 2]))
        self.assertEqual(labels.tolist(), [1, 2])
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.long)
        self.assertAlmostEqual(float(images[0, 0, 1]), 1.0)
        self.assertAlmostEqual(float(images[1, 0, 0]), 128 / 255)

    def test_image_label_tensors_from_examples_rejects_empty_examples(self) -> None:
        with self.assertRaisesRegex(ValueError, "examples must not be empty"):
            image_label_tensors_from_examples([])

    def test_image_label_tensors_from_examples_rejects_mismatched_image_shapes(self) -> None:
        with TemporaryDirectory() as directory:
            image_a = Path(directory) / "a.pgm"
            image_b = Path(directory) / "b.pgm"
            image_a.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))
            image_b.write_bytes(b"P5\n1 1\n255\n" + bytes([128]))
            examples = [
                ImageChoiceExample(image_path=image_a, choices=FASHION_MNIST_LABELS, answer_index=1),
                ImageChoiceExample(image_path=image_b, choices=FASHION_MNIST_LABELS, answer_index=2),
            ]

            with self.assertRaisesRegex(ValueError, "all images must have the same shape"):
                image_label_tensors_from_examples(examples)

    def test_patch_transformer_classifier_outputs_class_logits(self) -> None:
        model = PatchTransformerClassifier(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
            num_classes=10,
        )

        logits = model(torch.zeros((3, 4, 4), dtype=torch.float32))

        self.assertEqual(logits.shape, torch.Size([3, 10]))

    def test_patch_transformer_classifier_composes_adapter_core_and_head(self) -> None:
        model = PatchTransformerClassifier(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
            num_classes=10,
        )
        model.eval()
        images = torch.zeros((3, 4, 4), dtype=torch.float32)

        with torch.no_grad():
            logits = model(images)
            manual_logits = model.classify_embeddings(model.encode_images(images))

        self.assertTrue(torch.allclose(logits, manual_logits))

    def test_patch_transformer_classifier_exposes_hidden_sequence_path(self) -> None:
        model = PatchTransformerClassifier(
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
            num_classes=10,
        )
        images = torch.zeros((3, 4, 4), dtype=torch.float32)

        embeddings = model.embed_images(images)
        encoded = model.encode_images(images)
        logits = model.classify_embeddings(encoded)

        self.assertEqual(embeddings.shape, torch.Size([3, 4, 8]))
        self.assertEqual(encoded.shape, torch.Size([3, 4, 8]))
        self.assertEqual(logits.shape, torch.Size([3, 10]))

    def test_image_patch_adapter_outputs_hidden_sequence(self) -> None:
        adapter = ImagePatchAdapter(image_size=(4, 4), patch_size=2, embedding_dim=8)

        embeddings = adapter(torch.zeros((3, 4, 4), dtype=torch.float32))

        self.assertEqual(embeddings.shape, torch.Size([3, 4, 8]))

    def test_shared_transformer_core_preserves_sequence_shape(self) -> None:
        core = SharedTransformerCore(
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
        )

        hidden = core(torch.zeros((3, 4, 8), dtype=torch.float32))

        self.assertEqual(hidden.shape, torch.Size([3, 4, 8]))

    def test_classification_head_maps_hidden_sequence_to_class_logits(self) -> None:
        head = ClassificationHead(embedding_dim=8, num_classes=10)

        logits = head(torch.zeros((3, 4, 8), dtype=torch.float32))

        self.assertEqual(logits.shape, torch.Size([3, 10]))

    def test_trains_small_classifier_smoke(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fashion.jsonl"
            _write_image_choice_examples(path, Path(directory) / "images")
            examples = load_image_choice_examples_jsonl(path)

            metrics = train_fashion_mnist_classifier(
                train_examples=examples,
                eval_examples=examples,
                config=ImageClassificationConfig(
                    patch_size=1,
                    max_steps=1,
                    batch_size=2,
                    learning_rate=0.003,
                    seed=7,
                    model_preset="tiny",
                    device="cpu",
                ),
            )

        self.assertEqual(metrics.target, "label")
        self.assertEqual(metrics.input_representation, "image-patches")
        self.assertEqual(metrics.train_case_count, 2)
        self.assertEqual(metrics.eval_case_count, 2)
        self.assertIsNotNone(metrics.eval_accuracy)


def _write_image_choice_examples(path: Path, image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_a = image_dir / "a.pgm"
    image_b = image_dir / "b.pgm"
    image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
    image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
    rows = [
        {
            "image_path": str(image_a),
            "choices": list(FASHION_MNIST_LABELS),
            "answer_index": 9,
        },
        {
            "image_path": str(image_b),
            "choices": list(FASHION_MNIST_LABELS),
            "answer_index": 0,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
