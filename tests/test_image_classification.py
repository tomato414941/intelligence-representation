import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from intrep.image_classification import (
    CIFAR10_LABELS,
    ClassificationHead,
    FASHION_MNIST_LABELS,
    ImageTextChoiceExample,
    ImageClassificationExample,
    ImageClassificationConfig,
    ImageClassificationDataset,
    ImageFolderClassificationDataset,
    ImagePatchInputLayer,
    MNIST_LABELS,
    image_classification_examples_from_text_choices,
    image_classification_example_to_record,
    image_classification_tensors_from_examples,
    load_image_classification_examples_jsonl,
    load_image_text_choice_examples_jsonl,
    image_text_choice_tensors_from_examples,
    train_image_classifier,
    train_image_classifier_with_result,
)
from intrep.shared_multimodal_model import SharedMultimodalModel
from intrep.transformer_core import SharedTransformerCore


class ImageClassificationTest(unittest.TestCase):
    def test_loads_image_text_choice_examples_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fashion.jsonl"
            _write_image_text_choice_examples(path, Path(directory) / "images")

            examples = load_image_text_choice_examples_jsonl(path)
            images, labels = image_text_choice_tensors_from_examples(examples)

        self.assertEqual(len(examples), 2)
        self.assertIsInstance(examples[0], ImageTextChoiceExample)
        self.assertEqual(examples[0].answer_text, "Ankle boot")
        self.assertEqual(images.shape, torch.Size([2, 2, 2]))
        self.assertEqual(labels.tolist(), [9, 0])
        self.assertEqual(float(images[0, 0, 0]), 0.0)
        self.assertEqual(float(images[0, 0, 1]), 1.0)

    def test_loads_image_classification_examples_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            image_dir = Path(directory) / "images"
            image_dir.mkdir()
            image_a = image_dir / "a.pgm"
            image_b = image_dir / "b.pgm"
            image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
            image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
            examples = [
                ImageClassificationExample(image_path=image_a, label_names=FASHION_MNIST_LABELS, label_index=9),
                ImageClassificationExample(image_path=image_b, label_names=FASHION_MNIST_LABELS, label_index=0),
            ]
            path = Path(directory) / "classification.jsonl"
            path.write_text(
                "\n".join(json.dumps(image_classification_example_to_record(example)) for example in examples) + "\n",
                encoding="utf-8",
            )

            loaded = load_image_classification_examples_jsonl(path)
            images, labels = image_classification_tensors_from_examples(loaded)

        self.assertEqual(len(loaded), 2)
        self.assertIsInstance(loaded[0], ImageClassificationExample)
        self.assertEqual(loaded[0].label_text, "Ankle boot")
        self.assertEqual(images.shape, torch.Size([2, 2, 2]))
        self.assertEqual(labels.tolist(), [9, 0])

    def test_image_text_choice_tensors_from_examples_uses_shared_example_core(self) -> None:
        with TemporaryDirectory() as directory:
            image_a = Path(directory) / "a.pgm"
            image_b = Path(directory) / "b.pgm"
            image_a.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))
            image_b.write_bytes(b"P5\n2 1\n255\n" + bytes([128, 64]))
            examples = [
                ImageTextChoiceExample(image_path=image_a, choices=FASHION_MNIST_LABELS, answer_index=1),
                ImageTextChoiceExample(image_path=image_b, choices=FASHION_MNIST_LABELS, answer_index=2),
            ]

            images, labels = image_text_choice_tensors_from_examples(examples)

        self.assertEqual(images.shape, torch.Size([2, 1, 2]))
        self.assertEqual(labels.tolist(), [1, 2])
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.long)
        self.assertAlmostEqual(float(images[0, 0, 1]), 1.0)
        self.assertAlmostEqual(float(images[1, 0, 0]), 128 / 255)

    def test_image_text_choice_tensors_from_examples_rejects_empty_examples(self) -> None:
        with self.assertRaisesRegex(ValueError, "examples must not be empty"):
            image_text_choice_tensors_from_examples([])

    def test_image_text_choice_tensors_from_examples_rejects_mismatched_image_shapes(self) -> None:
        with TemporaryDirectory() as directory:
            image_a = Path(directory) / "a.pgm"
            image_b = Path(directory) / "b.pgm"
            image_a.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))
            image_b.write_bytes(b"P5\n1 1\n255\n" + bytes([128]))
            examples = [
                ImageTextChoiceExample(image_path=image_a, choices=FASHION_MNIST_LABELS, answer_index=1),
                ImageTextChoiceExample(image_path=image_b, choices=FASHION_MNIST_LABELS, answer_index=2),
            ]

            with self.assertRaisesRegex(ValueError, "all images must have the same shape"):
                image_text_choice_tensors_from_examples(examples)

    def test_image_classification_tensors_from_examples_uses_label_index(self) -> None:
        with TemporaryDirectory() as directory:
            image_a = Path(directory) / "a.pgm"
            image_b = Path(directory) / "b.pgm"
            image_a.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))
            image_b.write_bytes(b"P5\n2 1\n255\n" + bytes([128, 64]))
            examples = [
                ImageClassificationExample(image_path=image_a, label_names=FASHION_MNIST_LABELS, label_index=1),
                ImageClassificationExample(image_path=image_b, label_names=FASHION_MNIST_LABELS, label_index=2),
            ]

            images, labels = image_classification_tensors_from_examples(examples)

        self.assertEqual(images.shape, torch.Size([2, 1, 2]))
        self.assertEqual(labels.tolist(), [1, 2])
        self.assertEqual(examples[0].label_text, "Trouser")

    def test_image_classification_dataset_reads_examples_lazily(self) -> None:
        with TemporaryDirectory() as directory:
            image_a = Path(directory) / "a.pgm"
            image_b = Path(directory) / "b.pgm"
            image_a.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))
            image_b.write_bytes(b"P5\n2 1\n255\n" + bytes([128, 64]))
            examples = [
                ImageClassificationExample(image_path=image_a, label_names=FASHION_MNIST_LABELS, label_index=1),
                ImageClassificationExample(image_path=image_b, label_names=FASHION_MNIST_LABELS, label_index=2),
            ]

            dataset = ImageClassificationDataset(examples)
            image, label = dataset[1]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.image_shape, (1, 2))
        self.assertEqual(dataset.channel_count, 1)
        self.assertEqual(image.shape, torch.Size([1, 2]))
        self.assertEqual(int(label.item()), 2)
        self.assertAlmostEqual(float(image[0, 0]), 128 / 255)

    def test_image_folder_classification_dataset_reads_png_images(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            apple_dir = root / "apple"
            zebra_dir = root / "zebra"
            apple_dir.mkdir()
            zebra_dir.mkdir()
            Image.new("RGB", (2, 2), color=(255, 0, 0)).save(apple_dir / "a.png")
            Image.new("RGB", (2, 2), color=(0, 255, 0)).save(zebra_dir / "z.png")

            dataset = ImageFolderClassificationDataset(root)
            image, label = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.label_names, ("apple", "zebra"))
        self.assertEqual(dataset.image_shape, (2, 2, 3))
        self.assertEqual(dataset.channel_count, 3)
        self.assertEqual(image.shape, torch.Size([2, 2, 3]))
        self.assertEqual(int(label.item()), 0)
        self.assertAlmostEqual(float(image[0, 0, 0]), 1.0)

    def test_training_accepts_image_folder_classification_dataset(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            apple_dir = root / "apple"
            zebra_dir = root / "zebra"
            apple_dir.mkdir()
            zebra_dir.mkdir()
            Image.new("RGB", (2, 2), color=(255, 0, 0)).save(apple_dir / "a.png")
            Image.new("RGB", (2, 2), color=(0, 255, 0)).save(zebra_dir / "z.png")
            dataset = ImageFolderClassificationDataset(root)

            result = train_image_classifier_with_result(
                train_dataset=dataset,
                config=ImageClassificationConfig(
                    patch_size=1,
                    max_steps=0,
                    batch_size=2,
                    learning_rate=0.003,
                    seed=7,
                    model_preset="tiny",
                    device="cpu",
                ),
            )

        self.assertEqual(result.label_names, ("apple", "zebra"))
        self.assertEqual(result.image_shape, (2, 2, 3))
        self.assertEqual(result.metrics.train_case_count, 2)

    def test_shared_multimodal_model_outputs_class_logits(self) -> None:
        model = SharedMultimodalModel(
            vocab_size=8,
            text_context_length=4,
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
            num_classes=10,
        )

        logits = model.image_classification_logits(torch.zeros((3, 4, 4), dtype=torch.float32))

        self.assertEqual(logits.shape, torch.Size([3, 10]))
        self.assertEqual(model.text_logits(torch.zeros((3, 4), dtype=torch.long)).shape, torch.Size([3, 4, 8]))

    def test_shared_multimodal_model_accepts_rgb_images_for_classification(self) -> None:
        model = SharedMultimodalModel(
            vocab_size=1,
            text_context_length=1,
            image_size=(4, 4),
            patch_size=2,
            embedding_dim=8,
            num_heads=2,
            hidden_dim=16,
            num_layers=1,
            num_classes=10,
            channel_count=3,
        )

        logits = model.image_classification_logits(torch.zeros((3, 4, 4, 3), dtype=torch.float32))

        self.assertEqual(logits.shape, torch.Size([3, 10]))

    def test_image_text_choice_tensors_from_examples_preserves_rgb_images(self) -> None:
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "a.ppm"
            image_path.write_bytes(b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 255, 0]))
            examples = [
                ImageClassificationExample(image_path=image_path, label_names=CIFAR10_LABELS, label_index=3),
            ]

            images, labels = image_classification_tensors_from_examples(examples)

        self.assertEqual(images.shape, torch.Size([1, 1, 2, 3]))
        self.assertEqual(labels.tolist(), [3])
        self.assertAlmostEqual(float(images[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(images[0, 0, 1, 1]), 1.0)

    def test_training_uses_example_label_count_for_output_classes(self) -> None:
        with TemporaryDirectory() as directory:
            image_a = Path(directory) / "a.pgm"
            image_b = Path(directory) / "b.pgm"
            image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
            image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
            examples = [
                ImageClassificationExample(image_path=image_a, label_names=MNIST_LABELS, label_index=7),
                ImageClassificationExample(image_path=image_b, label_names=MNIST_LABELS, label_index=0),
            ]

            result = train_image_classifier_with_result(
                train_examples=examples,
                config=ImageClassificationConfig(
                    patch_size=1,
                    max_steps=0,
                    batch_size=2,
                    learning_rate=0.003,
                    seed=7,
                    model_preset="tiny",
                    device="cpu",
                ),
            )

        self.assertEqual(result.model.classification_head.output.out_features, len(MNIST_LABELS))

    def test_training_rejects_mixed_choice_sets(self) -> None:
        with TemporaryDirectory() as directory:
            image_a = Path(directory) / "a.pgm"
            image_b = Path(directory) / "b.pgm"
            image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
            image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
            examples = [
                ImageClassificationExample(image_path=image_a, label_names=MNIST_LABELS, label_index=7),
                ImageClassificationExample(image_path=image_b, label_names=FASHION_MNIST_LABELS, label_index=0),
            ]

            with self.assertRaisesRegex(ValueError, "same label_names"):
                train_image_classifier_with_result(
                    train_examples=examples,
                    config=ImageClassificationConfig(
                        patch_size=1,
                        max_steps=0,
                        batch_size=2,
                        learning_rate=0.003,
                        seed=7,
                        model_preset="tiny",
                        device="cpu",
                    ),
                )

    def test_shared_multimodal_model_composes_image_route_core_and_classification_head(self) -> None:
        model = SharedMultimodalModel(
            vocab_size=1,
            text_context_length=1,
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
            logits = model.image_classification_logits(images)
            manual_logits = model.classify_embeddings(model.encode_images(images))

        self.assertTrue(torch.allclose(logits, manual_logits))

    def test_shared_multimodal_model_exposes_image_embedding_sequence_path(self) -> None:
        model = SharedMultimodalModel(
            vocab_size=1,
            text_context_length=1,
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

    def test_image_patch_input_layer_outputs_input_embedding_sequence(self) -> None:
        input_layer = ImagePatchInputLayer(image_size=(4, 4), patch_size=2, embedding_dim=8)

        embeddings = input_layer(torch.zeros((3, 4, 4), dtype=torch.float32))

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
            _write_image_text_choice_examples(path, Path(directory) / "images")
            examples = image_classification_examples_from_text_choices(load_image_text_choice_examples_jsonl(path))

            metrics = train_image_classifier(
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

    def test_trains_small_classifier_with_result(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fashion.jsonl"
            _write_image_text_choice_examples(path, Path(directory) / "images")
            examples = image_classification_examples_from_text_choices(load_image_text_choice_examples_jsonl(path))

            result = train_image_classifier_with_result(
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

        self.assertEqual(result.metrics.train_case_count, 2)
        self.assertIsInstance(result.model, SharedMultimodalModel)
        self.assertIsInstance(result.model.image_input_layer, ImagePatchInputLayer)


def _write_image_text_choice_examples(path: Path, image_dir: Path) -> None:
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
