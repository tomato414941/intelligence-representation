import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from intrep import train_image_classification
from intrep.image_classification import (
    FASHION_MNIST_LABELS,
    ImageFolderClassificationDataset,
    ImageClassificationMetrics,
    ImageClassificationTrainingResult,
)
from intrep.image_classification_checkpoint import load_image_classification_checkpoint
from intrep.shared_multimodal_checkpoint import load_shared_multimodal_initialization
from intrep.shared_multimodal_model import SharedMultimodalModel


class TrainImageClassificationCLITest(unittest.TestCase):
    def test_builds_image_classification_config(self) -> None:
        captured_config = None
        captured_train_count = 0
        captured_eval_count = 0

        def fake_train_image_classifier_with_result(
            *,
            train_examples=None,
            eval_examples=None,
            train_dataset=None,
            eval_dataset=None,
            config,
        ):
            nonlocal captured_config, captured_train_count, captured_eval_count
            captured_config = config
            captured_train_count = len(train_examples or train_dataset)
            captured_eval_count = len(eval_examples or eval_dataset or [])
            metrics = ImageClassificationMetrics(
                target="label",
                input_representation="image-patches",
                train_case_count=1,
                eval_case_count=1,
                train_initial_loss=2.0,
                train_final_loss=1.0,
                train_accuracy=1.0,
                eval_accuracy=1.0,
                patch_size=config.patch_size,
                max_steps=config.max_steps,
                model_preset=config.model_preset,
            )
            model = SharedMultimodalModel(
                vocab_size=1,
                text_context_length=1,
                image_size=(4, 4),
                patch_size=config.patch_size,
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
                num_classes=len(FASHION_MNIST_LABELS),
            )
            return ImageClassificationTrainingResult(
                metrics=metrics,
                model=model,
                config=config,
                image_shape=(4, 4),
                label_names=FASHION_MNIST_LABELS,
            )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            metrics_path = root / "metrics.json"
            _write_image_label_events(train_path, root / "train-images", "train")
            _write_image_label_events(eval_path, root / "eval-images", "eval")

            with patch.object(
                train_image_classification,
                "train_image_classifier_with_result",
                fake_train_image_classifier_with_result,
            ):
                train_image_classification.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--eval-path",
                        str(eval_path),
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

        self.assertIsNotNone(captured_config)
        assert captured_config is not None
        self.assertEqual(captured_train_count, 2)
        self.assertEqual(captured_eval_count, 2)
        self.assertEqual(captured_config.patch_size, 4)
        self.assertEqual(captured_config.model_preset, "tiny")
        self.assertEqual(captured_config.max_steps, 20)
        self.assertEqual(captured_config.batch_size, 8)

    def test_builds_image_folder_datasets(self) -> None:
        captured_train_dataset = None
        captured_eval_dataset = None

        def fake_train_image_classifier_with_result(
            *,
            train_examples=None,
            eval_examples=None,
            train_dataset=None,
            eval_dataset=None,
            config,
        ):
            nonlocal captured_train_dataset, captured_eval_dataset
            captured_train_dataset = train_dataset
            captured_eval_dataset = eval_dataset
            metrics = ImageClassificationMetrics(
                target="label",
                input_representation="image-patches",
                train_case_count=len(train_dataset),
                eval_case_count=len(eval_dataset),
                train_initial_loss=2.0,
                train_final_loss=1.0,
                train_accuracy=1.0,
                eval_accuracy=1.0,
                patch_size=config.patch_size,
                max_steps=config.max_steps,
                model_preset=config.model_preset,
            )
            model = SharedMultimodalModel(
                vocab_size=1,
                text_context_length=1,
                image_size=(2, 2),
                patch_size=config.patch_size,
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
                num_classes=2,
                channel_count=3,
            )
            return ImageClassificationTrainingResult(
                metrics=metrics,
                model=model,
                config=config,
                image_shape=(2, 2, 3),
                label_names=("apple", "zebra"),
            )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_root = root / "train"
            eval_root = root / "eval"
            _write_image_folder(train_root)
            _write_image_folder(eval_root)

            with patch.object(
                train_image_classification,
                "train_image_classifier_with_result",
                fake_train_image_classifier_with_result,
            ):
                train_image_classification.main(
                    [
                        "--train-image-folder",
                        str(train_root),
                        "--eval-image-folder",
                        str(eval_root),
                        "--image-size",
                        "2",
                        "2",
                        "--image-patch-size",
                        "1",
                    ]
                )

        self.assertIsInstance(captured_train_dataset, ImageFolderClassificationDataset)
        self.assertIsInstance(captured_eval_dataset, ImageFolderClassificationDataset)
        assert captured_train_dataset is not None
        self.assertEqual(captured_train_dataset.label_names, ("apple", "zebra"))
        self.assertEqual(captured_train_dataset.image_shape, (2, 2, 3))

    def test_runs_small_image_label_smoke(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            metrics_path = root / "metrics.json"
            checkpoint_path = root / "classification.pt"
            _write_image_label_events(train_path, root / "train-images", "train")
            _write_image_label_events(eval_path, root / "eval-images", "eval")

            with redirect_stdout(output):
                train_image_classification.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--eval-path",
                        str(eval_path),
                        "--max-steps",
                        "1",
                        "--batch-size",
                        "2",
                        "--image-patch-size",
                        "1",
                        "--metrics-path",
                        str(metrics_path),
                        "--checkpoint-path",
                        str(checkpoint_path),
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint = load_image_classification_checkpoint(checkpoint_path, device="cpu")
            initialization = load_shared_multimodal_initialization(checkpoint_path, device="cpu")

        self.assertIn("target=label", output.getvalue())
        self.assertIn("intrep image classification", output.getvalue())
        self.assertIn("input_representation=image-patches", output.getvalue())
        self.assertEqual(payload["target"], "label")
        self.assertEqual(payload["input_representation"], "image-patches")
        self.assertEqual(payload["train_case_count"], 2)
        self.assertEqual(payload["eval_case_count"], 2)
        self.assertEqual(checkpoint.label_names, FASHION_MNIST_LABELS)
        self.assertEqual(checkpoint.image_shape, (1, 2))
        self.assertEqual(initialization.source_schema, "intrep.image_classification_checkpoint.v1")
        self.assertIsNone(initialization.tokenizer)

    def test_runs_small_image_folder_smoke(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_root = root / "train"
            eval_root = root / "eval"
            metrics_path = root / "metrics.json"
            _write_image_folder(train_root)
            _write_image_folder(eval_root)

            with redirect_stdout(output):
                train_image_classification.main(
                    [
                        "--train-image-folder",
                        str(train_root),
                        "--eval-image-folder",
                        str(eval_root),
                        "--max-steps",
                        "1",
                        "--batch-size",
                        "2",
                        "--image-patch-size",
                        "1",
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn("intrep image classification", output.getvalue())
        self.assertEqual(payload["train_case_count"], 2)
        self.assertEqual(payload["eval_case_count"], 2)


def _write_image_label_events(path: Path, image_dir: Path, prefix: str) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_a = image_dir / f"{prefix}_a.pgm"
    image_b = image_dir / f"{prefix}_b.pgm"
    image_a.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))
    image_b.write_bytes(b"P5\n2 1\n255\n" + bytes([255, 0]))
    rows = [
        {
            "image_path": str(image_a),
            "label_names": list(FASHION_MNIST_LABELS),
            "label_index": 9,
        },
        {
            "image_path": str(image_b),
            "label_names": list(FASHION_MNIST_LABELS),
            "label_index": 0,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_image_folder(root: Path) -> None:
    apple_dir = root / "apple"
    zebra_dir = root / "zebra"
    apple_dir.mkdir(parents=True)
    zebra_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(apple_dir / "a.png")
    Image.new("RGB", (2, 2), color=(0, 255, 0)).save(zebra_dir / "z.png")


if __name__ == "__main__":
    unittest.main()
