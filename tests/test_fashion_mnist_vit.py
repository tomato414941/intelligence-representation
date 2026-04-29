import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from intrep.fashion_mnist_vit import (
    ImageClassificationConfig,
    PatchTransformerClassifier,
    image_label_tensors,
    train_fashion_mnist_classifier,
)
from intrep.signal_io import load_signals_jsonl


class FashionMNISTViTTest(unittest.TestCase):
    def test_extracts_image_label_tensors_from_signal_pairs(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fashion.jsonl"
            _write_image_label_events(path, Path(directory) / "images")

            images, labels = image_label_tensors(load_signals_jsonl(path))

        self.assertEqual(images.shape, torch.Size([2, 2, 2]))
        self.assertEqual(labels.tolist(), [9, 0])
        self.assertEqual(float(images[0, 0, 0]), 0.0)
        self.assertEqual(float(images[0, 0, 1]), 1.0)

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

    def test_trains_small_classifier_smoke(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fashion.jsonl"
            _write_image_label_events(path, Path(directory) / "images")
            events = load_signals_jsonl(path)

            metrics = train_fashion_mnist_classifier(
                train_events=events,
                eval_events=events,
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

        self.assertEqual(metrics.target_channel, "label")
        self.assertEqual(metrics.rendering, "image-patches")
        self.assertEqual(metrics.train_case_count, 2)
        self.assertEqual(metrics.eval_case_count, 2)
        self.assertIsNotNone(metrics.eval_accuracy)


def _write_image_label_events(path: Path, image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_a = image_dir / "a.pgm"
    image_b = image_dir / "b.pgm"
    image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
    image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
    rows = [
        {
            "channel": "image",
            "payload_ref": {
                "uri": image_a.as_uri(),
                "media_type": "image/x-portable-graymap",
            },
        },
        {"channel": "label", "payload": "9:Ankle boot"},
        {
            "channel": "image",
            "payload_ref": {
                "uri": image_b.as_uri(),
                "media_type": "image/x-portable-graymap",
            },
        },
        {"channel": "label", "payload": "0:T-shirt/top"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
