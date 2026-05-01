import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import TensorDataset

from intrep.image_training_data import (
    channel_count_from_image_shape,
    image_tensor_from_path,
    seeded_data_loader,
)


class ImageTrainingDataTest(unittest.TestCase):
    def test_image_tensor_from_path_normalizes_pixels(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.pgm"
            path.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))

            image = image_tensor_from_path(path)

        self.assertEqual(image.shape, torch.Size([1, 2]))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(image.tolist(), [[0.0, 1.0]])

    def test_channel_count_from_image_shape(self) -> None:
        self.assertEqual(channel_count_from_image_shape((28, 28)), 1)
        self.assertEqual(channel_count_from_image_shape((32, 32, 3)), 3)

    def test_seeded_data_loader_uses_reproducible_shuffle(self) -> None:
        dataset = TensorDataset(torch.arange(6))
        first = [
            int(value)
            for batch in seeded_data_loader(dataset, batch_size=2, seed=7, shuffle=True, device=torch.device("cpu"))
            for value in batch[0].tolist()
        ]
        second = [
            int(value)
            for batch in seeded_data_loader(dataset, batch_size=2, seed=7, shuffle=True, device=torch.device("cpu"))
            for value in batch[0].tolist()
        ]

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
