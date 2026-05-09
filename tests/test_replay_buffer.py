import unittest

import torch

from intrep.learning.replay_buffer import ReplayBuffer


class ReplayBufferTest(unittest.TestCase):
    def test_appends_and_samples_items(self) -> None:
        buffer = ReplayBuffer[int](capacity=4)
        buffer.extend([1, 2, 3])
        generator = torch.Generator().manual_seed(0)

        sample = buffer.sample(2, generator=generator)

        self.assertEqual(len(buffer), 3)
        self.assertEqual(len(sample), 2)
        self.assertTrue(set(sample).issubset({1, 2, 3}))
        self.assertEqual(len(set(sample)), 2)

    def test_capacity_drops_oldest_items(self) -> None:
        buffer = ReplayBuffer[int](capacity=3)
        buffer.extend([1, 2, 3, 4])

        sample = buffer.sample(3, generator=torch.Generator().manual_seed(0))

        self.assertEqual(set(sample), {2, 3, 4})

    def test_rejects_invalid_capacity_and_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "capacity must be positive"):
            ReplayBuffer[int](capacity=0)

        buffer = ReplayBuffer[int](capacity=2)
        buffer.append(1)
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            buffer.sample(0)
        with self.assertRaisesRegex(ValueError, "batch_size must not exceed"):
            buffer.sample(2)


if __name__ == "__main__":
    unittest.main()
