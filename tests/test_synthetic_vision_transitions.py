import unittest

import numpy as np

from intrep.sources.vision.synthetic_transitions import (
    MOVING_DOT_ACTIONS,
    MovingDotTransitionDataset,
    moving_dot_frame,
    moving_dot_next_position,
)


class SyntheticVisionTransitionsTest(unittest.TestCase):
    def test_moving_dot_action_names_are_stable(self) -> None:
        self.assertEqual(MOVING_DOT_ACTIONS, ("stay", "up", "down", "left", "right"))

    def test_moving_dot_next_position_clamps_at_image_boundary(self) -> None:
        self.assertEqual(moving_dot_next_position(np.array([0, 0]), 1, image_size=4).tolist(), [0, 0])
        self.assertEqual(moving_dot_next_position(np.array([0, 0]), 3, image_size=4).tolist(), [0, 0])
        self.assertEqual(moving_dot_next_position(np.array([3, 3]), 2, image_size=4).tolist(), [3, 3])
        self.assertEqual(moving_dot_next_position(np.array([3, 3]), 4, image_size=4).tolist(), [3, 3])

    def test_moving_dot_frame_marks_one_pixel(self) -> None:
        frame = moving_dot_frame(np.array([1, 2]), image_size=4)

        self.assertEqual(frame.shape, (1, 4, 4))
        self.assertEqual(float(frame.sum()), 1.0)
        self.assertEqual(float(frame[0, 1, 2]), 1.0)

    def test_dataset_returns_reproducible_transitions(self) -> None:
        first = MovingDotTransitionDataset(sample_count=8, image_size=6, seed=11)
        second = MovingDotTransitionDataset(sample_count=8, image_size=6, seed=11)

        self.assertEqual(len(first), 8)
        for index in range(len(first)):
            np.testing.assert_array_equal(first[index].frame, second[index].frame)
            self.assertEqual(first[index].action, second[index].action)
            np.testing.assert_array_equal(first[index].next_frame, second[index].next_frame)

    def test_dataset_next_frame_matches_action(self) -> None:
        dataset = MovingDotTransitionDataset(sample_count=16, image_size=5, seed=19)

        for sample in dataset:
            current_position = np.argwhere(sample.frame[0] > 0)[0]
            next_position = np.argwhere(sample.next_frame[0] > 0)[0]
            expected = moving_dot_next_position(current_position, sample.action, image_size=5)
            self.assertEqual(next_position.tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
