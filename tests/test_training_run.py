import unittest

from intrep.core.training_run import BestMetricTracker


class BestMetricTrackerTest(unittest.TestCase):
    def test_tracks_min_metric(self) -> None:
        tracker = BestMetricTracker(mode="min")

        self.assertTrue(tracker.update(step=1, value=2.0))
        self.assertFalse(tracker.update(step=2, value=2.5))
        self.assertTrue(tracker.update(step=3, value=1.5))

        self.assertIsNotNone(tracker.best)
        assert tracker.best is not None
        self.assertEqual(tracker.best.step, 3)
        self.assertEqual(tracker.best.value, 1.5)

    def test_tracks_max_metric(self) -> None:
        tracker = BestMetricTracker(mode="max")

        self.assertTrue(tracker.update(step=1, value=0.2))
        self.assertFalse(tracker.update(step=2, value=0.1))
        self.assertTrue(tracker.update(step=3, value=0.4))

        self.assertIsNotNone(tracker.best)
        assert tracker.best is not None
        self.assertEqual(tracker.best.step, 3)
        self.assertEqual(tracker.best.value, 0.4)


if __name__ == "__main__":
    unittest.main()
