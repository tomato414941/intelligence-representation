import unittest

from intrep.grid_signal_corpus import default_grid_signals


class GridSignalCorpusTest(unittest.TestCase):
    def test_default_grid_signals_include_action_and_consequence(self) -> None:
        events = default_grid_signals()

        roles = {event.channel for event in events}

        self.assertIn("observation", roles)
        self.assertIn("action", roles)
        self.assertIn("consequence", roles)
        self.assertTrue(all(event.payload for event in events))


if __name__ == "__main__":
    unittest.main()
