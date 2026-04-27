import unittest

from intrep.grid_typed_corpus import default_grid_typed_events
from intrep.typed_events import EventRole


class GridTypedCorpusTest(unittest.TestCase):
    def test_default_grid_typed_events_include_action_and_consequence(self) -> None:
        events = default_grid_typed_events()

        roles = {event.role for event in events}

        self.assertIn(EventRole.OBSERVATION, roles)
        self.assertIn(EventRole.ACTION, roles)
        self.assertIn(EventRole.CONSEQUENCE, roles)
        self.assertTrue(all(event.id for event in events))
        self.assertTrue(any(event.episode_id for event in events))


if __name__ == "__main__":
    unittest.main()
