import unittest

from intrep.grid_corpus import (
    GridEpisode,
    GridTransition,
    build_default_grid_corpus,
    build_grid_corpus,
    default_grid_documents,
    episode_to_mixed_documents,
    episodes_to_mixed_documents,
)
from intrep.mixed_corpus import render_corpus


class GridCorpusTest(unittest.TestCase):
    def test_episode_to_mixed_documents_creates_action_conditioned_step_docs(self) -> None:
        episode = {
            "id": "episode one",
            "transitions": [
                {
                    "text": "The agent sees a key to the east.",
                    "grid": ("###", "#A#", "#K#"),
                    "action": "move_south",
                    "next_grid": ("###", "#.#", "#A#"),
                    "next_text": "The agent moves onto the key cell.",
                }
            ],
        }

        documents = episode_to_mixed_documents(episode)

        self.assertEqual(
            [(document.id, document.modality) for document in documents],
            [
                ("episode_one_step_001_text", "text"),
                ("episode_one_step_001_grid", "grid"),
                ("episode_one_step_001_action_log", "action_log"),
                ("episode_one_step_001_next_grid", "next_grid"),
                ("episode_one_step_001_next_text", "next_text"),
            ],
        )
        self.assertIn("The agent sees a key", documents[0].content)
        self.assertEqual(documents[1].content, "<grid>\n###\n#A#\n#K#\n</grid>")
        self.assertEqual(documents[2].content, "<action> move_south")
        self.assertEqual(documents[3].content, "<next_grid>\n###\n#.#\n#A#\n</next_grid>")
        self.assertIn("moves onto the key", documents[4].content)

    def test_episode_to_mixed_documents_accepts_dataclass_episode(self) -> None:
        episode = GridEpisode(
            id="grid_episode",
            transitions=[
                GridTransition(
                    text="The agent is left of the goal.",
                    grid=(("#", "#", "#"), ("#", "A", "G"), ("#", "#", "#")),
                    action={"type": "move", "direction": "east"},
                    next_grid=(("#", "#", "#"), ("#", ".", "A"), ("#", "#", "#")),
                    next_text="The agent moves east.",
                )
            ],
        )

        documents = episode_to_mixed_documents(episode)

        self.assertEqual(documents[1].modality, "grid")
        self.assertIn("#AG", documents[1].content)
        self.assertEqual(documents[2].modality, "action_log")
        self.assertIn("direction=east", documents[2].content)
        self.assertIn("type=move", documents[2].content)
        self.assertIn("#.A", documents[3].content)

    def test_episodes_to_mixed_documents_flattens_multiple_episodes(self) -> None:
        episodes = [
            GridEpisode(
                id="first",
                transitions=[
                    GridTransition("before", ("A.",), "right", (".A",), "after"),
                ],
            ),
            GridEpisode(
                id="second",
                transitions=[
                    GridTransition("before", ("A#",), "right", ("A#",), "blocked"),
                ],
            ),
        ]

        documents = episodes_to_mixed_documents(episodes)

        self.assertEqual(len(documents), 10)
        self.assertEqual(documents[0].id, "first_step_001_text")
        self.assertEqual(documents[5].id, "second_step_001_text")

    def test_default_grid_documents_include_nonverbal_grid_observations(self) -> None:
        documents = default_grid_documents()
        modalities = {document.modality for document in documents}
        rendered = render_corpus(documents)

        self.assertEqual(build_default_grid_corpus(), documents)
        self.assertEqual(build_grid_corpus(), documents)
        self.assertEqual({"text", "grid", "action_log", "next_grid", "next_text"}, modalities)
        self.assertIn("<doc type=grid", rendered)
        self.assertIn("<grid>", rendered)
        self.assertIn("<action>", rendered)
        self.assertIn("<next_grid>", rendered)
        self.assertIn("A...", rendered)
        self.assertIn(".#..", rendered)

    def test_episode_to_mixed_documents_validates_minimal_transition_shape(self) -> None:
        episode = {
            "id": "broken",
            "transitions": [
                {
                    "text": "missing next text",
                    "grid": ("A",),
                    "action": "wait",
                    "next_grid": ("A",),
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "next_text"):
            episode_to_mixed_documents(episode)


if __name__ == "__main__":
    unittest.main()
