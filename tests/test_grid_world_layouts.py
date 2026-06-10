import unittest

from intrep.domains.grid.world import (
    generate_grid_world_layouts,
    generate_grid_world_transition_table,
    generate_random_grid_world_state,
)


class GridWorldLayoutGenerationTest(unittest.TestCase):
    def test_generates_valid_layout(self) -> None:
        state = generate_random_grid_world_state(width=5, height=4, wall_count=3, seed=7)

        self.assertEqual(state.width, 5)
        self.assertEqual(state.height, 4)
        self.assertEqual(len(state.walls), 3)
        self.assertNotIn(state.goal, state.walls)
        self.assertNotIn(state.agent, state.walls)

    def test_same_seed_reproduces_layout_and_seeds_differ(self) -> None:
        first = generate_random_grid_world_state(width=5, height=4, wall_count=3, seed=7)
        again = generate_random_grid_world_state(width=5, height=4, wall_count=3, seed=7)
        other = generate_random_grid_world_state(width=5, height=4, wall_count=3, seed=8)

        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_rejects_wall_count_leaving_no_room(self) -> None:
        with self.assertRaises(ValueError):
            generate_random_grid_world_state(width=2, height=2, wall_count=3, seed=7)

    def test_generates_layout_batch_usable_for_transitions(self) -> None:
        layouts = generate_grid_world_layouts(8, width=5, height=4, wall_count=3, seed=100)

        self.assertEqual(len(layouts), 8)
        self.assertGreater(len(set(layouts)), 1)
        for layout in layouts:
            transitions = generate_grid_world_transition_table(layout)
            # 20 cells minus 3 walls leaves 17 agent cells, 5 actions each.
            self.assertEqual(len(transitions), 17 * 5)


if __name__ == "__main__":
    unittest.main()
