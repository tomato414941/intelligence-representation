import unittest

from intrep.worlds.cellular.world import (
    LIFE_RULE,
    CellularRule,
    CellularWorldState,
    generate_cellular_transitions,
    generate_random_cellular_rule,
    generate_random_cellular_state,
    step_cellular_state,
)


def _state(rows: list[str]) -> CellularWorldState:
    grid = tuple(tuple(1 if cell == "#" else 0 for cell in row) for row in rows)
    return CellularWorldState(width=len(rows[0]), height=len(rows), grid=grid)


class CellularRuleTest(unittest.TestCase):
    def test_rejects_neighbor_counts_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            CellularRule(birth=frozenset({9}), survival=frozenset())

    def test_random_rule_is_reproducible_and_excludes_zero_birth(self) -> None:
        first = generate_random_cellular_rule(7)
        again = generate_random_cellular_rule(7)
        self.assertEqual(first, again)
        for seed in range(20):
            self.assertNotIn(0, generate_random_cellular_rule(seed).birth)


class LifeStepTest(unittest.TestCase):
    def test_blinker_oscillates_with_period_two(self) -> None:
        vertical = _state(
            [
                ".....",
                "..#..",
                "..#..",
                "..#..",
                ".....",
            ]
        )
        horizontal = _state(
            [
                ".....",
                ".....",
                ".###.",
                ".....",
                ".....",
            ]
        )

        self.assertEqual(step_cellular_state(vertical, LIFE_RULE), horizontal)
        self.assertEqual(step_cellular_state(horizontal, LIFE_RULE), vertical)

    def test_block_is_still_life(self) -> None:
        block = _state(
            [
                "....",
                ".##.",
                ".##.",
                "....",
            ]
        )

        self.assertEqual(step_cellular_state(block, LIFE_RULE), block)

    def test_cells_outside_the_grid_count_as_dead(self) -> None:
        # A blinker touching the edge: the cell beyond the border cannot be
        # born, so the oscillation collapses instead of wrapping.
        edge_blinker = _state(
            [
                "#....",
                "#....",
                "#....",
            ]
        )
        after = step_cellular_state(edge_blinker, LIFE_RULE)

        self.assertEqual(
            after,
            _state(
                [
                    ".....",
                    "##...",
                    ".....",
                ]
            ),
        )


class CellularGenerationTest(unittest.TestCase):
    def test_random_state_is_reproducible(self) -> None:
        first = generate_random_cellular_state(width=6, height=4, seed=11)
        again = generate_random_cellular_state(width=6, height=4, seed=11)
        other = generate_random_cellular_state(width=6, height=4, seed=12)

        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_transitions_match_single_step(self) -> None:
        transitions = generate_cellular_transitions(
            LIFE_RULE, width=6, height=4, count=8, seed=100
        )

        self.assertEqual(len(transitions), 8)
        for transition in transitions:
            self.assertEqual(transition.rule, LIFE_RULE)
            self.assertEqual(
                transition.next_state,
                step_cellular_state(transition.state, LIFE_RULE),
            )


if __name__ == "__main__":
    unittest.main()
