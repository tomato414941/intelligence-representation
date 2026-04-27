import unittest

from intrep.grid_world import (
    GridAction,
    GridWorld,
    GridWorldState,
    Position,
    generate_grid_world_corpus,
    observation_from_state,
    transition_state,
)


class GridWorldTest(unittest.TestCase):
    def test_step_returns_next_observation_after_action(self) -> None:
        world = GridWorld(
            GridWorldState(
                width=3,
                height=2,
                agent=Position(row=0, col=0),
                goal=Position(row=0, col=1),
            )
        )

        observation = world.step("right")

        self.assertEqual(world.hidden_state.agent, Position(row=0, col=1))
        self.assertEqual(observation.agent, Position(row=0, col=1))
        self.assertTrue(observation.reached_goal)
        self.assertEqual(observation.grid, (".*.", "..."))
        self.assertIn("after right", observation.text)

    def test_wall_collision_keeps_hidden_state_and_marks_observation_blocked(self) -> None:
        state = GridWorldState(
            width=3,
            height=3,
            agent=Position(row=1, col=0),
            goal=Position(row=2, col=2),
            walls=frozenset({Position(row=1, col=1)}),
        )

        next_state, blocked = transition_state(state, GridAction(direction="right"))
        observation = observation_from_state(next_state, last_action="right", blocked=blocked)

        self.assertEqual(next_state, state)
        self.assertTrue(blocked)
        self.assertTrue(observation.blocked)
        self.assertEqual(observation.agent, Position(row=1, col=0))
        self.assertEqual(observation.grid, ("...", "A#.", "..."))

    def test_observation_is_separate_from_hidden_state(self) -> None:
        state = GridWorldState(
            width=4,
            height=3,
            agent=Position(row=0, col=0),
            goal=Position(row=2, col=3),
            walls=frozenset({Position(row=1, col=1)}),
        )

        observation = observation_from_state(state)

        self.assertIsNot(observation, state)
        self.assertEqual(state.goal, Position(row=2, col=3))
        self.assertNotIn("G", "".join(observation.grid))
        self.assertNotIn("(2, 3)", observation.text)
        self.assertEqual(observation.grid, ("A...", ".#..", "...."))

    def test_generate_grid_world_corpus_records_action_conditioned_next_observations(
        self,
    ) -> None:
        state = GridWorldState(
            width=3,
            height=2,
            agent=Position(row=0, col=0),
            goal=Position(row=1, col=2),
            walls=frozenset({Position(row=1, col=1)}),
        )

        examples = generate_grid_world_corpus(
            actions=("right", "down", "right"),
            initial_state=state,
        )

        self.assertEqual(
            [example.action.direction for example in examples],
            ["right", "down", "right"],
        )
        self.assertEqual(examples[0].state_before.agent, Position(row=0, col=0))
        self.assertEqual(examples[0].state_after.agent, Position(row=0, col=1))
        self.assertEqual(examples[0].next_observation.agent, Position(row=0, col=1))
        self.assertTrue(examples[1].next_observation.blocked)
        self.assertEqual(examples[1].state_after.agent, Position(row=0, col=1))
        self.assertEqual(examples[2].next_observation.agent, Position(row=0, col=2))


if __name__ == "__main__":
    unittest.main()
