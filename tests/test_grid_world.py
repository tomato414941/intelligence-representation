import unittest

from intrep.grid_world import (
    GridAction,
    GridExperienceTransition,
    GridWorld,
    GridWorldState,
    Position,
    generate_grid_world_experience,
    grid_experience_transition_to_text,
    language_modeling_examples_from_grid_experience,
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

        result = world.step("right")

        self.assertEqual(world.hidden_state.agent, Position(row=0, col=1))
        self.assertEqual(result.observation.agent, Position(row=0, col=1))
        self.assertTrue(result.observation.reached_goal)
        self.assertEqual(result.observation.grid, (".*.", "..."))
        self.assertIn("after right", result.observation.text)
        self.assertIn("goal at (0, 1)", result.observation.text)
        self.assertEqual(result.reward, 1.0)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)

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
        self.assertEqual(observation.grid, ("...", "A#.", "..G"))

    def test_observation_is_fully_observed_but_separate_from_hidden_state(self) -> None:
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
        self.assertIn("G", "".join(observation.grid))
        self.assertIn("goal at (2, 3)", observation.text)
        self.assertEqual(observation.grid, ("A...", ".#..", "...G"))

    def test_generate_grid_world_experience_records_action_conditioned_next_observations(
        self,
    ) -> None:
        state = GridWorldState(
            width=3,
            height=2,
            agent=Position(row=0, col=0),
            goal=Position(row=1, col=2),
            walls=frozenset({Position(row=1, col=1)}),
        )

        examples = generate_grid_world_experience(
            actions=("right", "down", "right"),
            initial_state=state,
        )

        self.assertIsInstance(examples[0], GridExperienceTransition)
        self.assertEqual(
            [example.action.direction for example in examples],
            ["right", "down", "right"],
        )
        self.assertEqual(examples[0].observation.agent, Position(row=0, col=0))
        self.assertEqual(examples[0].next_observation.agent, Position(row=0, col=1))
        self.assertTrue(examples[1].next_observation.blocked)
        self.assertEqual(examples[1].reward, -0.1)
        self.assertFalse(examples[1].terminated)
        self.assertEqual(examples[2].next_observation.agent, Position(row=0, col=2))

    def test_grid_experience_transition_to_text_renders_prediction_task(self) -> None:
        state = GridWorldState(
            width=3,
            height=2,
            agent=Position(row=0, col=0),
            goal=Position(row=1, col=2),
            walls=frozenset({Position(row=1, col=1)}),
        )
        example = generate_grid_world_experience(actions=("right",), initial_state=state)[0]

        text = grid_experience_transition_to_text(example)

        self.assertEqual(
            text,
            "\n".join(
                (
                    "<obs>",
                    "A..",
                    ".#G",
                    "<action> right",
                    "<next_obs>",
                    ".A.",
                    ".#G",
                    "<reward> -0.01",
                    "<terminated> false",
                    "<truncated> false",
                )
            ),
        )

    def test_language_modeling_examples_from_grid_experience(self) -> None:
        examples = generate_grid_world_experience(actions=("right", "down"))

        text_examples = language_modeling_examples_from_grid_experience(examples)

        self.assertEqual(len(text_examples), 2)
        self.assertIn("<action> right", text_examples[0].text)
        self.assertIn("<next_obs>", text_examples[0].text)

    def test_language_modeling_examples_from_grid_experience_rejects_empty_examples(self) -> None:
        with self.assertRaisesRegex(ValueError, "examples must not be empty"):
            language_modeling_examples_from_grid_experience(())


if __name__ == "__main__":
    unittest.main()
