import unittest
from io import StringIO
from unittest.mock import Mock
from unittest.mock import patch

import torch

from intrep.problems.shogi_policy_value.examples import ShogiMovePolicyValueExample
from tests.shogi_test_helpers import shogi_move_policy_value_examples_from_test_moves
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_POLICY_VALUE_ALPHA_ZERO_LIKE_POLICY_PLANE_ASSEMBLY_SPEC_ID,
    SHOGI_POLICY_VALUE_DLSHOGI_LIKE_POLICY_PLANE_ASSEMBLY_SPEC_ID,
    SHOGI_POLICY_VALUE_MINIMAL_SPLIT_GLOBAL_POLICY_PLANE_ASSEMBLY_SPEC_ID,
    SHOGI_POLICY_VALUE_MINIMAL_SINGLE_GLOBAL_POLICY_PLANE_ASSEMBLY_SPEC_ID,
    SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
    SHOGI_POLICY_VALUE_RICH_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID,
    SHOGI_POLICY_VALUE_RICH_POLICY_PLANE_ASSEMBLY_SPEC_ID,
)
import intrep.problems.shogi_policy_value.training as training
from intrep.problems.shogi_policy_value.training import (
    ShogiPolicyValueTrainingConfig,
    train_shogi_policy_value_model,
)


class ShogiPolicyValueTrainingTest(unittest.TestCase):
    def test_policy_target_loss_uses_soft_targets(self) -> None:
        uniform_loss = training._policy_target_loss(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[0.75, 0.25]]),
        )
        preferred_loss = training._policy_target_loss(
            torch.tensor([[2.0, 0.0]]),
            torch.tensor([[0.75, 0.25]]),
        )

        self.assertLess(float(preferred_loss.item()), float(uniform_loss.item()))

    def test_trains_for_one_step(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d", "2g2f"))

        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        self.assertEqual(result.metrics.train_case_count, 3)
        self.assertGreater(result.metrics.initial_loss, 0.0)
        self.assertGreater(result.metrics.final_loss, 0.0)
        self.assertGreaterEqual(result.metrics.top_3_accuracy, result.metrics.accuracy)
        self.assertGreaterEqual(result.metrics.top_5_accuracy, result.metrics.top_3_accuracy)
        self.assertGreater(result.metrics.mean_reciprocal_rank, 0.0)
        self.assertGreaterEqual(result.metrics.mean_correct_move_rank, 1.0)

    def test_trains_policy_plane_model_for_one_step(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d", "2g2f"))

        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                assembly_spec_id=SHOGI_POLICY_VALUE_RICH_POLICY_PLANE_ASSEMBLY_SPEC_ID,
            ),
        )

        self.assertEqual(result.metrics.train_case_count, 3)
        self.assertGreater(result.metrics.initial_loss, 0.0)
        self.assertGreater(result.metrics.final_loss, 0.0)
        self.assertGreaterEqual(result.metrics.top_3_accuracy, result.metrics.accuracy)
        self.assertGreaterEqual(result.metrics.top_5_accuracy, result.metrics.top_3_accuracy)

    def test_trains_alpha_zero_like_policy_plane_model_for_one_step(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d", "2g2f"))

        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                assembly_spec_id=SHOGI_POLICY_VALUE_ALPHA_ZERO_LIKE_POLICY_PLANE_ASSEMBLY_SPEC_ID,
            ),
        )

        self.assertEqual(result.metrics.train_case_count, 3)
        self.assertGreater(result.metrics.initial_loss, 0.0)
        self.assertGreater(result.metrics.final_loss, 0.0)

    def test_trains_dlshogi_like_policy_plane_model_for_one_step(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d", "2g2f"))

        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                assembly_spec_id=SHOGI_POLICY_VALUE_DLSHOGI_LIKE_POLICY_PLANE_ASSEMBLY_SPEC_ID,
            ),
        )

        self.assertEqual(result.metrics.train_case_count, 3)
        self.assertGreater(result.metrics.initial_loss, 0.0)
        self.assertGreater(result.metrics.final_loss, 0.0)

    def test_trains_minimal_split_global_policy_plane_model_for_one_step(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d", "2g2f"))

        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                assembly_spec_id=SHOGI_POLICY_VALUE_MINIMAL_SPLIT_GLOBAL_POLICY_PLANE_ASSEMBLY_SPEC_ID,
            ),
        )

        self.assertEqual(result.metrics.train_case_count, 3)
        self.assertGreater(result.metrics.initial_loss, 0.0)
        self.assertGreater(result.metrics.final_loss, 0.0)

    def test_trains_minimal_single_global_policy_plane_model_for_one_step(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d", "2g2f"))

        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                assembly_spec_id=SHOGI_POLICY_VALUE_MINIMAL_SINGLE_GLOBAL_POLICY_PLANE_ASSEMBLY_SPEC_ID,
            ),
        )

        self.assertEqual(result.metrics.train_case_count, 3)
        self.assertGreater(result.metrics.initial_loss, 0.0)
        self.assertGreater(result.metrics.final_loss, 0.0)

    def test_can_overfit_tiny_move_sequence(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))

        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=80,
                batch_size=2,
                learning_rate=0.02,
                embedding_dim=8,
                hidden_dim=16,
                assembly_spec_id=SHOGI_POLICY_VALUE_RICH_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID,
            ),
        )

        self.assertLess(result.metrics.final_loss, result.metrics.initial_loss)
        self.assertGreaterEqual(result.metrics.accuracy, 0.5)

    def test_limits_eval_examples(self) -> None:
        examples = tuple(
            shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d", "2g2f", "8c8d"))
            + shogi_move_policy_value_examples_from_test_moves(("2g2f", "8c8d", "2f2e", "8d8e"))
        )

        result = train_shogi_policy_value_model(
            examples,
            eval_examples=examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                max_train_eval_examples=3,
                max_eval_examples=2,
            ),
        )

        self.assertEqual(result.metrics.train_case_count, len(examples))
        self.assertEqual(result.metrics.eval_case_count, 2)

    def test_rejects_negative_num_workers(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f",))

        with self.assertRaisesRegex(ValueError, "num_workers"):
            train_shogi_policy_value_model(
                examples,
                config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                    max_steps=1,
                    num_workers=-1,
                ),
            )

    def test_rejects_zero_total_loss_weight(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f",))

        with self.assertRaisesRegex(ValueError, "at least one loss weight"):
            train_shogi_policy_value_model(
                examples,
                config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                    max_steps=1,
                    policy_loss_weight=0.0,
                    value_loss_weight=0.0,
                ),
            )

    def test_rejects_nonstandard_loss_weights_without_explicit_allowance(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f",))

        with self.assertRaisesRegex(ValueError, "allow_nonstandard_loss_weights"):
            train_shogi_policy_value_model(
                examples,
                config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                    max_steps=1,
                    value_loss_weight=0.5,
                ),
            )

    def test_progress_callback_runs_only_on_progress_interval(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        reported_steps: list[int] = []

        train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=3,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                progress_every=2,
            ),
            progress_callback=lambda progress: reported_steps.append(progress.step),
        )

        self.assertEqual(reported_steps, [2])

    def test_progress_callback_is_not_called_without_progress_interval(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        reported_steps: list[int] = []

        train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=3,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
            progress_callback=lambda progress: reported_steps.append(progress.step),
        )

        self.assertEqual(reported_steps, [])

    def test_phase_progress_callback_reports_evaluation_without_printing(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        phase_events: list[training.ShogiPolicyValuePhaseProgress] = []

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            train_shogi_policy_value_model(
                examples,
                eval_examples=examples,
                config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                    max_steps=1,
                    batch_size=1,
                    embedding_dim=8,
                    hidden_dim=16,
                    num_heads=2,
                    log_every=1,
                ),
                phase_progress_callback=phase_events.append,
            )

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(("initial_train_eval", "start"), [(event.phase, event.event) for event in phase_events])
        self.assertIn(("initial_train_eval", "progress"), [(event.phase, event.event) for event in phase_events])
        self.assertIn(("initial_train_eval", "done"), [(event.phase, event.event) for event in phase_events])
        self.assertIn(("initial_eval", "start"), [(event.phase, event.event) for event in phase_events])
        self.assertTrue(all(event.elapsed_seconds >= 0.0 for event in phase_events))

    def test_early_stopping_stops_after_eval_patience(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))

        result = train_shogi_policy_value_model(
            examples,
            eval_examples=examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=5,
                batch_size=2,
                learning_rate=0.0,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                eval_every=1,
                early_stopping_patience=1,
            ),
        )

        self.assertTrue(result.metrics.stopped_early)
        self.assertEqual(result.metrics.actual_steps, 1)
        self.assertEqual(result.metrics.stopped_step, 1)
        self.assertEqual(result.metrics.early_stopping_patience, 1)

    def test_early_stopping_requires_eval_every(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f",))

        with self.assertRaisesRegex(ValueError, "eval_every"):
            train_shogi_policy_value_model(
                examples,
                eval_examples=examples,
                config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                    max_steps=1,
                    early_stopping_patience=1,
                ),
            )

    def test_trains_value_head_when_targets_are_available(self) -> None:
        base_examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        examples = tuple(
            ShogiMovePolicyValueExample(
                position_sfen=example.position_sfen,
                legal_moves=example.legal_moves,
                chosen_move=example.chosen_move,
                value_target=1.0 if index == 0 else -1.0,
            )
            for index, example in enumerate(base_examples)
        )

        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=2,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                value_loss_weight=0.1,
                allow_nonstandard_loss_weights=True,
            ),
        )

        self.assertIsNotNone(result.metrics.initial_value_loss)
        self.assertIsNotNone(result.metrics.value_loss)

    def test_policy_and_value_can_improve_on_tiny_game_set(self) -> None:
        examples = tuple(
            shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d", "2g2f", "8c8d"))
            + shogi_move_policy_value_examples_from_test_moves(("2g2f", "8c8d", "2f2e", "8d8e"))
        )
        valued_examples = tuple(
            ShogiMovePolicyValueExample(
                position_sfen=example.position_sfen,
                legal_moves=example.legal_moves,
                chosen_move=example.chosen_move,
                value_target=1.0 if index % 2 == 0 else -1.0,
            )
            for index, example in enumerate(examples)
        )

        result = train_shogi_policy_value_model(
            valued_examples,
            eval_examples=valued_examples[:2],
            config=ShogiPolicyValueTrainingConfig(
                max_steps=80,
                batch_size=4,
                learning_rate=0.02,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                assembly_spec_id=SHOGI_POLICY_VALUE_RICH_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID,
                value_loss_weight=0.2,
                allow_nonstandard_loss_weights=True,
            ),
        )

        self.assertLess(result.metrics.final_loss, result.metrics.initial_loss)
        self.assertIsNotNone(result.metrics.initial_value_loss)
        self.assertIsNotNone(result.metrics.value_loss)
        self.assertLess(result.metrics.value_loss, result.metrics.initial_value_loss)
        self.assertEqual(result.metrics.eval_case_count, 2)
        self.assertIsNotNone(result.metrics.initial_eval_loss)
        self.assertIsNotNone(result.metrics.eval_loss)
        self.assertIsNotNone(result.metrics.initial_eval_accuracy)
        self.assertIsNotNone(result.metrics.initial_eval_value_loss)
        self.assertIsNotNone(result.metrics.eval_value_loss)

    def test_value_only_can_improve_on_tiny_game_set(self) -> None:
        base_examples = tuple(
            shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d", "2g2f", "8c8d"))
            + shogi_move_policy_value_examples_from_test_moves(("2g2f", "8c8d", "2f2e", "8d8e"))
        )
        examples = tuple(
            ShogiMovePolicyValueExample(
                position_sfen=example.position_sfen,
                legal_moves=example.legal_moves,
                chosen_move=example.chosen_move,
                value_target=1.0 if index % 2 == 0 else -1.0,
            )
            for index, example in enumerate(base_examples)
        )

        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=80,
                batch_size=4,
                learning_rate=0.02,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                assembly_spec_id=SHOGI_POLICY_VALUE_RICH_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID,
                policy_loss_weight=0.0,
                value_loss_weight=1.0,
                allow_nonstandard_loss_weights=True,
            ),
        )

        self.assertEqual(result.config.policy_loss_weight, 0.0)
        self.assertIsNotNone(result.metrics.initial_value_loss)
        self.assertIsNotNone(result.metrics.value_loss)
        self.assertLess(result.metrics.value_loss, result.metrics.initial_value_loss)

    def test_value_only_training_step_skips_policy_forward(self) -> None:
        base_examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        examples = tuple(
            ShogiMovePolicyValueExample(
                position_sfen=example.position_sfen,
                legal_moves=example.legal_moves,
                chosen_move=example.chosen_move,
                value_target=1.0,
            )
            for example in base_examples
        )
        model = training.build_shogi_policy_value_model(
            ShogiPolicyValueTrainingConfig(
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                num_layers=1,
                assembly_spec_id=SHOGI_POLICY_VALUE_RICH_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID,
            )
        )
        model.forward_policy_value = Mock(wraps=model.forward_policy_value)
        model.predict_value = Mock(wraps=model.predict_value)

        original_build_model = training.build_shogi_policy_value_model
        training.build_shogi_policy_value_model = lambda config: model
        try:
            train_shogi_policy_value_model(
                examples,
                config=ShogiPolicyValueTrainingConfig(
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                assembly_spec_id=SHOGI_POLICY_VALUE_RICH_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID,
                policy_loss_weight=0.0,
                value_loss_weight=1.0,
                allow_nonstandard_loss_weights=True,
                    max_train_eval_examples=1,
                ),
            )
        finally:
            training.build_shogi_policy_value_model = original_build_model

        self.assertGreaterEqual(model.predict_value.call_count, 1)
        self.assertEqual(model.forward_policy_value.call_count, 2)


if __name__ == "__main__":
    unittest.main()
