import unittest

import shogi
import torch
from torch.utils.data import DataLoader

from intrep.problems.shogi_policy_value.examples import (
    ShogiMoveChoiceDataset,
    ShogiMoveChoiceExample,
    ShogiLegalMoveTokenPolicyValueDataset,
    ShogiMovePolicyValueExample,
    ShogiPositionValueExample,
    collate_legal_move_token_policy_value_samples,
    shogi_move_choice_example_from_board,
    tensorize_legal_move_token_policy_value_example,
    tensorize_compact_policy_plane_value_example,
    tensorize_policy_plane_value_example,
)
from tests.shogi_test_helpers import shogi_move_choice_examples_from_test_moves, shogi_move_policy_value_examples_from_test_moves
from intrep.worlds.shogi.move_encoding import SHOGI_MOVE_FEATURE_COUNT, shogi_move_feature_ids
from intrep.worlds.shogi.policy_plane import SHOGI_POLICY_PLANE_ACTION_COUNT, shogi_policy_plane_action_index
from intrep.worlds.shogi.position_encoding import (
    SHOGI_POSITION_GLOBAL_SLOT_COUNT,
    SHOGI_POSITION_LINE_FEATURE_COUNT,
    SHOGI_POSITION_LINE_SLOT_COUNT,
    SHOGI_POSITION_PIECE_FEATURE_COUNT,
    SHOGI_POSITION_PIECE_SLOT_COUNT,
    SHOGI_POSITION_SQUARE_COUNT,
    SHOGI_POSITION_SQUARE_FEATURE_COUNT,
)


class ShogiMoveChoiceExampleTest(unittest.TestCase):
    def test_requires_chosen_move_in_legal_moves(self) -> None:
        with self.assertRaises(ValueError):
            ShogiMoveChoiceExample(
                position_sfen=shogi.Board().sfen(),
                legal_moves=("7g7f",),
                chosen_move="2b2c",
            )

    def test_builds_example_from_board(self) -> None:
        board = shogi.Board()
        example = shogi_move_choice_example_from_board(board, "7g7f")

        self.assertEqual(example.position_sfen, board.sfen())
        self.assertIn("7g7f", example.legal_moves)
        self.assertEqual(example.chosen_move, "7g7f")

    def test_builds_examples_from_usi_moves(self) -> None:
        examples = shogi_move_choice_examples_from_test_moves(("7g7f", "3c3d"))

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].chosen_move, "7g7f")
        self.assertEqual(examples[1].chosen_move, "3c3d")
        self.assertIn("3c3d", examples[1].legal_moves)

    def test_builds_policy_value_targets_from_winner(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"), winner="black")

        self.assertEqual(examples[0].value_target, 1.0)
        self.assertEqual(examples[1].value_target, -1.0)

    def test_dataset_returns_legal_move_token_mask_and_label_index(self) -> None:
        examples = shogi_move_choice_examples_from_test_moves(("7g7f", "3c3d"))
        dataset = ShogiMoveChoiceDataset(examples)

        position_features, legal_move_token_features, legal_move_token_mask, label_index, policy_targets = dataset[0]

        self.assertEqual(tuple(position_features.global_feature_ids.shape), (SHOGI_POSITION_GLOBAL_SLOT_COUNT,))
        self.assertEqual(
            tuple(position_features.square_feature_ids.shape),
            (SHOGI_POSITION_SQUARE_COUNT, SHOGI_POSITION_SQUARE_FEATURE_COUNT),
        )
        self.assertEqual(tuple(legal_move_token_features.shape), (len(examples[0].legal_moves), SHOGI_MOVE_FEATURE_COUNT))
        self.assertEqual(legal_move_token_mask.dtype, torch.bool)
        self.assertEqual(int(legal_move_token_mask.sum().item()), len(examples[0].legal_moves))
        self.assertEqual(int(label_index.item()), examples[0].legal_moves.index("7g7f"))
        self.assertEqual(float(policy_targets[label_index].item()), 1.0)

    def test_dataset_returns_policy_targets_when_available(self) -> None:
        board = shogi.Board()
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        example = ShogiMoveChoiceExample(
            position_sfen=board.sfen(),
            legal_moves=legal_moves,
            chosen_move="7g7f",
            policy_targets={"7g7f": 3.0, "2g2f": 1.0},
        )
        dataset = ShogiMoveChoiceDataset((example,))

        *_, policy_targets = dataset[0]

        self.assertEqual(float(policy_targets[legal_moves.index("7g7f")].item()), 0.75)
        self.assertEqual(float(policy_targets[legal_moves.index("2g2f")].item()), 0.25)

    def test_policy_targets_and_labels_follow_legal_move_order(self) -> None:
        board = shogi.Board()
        board.remove_piece_at(shogi.SQUARE_NAMES.index("7g"))
        board.remove_piece_at(shogi.SQUARE_NAMES.index("3c"))
        board.add_piece_into_hand(shogi.BISHOP, shogi.BLACK)
        legal_moves = ("B*5e", "8h2b+", "2g2f", "7i7h")
        self.assertTrue(set(legal_moves).issubset({move.usi() for move in board.legal_moves}))
        example = ShogiMoveChoiceExample(
            position_sfen=board.sfen(),
            legal_moves=legal_moves,
            chosen_move="7i7h",
            policy_targets={"7i7h": 6.0, "B*5e": 2.0},
        )
        dataset = ShogiMoveChoiceDataset((example,))

        _, legal_move_token_features, legal_move_token_mask, label_index, policy_targets = dataset[0]

        self.assertEqual(int(label_index.item()), 3)
        self.assertEqual(int(legal_move_token_mask.sum().item()), len(legal_moves))
        self.assertTrue(torch.equal(legal_move_token_features[0], shogi_move_feature_ids("B*5e", turn=board.turn)))
        self.assertTrue(torch.equal(legal_move_token_features[1], shogi_move_feature_ids("8h2b+", turn=board.turn)))
        self.assertEqual(float(policy_targets[0].item()), 0.25)
        self.assertEqual(float(policy_targets[3].item()), 0.75)

    def test_policy_value_dataset_rejects_missing_non_chosen_policy_targets(self) -> None:
        board = shogi.Board()
        example = ShogiMovePolicyValueExample(
            position_sfen=board.sfen(),
            legal_moves=tuple(sorted(move.usi() for move in board.legal_moves)),
            chosen_move="7g7f",
            policy_target_source="mcts_visit_counts",
        )
        dataset = ShogiLegalMoveTokenPolicyValueDataset((example,))

        with self.assertRaisesRegex(ValueError, "missing policy_targets"):
            dataset[0]

    def test_policy_value_dataset_returns_optional_value_target(self) -> None:
        board = shogi.Board()
        example = ShogiMovePolicyValueExample(
            position_sfen=board.sfen(),
            legal_moves=tuple(sorted(move.usi() for move in board.legal_moves)),
            chosen_move="7g7f",
            value_target=1.0,
        )
        dataset = ShogiLegalMoveTokenPolicyValueDataset((example,))

        sample = dataset[0]

        self.assertEqual(float(sample.value_target.item()), 1.0)

    def test_position_value_example_validates_value_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "value_target"):
            ShogiPositionValueExample(position_sfen=shogi.Board().sfen(), value_target=2.0)

    def test_dataset_can_be_batched(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        loader = DataLoader(
            ShogiLegalMoveTokenPolicyValueDataset(examples),
            batch_size=2,
            collate_fn=collate_legal_move_token_policy_value_samples,
        )

        batch = next(iter(loader))

        self.assertEqual(tuple(batch.position_features.global_feature_ids.shape), (2, SHOGI_POSITION_GLOBAL_SLOT_COUNT))
        self.assertEqual(tuple(batch.legal_move_token_features.shape), (2, len(examples[0].legal_moves), SHOGI_MOVE_FEATURE_COUNT))
        self.assertEqual(tuple(batch.legal_move_token_mask.shape), (2, len(examples[0].legal_moves)))
        self.assertEqual(tuple(batch.labels.shape), (2,))
        self.assertEqual(tuple(batch.policy_targets.shape), (2, len(examples[0].legal_moves)))
        self.assertEqual(tuple(batch.value_targets.shape), (2,))

    def test_policy_value_dataset_accepts_tensorized_samples(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        samples = [tensorize_legal_move_token_policy_value_example(example) for example in examples]
        loader = DataLoader(
            ShogiLegalMoveTokenPolicyValueDataset(samples),
            batch_size=2,
            collate_fn=collate_legal_move_token_policy_value_samples,
        )

        batch = next(iter(loader))

        self.assertEqual(tuple(batch.position_features.global_feature_ids.shape), (2, SHOGI_POSITION_GLOBAL_SLOT_COUNT))
        self.assertEqual(tuple(batch.legal_move_token_features.shape), (2, len(examples[0].legal_moves), SHOGI_MOVE_FEATURE_COUNT))
        self.assertEqual(tuple(batch.legal_move_token_mask.shape), (2, len(examples[0].legal_moves)))
        self.assertEqual(tuple(batch.labels.shape), (2,))
        self.assertEqual(tuple(batch.policy_targets.shape), (2, len(examples[0].legal_moves)))
        self.assertEqual(tuple(batch.value_targets.shape), (2,))

    def test_policy_plane_sample_maps_chosen_move_to_fixed_action_target(self) -> None:
        board = shogi.Board()
        example = ShogiMovePolicyValueExample(
            position_sfen=board.sfen(),
            legal_moves=tuple(sorted(move.usi() for move in board.legal_moves)),
            chosen_move="7g7f",
            value_target=1.0,
        )

        sample = tensorize_policy_plane_value_example(example)
        action_index = shogi_policy_plane_action_index("7g7f", turn=board.turn)

        self.assertEqual(tuple(sample.position_features.global_feature_ids.shape), (SHOGI_POSITION_GLOBAL_SLOT_COUNT,))
        self.assertEqual(
            tuple(sample.position_features.piece_feature_ids.shape),
            (SHOGI_POSITION_PIECE_SLOT_COUNT, SHOGI_POSITION_PIECE_FEATURE_COUNT),
        )
        self.assertEqual(
            tuple(sample.position_features.line_feature_ids.shape),
            (SHOGI_POSITION_LINE_SLOT_COUNT, SHOGI_POSITION_LINE_FEATURE_COUNT),
        )
        self.assertEqual(tuple(sample.policy_plane_targets.shape), (SHOGI_POLICY_PLANE_ACTION_COUNT,))
        self.assertEqual(tuple(sample.policy_plane_legal_mask.shape), (SHOGI_POLICY_PLANE_ACTION_COUNT,))
        self.assertEqual(int(sample.policy_plane_label.item()), action_index)
        self.assertEqual(float(sample.policy_plane_targets[action_index].item()), 1.0)
        self.assertEqual(float(sample.policy_plane_targets.sum().item()), 1.0)
        self.assertTrue(bool(sample.policy_plane_legal_mask[action_index].item()))
        self.assertEqual(float(sample.value_target.item()), 1.0)

    def test_compact_policy_plane_sample_uses_sparse_targets(self) -> None:
        board = shogi.Board()
        example = ShogiMovePolicyValueExample(
            position_sfen=board.sfen(),
            legal_moves=tuple(sorted(move.usi() for move in board.legal_moves)),
            chosen_move="7g7f",
            policy_targets={"7g7f": 3.0, "2g2f": 1.0},
        )

        sample = tensorize_compact_policy_plane_value_example(example)
        first_index = shogi_policy_plane_action_index("7g7f", turn=board.turn)
        second_index = shogi_policy_plane_action_index("2g2f", turn=board.turn)

        self.assertIn(first_index, set(int(index.item()) for index in sample.legal_action_indices))
        self.assertEqual(tuple(int(index.item()) for index in sample.target_action_indices), (first_index, second_index))
        self.assertEqual(tuple(float(weight.item()) for weight in sample.target_weights), (0.75, 0.25))

    def test_policy_plane_sample_maps_policy_targets_to_fixed_action_space(self) -> None:
        board = shogi.Board()
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        example = ShogiMovePolicyValueExample(
            position_sfen=board.sfen(),
            legal_moves=legal_moves,
            chosen_move="7g7f",
            policy_targets={"7g7f": 3.0, "2g2f": 1.0},
        )

        sample = tensorize_policy_plane_value_example(example)
        first_index = shogi_policy_plane_action_index("7g7f", turn=board.turn)
        second_index = shogi_policy_plane_action_index("2g2f", turn=board.turn)

        self.assertEqual(float(sample.policy_plane_targets[first_index].item()), 0.75)
        self.assertEqual(float(sample.policy_plane_targets[second_index].item()), 0.25)
        self.assertEqual(float(sample.policy_plane_targets.sum().item()), 1.0)

    def test_policy_plane_sample_rejects_missing_non_chosen_policy_targets(self) -> None:
        board = shogi.Board()
        example = ShogiMovePolicyValueExample(
            position_sfen=board.sfen(),
            legal_moves=tuple(sorted(move.usi() for move in board.legal_moves)),
            chosen_move="7g7f",
            policy_target_source="mcts_visit_counts",
        )

        with self.assertRaisesRegex(ValueError, "missing policy_targets"):
            tensorize_policy_plane_value_example(example)


if __name__ == "__main__":
    unittest.main()
