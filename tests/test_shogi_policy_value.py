import unittest

import shogi
import torch
from torch.utils.data import DataLoader

from intrep.tasks.shogi_policy_value.examples import (
    ShogiMoveChoiceDataset,
    ShogiMoveChoiceExample,
    ShogiPolicyValueDataset,
    ShogiPolicyValueExample,
    ShogiPositionValueExample,
    shogi_move_choice_example_from_board,
)
from tests.shogi_test_helpers import shogi_move_choice_examples_from_test_moves, shogi_policy_value_examples_from_test_moves
from intrep.worlds.shogi.move_encoding import SHOGI_MOVE_FEATURE_COUNT
from intrep.worlds.shogi.position_encoding import SHOGI_POSITION_TOKEN_COUNT


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
        examples = shogi_policy_value_examples_from_test_moves(("7g7f", "3c3d"), winner="black")

        self.assertEqual(examples[0].value_target, 1.0)
        self.assertEqual(examples[1].value_target, -1.0)

    def test_dataset_returns_candidate_mask_and_label_index(self) -> None:
        examples = shogi_move_choice_examples_from_test_moves(("7g7f", "3c3d"))
        dataset = ShogiMoveChoiceDataset(examples)

        position_token_ids, candidate_move_features, candidate_mask, label_index, policy_targets = dataset[0]

        self.assertEqual(tuple(position_token_ids.shape), (SHOGI_POSITION_TOKEN_COUNT,))
        self.assertEqual(tuple(candidate_move_features.shape), (len(examples[0].legal_moves), SHOGI_MOVE_FEATURE_COUNT))
        self.assertEqual(candidate_mask.dtype, torch.bool)
        self.assertEqual(int(candidate_mask.sum().item()), len(examples[0].legal_moves))
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

    def test_policy_value_dataset_returns_optional_value_target(self) -> None:
        board = shogi.Board()
        example = ShogiPolicyValueExample(
            position_sfen=board.sfen(),
            legal_moves=tuple(sorted(move.usi() for move in board.legal_moves)),
            chosen_move="7g7f",
            value_target=1.0,
        )
        dataset = ShogiPolicyValueDataset((example,))

        *_, value_target = dataset[0]

        self.assertEqual(float(value_target.item()), 1.0)

    def test_position_value_example_validates_value_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "value_target"):
            ShogiPositionValueExample(position_sfen=shogi.Board().sfen(), value_target=2.0)

    def test_dataset_can_be_batched(self) -> None:
        examples = shogi_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        loader = DataLoader(ShogiPolicyValueDataset(examples), batch_size=2)

        position_token_ids, candidate_move_features, candidate_masks, label_indexes, policy_targets, value_targets = next(iter(loader))

        self.assertEqual(tuple(position_token_ids.shape), (2, SHOGI_POSITION_TOKEN_COUNT))
        self.assertEqual(tuple(candidate_move_features.shape), (2, len(examples[0].legal_moves), SHOGI_MOVE_FEATURE_COUNT))
        self.assertEqual(tuple(candidate_masks.shape), (2, len(examples[0].legal_moves)))
        self.assertEqual(tuple(label_indexes.shape), (2,))
        self.assertEqual(tuple(policy_targets.shape), (2, len(examples[0].legal_moves)))
        self.assertEqual(tuple(value_targets.shape), (2,))


if __name__ == "__main__":
    unittest.main()
