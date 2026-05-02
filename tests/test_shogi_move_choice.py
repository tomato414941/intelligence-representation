import unittest

import shogi
import torch
from torch.utils.data import DataLoader

from intrep.shogi_move_choice import (
    ShogiMoveChoiceDataset,
    ShogiMoveChoiceExample,
    shogi_move_choice_example_from_board,
    shogi_move_choice_examples_from_usi_moves,
)
from intrep.shogi_position_encoding import SHOGI_POSITION_TOKEN_COUNT


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
        examples = shogi_move_choice_examples_from_usi_moves(("7g7f", "3c3d"))

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].chosen_move, "7g7f")
        self.assertEqual(examples[1].chosen_move, "3c3d")
        self.assertIn("3c3d", examples[1].legal_moves)

    def test_dataset_returns_candidate_mask_and_label_index(self) -> None:
        examples = shogi_move_choice_examples_from_usi_moves(("7g7f", "3c3d"))
        dataset = ShogiMoveChoiceDataset(examples)

        position_token_ids, candidate_mask, label_index = dataset[0]

        self.assertEqual(tuple(position_token_ids.shape), (SHOGI_POSITION_TOKEN_COUNT,))
        self.assertEqual(candidate_mask.dtype, torch.bool)
        self.assertEqual(int(candidate_mask.sum().item()), len(examples[0].legal_moves))
        self.assertEqual(int(label_index.item()), examples[0].legal_moves.index("7g7f"))

    def test_dataset_can_be_batched(self) -> None:
        examples = shogi_move_choice_examples_from_usi_moves(("7g7f", "3c3d"))
        loader = DataLoader(ShogiMoveChoiceDataset(examples), batch_size=2)

        position_token_ids, candidate_masks, label_indexes = next(iter(loader))

        self.assertEqual(tuple(position_token_ids.shape), (2, SHOGI_POSITION_TOKEN_COUNT))
        self.assertEqual(tuple(candidate_masks.shape), (2, len(examples[0].legal_moves)))
        self.assertEqual(tuple(label_indexes.shape), (2,))


if __name__ == "__main__":
    unittest.main()
