import unittest

import shogi
import torch
from torch import nn

from intrep.play_shogi_move_choice import choose_shogi_move, score_shogi_legal_moves


class IncreasingLogitModel(nn.Module):
    def forward(
        self,
        position_token_ids: torch.Tensor,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        del position_token_ids, candidate_move_features
        logits = torch.arange(candidate_mask.size(1), dtype=torch.float32).unsqueeze(0)
        return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)


class PlayShogiMoveChoiceTest(unittest.TestCase):
    def test_scores_legal_moves(self) -> None:
        scores = score_shogi_legal_moves(IncreasingLogitModel(), shogi.Board())

        self.assertEqual(len(scores), len(tuple(shogi.Board().legal_moves)))
        self.assertTrue(all(isinstance(move, str) for move, _ in scores))

    def test_choose_move_returns_highest_scored_legal_move(self) -> None:
        board = shogi.Board()
        expected = sorted(move.usi() for move in board.legal_moves)[-1]

        self.assertEqual(choose_shogi_move(IncreasingLogitModel(), board), expected)


if __name__ == "__main__":
    unittest.main()
