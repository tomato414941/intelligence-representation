import unittest

import torch

from intrep.shogi_move_encoding import (
    NO_DROP_PIECE_ID,
    NO_FROM_SQUARE_ID,
    SHOGI_MOVE_FEATURE_COUNT,
    shogi_candidate_move_features,
    shogi_move_feature_ids,
)


class ShogiMoveEncodingTest(unittest.TestCase):
    def test_encodes_normal_move(self) -> None:
        feature_ids = shogi_move_feature_ids("7g7f")

        self.assertEqual(feature_ids.dtype, torch.long)
        self.assertEqual(tuple(feature_ids.shape), (SHOGI_MOVE_FEATURE_COUNT,))
        self.assertNotEqual(int(feature_ids[0].item()), NO_FROM_SQUARE_ID)
        self.assertEqual(int(feature_ids[2].item()), 0)
        self.assertEqual(int(feature_ids[3].item()), NO_DROP_PIECE_ID)

    def test_encodes_promotion(self) -> None:
        feature_ids = shogi_move_feature_ids("2b3c+")

        self.assertEqual(int(feature_ids[2].item()), 1)

    def test_encodes_drop_move(self) -> None:
        feature_ids = shogi_move_feature_ids("P*5e")

        self.assertEqual(int(feature_ids[0].item()), NO_FROM_SQUARE_ID)
        self.assertGreater(int(feature_ids[3].item()), NO_DROP_PIECE_ID)

    def test_candidate_features_are_padded_to_max_choice_count(self) -> None:
        features = shogi_candidate_move_features(("7g7f", "3c3d"), max_choice_count=4)

        self.assertEqual(tuple(features.shape), (4, SHOGI_MOVE_FEATURE_COUNT))
        self.assertTrue(torch.equal(features[2], torch.zeros(SHOGI_MOVE_FEATURE_COUNT, dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
