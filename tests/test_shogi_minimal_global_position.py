import unittest

import shogi
import torch

from intrep.representation.inputs.shogi_minimal_global_position import (
    ShogiMinimalGlobalPositionAttentionLogitBias,
    ShogiMinimalGlobalPositionInputLayer,
)
from intrep.representation.inputs.shogi_position_features.position_encoding import (
    HAND_PIECE_TYPES,
    MOVE_COUNT_BUCKET_OFFSET,
    OPPONENT_HAND_OFFSET,
    OPPONENT_PIECE_OFFSET,
    OWN_HAND_OFFSET,
    OWN_PIECE_OFFSET,
    SHOGI_POSITION_STATE_FEATURE_ID,
    SIDE_TO_MOVE_BLACK_FEATURE_ID,
    absolute_to_relative_square,
)
from intrep.representation.inputs.shogi_position_features.position_features import stack_shogi_position_features
from intrep.representation.inputs.shogi_position_features.position_minimal_global import (
    SHOGI_MINIMAL_GLOBAL_ELEMENT_COUNT,
    SHOGI_MINIMAL_GLOBAL_GLOBAL_ELEMENT_COUNT,
    SHOGI_MINIMAL_GLOBAL_GLOBAL_FEATURE_COUNT,
    SHOGI_MINIMAL_GLOBAL_POSITION_FEATURE_MANIFEST,
    SHOGI_MINIMAL_GLOBAL_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_MINIMAL_GLOBAL_SQUARE_ELEMENT_OFFSET,
    SHOGI_MINIMAL_GLOBAL_SQUARE_FEATURE_COUNT,
    shogi_minimal_global_position_feature_manifest_hash,
    shogi_minimal_global_position_features_from_sfen,
)


class ShogiMinimalGlobalPositionTest(unittest.TestCase):
    def test_manifest_matches_minimal_global_intent(self) -> None:
        self.assertEqual(
            SHOGI_MINIMAL_GLOBAL_POSITION_FEATURE_MANIFEST_HASH,
            shogi_minimal_global_position_feature_manifest_hash(),
        )
        self.assertEqual(SHOGI_MINIMAL_GLOBAL_POSITION_FEATURE_MANIFEST["feature_groups"], ["global", "square"])
        self.assertEqual(SHOGI_MINIMAL_GLOBAL_POSITION_FEATURE_MANIFEST["global_element_count"], 1)
        self.assertEqual(SHOGI_MINIMAL_GLOBAL_POSITION_FEATURE_MANIFEST["square_features"], ["piece_identity"])

    def test_encodes_one_global_element_and_square_piece_placement(self) -> None:
        features = shogi_minimal_global_position_features_from_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        relative_5a = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5a"), shogi.BLACK)
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        relative_5i = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5i"), shogi.BLACK)

        self.assertEqual(
            tuple(features.global_feature_ids.shape),
            (SHOGI_MINIMAL_GLOBAL_GLOBAL_ELEMENT_COUNT, SHOGI_MINIMAL_GLOBAL_GLOBAL_FEATURE_COUNT),
        )
        self.assertEqual(tuple(features.square_feature_ids.shape), (81, SHOGI_MINIMAL_GLOBAL_SQUARE_FEATURE_COUNT))
        self.assertEqual(tuple(features.piece_feature_ids.shape), (0, 0))
        self.assertEqual(tuple(features.line_feature_ids.shape), (0, 0))
        self.assertEqual(int(features.pair_relation_edges.relation_ids.numel()), 0)
        self.assertEqual(int(features.square_feature_ids[relative_5a, 0].item()), OPPONENT_PIECE_OFFSET + shogi.KING - 1)
        self.assertEqual(int(features.square_feature_ids[relative_5e, 0].item()), OWN_PIECE_OFFSET + shogi.ROOK - 1)
        self.assertEqual(int(features.square_feature_ids[relative_5i, 0].item()), OWN_PIECE_OFFSET + shogi.KING - 1)

    def test_global_features_are_combined_into_one_element(self) -> None:
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P2b 1")
        features = shogi_minimal_global_position_features_from_sfen(board.sfen())
        global_features = features.global_feature_ids[0]
        own_pawn_feature_index = 3 + HAND_PIECE_TYPES.index(shogi.PAWN)
        opponent_bishop_feature_index = 3 + len(HAND_PIECE_TYPES) + HAND_PIECE_TYPES.index(shogi.BISHOP)

        self.assertEqual(int(global_features[0].item()), SHOGI_POSITION_STATE_FEATURE_ID)
        self.assertEqual(int(global_features[1].item()), SIDE_TO_MOVE_BLACK_FEATURE_ID)
        self.assertEqual(int(global_features[2].item()), MOVE_COUNT_BUCKET_OFFSET + 1)
        self.assertEqual(int(global_features[own_pawn_feature_index].item()), OWN_HAND_OFFSET + 1)
        self.assertEqual(int(global_features[opponent_bishop_feature_index].item()), OPPONENT_HAND_OFFSET + 2)

    def test_input_layer_builds_82_element_sequence(self) -> None:
        features = stack_shogi_position_features(
            [
                shogi_minimal_global_position_features_from_sfen(shogi.Board().sfen()),
                shogi_minimal_global_position_features_from_sfen(shogi.Board().sfen()),
            ]
        )
        layer = ShogiMinimalGlobalPositionInputLayer(embedding_dim=8)

        embeddings = layer(features)

        self.assertEqual(tuple(embeddings.shape), (2, SHOGI_MINIMAL_GLOBAL_ELEMENT_COUNT, 8))

    def test_attention_logit_bias_targets_square_pairs_only(self) -> None:
        features = stack_shogi_position_features(
            [
                shogi_minimal_global_position_features_from_sfen(shogi.Board().sfen()),
                shogi_minimal_global_position_features_from_sfen(shogi.Board().sfen()),
            ]
        )
        embeddings = torch.zeros((2, SHOGI_MINIMAL_GLOBAL_ELEMENT_COUNT, 8))
        attention_logit_bias = ShogiMinimalGlobalPositionAttentionLogitBias()
        attention_logit_bias.square_relation_bias.weight.data[:, 0] = torch.arange(17 * 17, dtype=torch.float32)

        bias = attention_logit_bias(features, embeddings)
        same_square_relation = 8 * 17 + 8

        self.assertEqual(tuple(bias.shape), (2, SHOGI_MINIMAL_GLOBAL_ELEMENT_COUNT, SHOGI_MINIMAL_GLOBAL_ELEMENT_COUNT))
        self.assertEqual(
            float(
                bias[
                    0,
                    SHOGI_MINIMAL_GLOBAL_SQUARE_ELEMENT_OFFSET,
                    SHOGI_MINIMAL_GLOBAL_SQUARE_ELEMENT_OFFSET,
                ].item()
            ),
            float(same_square_relation),
        )
        self.assertEqual(float(bias[0, 0, SHOGI_MINIMAL_GLOBAL_SQUARE_ELEMENT_OFFSET].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
