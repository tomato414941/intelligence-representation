import unittest

import shogi
import torch

from intrep.representation.inputs.shogi_dlshogi_like_position import (
    ShogiDlshogiLikePositionAttentionLogitBias,
    ShogiDlshogiLikePositionInputLayer,
)
from intrep.representation.inputs.shogi_position_features.position_dlshogi_like import (
    SHOGI_DLSHOGI_LIKE_ELEMENT_COUNT,
    SHOGI_DLSHOGI_LIKE_GLOBAL_ELEMENT_COUNT,
    SHOGI_DLSHOGI_LIKE_POSITION_FEATURE_MANIFEST,
    SHOGI_DLSHOGI_LIKE_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_OFFSET,
    SHOGI_DLSHOGI_LIKE_SQUARE_FIELD_COUNT,
    shogi_dlshogi_like_position_feature_manifest_hash,
    shogi_dlshogi_like_position_features_from_sfen,
)
from intrep.representation.inputs.shogi_position_features.position_features import stack_shogi_position_features
from intrep.representation.inputs.shogi_position_features.position_schema import (
    HAND_PIECE_TYPES,
    IN_CHECK_FEATURE_ID,
    MOVE_COUNT_BUCKET_OFFSET,
    NOT_IN_CHECK_FEATURE_ID,
    OPPONENT_ATTACK_OFFSET,
    OPPONENT_HAND_OFFSET,
    OPPONENT_PIECE_OFFSET,
    OPPONENT_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
    OWN_ATTACK_OFFSET,
    OWN_HAND_OFFSET,
    OWN_PIECE_OFFSET,
    OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
    SQUARE_ATTACK_PIECE_TYPES,
    SHOGI_POSITION_STATE_FEATURE_ID,
    SIDE_TO_MOVE_BLACK_FEATURE_ID,
)
from intrep.domains.shogi.coordinates import absolute_to_relative_square


class ShogiDlshogiLikePositionTest(unittest.TestCase):
    def test_manifest_matches_dlshogi_like_current_state_intent(self) -> None:
        self.assertEqual(
            SHOGI_DLSHOGI_LIKE_POSITION_FEATURE_MANIFEST_HASH,
            shogi_dlshogi_like_position_feature_manifest_hash(),
        )
        self.assertEqual(SHOGI_DLSHOGI_LIKE_POSITION_FEATURE_MANIFEST["feature_groups"], ["global", "square"])
        self.assertEqual(
            SHOGI_DLSHOGI_LIKE_POSITION_FEATURE_MANIFEST["square_feature_groups"],
            [
                "own_piece_planes",
                "opponent_piece_planes",
                "own_attack_count",
                "opponent_attack_count",
                "own_piece_type_attacks",
                "opponent_piece_type_attacks",
            ],
        )

    def test_encodes_piece_planes_and_attack_features(self) -> None:
        features = shogi_dlshogi_like_position_features_from_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        relative_5a = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5a"), shogi.BLACK)
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        relative_5i = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5i"), shogi.BLACK)
        own_attack_count_field = 28
        opponent_attack_count_field = 29
        own_rook_attack_field = 30 + SQUARE_ATTACK_PIECE_TYPES.index(shogi.ROOK)
        opponent_king_attack_field = 30 + len(SQUARE_ATTACK_PIECE_TYPES) + SQUARE_ATTACK_PIECE_TYPES.index(shogi.KING)

        self.assertEqual(tuple(features.global_feature_ids.shape), (SHOGI_DLSHOGI_LIKE_GLOBAL_ELEMENT_COUNT,))
        self.assertEqual(tuple(features.square_feature_ids.shape), (81, SHOGI_DLSHOGI_LIKE_SQUARE_FIELD_COUNT))
        self.assertEqual(tuple(features.piece_feature_ids.shape), (0, 0))
        self.assertEqual(tuple(features.line_feature_ids.shape), (0, 0))
        self.assertEqual(int(features.pair_relation_edges.relation_ids.numel()), 0)
        self.assertEqual(
            int(features.square_feature_ids[relative_5a, 14 + shogi.KING - 1].item()),
            OPPONENT_PIECE_OFFSET + shogi.KING - 1,
        )
        self.assertEqual(
            int(features.square_feature_ids[relative_5e, shogi.ROOK - 1].item()),
            OWN_PIECE_OFFSET + shogi.ROOK - 1,
        )
        self.assertEqual(
            int(features.square_feature_ids[relative_5i, shogi.KING - 1].item()),
            OWN_PIECE_OFFSET + shogi.KING - 1,
        )
        self.assertEqual(
            int(features.square_feature_ids[relative_5a, own_attack_count_field].item()),
            OWN_ATTACK_OFFSET + 1,
        )
        self.assertEqual(
            int(features.square_feature_ids[relative_5e, opponent_attack_count_field].item()),
            OPPONENT_ATTACK_OFFSET,
        )
        self.assertEqual(
            int(features.square_feature_ids[relative_5a, own_rook_attack_field].item()),
            OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET + SQUARE_ATTACK_PIECE_TYPES.index(shogi.ROOK) * 2 + 1,
        )
        self.assertEqual(
            int(features.square_feature_ids[relative_5e, opponent_king_attack_field].item()),
            OPPONENT_SQUARE_PIECE_TYPE_ATTACK_OFFSET + SQUARE_ATTACK_PIECE_TYPES.index(shogi.KING) * 2,
        )

    def test_global_features_are_state_side_check_move_count_and_hands(self) -> None:
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P2b 1")
        features = shogi_dlshogi_like_position_features_from_sfen(board.sfen())
        own_pawn_hand_index = 4 + HAND_PIECE_TYPES.index(shogi.PAWN)
        opponent_bishop_hand_index = 4 + len(HAND_PIECE_TYPES) + HAND_PIECE_TYPES.index(shogi.BISHOP)

        self.assertEqual(int(features.global_feature_ids[0].item()), SHOGI_POSITION_STATE_FEATURE_ID)
        self.assertEqual(int(features.global_feature_ids[1].item()), SIDE_TO_MOVE_BLACK_FEATURE_ID)
        self.assertEqual(int(features.global_feature_ids[2].item()), NOT_IN_CHECK_FEATURE_ID)
        self.assertEqual(int(features.global_feature_ids[3].item()), MOVE_COUNT_BUCKET_OFFSET + 1)
        self.assertEqual(int(features.global_feature_ids[own_pawn_hand_index].item()), OWN_HAND_OFFSET + 1)
        self.assertEqual(int(features.global_feature_ids[opponent_bishop_hand_index].item()), OPPONENT_HAND_OFFSET + 2)

    def test_global_features_encode_check_state(self) -> None:
        features = shogi_dlshogi_like_position_features_from_sfen("4k4/9/9/9/4r4/9/9/9/4K4 b - 1")

        self.assertEqual(int(features.global_feature_ids[2].item()), IN_CHECK_FEATURE_ID)

    def test_input_layer_builds_dlshogi_like_sequence(self) -> None:
        features = stack_shogi_position_features(
            [
                shogi_dlshogi_like_position_features_from_sfen(shogi.Board().sfen()),
                shogi_dlshogi_like_position_features_from_sfen(shogi.Board().sfen()),
            ]
        )
        layer = ShogiDlshogiLikePositionInputLayer(embedding_dim=8)

        embeddings = layer(features)

        self.assertEqual(tuple(embeddings.shape), (2, SHOGI_DLSHOGI_LIKE_ELEMENT_COUNT, 8))

    def test_attention_logit_bias_targets_square_pairs_only(self) -> None:
        features = stack_shogi_position_features(
            [
                shogi_dlshogi_like_position_features_from_sfen(shogi.Board().sfen()),
                shogi_dlshogi_like_position_features_from_sfen(shogi.Board().sfen()),
            ]
        )
        embeddings = torch.zeros((2, SHOGI_DLSHOGI_LIKE_ELEMENT_COUNT, 8))
        attention_logit_bias = ShogiDlshogiLikePositionAttentionLogitBias()
        attention_logit_bias.square_relation_bias.weight.data[:, 0] = torch.arange(17 * 17, dtype=torch.float32)

        bias = attention_logit_bias(features, embeddings)
        same_square_relation = 8 * 17 + 8

        self.assertEqual(tuple(bias.shape), (2, SHOGI_DLSHOGI_LIKE_ELEMENT_COUNT, SHOGI_DLSHOGI_LIKE_ELEMENT_COUNT))
        self.assertEqual(
            float(
                bias[
                    0,
                    SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_OFFSET,
                    SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_OFFSET,
                ].item()
            ),
            float(same_square_relation),
        )
        self.assertEqual(float(bias[0, 0, SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_OFFSET].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
