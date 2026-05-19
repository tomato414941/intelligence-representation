import unittest

import shogi
import torch

from intrep.worlds.shogi.position_encoding import (
    ATTACK_COUNT_TOKEN_MAX,
    CAPTURE_FLOW_OPPORTUNITY_OFFSET,
    COUNTERFACTUAL_REMOVAL_SELF_CHECK_OFFSET,
    COUNTERFACTUAL_REMOVAL_SLIDER_BLOCKER_OFFSET,
    GIFT_DANGER_OFFSET,
    HAND_COUNT_TOKEN_MAX,
    HAND_PIECE_TYPES,
    IN_CHECK_TOKEN_ID,
    LINE_KIND_OFFSET,
    LINE_OCCUPANCY_COUNT_OFFSET,
    LINE_OPPONENT_KING_ON_LINE_OFFSET,
    LINE_OPPONENT_SLIDER_ON_LINE_OFFSET,
    LINE_OWN_KING_ON_LINE_OFFSET,
    LINE_OWN_SLIDER_ON_LINE_OFFSET,
    MOVE_COUNT_BUCKET_OFFSET,
    MOVE_COUNT_BUCKET_OVERFLOW,
    NOT_IN_CHECK_TOKEN_ID,
    OPPONENT_ATTACK_OFFSET,
    OPPONENT_DROP_SHADOW_OFFSET,
    OPPONENT_HAND_OFFSET,
    OPPONENT_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
    OPPONENT_PIECE_OFFSET,
    OPPONENT_KING_RELATIVE_SQUARE_OFFSET,
    OWN_ATTACK_OFFSET,
    OWN_DROP_SHADOW_OFFSET,
    OWN_HAND_OFFSET,
    OWN_KING_RELATIVE_SQUARE_OFFSET,
    OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
    OWN_PIECE_OFFSET,
    PIECE_LOCATION_BOARD_TOKEN_ID,
    PIECE_LOCATION_EMPTY_TOKEN_ID,
    PIECE_LOCATION_HAND_TOKEN_ID,
    PIECE_TOKEN_OFFSET,
    PIECE_SLOT_COUNT,
    PIECE_SQUARE_OFFSET,
    PIECE_SQUARE_UNKNOWN_TOKEN_ID,
    PAIR_RELATION_PIECE_ATTACKS_PIECE,
    PAIR_RELATION_PIECE_ATTACKS_SQUARE,
    PAIR_RELATION_PIECE_ON_SQUARE,
    SQUARE_ATTACK_PIECE_TYPES,
    SHOGI_POSITION_FEATURE_MANIFEST,
    SHOGI_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
    SHOGI_POSITION_GLOBAL_SLOT_COUNT,
    SHOGI_POSITION_LINE_FEATURE_COUNT,
    SHOGI_POSITION_LINE_SLOT_COUNT,
    SHOGI_POSITION_PIECE_FEATURE_COUNT,
    SHOGI_POSITION_PIECE_SLOT_COUNT,
    SHOGI_POSITION_SQUARE_COUNT,
    SHOGI_POSITION_SQUARE_FEATURE_COUNT,
    SHOGI_POSITION_VOCAB_SIZE,
    SIDE_TO_MOVE_BLACK_TOKEN_ID,
    SIDE_TO_MOVE_WHITE_TOKEN_ID,
    SQUARE_TOKEN_OFFSET,
    absolute_to_relative_square,
    king_relative_offset_bucket,
    move_count_bucket_token_id,
    shogi_position_feature_manifest_hash,
    shogi_position_features_from_sfen,
)


PIECE_FEATURE_INDEX = 0
OWN_ATTACK_FEATURE_INDEX = 1
OPPONENT_ATTACK_FEATURE_INDEX = 2
OWN_PIECE_TYPE_ATTACK_FEATURE_OFFSET = 3
OPPONENT_PIECE_TYPE_ATTACK_FEATURE_OFFSET = OWN_PIECE_TYPE_ATTACK_FEATURE_OFFSET + len(SQUARE_ATTACK_PIECE_TYPES)
OWN_KING_RELATIVE_FEATURE_INDEX = OPPONENT_PIECE_TYPE_ATTACK_FEATURE_OFFSET + len(SQUARE_ATTACK_PIECE_TYPES)
OPPONENT_KING_RELATIVE_FEATURE_INDEX = OWN_KING_RELATIVE_FEATURE_INDEX + 1
OWN_DROP_SHADOW_FEATURE_OFFSET = OPPONENT_KING_RELATIVE_FEATURE_INDEX + 1
OPPONENT_DROP_SHADOW_FEATURE_OFFSET = OWN_DROP_SHADOW_FEATURE_OFFSET + len(HAND_PIECE_TYPES)
COUNTERFACTUAL_REMOVAL_FEATURE_OFFSET = OPPONENT_DROP_SHADOW_FEATURE_OFFSET + len(HAND_PIECE_TYPES)
GIFT_FLOW_FEATURE_OFFSET = COUNTERFACTUAL_REMOVAL_FEATURE_OFFSET + 3


class ShogiPositionEncodingTest(unittest.TestCase):
    def test_feature_manifest_hash_matches_current_manifest(self) -> None:
        self.assertEqual(SHOGI_POSITION_FEATURE_MANIFEST_HASH, shogi_position_feature_manifest_hash())
        self.assertEqual(
            SHOGI_POSITION_FEATURE_MANIFEST["feature_sequence_token_count"],
            SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
        )
        self.assertEqual(SHOGI_POSITION_FEATURE_MANIFEST["square_feature_count"], SHOGI_POSITION_SQUARE_FEATURE_COUNT)
        self.assertEqual(SHOGI_POSITION_FEATURE_MANIFEST["piece_feature_count"], SHOGI_POSITION_PIECE_FEATURE_COUNT)

    def test_encodes_start_position_as_feature_groups(self) -> None:
        features = shogi_position_features_from_sfen(shogi.Board().sfen())

        self.assertEqual(tuple(features.global_feature_ids.shape), (SHOGI_POSITION_GLOBAL_SLOT_COUNT,))
        self.assertEqual(
            tuple(features.square_feature_ids.shape),
            (SHOGI_POSITION_SQUARE_COUNT, SHOGI_POSITION_SQUARE_FEATURE_COUNT),
        )
        self.assertEqual(
            tuple(features.piece_feature_ids.shape),
            (SHOGI_POSITION_PIECE_SLOT_COUNT, SHOGI_POSITION_PIECE_FEATURE_COUNT),
        )
        self.assertEqual(
            tuple(features.line_feature_ids.shape),
            (SHOGI_POSITION_LINE_SLOT_COUNT, SHOGI_POSITION_LINE_FEATURE_COUNT),
        )
        self.assertEqual(SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT, 191)
        self.assertEqual(int(features.global_feature_ids[1].item()), SIDE_TO_MOVE_BLACK_TOKEN_ID)
        self.assertEqual(int(features.global_feature_ids[2].item()), NOT_IN_CHECK_TOKEN_ID)
        self.assertEqual(int(features.global_feature_ids[3].item()), MOVE_COUNT_BUCKET_OFFSET + 1)
        self.assertGreaterEqual(int(features.square_feature_ids.min().item()), 0)
        self.assertLess(int(features.line_feature_ids.max().item()), SHOGI_POSITION_VOCAB_SIZE)

    def test_side_to_move_changes_after_one_move(self) -> None:
        board = shogi.Board()
        board.push_usi("7g7f")

        features = shogi_position_features_from_sfen(board.sfen())

        self.assertEqual(int(features.global_feature_ids[1].item()), SIDE_TO_MOVE_WHITE_TOKEN_ID)

    def test_position_is_side_to_move_relative(self) -> None:
        board = shogi.Board()
        board.remove_piece_at(shogi.SQUARE_NAMES.index("1a"))
        board.set_piece_at(shogi.SQUARE_NAMES.index("5e"), shogi.Piece(shogi.PAWN, shogi.WHITE))
        black_features = shogi_position_features_from_sfen(board.sfen())
        board.turn = shogi.WHITE
        white_features = shogi_position_features_from_sfen(board.sfen())

        black_relative_square = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        white_relative_square = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.WHITE)

        self.assertEqual(
            int(black_features.square_feature_ids[black_relative_square, PIECE_FEATURE_INDEX].item()),
            OPPONENT_PIECE_OFFSET + shogi.PAWN - 1,
        )
        self.assertEqual(
            int(white_features.square_feature_ids[white_relative_square, PIECE_FEATURE_INDEX].item()),
            OWN_PIECE_OFFSET + shogi.PAWN - 1,
        )

    def test_encodes_whether_side_to_move_is_in_check(self) -> None:
        board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        checked_board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 w - 1")

        safe_features = shogi_position_features_from_sfen(board.sfen())
        checked_features = shogi_position_features_from_sfen(checked_board.sfen())

        self.assertEqual(int(safe_features.global_feature_ids[2].item()), NOT_IN_CHECK_TOKEN_ID)
        self.assertEqual(int(checked_features.global_feature_ids[2].item()), IN_CHECK_TOKEN_ID)

    def test_encodes_move_count_bucket(self) -> None:
        early_board = shogi.Board()
        late_board = shogi.Board()
        late_board.move_number = 95

        early_features = shogi_position_features_from_sfen(early_board.sfen())
        late_features = shogi_position_features_from_sfen(late_board.sfen())

        self.assertEqual(int(early_features.global_feature_ids[3].item()), MOVE_COUNT_BUCKET_OFFSET + 1)
        self.assertEqual(int(late_features.global_feature_ids[3].item()), MOVE_COUNT_BUCKET_OFFSET + 4)

    def test_move_count_bucket_supports_unknown_and_overflow(self) -> None:
        self.assertEqual(move_count_bucket_token_id(None), MOVE_COUNT_BUCKET_OFFSET)
        self.assertEqual(move_count_bucket_token_id(0), MOVE_COUNT_BUCKET_OFFSET)
        self.assertEqual(move_count_bucket_token_id(221), MOVE_COUNT_BUCKET_OFFSET + MOVE_COUNT_BUCKET_OVERFLOW)

    def test_encodes_attack_counts_relative_to_side_to_move(self) -> None:
        features = shogi_position_features_from_sfen(shogi.Board().sfen())
        relative_7f = absolute_to_relative_square(shogi.SQUARE_NAMES.index("7f"), shogi.BLACK)
        relative_3d = absolute_to_relative_square(shogi.SQUARE_NAMES.index("3d"), shogi.BLACK)

        self.assertEqual(ATTACK_COUNT_TOKEN_MAX, 3)
        self.assertEqual(int(features.square_feature_ids[relative_7f, OWN_ATTACK_FEATURE_INDEX].item()), OWN_ATTACK_OFFSET + 1)
        self.assertEqual(
            int(features.square_feature_ids[relative_3d, OPPONENT_ATTACK_FEATURE_INDEX].item()),
            OPPONENT_ATTACK_OFFSET + 1,
        )

    def test_white_to_move_attack_counts_are_side_to_move_relative(self) -> None:
        black_board = shogi.Board()
        white_board = shogi.Board()
        white_board.turn = shogi.WHITE
        black_relative_7f = absolute_to_relative_square(shogi.SQUARE_NAMES.index("7f"), shogi.BLACK)
        white_relative_7f = absolute_to_relative_square(shogi.SQUARE_NAMES.index("7f"), shogi.WHITE)

        black_features = shogi_position_features_from_sfen(black_board.sfen())
        white_features = shogi_position_features_from_sfen(white_board.sfen())

        self.assertEqual(
            int(black_features.square_feature_ids[black_relative_7f, OWN_ATTACK_FEATURE_INDEX].item()),
            OWN_ATTACK_OFFSET + 1,
        )
        self.assertEqual(
            int(white_features.square_feature_ids[white_relative_7f, OPPONENT_ATTACK_FEATURE_INDEX].item()),
            OPPONENT_ATTACK_OFFSET + 1,
        )

    def test_encodes_square_piece_type_attack_features(self) -> None:
        features = shogi_position_features_from_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        relative_5b = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5b"), shogi.BLACK)
        rook_feature = SQUARE_ATTACK_PIECE_TYPES.index(shogi.ROOK)
        pawn_feature = SQUARE_ATTACK_PIECE_TYPES.index(shogi.PAWN)

        self.assertEqual(
            int(features.square_feature_ids[relative_5b, OWN_PIECE_TYPE_ATTACK_FEATURE_OFFSET + rook_feature].item()),
            OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET + rook_feature * 2 + 1,
        )
        self.assertEqual(
            int(features.square_feature_ids[relative_5b, OWN_PIECE_TYPE_ATTACK_FEATURE_OFFSET + pawn_feature].item()),
            OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET + pawn_feature * 2,
        )

    def test_white_to_move_square_piece_type_attack_features_are_side_to_move_relative(self) -> None:
        black_board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        white_board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 w - 1")
        relative_5b_black = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5b"), shogi.BLACK)
        relative_5b_white = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5b"), shogi.WHITE)
        rook_feature = SQUARE_ATTACK_PIECE_TYPES.index(shogi.ROOK)

        black_features = shogi_position_features_from_sfen(black_board.sfen())
        white_features = shogi_position_features_from_sfen(white_board.sfen())

        self.assertEqual(
            int(black_features.square_feature_ids[relative_5b_black, OWN_PIECE_TYPE_ATTACK_FEATURE_OFFSET + rook_feature].item()),
            OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET + rook_feature * 2 + 1,
        )
        self.assertEqual(
            int(white_features.square_feature_ids[relative_5b_white, OPPONENT_PIECE_TYPE_ATTACK_FEATURE_OFFSET + rook_feature].item()),
            OPPONENT_SQUARE_PIECE_TYPE_ATTACK_OFFSET + rook_feature * 2 + 1,
        )

    def test_encodes_king_relative_square_features(self) -> None:
        features = shogi_position_features_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        relative_5i = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5i"), shogi.BLACK)
        relative_5h = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5h"), shogi.BLACK)
        expected_bucket = king_relative_offset_bucket(relative_5h, relative_5i)

        self.assertEqual(
            int(features.square_feature_ids[relative_5h, OWN_KING_RELATIVE_FEATURE_INDEX].item()),
            OWN_KING_RELATIVE_SQUARE_OFFSET + 1 + expected_bucket,
        )

    def test_white_to_move_king_relative_square_features_are_side_to_move_relative(self) -> None:
        white_board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 w - 1")
        relative_5a_for_white = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5a"), shogi.WHITE)
        relative_5b_for_white = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5b"), shogi.WHITE)
        expected_bucket = king_relative_offset_bucket(relative_5b_for_white, relative_5a_for_white)

        features = shogi_position_features_from_sfen(white_board.sfen())

        self.assertEqual(
            int(features.square_feature_ids[relative_5b_for_white, OWN_KING_RELATIVE_FEATURE_INDEX].item()),
            OWN_KING_RELATIVE_SQUARE_OFFSET + 1 + expected_bucket,
        )

    def test_encodes_fixed_piece_tokens_in_relative_square_order(self) -> None:
        features = shogi_position_features_from_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        relative_5a = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5a"), shogi.BLACK)
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        relative_5i = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5i"), shogi.BLACK)

        self.assertEqual(int(features.piece_feature_ids[0, 0].item()), PIECE_LOCATION_BOARD_TOKEN_ID)
        self.assertEqual(int(features.piece_feature_ids[0, 1].item()), OPPONENT_PIECE_OFFSET + shogi.KING - 1)
        self.assertEqual(int(features.piece_feature_ids[0, 2].item()), PIECE_SQUARE_OFFSET + relative_5a)
        self.assertEqual(int(features.piece_feature_ids[1, 1].item()), OWN_PIECE_OFFSET + shogi.ROOK - 1)
        self.assertEqual(int(features.piece_feature_ids[1, 2].item()), PIECE_SQUARE_OFFSET + relative_5e)
        self.assertEqual(int(features.piece_feature_ids[2, 1].item()), OWN_PIECE_OFFSET + shogi.KING - 1)
        self.assertEqual(int(features.piece_feature_ids[2, 2].item()), PIECE_SQUARE_OFFSET + relative_5i)
        self.assertEqual(int(features.piece_feature_ids[3, 0].item()), PIECE_LOCATION_EMPTY_TOKEN_ID)
        self.assertEqual(int(features.piece_feature_ids[3, 2].item()), PIECE_SQUARE_UNKNOWN_TOKEN_ID)

    def test_piece_tokens_include_king_relative_square_features(self) -> None:
        features = shogi_position_features_from_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        relative_5a = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5a"), shogi.BLACK)
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        relative_5i = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5i"), shogi.BLACK)
        own_king_bucket = king_relative_offset_bucket(relative_5e, relative_5i)
        opponent_king_bucket = king_relative_offset_bucket(relative_5e, relative_5a)

        self.assertEqual(
            int(features.piece_feature_ids[1, 3].item()),
            OWN_KING_RELATIVE_SQUARE_OFFSET + 1 + own_king_bucket,
        )
        self.assertEqual(
            int(features.piece_feature_ids[1, 4].item()),
            OPPONENT_KING_RELATIVE_SQUARE_OFFSET + 1 + opponent_king_bucket,
        )

    def test_encodes_drop_shadow_features_for_own_hand_piece(self) -> None:
        features = shogi_position_features_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        relative_5a = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5a"), shogi.BLACK)
        pawn_feature = HAND_PIECE_TYPES.index(shogi.PAWN)

        self.assertEqual(
            int(features.square_feature_ids[relative_5e, OWN_DROP_SHADOW_FEATURE_OFFSET + pawn_feature].item()),
            OWN_DROP_SHADOW_OFFSET + pawn_feature * 2 + 1,
        )
        self.assertEqual(
            int(features.square_feature_ids[relative_5a, OWN_DROP_SHADOW_FEATURE_OFFSET + pawn_feature].item()),
            OWN_DROP_SHADOW_OFFSET + pawn_feature * 2,
        )

    def test_encodes_drop_shadow_features_for_opponent_hand_piece(self) -> None:
        features = shogi_position_features_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b b 1")
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        bishop_feature = HAND_PIECE_TYPES.index(shogi.BISHOP)

        self.assertEqual(
            int(features.square_feature_ids[relative_5e, OPPONENT_DROP_SHADOW_FEATURE_OFFSET + bishop_feature].item()),
            OPPONENT_DROP_SHADOW_OFFSET + bishop_feature * 2 + 1,
        )

    def test_encodes_counterfactual_removal_features(self) -> None:
        features = shogi_position_features_from_sfen("4r3k/9/9/9/4G4/9/9/9/4K4 b - 1")
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)

        self.assertEqual(
            int(features.square_feature_ids[relative_5e, COUNTERFACTUAL_REMOVAL_FEATURE_OFFSET].item()),
            COUNTERFACTUAL_REMOVAL_SELF_CHECK_OFFSET + 1,
        )
        self.assertEqual(
            int(features.square_feature_ids[relative_5e, COUNTERFACTUAL_REMOVAL_FEATURE_OFFSET + 2].item()),
            COUNTERFACTUAL_REMOVAL_SLIDER_BLOCKER_OFFSET + 1,
        )
        self.assertEqual(int(features.piece_feature_ids[2, 5].item()), COUNTERFACTUAL_REMOVAL_SELF_CHECK_OFFSET + 1)

    def test_encodes_capture_to_hand_flow_features(self) -> None:
        features = shogi_position_features_from_sfen("4k4/9/9/9/4G4/9/9/9/4K4 b - 1")
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)

        self.assertEqual(
            int(features.square_feature_ids[relative_5e, GIFT_FLOW_FEATURE_OFFSET].item()),
            GIFT_DANGER_OFFSET + 1,
        )
        self.assertEqual(
            int(features.square_feature_ids[relative_5e, GIFT_FLOW_FEATURE_OFFSET + 1].item()),
            CAPTURE_FLOW_OPPORTUNITY_OFFSET,
        )
        self.assertEqual(int(features.piece_feature_ids[1, 8].item()), GIFT_DANGER_OFFSET + 1)

    def test_encodes_piece_square_and_piece_piece_pair_relations(self) -> None:
        features = shogi_position_features_from_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        relative_5a = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5a"), shogi.BLACK)
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        white_king_token = PIECE_TOKEN_OFFSET
        rook_token = PIECE_TOKEN_OFFSET + 1

        relation_edges = {
            (int(source.item()), int(target.item()), int(relation.item()))
            for source, target, relation in zip(
                features.pair_relation_edges.source_token_indices,
                features.pair_relation_edges.target_token_indices,
                features.pair_relation_edges.relation_ids,
                strict=True,
            )
        }

        self.assertIn(
            (rook_token, SQUARE_TOKEN_OFFSET + relative_5e, PAIR_RELATION_PIECE_ON_SQUARE),
            relation_edges,
        )
        self.assertIn(
            (rook_token, SQUARE_TOKEN_OFFSET + relative_5a, PAIR_RELATION_PIECE_ATTACKS_SQUARE),
            relation_edges,
        )
        self.assertIn((rook_token, white_king_token, PAIR_RELATION_PIECE_ATTACKS_PIECE), relation_edges)

    def test_encodes_line_features(self) -> None:
        features = shogi_position_features_from_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        file_5_line = 4

        self.assertEqual(int(features.line_feature_ids[file_5_line, 0].item()), LINE_KIND_OFFSET)
        self.assertEqual(int(features.line_feature_ids[file_5_line, 1].item()), LINE_OWN_KING_ON_LINE_OFFSET + 1)
        self.assertEqual(int(features.line_feature_ids[file_5_line, 2].item()), LINE_OPPONENT_KING_ON_LINE_OFFSET + 1)
        self.assertEqual(int(features.line_feature_ids[file_5_line, 3].item()), LINE_OWN_SLIDER_ON_LINE_OFFSET + 1)
        self.assertEqual(int(features.line_feature_ids[file_5_line, 4].item()), LINE_OPPONENT_SLIDER_ON_LINE_OFFSET)
        self.assertEqual(int(features.line_feature_ids[file_5_line, 5].item()), LINE_OCCUPANCY_COUNT_OFFSET + 3)

    def test_piece_tokens_include_hand_pieces_after_board_pieces(self) -> None:
        features = shogi_position_features_from_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b P2b 1")

        self.assertEqual(int(features.piece_feature_ids[3, 0].item()), PIECE_LOCATION_HAND_TOKEN_ID)
        self.assertEqual(int(features.piece_feature_ids[3, 1].item()), OWN_PIECE_OFFSET + shogi.PAWN - 1)
        self.assertEqual(int(features.piece_feature_ids[3, 2].item()), PIECE_SQUARE_UNKNOWN_TOKEN_ID)
        self.assertEqual(int(features.piece_feature_ids[4, 0].item()), PIECE_LOCATION_HAND_TOKEN_ID)
        self.assertEqual(int(features.piece_feature_ids[4, 1].item()), OPPONENT_PIECE_OFFSET + shogi.BISHOP - 1)
        self.assertEqual(int(features.piece_feature_ids[6, 0].item()), PIECE_LOCATION_EMPTY_TOKEN_ID)

    def test_full_start_position_uses_all_forty_piece_slots(self) -> None:
        features = shogi_position_features_from_sfen(shogi.Board().sfen())

        self.assertEqual(PIECE_SLOT_COUNT, 40)
        self.assertNotEqual(int(features.piece_feature_ids[PIECE_SLOT_COUNT - 1, 0].item()), PIECE_LOCATION_EMPTY_TOKEN_ID)

    def test_incomplete_piece_tokens_are_padded_to_forty_slots(self) -> None:
        features = shogi_position_features_from_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")

        self.assertEqual(PIECE_SLOT_COUNT, 40)
        self.assertEqual(int(features.piece_feature_ids[PIECE_SLOT_COUNT - 1, 0].item()), PIECE_LOCATION_EMPTY_TOKEN_ID)
        self.assertEqual(int(features.piece_feature_ids[PIECE_SLOT_COUNT - 1, 1].item()), 0)
        self.assertEqual(int(features.piece_feature_ids[PIECE_SLOT_COUNT - 1, 2].item()), PIECE_SQUARE_UNKNOWN_TOKEN_ID)

    def test_large_own_pawn_hand_counts_are_not_collapsed_to_six(self) -> None:
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        for _ in range(14):
            board.add_piece_into_hand(shogi.PAWN, shogi.BLACK)

        features = shogi_position_features_from_sfen(board.sfen())
        own_pawn_hand_index = 4 + HAND_PIECE_TYPES.index(shogi.PAWN)

        self.assertEqual(HAND_COUNT_TOKEN_MAX, 18)
        self.assertEqual(int(features.global_feature_ids[own_pawn_hand_index].item()), OWN_HAND_OFFSET + 14)
        self.assertLess(int(features.global_feature_ids[own_pawn_hand_index].item()), SHOGI_POSITION_VOCAB_SIZE)

    def test_white_to_move_hands_are_side_to_move_relative(self) -> None:
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        board.add_piece_into_hand(shogi.PAWN, shogi.BLACK)
        board.add_piece_into_hand(shogi.PAWN, shogi.WHITE)
        board.add_piece_into_hand(shogi.PAWN, shogi.WHITE)
        board.turn = shogi.WHITE

        features = shogi_position_features_from_sfen(board.sfen())
        own_pawn_hand_index = 4 + HAND_PIECE_TYPES.index(shogi.PAWN)
        opponent_pawn_hand_index = 4 + len(HAND_PIECE_TYPES) + HAND_PIECE_TYPES.index(shogi.PAWN)

        self.assertEqual(int(features.global_feature_ids[own_pawn_hand_index].item()), OWN_HAND_OFFSET + 2)
        self.assertEqual(int(features.global_feature_ids[opponent_pawn_hand_index].item()), OPPONENT_HAND_OFFSET + 1)


if __name__ == "__main__":
    unittest.main()
