import unittest

import shogi
import torch

from intrep.worlds.shogi.position_encoding import (
    ATTACK_TOKEN_OFFSET,
    BOARD_TOKEN_OFFSET,
    ATTACK_COUNT_TOKEN_MAX,
    DROP_SHADOW_TOKEN_OFFSET,
    HAND_COUNT_TOKEN_MAX,
    HAND_TOKEN_OFFSET,
    HAND_PIECE_TYPES,
    IN_CHECK_TOKEN_ID,
    IN_CHECK_TOKEN_INDEX,
    KING_RELATIVE_SQUARE_TOKEN_OFFSET,
    LINE_FEATURE_COUNT,
    LINE_FEATURE_TOKEN_OFFSET,
    LINE_KIND_OFFSET,
    LINE_OCCUPANCY_COUNT_OFFSET,
    LINE_OPPONENT_KING_ON_LINE_OFFSET,
    LINE_OPPONENT_SLIDER_ON_LINE_OFFSET,
    LINE_OWN_KING_ON_LINE_OFFSET,
    LINE_OWN_SLIDER_ON_LINE_OFFSET,
    MOVE_COUNT_BUCKET_OFFSET,
    MOVE_COUNT_BUCKET_OVERFLOW,
    MOVE_COUNT_TOKEN_INDEX,
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
    PIECE_FEATURE_COUNT,
    PIECE_FEATURE_TOKEN_OFFSET,
    PIECE_LOCATION_BOARD_TOKEN_ID,
    PIECE_LOCATION_EMPTY_TOKEN_ID,
    PIECE_LOCATION_HAND_TOKEN_ID,
    PIECE_SLOT_COUNT,
    PIECE_SQUARE_OFFSET,
    PIECE_SQUARE_UNKNOWN_TOKEN_ID,
    SQUARE_ATTACK_PIECE_TYPES,
    SQUARE_PIECE_TYPE_ATTACK_TOKEN_OFFSET,
    SHOGI_POSITION_TOKEN_COUNT,
    SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
    SHOGI_POSITION_VOCAB_SIZE,
    SIDE_TO_MOVE_BLACK_TOKEN_ID,
    SIDE_TO_MOVE_WHITE_TOKEN_ID,
    absolute_to_relative_square,
    king_relative_offset_bucket,
    move_count_bucket_token_id,
    shogi_position_token_ids_from_sfen,
)


class ShogiPositionEncodingTest(unittest.TestCase):
    def test_encodes_start_position_as_fixed_length_token_ids(self) -> None:
        token_ids = shogi_position_token_ids_from_sfen(shogi.Board().sfen())

        self.assertEqual(token_ids.dtype, torch.long)
        self.assertEqual(tuple(token_ids.shape), (SHOGI_POSITION_TOKEN_COUNT,))
        self.assertEqual(SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT, 191)
        self.assertEqual(int(token_ids[0].item()), SIDE_TO_MOVE_BLACK_TOKEN_ID)
        self.assertEqual(int(token_ids[IN_CHECK_TOKEN_INDEX].item()), NOT_IN_CHECK_TOKEN_ID)
        self.assertEqual(int(token_ids[MOVE_COUNT_TOKEN_INDEX].item()), MOVE_COUNT_BUCKET_OFFSET + 1)
        self.assertGreaterEqual(int(token_ids.min().item()), 0)
        self.assertLess(int(token_ids.max().item()), SHOGI_POSITION_VOCAB_SIZE)

    def test_side_to_move_changes_after_one_move(self) -> None:
        board = shogi.Board()
        board.push_usi("7g7f")

        token_ids = shogi_position_token_ids_from_sfen(board.sfen())

        self.assertEqual(int(token_ids[0].item()), SIDE_TO_MOVE_WHITE_TOKEN_ID)

    def test_position_is_side_to_move_relative(self) -> None:
        board = shogi.Board()
        board.remove_piece_at(shogi.SQUARE_NAMES.index("1a"))
        board.set_piece_at(shogi.SQUARE_NAMES.index("5e"), shogi.Piece(shogi.PAWN, shogi.WHITE))
        black_tokens = shogi_position_token_ids_from_sfen(board.sfen())
        board.turn = shogi.WHITE
        white_tokens = shogi_position_token_ids_from_sfen(board.sfen())

        black_relative_square = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        white_relative_square = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.WHITE)

        self.assertEqual(
            int(black_tokens[BOARD_TOKEN_OFFSET + black_relative_square].item()),
            OPPONENT_PIECE_OFFSET + shogi.PAWN - 1,
        )
        self.assertEqual(
            int(white_tokens[BOARD_TOKEN_OFFSET + white_relative_square].item()),
            OWN_PIECE_OFFSET + shogi.PAWN - 1,
        )

    def test_encodes_whether_side_to_move_is_in_check(self) -> None:
        board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        checked_board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 w - 1")

        safe_tokens = shogi_position_token_ids_from_sfen(board.sfen())
        checked_tokens = shogi_position_token_ids_from_sfen(checked_board.sfen())

        self.assertEqual(int(safe_tokens[IN_CHECK_TOKEN_INDEX].item()), NOT_IN_CHECK_TOKEN_ID)
        self.assertEqual(int(checked_tokens[IN_CHECK_TOKEN_INDEX].item()), IN_CHECK_TOKEN_ID)

    def test_encodes_move_count_bucket(self) -> None:
        early_board = shogi.Board()
        late_board = shogi.Board()
        late_board.move_number = 95

        early_tokens = shogi_position_token_ids_from_sfen(early_board.sfen())
        late_tokens = shogi_position_token_ids_from_sfen(late_board.sfen())

        self.assertEqual(int(early_tokens[MOVE_COUNT_TOKEN_INDEX].item()), MOVE_COUNT_BUCKET_OFFSET + 1)
        self.assertEqual(int(late_tokens[MOVE_COUNT_TOKEN_INDEX].item()), MOVE_COUNT_BUCKET_OFFSET + 4)

    def test_move_count_bucket_supports_unknown_and_overflow(self) -> None:
        self.assertEqual(move_count_bucket_token_id(None), MOVE_COUNT_BUCKET_OFFSET)
        self.assertEqual(move_count_bucket_token_id(0), MOVE_COUNT_BUCKET_OFFSET)
        self.assertEqual(move_count_bucket_token_id(221), MOVE_COUNT_BUCKET_OFFSET + MOVE_COUNT_BUCKET_OVERFLOW)

    def test_encodes_attack_counts_relative_to_side_to_move(self) -> None:
        board = shogi.Board()
        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        relative_7f = absolute_to_relative_square(shogi.SQUARE_NAMES.index("7f"), shogi.BLACK)
        relative_3d = absolute_to_relative_square(shogi.SQUARE_NAMES.index("3d"), shogi.BLACK)
        own_attack_index = ATTACK_TOKEN_OFFSET + relative_7f
        opponent_attack_index = ATTACK_TOKEN_OFFSET + 81 + relative_3d

        self.assertEqual(ATTACK_COUNT_TOKEN_MAX, 3)
        self.assertEqual(int(token_ids[own_attack_index].item()), OWN_ATTACK_OFFSET + 1)
        self.assertEqual(int(token_ids[opponent_attack_index].item()), OPPONENT_ATTACK_OFFSET + 1)

    def test_white_to_move_attack_counts_are_side_to_move_relative(self) -> None:
        black_board = shogi.Board()
        white_board = shogi.Board()
        white_board.turn = shogi.WHITE
        black_relative_7f = absolute_to_relative_square(shogi.SQUARE_NAMES.index("7f"), shogi.BLACK)
        white_relative_7f = absolute_to_relative_square(shogi.SQUARE_NAMES.index("7f"), shogi.WHITE)

        black_tokens = shogi_position_token_ids_from_sfen(black_board.sfen())
        white_tokens = shogi_position_token_ids_from_sfen(white_board.sfen())

        self.assertEqual(
            int(black_tokens[ATTACK_TOKEN_OFFSET + black_relative_7f].item()),
            OWN_ATTACK_OFFSET + 1,
        )
        self.assertEqual(
            int(white_tokens[ATTACK_TOKEN_OFFSET + 81 + white_relative_7f].item()),
            OPPONENT_ATTACK_OFFSET + 1,
        )

    def test_encodes_square_piece_type_attack_features(self) -> None:
        board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        relative_5b = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5b"), shogi.BLACK)
        rook_feature = SQUARE_ATTACK_PIECE_TYPES.index(shogi.ROOK)
        pawn_feature = SQUARE_ATTACK_PIECE_TYPES.index(shogi.PAWN)
        rook_attack_index = (
            SQUARE_PIECE_TYPE_ATTACK_TOKEN_OFFSET + relative_5b * len(SQUARE_ATTACK_PIECE_TYPES) + rook_feature
        )
        pawn_attack_index = (
            SQUARE_PIECE_TYPE_ATTACK_TOKEN_OFFSET + relative_5b * len(SQUARE_ATTACK_PIECE_TYPES) + pawn_feature
        )

        self.assertEqual(
            int(token_ids[rook_attack_index].item()),
            OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET + rook_feature * 2 + 1,
        )
        self.assertEqual(
            int(token_ids[pawn_attack_index].item()),
            OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET + pawn_feature * 2,
        )

    def test_white_to_move_square_piece_type_attack_features_are_side_to_move_relative(self) -> None:
        black_board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        white_board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 w - 1")
        relative_5b_black = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5b"), shogi.BLACK)
        relative_5b_white = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5b"), shogi.WHITE)
        rook_feature = SQUARE_ATTACK_PIECE_TYPES.index(shogi.ROOK)

        black_tokens = shogi_position_token_ids_from_sfen(black_board.sfen())
        white_tokens = shogi_position_token_ids_from_sfen(white_board.sfen())
        black_index = (
            SQUARE_PIECE_TYPE_ATTACK_TOKEN_OFFSET
            + relative_5b_black * len(SQUARE_ATTACK_PIECE_TYPES)
            + rook_feature
        )
        white_index = (
            SQUARE_PIECE_TYPE_ATTACK_TOKEN_OFFSET
            + 81 * len(SQUARE_ATTACK_PIECE_TYPES)
            + relative_5b_white * len(SQUARE_ATTACK_PIECE_TYPES)
            + rook_feature
        )

        self.assertEqual(
            int(black_tokens[black_index].item()),
            OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET + rook_feature * 2 + 1,
        )
        self.assertEqual(
            int(white_tokens[white_index].item()),
            OPPONENT_SQUARE_PIECE_TYPE_ATTACK_OFFSET + rook_feature * 2 + 1,
        )

    def test_encodes_king_relative_square_features(self) -> None:
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        relative_5i = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5i"), shogi.BLACK)
        relative_5h = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5h"), shogi.BLACK)
        feature_index = KING_RELATIVE_SQUARE_TOKEN_OFFSET + relative_5h
        expected_bucket = king_relative_offset_bucket(relative_5h, relative_5i)

        self.assertEqual(int(token_ids[feature_index].item()), OWN_KING_RELATIVE_SQUARE_OFFSET + 1 + expected_bucket)

    def test_white_to_move_king_relative_square_features_are_side_to_move_relative(self) -> None:
        white_board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 w - 1")
        relative_5a_for_white = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5a"), shogi.WHITE)
        relative_5b_for_white = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5b"), shogi.WHITE)
        feature_index = KING_RELATIVE_SQUARE_TOKEN_OFFSET + relative_5b_for_white
        expected_bucket = king_relative_offset_bucket(relative_5b_for_white, relative_5a_for_white)

        white_tokens = shogi_position_token_ids_from_sfen(white_board.sfen())

        self.assertEqual(
            int(white_tokens[feature_index].item()),
            OWN_KING_RELATIVE_SQUARE_OFFSET + 1 + expected_bucket,
        )

    def test_encodes_fixed_piece_tokens_in_relative_square_order(self) -> None:
        board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        relative_5a = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5a"), shogi.BLACK)
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        relative_5i = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5i"), shogi.BLACK)

        first_piece_offset = PIECE_FEATURE_TOKEN_OFFSET
        second_piece_offset = PIECE_FEATURE_TOKEN_OFFSET + PIECE_FEATURE_COUNT
        third_piece_offset = PIECE_FEATURE_TOKEN_OFFSET + PIECE_FEATURE_COUNT * 2
        fourth_piece_offset = PIECE_FEATURE_TOKEN_OFFSET + PIECE_FEATURE_COUNT * 3

        self.assertEqual(int(token_ids[first_piece_offset].item()), PIECE_LOCATION_BOARD_TOKEN_ID)
        self.assertEqual(int(token_ids[first_piece_offset + 1].item()), OPPONENT_PIECE_OFFSET + shogi.KING - 1)
        self.assertEqual(int(token_ids[first_piece_offset + 2].item()), PIECE_SQUARE_OFFSET + relative_5a)
        self.assertEqual(int(token_ids[second_piece_offset + 1].item()), OWN_PIECE_OFFSET + shogi.ROOK - 1)
        self.assertEqual(int(token_ids[second_piece_offset + 2].item()), PIECE_SQUARE_OFFSET + relative_5e)
        self.assertEqual(int(token_ids[third_piece_offset + 1].item()), OWN_PIECE_OFFSET + shogi.KING - 1)
        self.assertEqual(int(token_ids[third_piece_offset + 2].item()), PIECE_SQUARE_OFFSET + relative_5i)
        self.assertEqual(int(token_ids[fourth_piece_offset].item()), PIECE_LOCATION_EMPTY_TOKEN_ID)
        self.assertEqual(int(token_ids[fourth_piece_offset + 2].item()), PIECE_SQUARE_UNKNOWN_TOKEN_ID)

    def test_piece_tokens_include_king_relative_square_features(self) -> None:
        board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        relative_5a = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5a"), shogi.BLACK)
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        relative_5i = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5i"), shogi.BLACK)
        rook_piece_offset = PIECE_FEATURE_TOKEN_OFFSET + PIECE_FEATURE_COUNT

        own_king_bucket = king_relative_offset_bucket(relative_5e, relative_5i)
        opponent_king_bucket = king_relative_offset_bucket(relative_5e, relative_5a)

        self.assertEqual(
            int(token_ids[rook_piece_offset + 3].item()),
            OWN_KING_RELATIVE_SQUARE_OFFSET + 1 + own_king_bucket,
        )
        self.assertEqual(
            int(token_ids[rook_piece_offset + 4].item()),
            OPPONENT_KING_RELATIVE_SQUARE_OFFSET + 1 + opponent_king_bucket,
        )

    def test_encodes_drop_shadow_features_for_own_hand_piece(self) -> None:
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        relative_5a = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5a"), shogi.BLACK)
        pawn_feature = HAND_PIECE_TYPES.index(shogi.PAWN)

        drop_5e_index = DROP_SHADOW_TOKEN_OFFSET + relative_5e * len(HAND_PIECE_TYPES) + pawn_feature
        drop_5a_index = DROP_SHADOW_TOKEN_OFFSET + relative_5a * len(HAND_PIECE_TYPES) + pawn_feature

        self.assertEqual(int(token_ids[drop_5e_index].item()), OWN_DROP_SHADOW_OFFSET + pawn_feature * 2 + 1)
        self.assertEqual(int(token_ids[drop_5a_index].item()), OWN_DROP_SHADOW_OFFSET + pawn_feature * 2)

    def test_encodes_drop_shadow_features_for_opponent_hand_piece(self) -> None:
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b b 1")
        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        relative_5e = absolute_to_relative_square(shogi.SQUARE_NAMES.index("5e"), shogi.BLACK)
        bishop_feature = HAND_PIECE_TYPES.index(shogi.BISHOP)
        opponent_drop_5e_index = (
            DROP_SHADOW_TOKEN_OFFSET
            + 81 * len(HAND_PIECE_TYPES)
            + relative_5e * len(HAND_PIECE_TYPES)
            + bishop_feature
        )

        self.assertEqual(
            int(token_ids[opponent_drop_5e_index].item()),
            OPPONENT_DROP_SHADOW_OFFSET + bishop_feature * 2 + 1,
        )

    def test_encodes_line_features(self) -> None:
        board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        file_5_line = 4
        line_offset = LINE_FEATURE_TOKEN_OFFSET + file_5_line * LINE_FEATURE_COUNT

        self.assertEqual(int(token_ids[line_offset].item()), LINE_KIND_OFFSET)
        self.assertEqual(int(token_ids[line_offset + 1].item()), LINE_OWN_KING_ON_LINE_OFFSET + 1)
        self.assertEqual(int(token_ids[line_offset + 2].item()), LINE_OPPONENT_KING_ON_LINE_OFFSET + 1)
        self.assertEqual(int(token_ids[line_offset + 3].item()), LINE_OWN_SLIDER_ON_LINE_OFFSET + 1)
        self.assertEqual(int(token_ids[line_offset + 4].item()), LINE_OPPONENT_SLIDER_ON_LINE_OFFSET)
        self.assertEqual(int(token_ids[line_offset + 5].item()), LINE_OCCUPANCY_COUNT_OFFSET + 3)

    def test_piece_tokens_include_hand_pieces_after_board_pieces(self) -> None:
        board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 b P2b 1")
        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        own_hand_piece_offset = PIECE_FEATURE_TOKEN_OFFSET + PIECE_FEATURE_COUNT * 3
        opponent_hand_piece_offset = PIECE_FEATURE_TOKEN_OFFSET + PIECE_FEATURE_COUNT * 4
        empty_piece_offset = PIECE_FEATURE_TOKEN_OFFSET + PIECE_FEATURE_COUNT * 6

        self.assertEqual(int(token_ids[own_hand_piece_offset].item()), PIECE_LOCATION_HAND_TOKEN_ID)
        self.assertEqual(int(token_ids[own_hand_piece_offset + 1].item()), OWN_PIECE_OFFSET + shogi.PAWN - 1)
        self.assertEqual(int(token_ids[own_hand_piece_offset + 2].item()), PIECE_SQUARE_UNKNOWN_TOKEN_ID)
        self.assertEqual(int(token_ids[opponent_hand_piece_offset].item()), PIECE_LOCATION_HAND_TOKEN_ID)
        self.assertEqual(int(token_ids[opponent_hand_piece_offset + 1].item()), OPPONENT_PIECE_OFFSET + shogi.BISHOP - 1)
        self.assertEqual(int(token_ids[empty_piece_offset].item()), PIECE_LOCATION_EMPTY_TOKEN_ID)

    def test_full_start_position_uses_all_forty_piece_slots(self) -> None:
        token_ids = shogi_position_token_ids_from_sfen(shogi.Board().sfen())
        final_piece_offset = PIECE_FEATURE_TOKEN_OFFSET + (PIECE_SLOT_COUNT - 1) * PIECE_FEATURE_COUNT

        self.assertEqual(PIECE_SLOT_COUNT, 40)
        self.assertNotEqual(int(token_ids[final_piece_offset].item()), PIECE_LOCATION_EMPTY_TOKEN_ID)

    def test_incomplete_piece_tokens_are_padded_to_forty_slots(self) -> None:
        board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        final_piece_offset = PIECE_FEATURE_TOKEN_OFFSET + (PIECE_SLOT_COUNT - 1) * PIECE_FEATURE_COUNT

        self.assertEqual(PIECE_SLOT_COUNT, 40)
        self.assertEqual(int(token_ids[final_piece_offset].item()), PIECE_LOCATION_EMPTY_TOKEN_ID)
        self.assertEqual(int(token_ids[final_piece_offset + 1].item()), 0)
        self.assertEqual(int(token_ids[final_piece_offset + 2].item()), PIECE_SQUARE_UNKNOWN_TOKEN_ID)

    def test_large_own_pawn_hand_counts_are_not_collapsed_to_six(self) -> None:
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        for _ in range(14):
            board.add_piece_into_hand(shogi.PAWN, shogi.BLACK)

        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        black_pawn_hand_index = HAND_TOKEN_OFFSET + HAND_PIECE_TYPES.index(shogi.PAWN)

        self.assertEqual(HAND_COUNT_TOKEN_MAX, 18)
        self.assertEqual(int(token_ids[black_pawn_hand_index].item()), OWN_HAND_OFFSET + 14)
        self.assertLess(int(token_ids[black_pawn_hand_index].item()), SHOGI_POSITION_VOCAB_SIZE)

    def test_white_to_move_hands_are_side_to_move_relative(self) -> None:
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        board.add_piece_into_hand(shogi.PAWN, shogi.BLACK)
        board.add_piece_into_hand(shogi.PAWN, shogi.WHITE)
        board.add_piece_into_hand(shogi.PAWN, shogi.WHITE)
        board.turn = shogi.WHITE

        token_ids = shogi_position_token_ids_from_sfen(board.sfen())
        own_pawn_hand_index = HAND_TOKEN_OFFSET + HAND_PIECE_TYPES.index(shogi.PAWN)
        opponent_pawn_hand_index = HAND_TOKEN_OFFSET + len(HAND_PIECE_TYPES) + HAND_PIECE_TYPES.index(shogi.PAWN)

        self.assertEqual(int(token_ids[own_pawn_hand_index].item()), OWN_HAND_OFFSET + 2)
        self.assertEqual(int(token_ids[opponent_pawn_hand_index].item()), OPPONENT_HAND_OFFSET + 1)


if __name__ == "__main__":
    unittest.main()
