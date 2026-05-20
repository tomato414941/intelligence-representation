from __future__ import annotations

from dataclasses import dataclass

import shogi


@dataclass(frozen=True)
class PieceSlotRelationInfo:
    piece: shogi.Piece | None
    location_kind: str
    relative_square: int | None


@dataclass(frozen=True)
class _ShogiPositionDerivedRelations:
    counterfactual_removal_token_rows: list[list[int]]
    drop_potential_token_rows: list[list[int]]
    drop_shadow_token_rows: list[list[int]]
    legal_drop_targets_by_color: dict[int, dict[int, set[int]]]
    piece_slot_relation_infos: list[PieceSlotRelationInfo]
