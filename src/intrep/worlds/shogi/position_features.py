from __future__ import annotations

from dataclasses import dataclass

import torch

from intrep.worlds.shogi.position_schema import (
    PAIR_RELATION_COUNT,
    SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
    SHOGI_POSITION_GLOBAL_SLOT_COUNT,
    SHOGI_POSITION_LINE_FEATURE_COUNT,
    SHOGI_POSITION_LINE_SLOT_COUNT,
    SHOGI_POSITION_PIECE_FEATURE_COUNT,
    SHOGI_POSITION_PIECE_SLOT_COUNT,
    SHOGI_POSITION_SQUARE_COUNT,
    SHOGI_POSITION_SQUARE_FEATURE_COUNT,
)


@dataclass(frozen=True)
class ShogiPairRelationEdges:
    source_token_indices: torch.Tensor
    target_token_indices: torch.Tensor
    relation_ids: torch.Tensor
    batch_indices: torch.Tensor | None = None

    def to(self, device: torch.device) -> "ShogiPairRelationEdges":
        return ShogiPairRelationEdges(
            source_token_indices=self.source_token_indices.to(device),
            target_token_indices=self.target_token_indices.to(device),
            relation_ids=self.relation_ids.to(device),
            batch_indices=None if self.batch_indices is None else self.batch_indices.to(device),
        )


@dataclass(frozen=True)
class ShogiPositionFeatures:
    global_feature_ids: torch.Tensor
    square_feature_ids: torch.Tensor
    piece_feature_ids: torch.Tensor
    line_feature_ids: torch.Tensor
    pair_relation_edges: ShogiPairRelationEdges

    def to(self, device: torch.device) -> "ShogiPositionFeatures":
        return ShogiPositionFeatures(
            global_feature_ids=self.global_feature_ids.to(device),
            square_feature_ids=self.square_feature_ids.to(device),
            piece_feature_ids=self.piece_feature_ids.to(device),
            line_feature_ids=self.line_feature_ids.to(device),
            pair_relation_edges=self.pair_relation_edges.to(device),
        )


def stack_shogi_position_features(features: list[ShogiPositionFeatures]) -> ShogiPositionFeatures:
    return ShogiPositionFeatures(
        global_feature_ids=torch.stack([feature.global_feature_ids for feature in features]),
        square_feature_ids=torch.stack([feature.square_feature_ids for feature in features]),
        piece_feature_ids=torch.stack([feature.piece_feature_ids for feature in features]),
        line_feature_ids=torch.stack([feature.line_feature_ids for feature in features]),
        pair_relation_edges=stack_shogi_pair_relation_edges([feature.pair_relation_edges for feature in features]),
    )


def stack_shogi_pair_relation_edges(edges: list[ShogiPairRelationEdges]) -> ShogiPairRelationEdges:
    source_token_indices: list[torch.Tensor] = []
    target_token_indices: list[torch.Tensor] = []
    relation_ids: list[torch.Tensor] = []
    batch_indices: list[torch.Tensor] = []
    for batch_index, edge_set in enumerate(edges):
        edge_count = int(edge_set.relation_ids.numel())
        if edge_count == 0:
            continue
        source_token_indices.append(edge_set.source_token_indices)
        target_token_indices.append(edge_set.target_token_indices)
        relation_ids.append(edge_set.relation_ids)
        batch_indices.append(torch.full((edge_count,), batch_index, dtype=torch.long))
    if not relation_ids:
        empty = torch.empty((0,), dtype=torch.long)
        return ShogiPairRelationEdges(
            source_token_indices=empty,
            target_token_indices=empty,
            relation_ids=empty,
            batch_indices=empty,
        )
    return ShogiPairRelationEdges(
        source_token_indices=torch.cat(source_token_indices),
        target_token_indices=torch.cat(target_token_indices),
        relation_ids=torch.cat(relation_ids),
        batch_indices=torch.cat(batch_indices),
    )


def validate_shogi_position_feature_structure(features: ShogiPositionFeatures) -> None:
    _validate_integer_tensor_shape(
        "global_feature_ids",
        features.global_feature_ids,
        (SHOGI_POSITION_GLOBAL_SLOT_COUNT,),
    )
    _validate_integer_tensor_shape(
        "square_feature_ids",
        features.square_feature_ids,
        (SHOGI_POSITION_SQUARE_COUNT, SHOGI_POSITION_SQUARE_FEATURE_COUNT),
    )
    _validate_integer_tensor_shape(
        "piece_feature_ids",
        features.piece_feature_ids,
        (SHOGI_POSITION_PIECE_SLOT_COUNT, SHOGI_POSITION_PIECE_FEATURE_COUNT),
    )
    _validate_integer_tensor_shape(
        "line_feature_ids",
        features.line_feature_ids,
        (SHOGI_POSITION_LINE_SLOT_COUNT, SHOGI_POSITION_LINE_FEATURE_COUNT),
    )
    _validate_pair_relation_edge_structure(features.pair_relation_edges)


def _validate_integer_tensor_shape(name: str, tensor: object, expected_shape: tuple[int, ...]) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if not tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must use an integer dtype")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")


def _validate_pair_relation_edge_structure(edges: ShogiPairRelationEdges) -> None:
    _validate_integer_vector("pair_relation_edges.source_token_indices", edges.source_token_indices)
    _validate_integer_vector("pair_relation_edges.target_token_indices", edges.target_token_indices)
    _validate_integer_vector("pair_relation_edges.relation_ids", edges.relation_ids)
    edge_count = int(edges.relation_ids.numel())
    if int(edges.source_token_indices.numel()) != edge_count:
        raise ValueError("pair relation source and relation edge counts must match")
    if int(edges.target_token_indices.numel()) != edge_count:
        raise ValueError("pair relation target and relation edge counts must match")
    _validate_integer_vector_range(
        "pair_relation_edges.source_token_indices",
        edges.source_token_indices,
        minimum=0,
        maximum_exclusive=SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
    )
    _validate_integer_vector_range(
        "pair_relation_edges.target_token_indices",
        edges.target_token_indices,
        minimum=0,
        maximum_exclusive=SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
    )
    _validate_integer_vector_range(
        "pair_relation_edges.relation_ids",
        edges.relation_ids,
        minimum=0,
        maximum_exclusive=PAIR_RELATION_COUNT,
    )


def _validate_integer_vector(name: str, tensor: object) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if not tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must use an integer dtype")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor")


def _validate_integer_vector_range(
    name: str,
    tensor: torch.Tensor,
    *,
    minimum: int,
    maximum_exclusive: int,
) -> None:
    if tensor.numel() == 0:
        return
    if int(tensor.min().item()) < minimum or int(tensor.max().item()) >= maximum_exclusive:
        raise ValueError(f"{name} values must be in [{minimum}, {maximum_exclusive})")
