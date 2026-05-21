from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShogiPairRelationEdges:
    source_element_indices: torch.Tensor
    target_element_indices: torch.Tensor
    relation_ids: torch.Tensor
    batch_indices: torch.Tensor | None = None

    def to(self, device: torch.device) -> "ShogiPairRelationEdges":
        return ShogiPairRelationEdges(
            source_element_indices=self.source_element_indices.to(device),
            target_element_indices=self.target_element_indices.to(device),
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
    source_element_indices: list[torch.Tensor] = []
    target_element_indices: list[torch.Tensor] = []
    relation_ids: list[torch.Tensor] = []
    batch_indices: list[torch.Tensor] = []
    for batch_index, edge_set in enumerate(edges):
        edge_count = int(edge_set.relation_ids.numel())
        if edge_count == 0:
            continue
        source_element_indices.append(edge_set.source_element_indices)
        target_element_indices.append(edge_set.target_element_indices)
        relation_ids.append(edge_set.relation_ids)
        batch_indices.append(torch.full((edge_count,), batch_index, dtype=torch.long))
    if not relation_ids:
        empty = torch.empty((0,), dtype=torch.long)
        return ShogiPairRelationEdges(
            source_element_indices=empty,
            target_element_indices=empty,
            relation_ids=empty,
            batch_indices=empty,
        )
    return ShogiPairRelationEdges(
        source_element_indices=torch.cat(source_element_indices),
        target_element_indices=torch.cat(target_element_indices),
        relation_ids=torch.cat(relation_ids),
        batch_indices=torch.cat(batch_indices),
    )


def validate_integer_tensor_shape(name: str, tensor: object, expected_shape: tuple[int, ...]) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if not tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must use an integer dtype")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")


def validate_empty_pair_relation_edges(edges: ShogiPairRelationEdges, *, context: str) -> None:
    if int(edges.relation_ids.numel()) != 0:
        raise ValueError(f"{context} position features must not contain pair relation edges")


def validate_pair_relation_edge_structure(
    edges: ShogiPairRelationEdges,
    *,
    element_count: int,
    relation_count: int,
) -> None:
    validate_integer_vector("pair_relation_edges.source_element_indices", edges.source_element_indices)
    validate_integer_vector("pair_relation_edges.target_element_indices", edges.target_element_indices)
    validate_integer_vector("pair_relation_edges.relation_ids", edges.relation_ids)
    edge_count = int(edges.relation_ids.numel())
    if int(edges.source_element_indices.numel()) != edge_count:
        raise ValueError("pair relation source and relation edge counts must match")
    if int(edges.target_element_indices.numel()) != edge_count:
        raise ValueError("pair relation target and relation edge counts must match")
    validate_integer_vector_range(
        "pair_relation_edges.source_element_indices",
        edges.source_element_indices,
        minimum=0,
        maximum_exclusive=element_count,
    )
    validate_integer_vector_range(
        "pair_relation_edges.target_element_indices",
        edges.target_element_indices,
        minimum=0,
        maximum_exclusive=element_count,
    )
    validate_integer_vector_range(
        "pair_relation_edges.relation_ids",
        edges.relation_ids,
        minimum=0,
        maximum_exclusive=relation_count,
    )


def validate_integer_vector(name: str, tensor: object) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if not tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must use an integer dtype")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor")


def validate_integer_vector_range(
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
