from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.representation.cores.transformer import SharedTransformerCore


@dataclass(frozen=True)
class CellularRuleInferenceModelConfig:
    height: int = 6
    width: int = 6
    max_context: int = 8
    embedding_dim: int = 256
    hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 6


class CellularRuleInferenceModel(nn.Module):
    """Raw aligned before/after cell pairs plus an unanswered query board.

    No rule IDs, neighbor counts, local receptive fields, or rule tables enter
    the model. Spatial coordinates, example grouping and cell alignment are explicit.
    """

    def __init__(self, config: CellularRuleInferenceModelConfig) -> None:
        super().__init__()
        self.config = config
        self.cell_projection = nn.Linear(2, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.height * config.width, config.embedding_dim)
        self.example_embedding = nn.Embedding(config.max_context + 1, config.embedding_dim)
        self.role_embedding = nn.Embedding(2, config.embedding_dim)
        self.core = SharedTransformerCore(
            embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
            num_heads=config.num_heads, num_layers=config.num_layers,
        )
        self.next_cell_output = nn.Linear(config.embedding_dim, 2)

    def forward(self, support: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if query.ndim != 3 or tuple(query.shape[-2:]) != (cfg.height, cfg.width):
            raise ValueError("query must have shape [batch, height, width]")
        if support.ndim != 5 or support.shape[0] != query.shape[0] or tuple(support.shape[2:]) != (2, cfg.height, cfg.width):
            raise ValueError("support must have shape [batch, context, 2, height, width]")
        batch, count = support.shape[:2]
        if count > cfg.max_context:
            raise ValueError("too many context examples")
        cells = cfg.height * cfg.width
        query_pair = torch.stack((query, torch.zeros_like(query)), dim=1).unsqueeze(1)
        frames = torch.cat((support, query_pair), dim=1)
        pairs = frames.permute(0, 1, 3, 4, 2).reshape(batch, count + 1, cells, 2)
        positions = self.position_embedding(torch.arange(cells, device=query.device))
        example_ids = torch.cat((torch.arange(count, device=query.device), query.new_tensor([cfg.max_context], dtype=torch.long)))
        roles = torch.cat((torch.zeros(count, dtype=torch.long, device=query.device), torch.ones(1, dtype=torch.long, device=query.device)))
        embedded = (self.cell_projection(pairs) + positions[None, None]
                    + self.example_embedding(example_ids)[None, :, None]
                    + self.role_embedding(roles)[None, :, None])
        hidden = self.core(embedded.reshape(batch, (count + 1) * cells, -1))
        return self.next_cell_output(hidden[:, -cells:])
