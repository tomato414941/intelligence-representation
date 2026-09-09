"""Batched cellular ground truth; never used inside a learned model."""

from __future__ import annotations

import numpy as np

from intrep.worlds.cellular.world import CellularRule


def neighborhood_keys(grids: np.ndarray) -> np.ndarray:
    """Encode (current cell, live neighbor count), with dead borders."""
    grids = np.asarray(grids, dtype=np.int64)
    padded = np.pad(grids, [(0, 0)] * (grids.ndim - 2) + [(1, 1), (1, 1)])
    height, width = grids.shape[-2:]
    counts = np.zeros_like(grids)
    for row in range(3):
        for col in range(3):
            if (row, col) != (1, 1):
                counts += padded[..., row:row + height, col:col + width]
    return grids * 9 + counts


def rule_table(rule: CellularRule) -> np.ndarray:
    return np.array([int(n in rule.birth) for n in range(9)]
                    + [int(n in rule.survival) for n in range(9)], dtype=np.int64)


def step_grids(grids: np.ndarray, rules: list[CellularRule]) -> np.ndarray:
    """Apply one rule per leading batch item to one or more grids."""
    if len(grids) != len(rules):
        raise ValueError("one rule is required per batch item")
    tables = np.stack([rule_table(rule) for rule in rules])
    keys = neighborhood_keys(grids)
    batch_indices = np.arange(len(grids)).reshape((-1,) + (1,) * (grids.ndim - 1))
    return tables[batch_indices, keys]
