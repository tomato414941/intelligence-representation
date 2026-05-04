from __future__ import annotations


TRANSFORMER_CORE_PRESETS: dict[str, dict[str, int | float]] = {
    "tiny": {
        "embedding_dim": 8,
        "num_heads": 2,
        "hidden_dim": 16,
        "num_layers": 1,
        "dropout": 0.0,
    },
    "small": {
        "embedding_dim": 32,
        "num_heads": 4,
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.0,
    },
}
