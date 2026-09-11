from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def load_pretraining(directory: Path):
    provenance = json.loads((directory / 'provenance.json').read_text())
    path = directory / 'tokens.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance['token_sha256']:
        raise ValueError('pretraining token checksum differs')
    if set(provenance['train']['document_hashes']).intersection(provenance['validation']['document_hashes']):
        raise ValueError('pretraining documents cross splits')
    with np.load(path, allow_pickle=False) as archive:
        splits = {name: archive[name] for name in ('train', 'validation')}
    for name, tokens in splits.items():
        if (tokens.ndim != 1 or not len(tokens) or not np.issubdtype(tokens.dtype, np.integer)
                or np.any((tokens < 0) | ((tokens > 255) & (tokens != 258)))):
            raise ValueError('invalid byte corpus')
        if len(tokens) != provenance[name]['tokens']:
            raise ValueError('pretraining token count differs')
    return splits, provenance


def sample_blocks(tokens: np.ndarray, batch_size: int, block_bytes: int, generator: torch.Generator) -> torch.Tensor:
    if min(batch_size, block_bytes) < 1 or len(tokens) < block_bytes:
        raise ValueError('corpus must fill a nonempty token block batch')
    starts = torch.randint(len(tokens) - block_bytes + 1, (batch_size,), generator=generator).tolist()
    return torch.from_numpy(np.stack([tokens[start:start + block_bytes] for start in starts]).astype(np.int64))
