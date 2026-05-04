from __future__ import annotations

import torch


def concatenate_input_embedding_sequences(*sequences: torch.Tensor) -> torch.Tensor:
    if not sequences:
        raise ValueError("at least one input embedding sequence is required")
    first = sequences[0]
    _validate_input_embedding_sequence(first)
    batch_size = first.size(0)
    hidden_size = first.size(2)
    for sequence in sequences[1:]:
        _validate_input_embedding_sequence(sequence)
        if sequence.size(0) != batch_size:
            raise ValueError("input embedding sequences must have the same batch size")
        if sequence.size(2) != hidden_size:
            raise ValueError("input embedding sequences must have the same hidden size")
    return torch.cat(sequences, dim=1)


def _validate_input_embedding_sequence(sequence: torch.Tensor) -> None:
    if sequence.ndim != 3:
        raise ValueError("input embedding sequence must have shape [batch, sequence, hidden]")
    if not torch.is_floating_point(sequence):
        raise ValueError("input embedding sequence must be floating point")
    if sequence.numel() == 0:
        raise ValueError("input embedding sequence must not be empty")
