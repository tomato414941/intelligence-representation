from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from intrep.representation.inputs.multimodal_observation import (
    image_coordinates,
    image_patches,
    positions,
)


class FeatureSequenceInput(nn.Module):
    """Project a source's feature vectors, without interpreting their semantics."""

    def __init__(self, features: int, dimension: int) -> None:
        super().__init__()
        self.projection = nn.Linear(features, dimension)
        self.tag = nn.Parameter(torch.randn(dimension) * 0.02)
        self.dimension = dimension

    def forward(self, features: torch.Tensor, coordinates: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.projection.in_features:
            raise ValueError("features must have shape [batch, sequence, feature]")
        if coordinates is None:
            coordinates = torch.arange(features.shape[1], device=features.device).view(1, -1, 1)
        if coordinates.shape[:-1] not in (features.shape[:-1], (1, features.shape[1])):
            raise ValueError("coordinates must align with the feature sequence")
        return self.projection(features) + self.tag + 0.02 * positions(coordinates, self.dimension).to(features)


class ImageSequenceInput(nn.Module):
    def __init__(self, dimension: int, patch_size: int) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch size must be positive")
        self.patch_size = patch_size
        self.features = FeatureSequenceInput(3 * patch_size**2, dimension)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("image must be a nonempty HWC RGB tensor")
        patches, shape = image_patches(image, self.patch_size)
        coordinates = image_coordinates(*shape, device=image.device).unsqueeze(0)
        return self.features(patches.unsqueeze(0), coordinates)


class WaveformSequenceInput(nn.Module):
    def __init__(self, dimension: int, chunk_size: int) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError("audio chunk size must be positive")
        self.chunk_size = chunk_size
        self.features = FeatureSequenceInput(chunk_size, dimension)
        self.sample_rate = nn.Linear(1, dimension, bias=False)

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.ndim != 1 or not waveform.numel() or sample_rate < 1:
            raise ValueError("audio must be nonempty mono samples with a positive rate")
        chunks = F.pad(waveform, (0, -len(waveform) % self.chunk_size)).reshape(1, -1, self.chunk_size)
        times = torch.arange(chunks.shape[1], device=waveform.device).view(1, -1, 1) * self.chunk_size / sample_rate
        rate = waveform.new_tensor([[[math.log(sample_rate / 16000)]]])
        return self.features(chunks, times) + self.sample_rate(rate)


class CoordinateQueryInput(nn.Module):
    """Ask for outputs at coordinates without exposing any target values."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dimension) * 0.02)
        self.dimension = dimension

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 3 or min(coordinates.shape) < 1:
            raise ValueError("query coordinates must have shape [batch, sequence, axes]")
        return self.query + 0.02 * positions(coordinates, self.dimension).to(self.query)
