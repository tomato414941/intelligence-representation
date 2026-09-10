from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from intrep.sources.language.byte_tokenizer import ByteTokenizer

PAD, BOS, EOS = 256, 257, 258
TEXT_VOCAB_SIZE = 259


@dataclass(frozen=True)
class MultimodalObservation:
    """Model-side inputs, not a schema for source-side experience records."""

    text: str = ""
    image: torch.Tensor | None = None  # [height, width, 3], values in [0, 1]
    audio: torch.Tensor | None = None  # mono waveform, values in [-1, 1]
    sample_rate: int = 16000
    previous_action: int | None = None
    feedback: torch.Tensor | None = None  # reward, terminated, truncated


def positions(coordinates: torch.Tensor, dimension: int) -> torch.Tensor:
    """Fourier coordinates have no learned maximum sequence or image size."""
    axes = coordinates.shape[-1]
    count = math.ceil(dimension / (2 * axes))
    frequencies = torch.exp(
        torch.arange(count, device=coordinates.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, count - 1))
    )
    angles = coordinates.float().unsqueeze(-1) * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :dimension]


def image_coordinates(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    rows, cols = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij",
    )
    return torch.stack((rows.flatten(), cols.flatten()), dim=-1)


def image_patches(image: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width, channels = image.shape
    if channels != 3 or min(height, width) < 1:
        raise ValueError("image must have shape [positive height, positive width, 3]")
    if patch_size == 1:
        return image.reshape(-1, 3), (height, width)
    padded = F.pad(image.permute(2, 0, 1), (0, -width % patch_size, 0, -height % patch_size))
    patches = F.unfold(padded.unsqueeze(0), kernel_size=patch_size, stride=patch_size)
    return patches[0].T, (math.ceil(height / patch_size), math.ceil(width / patch_size))


def patches_to_image(patches: torch.Tensor, shape: tuple[int, int], patch_size: int) -> torch.Tensor:
    height, width = shape
    if patch_size == 1:
        return patches.reshape(height, width, 3)
    padded_shape = (math.ceil(height / patch_size) * patch_size, math.ceil(width / patch_size) * patch_size)
    image = F.fold(patches.T.unsqueeze(0), padded_shape, kernel_size=patch_size, stride=patch_size)
    return image[0, :, :height, :width].permute(1, 2, 0)


class MultimodalObservationInput(nn.Module):
    """Native modality adapters meeting at the existing embedding boundary."""

    def __init__(self, dimension: int, action_count: int, image_patch_size: int, audio_chunk_size: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.image_patch_size = image_patch_size
        self.audio_chunk_size = audio_chunk_size
        self.action_count = action_count
        self.text = nn.Embedding(TEXT_VOCAB_SIZE, dimension, padding_idx=PAD)
        self.image = nn.Linear(3 * image_patch_size**2, dimension)
        self.audio = nn.Linear(audio_chunk_size, dimension)
        self.action = nn.Embedding(action_count, dimension)
        self.feedback = nn.Linear(3, dimension)
        self.modality = nn.Embedding(5, dimension)
        self.image_extent = nn.Linear(2, dimension, bias=False)
        self.audio_rate = nn.Linear(1, dimension, bias=False)

    def forward(self, observation: MultimodalObservation, step: int) -> torch.Tensor:
        return self.encode_many([observation], step)[0]

    def encode_many(self, observations: Sequence[MultimodalObservation], step: int) -> list[torch.Tensor]:
        if not observations or step < 0:
            raise ValueError("observations must not be empty and step must be nonnegative")
        device = self.modality.weight.device
        groups: list[list[tuple]] = [[] for _ in range(5)]
        cpu = torch.device("cpu")
        for index, observation in enumerate(observations):
            if observation.text:
                ids = torch.tensor(ByteTokenizer().encode(observation.text))
                groups[0].append((index, ids, torch.arange(len(ids)).unsqueeze(-1), None))
            if observation.image is not None:
                image = observation.image
                if image.ndim != 3 or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError("image must be a finite HWC tensor in [0,1]")
                patches, shape = image_patches(image.float(), self.image_patch_size)
                coords = image_coordinates(*shape, device=cpu)
                extent = torch.tensor(shape, dtype=torch.float32).log1p().expand(len(patches), -1)
                groups[1].append((index, patches, coords, extent))
            if observation.audio is not None:
                audio = observation.audio
                if (audio.ndim != 1 or not audio.numel() or observation.sample_rate <= 0
                        or not torch.isfinite(audio).all() or audio.abs().max() > 1):
                    raise ValueError("audio must be a nonempty finite mono waveform in [-1,1] with positive sample rate")
                chunks = F.pad(audio.float(), (0, -len(audio) % self.audio_chunk_size)).reshape(-1, self.audio_chunk_size)
                times = (torch.arange(len(chunks)).float() * self.audio_chunk_size / observation.sample_rate * 1000).unsqueeze(-1)
                rate = torch.full((len(chunks), 1), math.log(observation.sample_rate / 16000))
                groups[2].append((index, chunks, times, rate))
            if observation.previous_action is not None:
                if not 0 <= observation.previous_action < self.action_count:
                    raise ValueError("previous action is outside the configured action vocabulary")
                groups[3].append((index, torch.tensor([observation.previous_action]), None, None))
            if observation.feedback is not None:
                feedback = observation.feedback
                if feedback.shape != (3,) or not torch.isfinite(feedback).all():
                    raise ValueError("feedback must contain finite reward, terminated and truncated values")
                groups[4].append((index, feedback.float().unsqueeze(0), None, None))
        pieces: list[list[torch.Tensor]] = [[] for _ in observations]
        for kind, (group, adapter) in enumerate(zip(groups, (self.text, self.image, self.audio, self.action, self.feedback))):
            if not group:
                continue
            source_device = group[0][1].device
            raw = torch.cat([item[1].to(source_device) for item in group]).to(device)
            embedded = adapter(raw) + self.modality.weight[kind]
            if kind < 3:
                coords = torch.cat([item[2] for item in group])
                embedded = embedded + positions(coords, self.dimension).to(device)
            if kind in (1, 2):
                extra = torch.cat([item[3] for item in group]).to(device)
                embedded = embedded + (self.image_extent(extra) if kind == 1 else self.audio_rate(extra))
            for item, tokens in zip(group, embedded.split([len(item[1]) for item in group])):
                pieces[item[0]].append(tokens)
        if any(not sequence for sequence in pieces):
            raise ValueError("an observation must contain at least one input")
        time = positions(torch.tensor([[step]], device=device), self.dimension)
        return [torch.cat(sequence, dim=0) + time for sequence in pieces]
