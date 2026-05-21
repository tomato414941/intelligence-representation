from __future__ import annotations

import torch
from torch import nn


class ImagePatchInputLayer(nn.Module):
    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        patch_size: int,
        embedding_dim: int,
        channel_count: int = 1,
    ) -> None:
        super().__init__()
        height, width = image_size
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if channel_count <= 0:
            raise ValueError("channel_count must be positive")
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("image dimensions must be divisible by patch_size")
        patch_dim = patch_size * patch_size * channel_count
        patch_count = (height // patch_size) * (width // patch_size)
        self.image_size = image_size
        self.channel_count = channel_count
        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_dim, embedding_dim)
        self.position_embedding = nn.Embedding(patch_count, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim not in (3, 4):
            raise ValueError("images must have shape [batch, height, width] or [batch, height, width, channels]")
        if tuple(images.shape[1:]) != self.image_size:
            if images.ndim == 3 or tuple(images.shape[1:3]) != self.image_size:
                raise ValueError("images do not match input layer image_size")
        channel_count = 1 if images.ndim == 3 else int(images.shape[3])
        if channel_count != self.channel_count:
            raise ValueError("images do not match input layer channel_count")
        patches = patchify_images(images, self.patch_size)
        positions = torch.arange(patches.size(1), device=images.device).unsqueeze(0)
        return self.patch_embedding(patches) + self.position_embedding(positions)


def patchify_images(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.ndim == 3:
        patches = images.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        return patches.contiguous().view(images.size(0), -1, patch_size * patch_size)
    if images.ndim == 4:
        patches = images.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.permute(0, 1, 2, 4, 5, 3)
        return patches.contiguous().view(
            images.size(0),
            -1,
            patch_size * patch_size * images.size(3),
        )
    raise ValueError("images must have shape [batch, height, width] or [batch, height, width, channels]")
