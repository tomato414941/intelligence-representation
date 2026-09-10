from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from intrep.representation.cores.transformer import SharedTransformerCore
from intrep.representation.inputs.multimodal_observation import (
    BOS,
    EOS,
    PAD,
    TEXT_VOCAB_SIZE,
    MultimodalObservation,
    MultimodalObservationInput,
    image_coordinates,
    patches_to_image,
    positions,
)
from intrep.sources.language.byte_tokenizer import ByteTokenizer


@dataclass(frozen=True)
class MultimodalAgentConfig:
    embedding_dim: int = 256
    hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 6
    memory_tokens: int = 32
    action_count: int = 5
    image_patch_size: int = 1
    audio_chunk_size: int = 128

    def __post_init__(self) -> None:
        if min(vars(self).values()) < 1 or self.embedding_dim % self.num_heads:
            raise ValueError("model dimensions must be positive, and heads must divide embedding_dim")


@dataclass
class PredictedOutcome:
    images: list[torch.Tensor | None]
    audio: torch.Tensor
    feedback: torch.Tensor  # reward, termination logit, truncation logit


class MultimodalAgentModel(nn.Module):
    """One shared predictive core for perception, memory, language and action."""

    def __init__(self, config: MultimodalAgentConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embedding_dim
        self.observation_input = MultimodalObservationInput(
            dim, config.action_count, config.image_patch_size, config.audio_chunk_size,
        )
        self.core = SharedTransformerCore(
            embedding_dim=dim, hidden_dim=config.hidden_dim,
            num_heads=config.num_heads, num_layers=config.num_layers,
        )
        self.initial_memory = nn.Parameter(torch.randn(config.memory_tokens, dim) * 0.02)
        self.memory_queries = nn.Parameter(torch.randn(config.memory_tokens, dim) * 0.02)
        self.memory_gate = nn.Linear(2 * dim, dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.requests = nn.Embedding(5, dim)  # policy, image, audio, feedback, text
        self.action_output = nn.Linear(dim, config.action_count)
        self.image_output = nn.Linear(dim, 3 * config.image_patch_size**2)
        self.audio_output = nn.Linear(dim, config.audio_chunk_size)
        self.feedback_output = nn.Linear(dim, 3)
        self.text_output = nn.Linear(dim, TEXT_VOCAB_SIZE)

    def new_memory(self, batch_size: int = 1) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return self.initial_memory.unsqueeze(0).expand(batch_size, -1, -1)

    def _memory(self, memory: torch.Tensor) -> None:
        if memory.ndim != 3 or tuple(memory.shape[1:]) != (self.config.memory_tokens, self.config.embedding_dim):
            raise ValueError("memory shape does not match the model")

    def _run(self, sequences: Sequence[torch.Tensor], *, causal: bool = False) -> list[torch.Tensor]:
        if not sequences:
            raise ValueError("batch must not be empty")
        padded = pad_sequence(list(sequences), batch_first=True)
        lengths = torch.tensor([len(sequence) for sequence in sequences], device=padded.device)
        valid = torch.arange(padded.shape[1], device=padded.device)[None] < lengths[:, None]
        # Mask padded keys. Padded queries can attend real keys, avoiding all-masked NaNs.
        bias = padded.new_zeros((len(sequences), padded.shape[1], padded.shape[1]))
        bias.masked_fill_(~valid[:, None, :], float("-inf"))
        hidden = self.core(padded, causal=causal, attention_logit_bias=bias)
        return [hidden[index, :len(sequence)] for index, sequence in enumerate(sequences)]

    def observe(
        self, observations: Sequence[MultimodalObservation], memory: torch.Tensor | None = None,
        *, step: int = 0,
    ) -> torch.Tensor:
        if memory is None:
            memory = self.new_memory(len(observations))
        self._memory(memory)
        if len(observations) != len(memory):
            raise ValueError("observation and memory batch sizes differ")
        encoded = self.observation_input.encode_many(observations, step)
        sequences = [torch.cat((row, observation, self.memory_queries)) for row, observation in zip(memory, encoded)]
        hidden = self._run(sequences)
        proposal = torch.stack([row[-self.config.memory_tokens:] for row in hidden])
        gate = self.memory_gate(torch.cat((memory, proposal), dim=-1)).sigmoid()
        return self.memory_norm(gate * proposal + (1 - gate) * memory)

    def policy(self, memory: torch.Tensor) -> torch.Tensor:
        self._memory(memory)
        query = self.requests.weight[0].unsqueeze(0)
        hidden = self._run([torch.cat((row, query)) for row in memory])
        return self.action_output(torch.stack([row[-1] for row in hidden]))

    def predict_outcome(
        self, memory: torch.Tensor, actions: torch.Tensor, *, image_shapes: Sequence[tuple[int, int] | None],
        audio_samples: int, sample_rate: int = 16000,
    ) -> PredictedOutcome:
        self._memory(memory)
        if (actions.shape != (len(memory),) or len(image_shapes) != len(memory)
                or audio_samples < 0 or sample_rate < 1
                or actions.min() < 0 or actions.max() >= self.config.action_count):
            raise ValueError("invalid action, output shape, or audio request")
        device, dim = memory.device, self.config.embedding_dim
        patch_size = self.config.image_patch_size
        chunks = math.ceil(audio_samples / self.config.audio_chunk_size)
        audio_times = (torch.arange(chunks).float() * self.config.audio_chunk_size
                       / sample_rate * 1000).unsqueeze(-1)
        audio_query = positions(audio_times, dim).to(device) + self.requests.weight[2]
        counts, coordinates = [], []
        for shape in image_shapes:
            if shape is not None and (len(shape) != 2 or min(shape) < 1):
                raise ValueError("requested image dimensions must be positive")
            if shape is None:
                coords = torch.empty((0, 2))
            else:
                patch_shape = tuple(math.ceil(size / patch_size) for size in shape)
                coords = image_coordinates(*patch_shape, device=torch.device("cpu"))
            counts.append(len(coords))
            coordinates.append(coords)
        image_queries = (positions(torch.cat(coordinates), dim).to(device) + self.requests.weight[1]).split(counts)
        action_embeddings = self.observation_input.action(actions.to(device)).unsqueeze(1)
        sequences = []
        for row, action_embedding, image_query in zip(memory, action_embeddings, image_queries):
            sequences.append(torch.cat((row, action_embedding, image_query, audio_query, self.requests.weight[3:4])))
        hidden = self._run(sequences)
        start = self.config.memory_tokens + 1
        image_hidden = torch.cat([row[start:start + count] for row, count in zip(hidden, counts)])
        patch_groups = self.image_output(image_hidden).sigmoid().split(counts)
        images = [None if shape is None else patches_to_image(patches, shape, patch_size)
                  for patches, shape in zip(patch_groups, image_shapes)]
        audio_hidden = torch.stack([row[start + count:start + count + chunks] for row, count in zip(hidden, counts)])
        waves = self.audio_output(audio_hidden).tanh().flatten(1)[:, :audio_samples]
        feedback = self.feedback_output(torch.stack([row[-1] for row in hidden]))
        return PredictedOutcome(images, waves, feedback)

    def text_logits(self, memory: torch.Tensor, prefixes: Sequence[Sequence[int]]) -> list[torch.Tensor]:
        self._memory(memory)
        if len(prefixes) != len(memory):
            raise ValueError("text and memory batch sizes differ")
        lengths = [len(prefix) + 1 for prefix in prefixes]
        ids, coordinates = [], []
        for prefix in prefixes:
            if any(token < 0 or token >= TEXT_VOCAB_SIZE or token == PAD for token in prefix):
                raise ValueError("invalid text prefix token")
            ids.extend([BOS, *prefix])
            coordinates.append(torch.arange(len(prefix) + 1).unsqueeze(-1))
        embedded = (self.observation_input.text(torch.tensor(ids, dtype=torch.long, device=memory.device))
                    + positions(torch.cat(coordinates), self.config.embedding_dim).to(memory.device)
                    + self.requests.weight[4])
        sequences = [torch.cat((row, query)) for row, query in zip(memory, embedded.split(lengths))]
        hidden = self._run(sequences, causal=True)
        logits = self.text_output(torch.cat([row[self.config.memory_tokens:] for row in hidden]))
        return list(logits.split(lengths))

    def text_loss(self, memory: torch.Tensor, targets: Sequence[str]) -> torch.Tensor:
        tokens = [ByteTokenizer().encode(text) for text in targets]
        logits = self.text_logits(memory, tokens)
        labels = torch.tensor([token for ids in tokens for token in (*ids, EOS)], device=memory.device)
        weights = torch.tensor([1 / len(row) for row in logits for _ in range(len(row))], device=memory.device)
        return (F.cross_entropy(torch.cat(logits), labels, reduction="none") * weights).sum() / len(logits)

    @torch.no_grad()
    def generate_text(self, memory: torch.Tensor, *, max_bytes: int = 128) -> list[str]:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        prefixes: list[list[int]] = [[] for _ in memory]
        done = [False] * len(memory)
        for _ in range(max_bytes):
            logits = self.text_logits(memory, prefixes)
            for index, row in enumerate(logits):
                if done[index]:
                    continue
                scores = row[-1].clone()
                scores[PAD] = scores[BOS] = float("-inf")
                token = int(scores.argmax())
                if token == EOS:
                    done[index] = True
                else:
                    prefixes[index].append(token)
            if all(done):
                break
        return [ByteTokenizer().decode(prefix) for prefix in prefixes]
