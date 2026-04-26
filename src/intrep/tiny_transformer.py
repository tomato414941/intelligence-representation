from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.dataset import ActionConditionedExample
from intrep.sequence import SequenceExample, sequences_from_examples
from intrep.tokens import fact_from_token, model_input_tokens
from intrep.torch_sequence import SequenceVocabulary, build_vocabulary, max_input_length
from intrep.types import Action, Fact


@dataclass(frozen=True)
class TinyTransformerConfig:
    embedding_dim: int = 16
    num_heads: int = 2
    hidden_dim: int = 32
    num_layers: int = 1
    epochs: int = 35
    learning_rate: float = 0.05
    seed: int = 7


class TinyTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        target_size: int,
        max_length: int,
        config: TinyTransformerConfig,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(max_length, config.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output = nn.Linear(config.embedding_dim, target_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        encoded = self.encoder(hidden)
        pooled = encoded[:, -1, :]
        return self.output(pooled)


class TinyTransformerPredictor:
    def __init__(self, config: TinyTransformerConfig | None = None) -> None:
        self.config = config or TinyTransformerConfig()
        self.vocabulary: SequenceVocabulary | None = None
        self.max_length = 1
        self.model: TinyTransformerModel | None = None

    def fit(self, examples: list[ActionConditionedExample]) -> None:
        torch.manual_seed(self.config.seed)
        sequences = sequences_from_examples(examples)
        self.vocabulary = build_vocabulary(sequences)
        self.max_length = max_input_length(sequences)
        self.model = TinyTransformerModel(
            vocab_size=len(self.vocabulary.token_to_id),
            target_size=len(self.vocabulary.target_to_id),
            max_length=self.max_length,
            config=self.config,
        )
        inputs, targets = _tensorize(sequences, self.vocabulary, self.max_length)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.config.epochs):
            optimizer.zero_grad()
            logits = self.model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

    def predict(self, state: list[Fact], action: Action) -> Fact | None:
        if self.vocabulary is None or self.model is None:
            return None
        sequence = SequenceExample(
            id="prediction",
            input_tokens=model_input_tokens(state, action),
            target_token="",
            source="runtime",
        )
        token_ids = torch.tensor(
            [self.vocabulary.encode_tokens(sequence.input_tokens, self.max_length)],
            dtype=torch.long,
        )
        self.model.eval()
        with torch.no_grad():
            target_id = int(self.model(token_ids).argmax(dim=-1).item())
        return fact_from_token(self.vocabulary.decode_target(target_id))


def _tensorize(
    sequences: list[SequenceExample],
    vocabulary: SequenceVocabulary,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor(
        [vocabulary.encode_tokens(sequence.input_tokens, max_length) for sequence in sequences],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [vocabulary.encode_target(sequence.target_token) for sequence in sequences],
        dtype=torch.long,
    )
    return inputs, targets
