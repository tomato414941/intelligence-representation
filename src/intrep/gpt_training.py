from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.byte_tokenizer import ByteTokenizer
from intrep.gpt_model import DecoderOnlyGPT, GPTConfig
from intrep.mixed_corpus import MixedDocument, default_mixed_documents, render_corpus


@dataclass(frozen=True)
class GPTTrainingConfig:
    context_length: int = 64
    batch_size: int = 8
    max_steps: int = 20
    learning_rate: float = 0.003
    seed: int = 7


@dataclass(frozen=True)
class GPTTrainingResult:
    initial_loss: float
    final_loss: float
    steps: int
    token_count: int


def train_mixed_gpt(
    documents: list[MixedDocument] | None = None,
    training_config: GPTTrainingConfig | None = None,
    model_config: GPTConfig | None = None,
) -> GPTTrainingResult:
    config = training_config or GPTTrainingConfig()
    torch.manual_seed(config.seed)
    tokenizer = ByteTokenizer()
    corpus = render_corpus(documents or default_mixed_documents())
    token_ids = tokenizer.encode(corpus)
    inputs, targets = language_model_batches(
        token_ids=token_ids,
        context_length=config.context_length,
        batch_size=config.batch_size,
    )
    gpt_config = model_config or GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=config.context_length,
    )
    model = DecoderOnlyGPT(gpt_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    initial_loss: float | None = None
    final_loss = 0.0

    model.train()
    for step in range(config.max_steps):
        batch_index = step % inputs.size(0)
        batch_inputs = inputs[batch_index]
        batch_targets = targets[batch_index]
        optimizer.zero_grad()
        logits = model(batch_inputs)
        loss = loss_fn(logits.reshape(-1, tokenizer.vocab_size), batch_targets.reshape(-1))
        if initial_loss is None:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())
        loss.backward()
        optimizer.step()

    return GPTTrainingResult(
        initial_loss=initial_loss if initial_loss is not None else 0.0,
        final_loss=final_loss,
        steps=config.max_steps,
        token_count=len(token_ids),
    )


def language_model_batches(
    token_ids: list[int],
    context_length: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) <= context_length:
        raise ValueError("token_ids must be longer than context_length")

    input_rows = []
    target_rows = []
    for start in range(0, len(token_ids) - context_length, context_length):
        window = token_ids[start : start + context_length + 1]
        if len(window) != context_length + 1:
            continue
        input_rows.append(window[:-1])
        target_rows.append(window[1:])
    if not input_rows:
        raise ValueError("not enough tokens to build language-model batches")

    usable = (len(input_rows) // batch_size) * batch_size
    if usable == 0:
        usable = len(input_rows)
        batch_size = len(input_rows)
    inputs = torch.tensor(input_rows[:usable], dtype=torch.long).reshape(
        -1,
        batch_size,
        context_length,
    )
    targets = torch.tensor(target_rows[:usable], dtype=torch.long).reshape(
        -1,
        batch_size,
        context_length,
    )
    return inputs, targets
