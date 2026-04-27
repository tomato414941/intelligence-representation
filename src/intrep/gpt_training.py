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
    loss_history: tuple[float, ...] = ()
    initial_eval_loss: float | None = None
    final_eval_loss: float | None = None

    @property
    def best_loss(self) -> float:
        if not self.loss_history:
            return 0.0
        return min(self.loss_history)

    @property
    def loss_reduction(self) -> float:
        return self.initial_loss - self.final_loss

    @property
    def loss_reduction_ratio(self) -> float:
        if self.initial_loss == 0.0:
            return 0.0
        return self.loss_reduction / self.initial_loss


def train_mixed_gpt(
    documents: list[MixedDocument] | None = None,
    training_config: GPTTrainingConfig | None = None,
    model_config: GPTConfig | None = None,
    eval_documents: list[MixedDocument] | None = None,
) -> GPTTrainingResult:
    config = training_config or GPTTrainingConfig()
    _validate_training_config(config)
    torch.manual_seed(config.seed)
    tokenizer = ByteTokenizer()
    corpus_documents = documents if documents is not None else default_mixed_documents()
    if not corpus_documents:
        raise ValueError("documents must not be empty")
    corpus = render_corpus(corpus_documents)
    token_ids = tokenizer.encode(corpus)
    inputs, targets = language_model_batches(
        token_ids=token_ids,
        context_length=config.context_length,
        batch_size=config.batch_size,
    )
    eval_inputs: torch.Tensor | None = None
    eval_targets: torch.Tensor | None = None
    if eval_documents is not None:
        if not eval_documents:
            raise ValueError("eval_documents must not be empty")
        eval_token_ids = tokenizer.encode(render_corpus(eval_documents))
        eval_inputs, eval_targets = language_model_batches(
            token_ids=eval_token_ids,
            context_length=config.context_length,
            batch_size=config.batch_size,
        )
    gpt_config = model_config or GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=config.context_length,
    )
    if gpt_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model_config.vocab_size must match the byte tokenizer vocab size")
    if gpt_config.context_length != config.context_length:
        raise ValueError("model_config.context_length must match training_config.context_length")
    model = DecoderOnlyGPT(gpt_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    initial_loss: float | None = None
    final_loss = 0.0
    loss_history: list[float] = []
    initial_eval_loss = _evaluate_loss(model, loss_fn, eval_inputs, eval_targets)

    model.train()
    for step in range(config.max_steps):
        batch_index = step % inputs.size(0)
        batch_inputs = inputs[batch_index]
        batch_targets = targets[batch_index]
        optimizer.zero_grad()
        logits = model(batch_inputs)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1))
        if initial_loss is None:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())
        loss_history.append(final_loss)
        loss.backward()
        optimizer.step()

    final_eval_loss = _evaluate_loss(model, loss_fn, eval_inputs, eval_targets)

    return GPTTrainingResult(
        initial_loss=initial_loss if initial_loss is not None else 0.0,
        final_loss=final_loss,
        steps=config.max_steps,
        token_count=len(token_ids),
        loss_history=tuple(loss_history),
        initial_eval_loss=initial_eval_loss,
        final_eval_loss=final_eval_loss,
    )


def _evaluate_loss(
    model: DecoderOnlyGPT,
    loss_fn: nn.CrossEntropyLoss,
    inputs: torch.Tensor | None,
    targets: torch.Tensor | None,
) -> float | None:
    if inputs is None or targets is None:
        return None

    was_training = model.training
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_targets in zip(inputs, targets, strict=True):
            logits = model(batch_inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1))
            total_loss += float(loss.item())
    if was_training:
        model.train()
    return total_loss / inputs.size(0)


def _validate_training_config(config: GPTTrainingConfig) -> None:
    if config.context_length <= 0:
        raise ValueError("context_length must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")


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
