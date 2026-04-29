from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from intrep.byte_tokenizer import ByteTokenizer
from intrep.gpt_model import DecoderOnlyGPT, GPTConfig, gpt_config_to_dict
from intrep.text_examples import LanguageModelingExample, language_modeling_corpus_from_examples
from intrep.text_tokenizer import TextTokenizerKind, build_text_tokenizer

GPTTrainingDevice = Literal["auto", "cpu", "cuda"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPTTrainingConfig:
    context_length: int = 64
    batch_stride: int | None = None
    batch_size: int = 8
    max_steps: int = 20
    learning_rate: float = 0.003
    seed: int = 7
    device: GPTTrainingDevice = "cpu"
    checkpoint_path: Path | None = None
    tokenizer: TextTokenizerKind = "byte"
    tokenizer_vocab_size: int = 512
    tokenizer_min_pair_count: int = 2


@dataclass(frozen=True)
class GPTTrainingResult:
    initial_loss: float
    final_loss: float
    steps: int
    token_count: int
    loss_history: tuple[float, ...] = ()
    initial_train_loss: float | None = None
    final_train_loss: float | None = None
    initial_eval_loss: float | None = None
    final_eval_loss: float | None = None
    eval_split: str = "train"
    generalization_eval: bool = False
    warnings: tuple[str, ...] = ()
    device: str = "cpu"

    @property
    def initial_step_loss(self) -> float:
        return self.initial_loss

    @property
    def final_step_loss(self) -> float:
        return self.final_loss

    @property
    def best_loss(self) -> float:
        if not self.loss_history:
            return 0.0
        return min(self.loss_history)

    @property
    def best_step_loss(self) -> float:
        return self.best_loss

    @property
    def loss_reduction(self) -> float:
        return self.initial_loss - self.final_loss

    @property
    def step_loss_reduction(self) -> float:
        return self.loss_reduction

    @property
    def loss_reduction_ratio(self) -> float:
        if self.initial_loss == 0.0:
            return 0.0
        return self.loss_reduction / self.initial_loss

    @property
    def step_loss_reduction_ratio(self) -> float:
        return self.loss_reduction_ratio


@dataclass(frozen=True)
class GPTTrainingArtifacts:
    result: GPTTrainingResult
    model: DecoderOnlyGPT
    tokenizer: object


def train_language_modeling_gpt_with_artifacts(
    *,
    train_examples: list[LanguageModelingExample] | tuple[LanguageModelingExample, ...],
    training_config: GPTTrainingConfig | None = None,
    model_config: GPTConfig | None = None,
    eval_examples: list[LanguageModelingExample] | tuple[LanguageModelingExample, ...] | None = None,
) -> GPTTrainingArtifacts:
    return train_rendered_gpt_with_artifacts(
        corpus=language_modeling_corpus_from_examples(train_examples),
        training_config=training_config,
        model_config=model_config,
        eval_corpus=language_modeling_corpus_from_examples(eval_examples) if eval_examples is not None else None,
    )


def train_rendered_gpt_with_artifacts(
    *,
    corpus: str,
    training_config: GPTTrainingConfig | None = None,
    model_config: GPTConfig | None = None,
    eval_corpus: str | None = None,
) -> GPTTrainingArtifacts:
    config = training_config or GPTTrainingConfig()
    _validate_training_config(config)
    torch.manual_seed(config.seed)
    device = resolve_training_device(config.device)
    if not corpus:
        raise ValueError("corpus must not be empty")
    tokenizer = build_text_tokenizer(
        corpus,
        kind=config.tokenizer,
        vocab_size=config.tokenizer_vocab_size,
        min_pair_count=config.tokenizer_min_pair_count,
    )
    token_ids = tokenizer.encode(corpus)
    inputs, targets = language_model_batches(
        token_ids=token_ids,
        context_length=config.context_length,
        batch_size=config.batch_size,
        batch_stride=config.batch_stride,
    )
    inputs = inputs.to(device)
    targets = targets.to(device)
    eval_inputs: torch.Tensor | None = None
    eval_targets: torch.Tensor | None = None
    if eval_corpus is not None:
        if not eval_corpus:
            raise ValueError("eval_corpus must not be empty")
        eval_token_ids = tokenizer.encode(eval_corpus)
        eval_inputs, eval_targets = language_model_batches(
            token_ids=eval_token_ids,
            context_length=config.context_length,
            batch_size=config.batch_size,
            batch_stride=config.batch_stride,
        )
        eval_inputs = eval_inputs.to(device)
        eval_targets = eval_targets.to(device)
    gpt_config = model_config or GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=config.context_length,
    )
    if gpt_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model_config.vocab_size must match the tokenizer vocab size")
    if gpt_config.context_length != config.context_length:
        raise ValueError("model_config.context_length must match training_config.context_length")
    model = DecoderOnlyGPT(gpt_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    initial_loss: float | None = None
    final_loss = 0.0
    loss_history: list[float] = []
    initial_train_loss = _evaluate_loss(model, loss_fn, inputs, targets)
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

    final_train_loss = _evaluate_loss(model, loss_fn, inputs, targets)
    final_eval_loss = _evaluate_loss(model, loss_fn, eval_inputs, eval_targets)
    generalization_eval = eval_corpus is not None
    warnings = ()
    if not generalization_eval:
        warnings = (
            "No held-out eval corpus was provided; reported evaluation is train-split only and is not a generalization estimate.",
        )

    result = GPTTrainingResult(
        initial_loss=initial_loss if initial_loss is not None else 0.0,
        final_loss=final_loss,
        steps=config.max_steps,
        token_count=len(token_ids),
        loss_history=tuple(loss_history),
        initial_train_loss=initial_train_loss,
        final_train_loss=final_train_loss,
        initial_eval_loss=initial_eval_loss,
        final_eval_loss=final_eval_loss,
        eval_split="held_out" if generalization_eval else "train",
        generalization_eval=generalization_eval,
        warnings=warnings,
        device=str(device),
    )
    if config.checkpoint_path is not None:
        save_gpt_checkpoint(
            path=config.checkpoint_path,
            model=model,
            model_config=gpt_config,
            training_config=config,
            result=result,
        )
    return GPTTrainingArtifacts(result=result, model=model, tokenizer=tokenizer)


def resolve_training_device(requested_device: GPTTrainingDevice) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is false")
    return torch.device(requested_device)


def save_gpt_checkpoint(
    *,
    path: Path,
    model: DecoderOnlyGPT,
    model_config: GPTConfig,
    training_config: GPTTrainingConfig,
    result: GPTTrainingResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    training_payload = asdict(training_config)
    if training_config.checkpoint_path is not None:
        training_payload["checkpoint_path"] = str(training_config.checkpoint_path)
    payload = {
        "schema_version": "intrep.gpt_checkpoint.v1",
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "model_config": gpt_config_to_dict(model_config),
        "training_config": training_payload,
        "result": asdict(result),
    }
    torch.save(payload, path)


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
    if config.batch_stride is not None and config.batch_stride <= 0:
        raise ValueError("batch_stride must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda")
    if config.tokenizer not in ("byte", "byte-pair"):
        raise ValueError("tokenizer must be one of: byte, byte-pair")
    if config.tokenizer_vocab_size <= 0:
        raise ValueError("tokenizer_vocab_size must be positive")
    if config.tokenizer_min_pair_count <= 0:
        raise ValueError("tokenizer_min_pair_count must be positive")


def language_model_batches(
    token_ids: list[int],
    context_length: int,
    batch_size: int,
    batch_stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(token_ids) <= context_length:
        raise ValueError("token_ids must be longer than context_length")
    stride = batch_stride if batch_stride is not None else context_length
    if stride <= 0:
        raise ValueError("batch_stride must be positive")

    input_rows = []
    target_rows = []
    for start in range(0, len(token_ids) - context_length, stride):
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
    dropped_window_count = len(input_rows) - usable
    logger.debug(
        "language_model_batches token_count=%s context_length=%s stride=%s batch_size=%s window_count=%s usable_window_count=%s dropped_window_count=%s batch_count=%s",
        len(token_ids),
        context_length,
        stride,
        batch_size,
        len(input_rows),
        usable,
        dropped_window_count,
        usable // batch_size,
    )
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
