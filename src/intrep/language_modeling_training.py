from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import CausalTextModel, CausalTextConfig, causal_text_config_to_dict
from intrep.text_examples import LanguageModelingExample, language_modeling_corpus_from_examples
from intrep.text_tokenizer import (
    TextTokenizer,
    TextTokenizerKind,
    build_text_tokenizer,
    text_tokenizer_to_payload,
)
from intrep.training_utils import LearningRateSchedule, build_adamw, build_lr_scheduler, clip_gradients

LanguageModelingTrainingDevice = Literal["auto", "cpu", "cuda"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanguageModelingTrainingConfig:
    context_length: int = 64
    batch_stride: int | None = None
    batch_size: int = 8
    max_steps: int = 20
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    max_grad_norm: float | None = 1.0
    lr_schedule: LearningRateSchedule = "constant"
    warmup_steps: int = 0
    seed: int = 7
    device: LanguageModelingTrainingDevice = "cpu"
    checkpoint_path: Path | None = None
    tokenizer: TextTokenizerKind = "byte-pair"
    tokenizer_vocab_size: int = 512
    tokenizer_min_pair_count: int = 2
    eval_batch_limit: int | None = 64


@dataclass(frozen=True)
class LanguageModelingTrainingResult:
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
class LanguageModelingTrainingArtifacts:
    result: LanguageModelingTrainingResult
    model: CausalTextModel
    tokenizer: TextTokenizer


class LanguageModelingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        token_ids: list[int],
        *,
        context_length: int,
        batch_stride: int | None = None,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if len(token_ids) <= context_length:
            raise ValueError("token_ids must be longer than context_length")
        stride = batch_stride if batch_stride is not None else context_length
        if stride <= 0:
            raise ValueError("batch_stride must be positive")
        self._token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride
        self.window_count = ((len(token_ids) - context_length - 1) // stride) + 1
        if self.window_count <= 0:
            raise ValueError("not enough tokens to build language-model batches")

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.window_count:
            raise IndexError(index)
        start = index * self.stride
        window = self._token_ids[start : start + self.context_length + 1]
        return window[:-1], window[1:]


def train_language_modeling_with_artifacts(
    *,
    train_examples: list[LanguageModelingExample] | tuple[LanguageModelingExample, ...],
    training_config: LanguageModelingTrainingConfig | None = None,
    model_config: CausalTextConfig | None = None,
    initial_model: CausalTextModel | None = None,
    eval_examples: list[LanguageModelingExample] | tuple[LanguageModelingExample, ...] | None = None,
    tokenizer_override: TextTokenizer | None = None,
) -> LanguageModelingTrainingArtifacts:
    return _train_text_corpus_with_artifacts(
        corpus=language_modeling_corpus_from_examples(train_examples),
        training_config=training_config,
        model_config=model_config,
        initial_model=initial_model,
        eval_corpus=language_modeling_corpus_from_examples(eval_examples) if eval_examples is not None else None,
        tokenizer_override=tokenizer_override,
    )


def _train_text_corpus_with_artifacts(
    *,
    corpus: str,
    training_config: LanguageModelingTrainingConfig | None = None,
    model_config: CausalTextConfig | None = None,
    initial_model: CausalTextModel | None = None,
    eval_corpus: str | None = None,
    tokenizer_override: TextTokenizer | None = None,
) -> LanguageModelingTrainingArtifacts:
    config = training_config or LanguageModelingTrainingConfig()
    _validate_training_config(config)
    torch.manual_seed(config.seed)
    device = resolve_training_device(config.device)
    if not corpus:
        raise ValueError("corpus must not be empty")
    tokenizer = tokenizer_override or build_text_tokenizer(
        corpus,
        kind=config.tokenizer,
        vocab_size=config.tokenizer_vocab_size,
        min_pair_count=config.tokenizer_min_pair_count,
    )
    token_ids = tokenizer.encode(corpus)
    train_dataset = LanguageModelingDataset(
        token_ids=token_ids,
        context_length=config.context_length,
        batch_stride=config.batch_stride,
    )
    train_loader = _language_model_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
    )
    train_eval_loader = _language_model_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=False,
    )
    eval_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    if eval_corpus is not None:
        if not eval_corpus:
            raise ValueError("eval_corpus must not be empty")
        eval_token_ids = tokenizer.encode(eval_corpus)
        eval_dataset = LanguageModelingDataset(
            token_ids=eval_token_ids,
            context_length=config.context_length,
            batch_stride=config.batch_stride,
        )
        eval_loader = _language_model_data_loader(
            eval_dataset,
            batch_size=config.batch_size,
            seed=config.seed,
            shuffle=False,
        )
    if initial_model is not None and model_config is not None:
        _validate_initial_model_config(initial_model, model_config)
    if initial_model is not None:
        causal_text_config = initial_model.config
    else:
        causal_text_config = model_config or CausalTextConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=config.context_length,
        )
    if causal_text_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model_config.vocab_size must match the tokenizer vocab size")
    if causal_text_config.context_length != config.context_length:
        raise ValueError("model_config.context_length must match training_config.context_length")
    model = (
        initial_model.to(device)
        if initial_model is not None
        else CausalTextModel(causal_text_config).to(device)
    )
    optimizer = build_adamw(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=config.lr_schedule,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
    )
    loss_fn = nn.CrossEntropyLoss()
    initial_loss: float | None = None
    final_loss = 0.0
    loss_history: list[float] = []
    initial_train_loss = _evaluate_loss(
        model,
        loss_fn,
        train_eval_loader,
        device,
        batch_limit=config.eval_batch_limit,
    )
    initial_eval_loss = _evaluate_loss(
        model,
        loss_fn,
        eval_loader,
        device,
        batch_limit=config.eval_batch_limit,
    )

    model.train()
    train_iterator = iter(train_loader)
    for step in range(config.max_steps):
        try:
            batch_inputs, batch_targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch_inputs, batch_targets = next(train_iterator)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        logits = model(batch_inputs)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1))
        if initial_loss is None:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())
        loss_history.append(final_loss)
        loss.backward()
        clip_gradients(model, config.max_grad_norm)
        optimizer.step()
        scheduler.step()

    final_train_loss = _evaluate_loss(
        model,
        loss_fn,
        train_eval_loader,
        device,
        batch_limit=config.eval_batch_limit,
    )
    final_eval_loss = _evaluate_loss(
        model,
        loss_fn,
        eval_loader,
        device,
        batch_limit=config.eval_batch_limit,
    )
    generalization_eval = eval_corpus is not None
    warnings = ()
    if not generalization_eval:
        warnings = (
            "No held-out eval corpus was provided; reported evaluation is train-split only and is not a generalization estimate.",
        )

    result = LanguageModelingTrainingResult(
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
        save_causal_text_checkpoint(
            path=config.checkpoint_path,
            model=model,
            model_config=causal_text_config,
            training_config=config,
            result=result,
            tokenizer=tokenizer,
        )
    return LanguageModelingTrainingArtifacts(result=result, model=model, tokenizer=tokenizer)


def resolve_training_device(requested_device: LanguageModelingTrainingDevice) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is false")
    return torch.device(requested_device)


def save_causal_text_checkpoint(
    *,
    path: Path,
    model: CausalTextModel,
    model_config: CausalTextConfig,
    training_config: LanguageModelingTrainingConfig,
    result: LanguageModelingTrainingResult,
    tokenizer: TextTokenizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    training_payload = asdict(training_config)
    if training_config.checkpoint_path is not None:
        training_payload["checkpoint_path"] = str(training_config.checkpoint_path)
    payload = {
        "schema_version": "intrep.causal_text_checkpoint.v1",
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "model_config": causal_text_config_to_dict(model_config),
        "tokenizer": text_tokenizer_to_payload(tokenizer),
        "training_config": training_payload,
        "result": asdict(result),
    }
    torch.save(payload, path)


def _evaluate_loss(
    model: CausalTextModel,
    loss_fn: nn.CrossEntropyLoss,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device,
    *,
    batch_limit: int | None = None,
) -> float | None:
    if data_loader is None:
        return None

    was_training = model.training
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            if batch_limit is not None and batch_count >= batch_limit:
                break
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1))
            total_loss += float(loss.item())
            batch_count += 1
    if was_training:
        model.train()
    return total_loss / batch_count


def _language_model_data_loader(
    dataset: LanguageModelingDataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _validate_training_config(config: LanguageModelingTrainingConfig) -> None:
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
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.max_grad_norm is not None and config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if config.lr_schedule not in ("constant", "warmup_cosine"):
        raise ValueError("lr_schedule must be one of: constant, warmup_cosine")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda")
    if config.tokenizer not in ("byte", "byte-pair"):
        raise ValueError("tokenizer must be one of: byte, byte-pair")
    if config.tokenizer_vocab_size <= 0:
        raise ValueError("tokenizer_vocab_size must be positive")
    if config.tokenizer_min_pair_count <= 0:
        raise ValueError("tokenizer_min_pair_count must be positive")
    if config.eval_batch_limit is not None and config.eval_batch_limit <= 0:
        raise ValueError("eval_batch_limit must be positive")


def _validate_initial_model_config(model: CausalTextModel, config: CausalTextConfig) -> None:
    if causal_text_config_to_dict(model.config) != causal_text_config_to_dict(config):
        raise ValueError("initial_model config must match model_config")


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
