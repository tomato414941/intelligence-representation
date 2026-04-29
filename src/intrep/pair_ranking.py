from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from intrep.byte_tokenizer import ByteTokenizer
from intrep.mixed_corpus_evaluation import MixedEnvironmentDocumentPair


ContinuationScorer = Callable[[Any, ByteTokenizer, str, str], float]


@dataclass(frozen=True)
class PairRankingMetrics:
    top1_accuracy: float
    mean_correct_loss: float
    mean_best_distractor_loss: float
    mean_margin: float


def evaluate_symbolic_to_natural_ranking(
    pairs: Sequence[MixedEnvironmentDocumentPair],
    model: Any,
    tokenizer: ByteTokenizer,
    *,
    score_continuation_loss: ContinuationScorer | None = None,
) -> PairRankingMetrics:
    if not pairs:
        raise ValueError("pairs must not be empty")
    if len(pairs) < 2:
        raise ValueError("at least two pairs are required to build distractors")

    scorer = score_continuation_loss or torch_next_token_continuation_loss
    correct_losses: list[float] = []
    best_distractor_losses: list[float] = []
    margins: list[float] = []
    top1_count = 0

    for pair in pairs:
        prefix = _symbolic_prefix(pair)
        correct_continuation = pair.natural.content
        correct_loss = scorer(model, tokenizer, prefix, correct_continuation)
        distractor_losses = [
            scorer(model, tokenizer, prefix, distractor.natural.content)
            for distractor in pairs
            if distractor.episode_id != pair.episode_id
        ]
        best_distractor_loss = min(distractor_losses)
        margin = best_distractor_loss - correct_loss

        correct_losses.append(correct_loss)
        best_distractor_losses.append(best_distractor_loss)
        margins.append(margin)
        if correct_loss < best_distractor_loss:
            top1_count += 1

    count = len(pairs)
    return PairRankingMetrics(
        top1_accuracy=top1_count / count,
        mean_correct_loss=sum(correct_losses) / count,
        mean_best_distractor_loss=sum(best_distractor_losses) / count,
        mean_margin=sum(margins) / count,
    )


def torch_next_token_continuation_loss(
    model: Any,
    tokenizer: ByteTokenizer,
    prefix: str,
    continuation: str,
) -> float:
    return torch_next_token_continuation_losses(model, tokenizer, prefix, [continuation])[0]


def torch_next_token_continuation_losses(
    model: Any,
    tokenizer: ByteTokenizer,
    prefix: str,
    continuations: Sequence[str],
) -> list[float]:
    import torch
    import torch.nn.functional as F

    prefix_ids = tokenizer.encode(prefix)
    if not prefix_ids:
        raise ValueError("prefix must encode to at least one token")
    continuation_id_rows = [tokenizer.encode(continuation) for continuation in continuations]
    if not continuation_id_rows:
        raise ValueError("continuations must not be empty")
    if any(not continuation_ids for continuation_ids in continuation_id_rows):
        raise ValueError("continuation must encode to at least one token")

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    parameter = next(iter(model.parameters()), None) if hasattr(model, "parameters") else None
    device = parameter.device if parameter is not None else torch.device("cpu")
    max_token_count = max(len(prefix_ids) + len(continuation_ids) for continuation_ids in continuation_id_rows)
    context_length = int(getattr(getattr(model, "config", None), "context_length", max_token_count))

    try:
        with torch.no_grad():
            windows_by_length: dict[int, list[tuple[int, list[int]]]] = {}
            targets_by_length: dict[int, list[int]] = {}
            for continuation_index, continuation_ids in enumerate(continuation_id_rows):
                token_ids = prefix_ids + continuation_ids
                if len(token_ids) < 2:
                    raise ValueError("prefix and continuation must encode to at least two tokens")
                for target_index in range(len(prefix_ids), len(token_ids)):
                    window_start = max(0, target_index - context_length)
                    window = token_ids[window_start:target_index]
                    windows_by_length.setdefault(len(window), []).append((continuation_index, window))
                    targets_by_length.setdefault(len(window), []).append(token_ids[target_index])

            total_losses = [0.0 for _ in continuation_id_rows]
            total_counts = [0 for _ in continuation_id_rows]
            for length, windows in windows_by_length.items():
                input_ids = torch.tensor([window for _index, window in windows], dtype=torch.long, device=device)
                targets = torch.tensor(targets_by_length[length], dtype=torch.long, device=device)
                logits = model(input_ids)[:, -1, :]
                losses = F.cross_entropy(logits, targets, reduction="none")
                for (continuation_index, _window), loss in zip(windows, losses, strict=True):
                    total_losses[continuation_index] += float(loss.item())
                    total_counts[continuation_index] += 1
            return [
                total_loss / total_count
                for total_loss, total_count in zip(total_losses, total_counts, strict=True)
            ]
    finally:
        if was_training and hasattr(model, "train"):
            model.train()


def _symbolic_prefix(pair: MixedEnvironmentDocumentPair) -> str:
    return pair.symbolic.content + "\n"
