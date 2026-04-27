from __future__ import annotations

import math


def perplexity_from_loss(loss: float | None) -> float | None:
    if loss is None:
        return None
    try:
        return math.exp(loss)
    except OverflowError:
        return math.inf


def language_modeling_metrics_from_training_result(result: object) -> dict[str, float | None]:
    initial_train_loss = getattr(result, "initial_train_loss", None)
    final_train_loss = getattr(result, "final_train_loss", None)
    initial_eval_loss = getattr(result, "initial_eval_loss", None)
    final_eval_loss = getattr(result, "final_eval_loss", None)
    return {
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "train_loss_delta": _delta(initial_train_loss, final_train_loss),
        "initial_train_perplexity": perplexity_from_loss(initial_train_loss),
        "final_train_perplexity": perplexity_from_loss(final_train_loss),
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "eval_loss_delta": _delta(initial_eval_loss, final_eval_loss),
        "initial_eval_perplexity": perplexity_from_loss(initial_eval_loss),
        "final_eval_perplexity": perplexity_from_loss(final_eval_loss),
    }


def _delta(initial: float | None, final: float | None) -> float | None:
    if initial is None or final is None:
        return None
    return initial - final
