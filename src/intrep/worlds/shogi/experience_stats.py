from __future__ import annotations

from dataclasses import dataclass

from intrep.worlds.shogi.game_record import ShogiActorSpec, ShogiGameRecord
from intrep.worlds.shogi.game_trace import trace_shogi_game_record


@dataclass(frozen=True)
class ShogiPositionStats:
    transition_count: int
    unique_position_count: int
    duplicate_position_count: int
    position_duplicate_ratio: float
    max_position_repeat_count: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "transition_count": self.transition_count,
            "unique_position_count": self.unique_position_count,
            "duplicate_position_count": self.duplicate_position_count,
            "position_duplicate_ratio": self.position_duplicate_ratio,
            "max_position_repeat_count": self.max_position_repeat_count,
        }


@dataclass(frozen=True)
class ShogiTrainEvalPositionStats:
    train: ShogiPositionStats
    eval: ShogiPositionStats
    train_eval_position_overlap_count: int
    train_eval_position_overlap_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "train_position_stats": self.train.to_dict(),
            "eval_position_stats": self.eval.to_dict(),
            "train_eval_position_overlap_count": self.train_eval_position_overlap_count,
            "train_eval_position_overlap_ratio": self.train_eval_position_overlap_ratio,
        }


def shogi_position_stats(records: list[ShogiGameRecord]) -> ShogiPositionStats:
    position_counts: dict[str, int] = {}
    for record in records:
        for transition in trace_shogi_game_record(record).transitions:
            position_counts[transition.position_sfen] = position_counts.get(transition.position_sfen, 0) + 1
    transition_count = sum(position_counts.values())
    unique_position_count = len(position_counts)
    duplicate_position_count = transition_count - unique_position_count
    return ShogiPositionStats(
        transition_count=transition_count,
        unique_position_count=unique_position_count,
        duplicate_position_count=duplicate_position_count,
        position_duplicate_ratio=duplicate_position_count / transition_count if transition_count else 0.0,
        max_position_repeat_count=max(position_counts.values()) if position_counts else 0,
    )


def shogi_train_eval_position_stats(
    train_records: list[ShogiGameRecord],
    eval_records: list[ShogiGameRecord],
) -> ShogiTrainEvalPositionStats:
    train_positions = _position_set(train_records)
    eval_positions = _position_set(eval_records)
    overlap_count = len(train_positions & eval_positions)
    return ShogiTrainEvalPositionStats(
        train=shogi_position_stats(train_records),
        eval=shogi_position_stats(eval_records),
        train_eval_position_overlap_count=overlap_count,
        train_eval_position_overlap_ratio=overlap_count / len(eval_positions) if eval_positions else 0.0,
    )


def shogi_actor_pair(record: ShogiGameRecord) -> str:
    return f"{record.black_actor.kind}:{record.white_actor.kind}"


def shogi_actor_pair_counts(records: list[ShogiGameRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = shogi_actor_pair(record)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def shogi_checkpoint_actor_summaries(records: list[ShogiGameRecord]) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    summaries: dict[str, dict[str, object]] = {}
    actor_names: dict[str, set[str]] = {}
    for record in records:
        for actor in (record.black_actor, record.white_actor):
            if actor.kind != "checkpoint":
                continue
            summary = _checkpoint_actor_summary(actor)
            key = _summary_key(summary)
            summaries[key] = summary
            actor_names.setdefault(key, set()).add(actor.name)
            counts[key] = counts.get(key, 0) + 1

    output: list[dict[str, object]] = []
    for key in sorted(summaries):
        output.append({"count": counts[key], "actor_names": sorted(actor_names[key]), **summaries[key]})
    return output


def _position_set(records: list[ShogiGameRecord]) -> set[str]:
    return {transition.position_sfen for record in records for transition in trace_shogi_game_record(record).transitions}


def _checkpoint_actor_summary(actor: ShogiActorSpec) -> dict[str, object]:
    settings = actor.settings
    summary: dict[str, object] = {
        "checkpoint_id": _setting_text(settings, "checkpoint_id")
        or _setting_text(settings, "checkpoint_name")
        or actor.name,
        "checkpoint_path": _setting_text(settings, "checkpoint_path")
        or _setting_text(settings, "checkpoint")
        or "unknown",
        "move_selector": _setting_text(settings, "move_selector")
        or _setting_text(settings, "policy")
        or "unknown",
    }
    _add_optional_setting(summary, settings, "move_selection_profile")
    _add_optional_setting(summary, settings, "mcts_simulations_per_move", fallback_key="simulations")
    _add_optional_setting(summary, settings, "nn_leaf_eval_batch_limit", fallback_key="evaluation_batch_size")
    _add_optional_setting(summary, settings, "mcts_move_time_limit_sec")
    _add_optional_setting(summary, settings, "board_backend")
    return summary


def _add_optional_setting(
    summary: dict[str, object],
    settings: dict[str, str | int | float | bool | None],
    key: str,
    *,
    fallback_key: str | None = None,
) -> None:
    value = settings.get(key)
    if value is None and fallback_key is not None:
        value = settings.get(fallback_key)
    if value is not None:
        summary[key] = value


def _setting_text(settings: dict[str, str | int | float | bool | None], key: str) -> str | None:
    value = settings.get(key)
    if value is None:
        return None
    return str(value)


def _summary_key(summary: dict[str, object]) -> str:
    return "|".join(f"{key}={summary[key]}" for key in sorted(summary))
