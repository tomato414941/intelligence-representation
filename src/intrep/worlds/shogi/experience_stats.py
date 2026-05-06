from __future__ import annotations

from dataclasses import dataclass

from intrep.worlds.shogi.game_record import ShogiGameRecord


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
        for transition in record.transitions:
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


def _position_set(records: list[ShogiGameRecord]) -> set[str]:
    return {transition.position_sfen for record in records for transition in record.transitions}
