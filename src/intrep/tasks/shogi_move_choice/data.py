from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from intrep.tasks.shogi_move_choice.examples import (
    ShogiMoveChoiceExample,
)
from intrep.worlds.shogi.game_record import ShogiGameRecord, load_shogi_game_records_jsonl
from intrep.worlds.shogi.info_stats import parse_shogi_usi_info_line


def load_shogi_move_choice_examples_from_game_records_jsonl(
    path: str | Path,
    *,
    value_target_source: str,
    score_cp_scale: float = 600.0,
) -> list[ShogiMoveChoiceExample]:
    examples: list[ShogiMoveChoiceExample] = []
    for game_index, record in enumerate(load_shogi_game_records_jsonl(path)):
        game_examples = shogi_move_choice_examples_from_game_record(
            record,
            value_target_source=value_target_source,
            score_cp_scale=score_cp_scale,
        )
        examples.extend(_with_game_metadata(game_examples, game_index=game_index))
    return examples


def shogi_move_choice_examples_from_game_record(
    record: ShogiGameRecord,
    *,
    value_target_source: str = "winner",
    score_cp_scale: float = 600.0,
) -> list[ShogiMoveChoiceExample]:
    value_targets = shogi_value_targets_from_game_record(
        record,
        source=value_target_source,
        score_cp_scale=score_cp_scale,
    )
    return [
        ShogiMoveChoiceExample(
            position_sfen=transition.position_sfen,
            legal_moves=transition.legal_moves,
            chosen_move=transition.action_usi,
            policy_targets=transition.policy_targets,
            value_target=value_targets[index],
        )
        for index, transition in enumerate(record.transitions)
    ]


def shogi_value_targets_from_game_record(
    record: ShogiGameRecord,
    *,
    source: str,
    score_cp_scale: float = 600.0,
) -> tuple[float | None, ...]:
    if source == "winner":
        return shogi_return_targets_from_game_record(record)
    if source == "yaneuraou_best_score":
        return shogi_score_targets_from_game_record(record, score_cp_scale=score_cp_scale)
    raise ValueError(f"unsupported shogi value target source: {source}")


def shogi_return_targets_from_game_record(record: ShogiGameRecord) -> tuple[float | None, ...]:
    if record.winner is None:
        return tuple(None for _transition in record.transitions)
    return tuple(1.0 if transition.side == record.winner else -1.0 for transition in record.transitions)


def shogi_score_targets_from_game_record(record: ShogiGameRecord, *, score_cp_scale: float = 600.0) -> tuple[float | None, ...]:
    if score_cp_scale <= 0:
        raise ValueError("score_cp_scale must be positive")
    return tuple(_score_target_from_info_lines(transition.usi_info_lines, score_cp_scale=score_cp_scale) for transition in record.transitions)


def _score_target_from_info_lines(info_lines: Sequence[str], *, score_cp_scale: float) -> float | None:
    for line in info_lines:
        fields = parse_shogi_usi_info_line(line)
        multipv = fields.get("multipv")
        if multipv not in {None, 1}:
            continue
        score_kind = fields.get("score_kind")
        score_value = fields.get("score_value")
        if not isinstance(score_value, int):
            continue
        if score_kind == "cp":
            return math.tanh(score_value / score_cp_scale)
        if score_kind == "mate":
            return 1.0 if score_value > 0 else -1.0
    return None


def _with_game_metadata(
    examples: Sequence[ShogiMoveChoiceExample],
    *,
    game_index: int,
) -> list[ShogiMoveChoiceExample]:
    return [
        replace(example, game_index=game_index, ply_index=ply_index)
        for ply_index, example in enumerate(examples)
    ]
