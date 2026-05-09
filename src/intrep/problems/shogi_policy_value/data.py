from __future__ import annotations

import math
from dataclasses import replace
from math import exp
from pathlib import Path
from typing import Sequence

from intrep.problems.shogi_policy_value.examples import (
    ShogiMoveChoiceExample,
    ShogiPolicyValueExample,
    ShogiPositionValueExample,
)
from intrep.worlds.shogi.game_record import ShogiGameRecord, load_shogi_game_records_jsonl
from intrep.worlds.shogi.info_stats import parse_shogi_usi_info_line
from intrep.worlds.shogi.kif_io import load_shogi_game_record_from_kif_file


def load_shogi_policy_value_examples_from_game_records_jsonl(
    path: str | Path,
    *,
    policy_target_source: str,
    value_target_source: str,
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
    score_cp_scale: float = 600.0,
    max_games: int | None = None,
) -> list[ShogiPolicyValueExample]:
    if max_games is not None and max_games <= 0:
        raise ValueError("max_games must be positive")
    examples: list[ShogiPolicyValueExample] = []
    for game_index, record in enumerate(load_shogi_game_records_jsonl(path)):
        if max_games is not None and game_index >= max_games:
            break
        game_examples = shogi_policy_value_examples_from_game_record(
            record,
            policy_target_source=policy_target_source,
            value_target_source=value_target_source,
            policy_temperature_cp=policy_temperature_cp,
            policy_mate_cp=policy_mate_cp,
            score_cp_scale=score_cp_scale,
        )
        examples.extend(_with_game_metadata(game_examples, game_index=game_index))
    return examples


def load_shogi_move_choice_examples_from_kif_file(path: str | Path) -> list[ShogiMoveChoiceExample]:
    return shogi_move_choice_examples_from_game_record(load_shogi_game_record_from_kif_file(path))


def shogi_policy_value_examples_from_game_record(
    record: ShogiGameRecord,
    *,
    policy_target_source: str = "chosen_move",
    value_target_source: str = "winner",
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
    score_cp_scale: float = 600.0,
) -> list[ShogiPolicyValueExample]:
    policy_targets = shogi_policy_targets_from_game_record(
        record,
        source=policy_target_source,
        policy_temperature_cp=policy_temperature_cp,
        policy_mate_cp=policy_mate_cp,
    )
    value_targets = shogi_value_targets_from_game_record(
        record,
        source=value_target_source,
        score_cp_scale=score_cp_scale,
    )
    return [
        ShogiPolicyValueExample(
            position_sfen=transition.position_sfen,
            legal_moves=transition.legal_moves,
            chosen_move=transition.action_usi,
            policy_targets=policy_targets[index],
            value_target=value_targets[index],
        )
        for index, transition in enumerate(record.transitions)
    ]


def shogi_move_choice_examples_from_game_record(
    record: ShogiGameRecord,
    *,
    policy_target_source: str = "chosen_move",
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
) -> list[ShogiMoveChoiceExample]:
    policy_targets = shogi_policy_targets_from_game_record(
        record,
        source=policy_target_source,
        policy_temperature_cp=policy_temperature_cp,
        policy_mate_cp=policy_mate_cp,
    )
    return [
        ShogiMoveChoiceExample(
            position_sfen=transition.position_sfen,
            legal_moves=transition.legal_moves,
            chosen_move=transition.action_usi,
            policy_targets=policy_targets[index],
        )
        for index, transition in enumerate(record.transitions)
    ]


def shogi_position_value_examples_from_game_record(
    record: ShogiGameRecord,
    *,
    value_target_source: str = "winner",
    score_cp_scale: float = 600.0,
) -> list[ShogiPositionValueExample]:
    value_targets = shogi_value_targets_from_game_record(
        record,
        source=value_target_source,
        score_cp_scale=score_cp_scale,
    )
    return [
        ShogiPositionValueExample(
            position_sfen=transition.position_sfen,
            value_target=value_targets[index],
        )
        for index, transition in enumerate(record.transitions)
    ]


def shogi_policy_targets_from_game_record(
    record: ShogiGameRecord,
    *,
    source: str,
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
) -> tuple[dict[str, float] | None, ...]:
    if source == "chosen_move":
        return tuple(None for _transition in record.transitions)
    if source == "usi_multipv":
        return tuple(
            _policy_target_from_info_lines(
                transition.decision_usi_info_lines,
                legal_moves=transition.legal_moves,
                policy_temperature_cp=policy_temperature_cp,
                policy_mate_cp=policy_mate_cp,
            )
            for transition in record.transitions
        )
    raise ValueError(f"unsupported shogi policy target source: {source}")


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
    return tuple(_score_target_from_info_lines(transition.decision_usi_info_lines, score_cp_scale=score_cp_scale) for transition in record.transitions)


def _policy_target_from_info_lines(
    info_lines: Sequence[str],
    *,
    legal_moves: Sequence[str],
    policy_temperature_cp: float,
    policy_mate_cp: float,
) -> dict[str, float] | None:
    if policy_temperature_cp <= 0.0:
        raise ValueError("policy_temperature_cp must be positive")
    scored_moves: dict[str, float] = {}
    legal_move_set = set(legal_moves)
    for line in info_lines:
        fields = parse_shogi_usi_info_line(line)
        pv = fields.get("pv")
        score_cp = _score_cp_from_fields(fields, mate_cp=policy_mate_cp)
        if not isinstance(pv, tuple) or not pv or score_cp is None:
            continue
        move = str(pv[0])
        if move in legal_move_set:
            scored_moves[move] = score_cp
    if not scored_moves:
        return None
    max_score = max(scored_moves.values())
    weights = {
        move: exp((score_cp - max_score) / policy_temperature_cp)
        for move, score_cp in scored_moves.items()
    }
    total = sum(weights.values())
    return {move: weight / total for move, weight in weights.items()}


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


def _score_cp_from_fields(fields: dict[str, object], *, mate_cp: float) -> float | None:
    score_kind = fields.get("score_kind")
    score_value = fields.get("score_value")
    if not isinstance(score_value, int):
        return None
    if score_kind == "cp":
        return float(score_value)
    if score_kind == "mate":
        return mate_cp if score_value > 0 else -mate_cp
    return None


def _with_game_metadata(
    examples: Sequence[ShogiPolicyValueExample],
    *,
    game_index: int,
) -> list[ShogiPolicyValueExample]:
    return [
        replace(example, game_index=game_index, ply_index=ply_index)
        for ply_index, example in enumerate(examples)
    ]
