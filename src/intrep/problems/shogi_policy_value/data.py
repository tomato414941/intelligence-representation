from __future__ import annotations

import math
from dataclasses import replace
from math import exp
from pathlib import Path
from typing import Sequence

import shogi

from intrep.problems.shogi_policy_value.examples import (
    ShogiMoveChoiceExample,
    ShogiPolicyValueExample,
    ShogiPositionValueExample,
)
from intrep.worlds.shogi.engine_analysis import ShogiEngineAnalysis, load_shogi_engine_analysis_jsonl
from intrep.worlds.shogi.game_record import ShogiGameRecord, load_shogi_game_records_jsonl
from intrep.worlds.shogi.game_trace import ShogiGameTrace, trace_shogi_game_record
from intrep.worlds.shogi.info_stats import parse_shogi_usi_info_line
from intrep.worlds.shogi.kif_io import load_shogi_game_record_from_kif_file


def load_shogi_policy_value_examples_from_game_records_jsonl(
    path: str | Path,
    *,
    policy_target_construction: str,
    value_target_construction: str,
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
    score_cp_scale: float = 600.0,
    max_games: int | None = None,
) -> list[ShogiPolicyValueExample]:
    return load_shogi_policy_value_examples_from_game_records_jsonl_with_engine_analysis(
        path,
        policy_target_construction=policy_target_construction,
        value_target_construction=value_target_construction,
        analyses_by_position={},
        policy_temperature_cp=policy_temperature_cp,
        policy_mate_cp=policy_mate_cp,
        score_cp_scale=score_cp_scale,
        max_games=max_games,
    )


def load_shogi_policy_value_examples_from_game_records_jsonl_with_engine_analysis(
    path: str | Path,
    *,
    policy_target_construction: str,
    value_target_construction: str,
    analyses_by_position: dict[str, ShogiEngineAnalysis],
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
            policy_target_construction=policy_target_construction,
            value_target_construction=value_target_construction,
            analyses_by_position=analyses_by_position,
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
    policy_target_construction: str = "chosen_move",
    value_target_construction: str = "winner",
    analyses_by_position: dict[str, ShogiEngineAnalysis] | None = None,
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
    score_cp_scale: float = 600.0,
) -> list[ShogiPolicyValueExample]:
    analyses = analyses_by_position or {}
    trace = trace_shogi_game_record(record)
    policy_targets = shogi_policy_targets_from_game_trace(
        trace,
        source=policy_target_construction,
        analyses_by_position=analyses,
        policy_temperature_cp=policy_temperature_cp,
        policy_mate_cp=policy_mate_cp,
    )
    value_targets = shogi_value_targets_from_game_trace(
        trace,
        source=value_target_construction,
        analyses_by_position=analyses,
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
        for index, transition in enumerate(trace.transitions)
    ]


def shogi_policy_value_examples_from_game_record_plies(
    record: ShogiGameRecord,
    *,
    ply_indices: set[int],
    policy_target_construction: str = "chosen_move",
    value_target_construction: str = "winner",
    game_index: int | None = None,
) -> list[ShogiPolicyValueExample]:
    if policy_target_construction != "chosen_move":
        raise ValueError("selected-ply construction supports only chosen_move policy targets")
    if value_target_construction != "winner":
        raise ValueError("selected-ply construction supports only winner value targets")
    if not ply_indices:
        return []
    max_ply = max(ply_indices)
    board = shogi.Board(record.initial_position_sfen)
    examples: list[ShogiPolicyValueExample] = []
    for ply_index, move_record in enumerate(record.moves):
        if ply_index > max_ply:
            break
        side = "black" if board.turn == shogi.BLACK else "white"
        if ply_index in ply_indices:
            legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
            if move_record.action_usi not in legal_moves:
                raise ValueError(f"illegal move at ply {ply_index}: {move_record.action_usi}")
            examples.append(
                ShogiPolicyValueExample(
                    position_sfen=board.sfen(),
                    legal_moves=legal_moves,
                    chosen_move=move_record.action_usi,
                    policy_targets=None,
                    value_target=_winner_value_target(side=side, winner=record.winner),
                    game_index=game_index,
                    ply_index=ply_index,
                )
            )
        board.push_usi(move_record.action_usi)
    return examples


def shogi_move_choice_examples_from_game_record(
    record: ShogiGameRecord,
    *,
    policy_target_construction: str = "chosen_move",
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
) -> list[ShogiMoveChoiceExample]:
    trace = trace_shogi_game_record(record)
    policy_targets = shogi_policy_targets_from_game_trace(
        trace,
        source=policy_target_construction,
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
        for index, transition in enumerate(trace.transitions)
    ]


def shogi_position_value_examples_from_game_record(
    record: ShogiGameRecord,
    *,
    value_target_construction: str = "winner",
    score_cp_scale: float = 600.0,
) -> list[ShogiPositionValueExample]:
    trace = trace_shogi_game_record(record)
    value_targets = shogi_value_targets_from_game_trace(
        trace,
        source=value_target_construction,
        score_cp_scale=score_cp_scale,
    )
    return [
        ShogiPositionValueExample(
            position_sfen=transition.position_sfen,
            value_target=value_targets[index],
        )
        for index, transition in enumerate(trace.transitions)
    ]


def shogi_policy_targets_from_game_trace(
    trace: ShogiGameTrace,
    *,
    source: str,
    analyses_by_position: dict[str, ShogiEngineAnalysis] | None = None,
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
) -> tuple[dict[str, float] | None, ...]:
    if source == "chosen_move":
        return tuple(None for _transition in trace.transitions)
    if source == "decision_usi_multipv":
        return tuple(
            _policy_target_from_info_lines(
                transition.decision_usi_info_lines,
                legal_moves=transition.legal_moves,
                policy_temperature_cp=policy_temperature_cp,
                policy_mate_cp=policy_mate_cp,
            )
            for transition in trace.transitions
        )
    if source == "engine_analysis_multipv":
        return shogi_policy_targets_from_engine_analysis(
            analyses_by_position or {},
            trace,
            policy_temperature_cp=policy_temperature_cp,
            policy_mate_cp=policy_mate_cp,
        )
    raise ValueError(f"unsupported shogi policy target source: {source}")


def shogi_value_targets_from_game_trace(
    trace: ShogiGameTrace,
    *,
    source: str,
    analyses_by_position: dict[str, ShogiEngineAnalysis] | None = None,
    score_cp_scale: float = 600.0,
) -> tuple[float | None, ...]:
    if source == "winner":
        return shogi_return_targets_from_game_trace(trace)
    if source == "decision_usi_score":
        return shogi_score_targets_from_game_trace(trace, score_cp_scale=score_cp_scale)
    if source == "engine_analysis_score":
        return shogi_score_targets_from_engine_analysis(analyses_by_position or {}, trace, score_cp_scale=score_cp_scale)
    raise ValueError(f"unsupported shogi value target source: {source}")


def shogi_policy_targets_from_game_record(
    record: ShogiGameRecord,
    *,
    source: str,
    analyses_by_position: dict[str, ShogiEngineAnalysis] | None = None,
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
) -> tuple[dict[str, float] | None, ...]:
    return shogi_policy_targets_from_game_trace(
        trace_shogi_game_record(record),
        source=source,
        analyses_by_position=analyses_by_position,
        policy_temperature_cp=policy_temperature_cp,
        policy_mate_cp=policy_mate_cp,
    )


def shogi_value_targets_from_game_record(
    record: ShogiGameRecord,
    *,
    source: str,
    analyses_by_position: dict[str, ShogiEngineAnalysis] | None = None,
    score_cp_scale: float = 600.0,
) -> tuple[float | None, ...]:
    return shogi_value_targets_from_game_trace(
        trace_shogi_game_record(record),
        source=source,
        analyses_by_position=analyses_by_position,
        score_cp_scale=score_cp_scale,
    )


def shogi_return_targets_from_game_trace(trace: ShogiGameTrace) -> tuple[float | None, ...]:
    if trace.winner is None:
        return tuple(None for _transition in trace.transitions)
    return tuple(1.0 if transition.side == trace.winner else -1.0 for transition in trace.transitions)


def _winner_value_target(*, side: str, winner: str | None) -> float | None:
    if winner is None:
        return None
    return 1.0 if side == winner else -1.0


def shogi_return_targets_from_game_record(record: ShogiGameRecord) -> tuple[float | None, ...]:
    return shogi_return_targets_from_game_trace(trace_shogi_game_record(record))


def shogi_score_targets_from_game_trace(trace: ShogiGameTrace, *, score_cp_scale: float = 600.0) -> tuple[float | None, ...]:
    if score_cp_scale <= 0:
        raise ValueError("score_cp_scale must be positive")
    return tuple(_score_target_from_info_lines(transition.decision_usi_info_lines, score_cp_scale=score_cp_scale) for transition in trace.transitions)


def shogi_score_targets_from_game_record(record: ShogiGameRecord, *, score_cp_scale: float = 600.0) -> tuple[float | None, ...]:
    return shogi_score_targets_from_game_trace(trace_shogi_game_record(record), score_cp_scale=score_cp_scale)


def shogi_engine_analysis_by_position(analyses: Sequence[ShogiEngineAnalysis]) -> dict[str, ShogiEngineAnalysis]:
    by_position: dict[str, ShogiEngineAnalysis] = {}
    for analysis in analyses:
        if analysis.position_sfen in by_position:
            raise ValueError(f"duplicate shogi engine analysis for position: {analysis.position_sfen}")
        by_position[analysis.position_sfen] = analysis
    return by_position


def load_shogi_engine_analysis_by_position_jsonl(paths: Sequence[str | Path]) -> dict[str, ShogiEngineAnalysis]:
    analyses: list[ShogiEngineAnalysis] = []
    for path in paths:
        analyses.extend(load_shogi_engine_analysis_jsonl(path))
    return shogi_engine_analysis_by_position(analyses)


def shogi_policy_targets_from_engine_analysis(
    analyses_by_position: dict[str, ShogiEngineAnalysis],
    trace: ShogiGameTrace,
    *,
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
) -> tuple[dict[str, float] | None, ...]:
    return tuple(
        _policy_target_from_analysis(
            analyses_by_position.get(transition.position_sfen),
            legal_moves=transition.legal_moves,
            policy_temperature_cp=policy_temperature_cp,
            policy_mate_cp=policy_mate_cp,
        )
        for transition in trace.transitions
    )


def shogi_score_targets_from_engine_analysis(
    analyses_by_position: dict[str, ShogiEngineAnalysis],
    trace: ShogiGameTrace,
    *,
    score_cp_scale: float = 600.0,
) -> tuple[float | None, ...]:
    if score_cp_scale <= 0:
        raise ValueError("score_cp_scale must be positive")
    return tuple(
        _score_target_from_analysis(analyses_by_position.get(transition.position_sfen), score_cp_scale=score_cp_scale)
        for transition in trace.transitions
    )


def _policy_target_from_analysis(
    analysis: ShogiEngineAnalysis | None,
    *,
    legal_moves: Sequence[str],
    policy_temperature_cp: float,
    policy_mate_cp: float,
) -> dict[str, float] | None:
    if analysis is None:
        return None
    return _policy_target_from_info_lines(
        analysis.usi_info_lines,
        legal_moves=legal_moves,
        policy_temperature_cp=policy_temperature_cp,
        policy_mate_cp=policy_mate_cp,
    )


def _score_target_from_analysis(analysis: ShogiEngineAnalysis | None, *, score_cp_scale: float) -> float | None:
    if analysis is None:
        return None
    return _score_target_from_info_lines(analysis.usi_info_lines, score_cp_scale=score_cp_scale)


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
