from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from intrep.worlds.shogi.game_record import ShogiGameRecord, iter_shogi_game_records_jsonl
from intrep.worlds.shogi.game_trace import trace_shogi_game_record


@dataclass(frozen=True)
class ShogiUsiInfoStats:
    game_count: int
    ply_count: int
    info_ply_count: int
    info_line_count: int
    score_cp_line_count: int
    score_mate_line_count: int
    best_score_ply_count: int
    best_score_cp_line_count: int
    best_score_mate_line_count: int
    best_score_cp_min: int | None
    best_score_cp_max: int | None
    best_score_cp_mean: float | None
    best_score_mate_min: int | None
    best_score_mate_max: int | None
    depth_line_count: int
    nodes_line_count: int
    pv_line_count: int
    multipv_line_count: int
    action_pv_match_count: int
    multipv_counts: dict[int, int]
    depth_counts: dict[int, int]
    nodes_min: int | None
    nodes_max: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "game_count": self.game_count,
            "ply_count": self.ply_count,
            "info_ply_count": self.info_ply_count,
            "info_ply_ratio": _ratio(self.info_ply_count, self.ply_count),
            "info_line_count": self.info_line_count,
            "score_cp_line_count": self.score_cp_line_count,
            "score_mate_line_count": self.score_mate_line_count,
            "best_score_ply_count": self.best_score_ply_count,
            "best_score_ply_ratio": _ratio(self.best_score_ply_count, self.ply_count),
            "best_score_cp_line_count": self.best_score_cp_line_count,
            "best_score_mate_line_count": self.best_score_mate_line_count,
            "best_score_cp_min": self.best_score_cp_min,
            "best_score_cp_max": self.best_score_cp_max,
            "best_score_cp_mean": self.best_score_cp_mean,
            "best_score_mate_min": self.best_score_mate_min,
            "best_score_mate_max": self.best_score_mate_max,
            "depth_line_count": self.depth_line_count,
            "nodes_line_count": self.nodes_line_count,
            "pv_line_count": self.pv_line_count,
            "multipv_line_count": self.multipv_line_count,
            "action_pv_match_count": self.action_pv_match_count,
            "action_pv_match_ratio": _ratio(self.action_pv_match_count, self.pv_line_count),
            "multipv_counts": {str(key): value for key, value in sorted(self.multipv_counts.items())},
            "depth_counts": {str(key): value for key, value in sorted(self.depth_counts.items())},
            "nodes_min": self.nodes_min,
            "nodes_max": self.nodes_max,
        }


def inspect_shogi_usi_info_jsonl(path: str | Path) -> ShogiUsiInfoStats:
    return inspect_shogi_usi_info(iter_shogi_game_records_jsonl(path))


def inspect_shogi_usi_info(records: Iterable[ShogiGameRecord]) -> ShogiUsiInfoStats:
    game_count = 0
    ply_count = 0
    info_ply_count = 0
    info_line_count = 0
    score_cp_line_count = 0
    score_mate_line_count = 0
    best_score_ply_count = 0
    best_score_cp_line_count = 0
    best_score_mate_line_count = 0
    best_score_cp_values: list[int] = []
    best_score_mate_values: list[int] = []
    depth_line_count = 0
    nodes_line_count = 0
    pv_line_count = 0
    multipv_line_count = 0
    action_pv_match_count = 0
    multipv_counts: Counter[int] = Counter()
    depth_counts: Counter[int] = Counter()
    nodes_values: list[int] = []

    for record in records:
        game_count += 1
        for ply in trace_shogi_game_record(record).transitions:
            ply_count += 1
            ply_has_best_score = False
            if ply.decision_usi_info_lines:
                info_ply_count += 1
            for line in ply.decision_usi_info_lines:
                info_line_count += 1
                fields = parse_shogi_usi_info_line(line)
                score_kind = fields.get("score_kind")
                score_value = fields.get("score_value")
                if fields.get("score_kind") == "cp":
                    score_cp_line_count += 1
                if fields.get("score_kind") == "mate":
                    score_mate_line_count += 1
                if _is_best_info_line(fields) and isinstance(score_value, int):
                    ply_has_best_score = True
                    if score_kind == "cp":
                        best_score_cp_line_count += 1
                        best_score_cp_values.append(score_value)
                    elif score_kind == "mate":
                        best_score_mate_line_count += 1
                        best_score_mate_values.append(score_value)
                if fields.get("depth") is not None:
                    depth_line_count += 1
                    depth_counts[int(fields["depth"])] += 1
                if fields.get("nodes") is not None:
                    nodes_line_count += 1
                    nodes_values.append(int(fields["nodes"]))
                if fields.get("pv"):
                    pv_line_count += 1
                    pv = fields["pv"]
                    if isinstance(pv, tuple) and pv and pv[0] == ply.action_usi:
                        action_pv_match_count += 1
                if fields.get("multipv") is not None:
                    multipv_line_count += 1
                    multipv_counts[int(fields["multipv"])] += 1
            if ply_has_best_score:
                best_score_ply_count += 1

    return ShogiUsiInfoStats(
        game_count=game_count,
        ply_count=ply_count,
        info_ply_count=info_ply_count,
        info_line_count=info_line_count,
        score_cp_line_count=score_cp_line_count,
        score_mate_line_count=score_mate_line_count,
        best_score_ply_count=best_score_ply_count,
        best_score_cp_line_count=best_score_cp_line_count,
        best_score_mate_line_count=best_score_mate_line_count,
        best_score_cp_min=min(best_score_cp_values) if best_score_cp_values else None,
        best_score_cp_max=max(best_score_cp_values) if best_score_cp_values else None,
        best_score_cp_mean=_mean(best_score_cp_values),
        best_score_mate_min=min(best_score_mate_values) if best_score_mate_values else None,
        best_score_mate_max=max(best_score_mate_values) if best_score_mate_values else None,
        depth_line_count=depth_line_count,
        nodes_line_count=nodes_line_count,
        pv_line_count=pv_line_count,
        multipv_line_count=multipv_line_count,
        action_pv_match_count=action_pv_match_count,
        multipv_counts=dict(multipv_counts),
        depth_counts=dict(depth_counts),
        nodes_min=min(nodes_values) if nodes_values else None,
        nodes_max=max(nodes_values) if nodes_values else None,
    )


def write_shogi_usi_info_stats_json(path: str | Path, stats: ShogiUsiInfoStats) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_shogi_usi_info_line(line: str) -> dict[str, object]:
    words = line.split()
    if not words or words[0] != "info":
        return {}
    fields: dict[str, object] = {}
    index = 1
    while index < len(words):
        key = words[index]
        if key in {"depth", "seldepth", "nodes", "nps", "time", "multipv"} and index + 1 < len(words):
            value = _parse_int(words[index + 1])
            if value is not None:
                fields[key] = value
            index += 2
            continue
        if key == "score" and index + 2 < len(words):
            fields["score_kind"] = words[index + 1]
            value = _parse_int(words[index + 2])
            if value is not None:
                fields["score_value"] = value
            index += 3
            continue
        if key == "pv":
            fields["pv"] = tuple(words[index + 1 :])
            break
        index += 1
    return fields


def _is_best_info_line(fields: dict[str, object]) -> bool:
    multipv = fields.get("multipv")
    return multipv is None or multipv == 1


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
