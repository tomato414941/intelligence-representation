from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import shogi


@dataclass(frozen=True)
class ShogiActorSpec:
    kind: str
    name: str
    settings: dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class ShogiTransitionRecord:
    ply: int
    side: str
    position_sfen: str
    legal_moves: tuple[str, ...]
    action_usi: str
    next_position_sfen: str
    reward: float
    done: bool
    policy_targets: dict[str, float] | None = None
    usi_info_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShogiGameRecord:
    black_actor: ShogiActorSpec
    white_actor: ShogiActorSpec
    initial_position_sfen: str
    transitions: tuple[ShogiTransitionRecord, ...]
    winner: str | None = None
    end_reason: str | None = None


def load_shogi_game_records_jsonl(path: str | Path) -> list[ShogiGameRecord]:
    records: list[ShogiGameRecord] = []
    for record in iter_shogi_game_records_jsonl(path):
        records.append(record)
    if not records:
        raise ValueError("shogi game records jsonl must contain at least one game")
    return records


def iter_shogi_game_records_jsonl(path: str | Path) -> Iterator[ShogiGameRecord]:
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            record = shogi_game_record_from_json(json.loads(stripped))
            if record.transitions:
                yield record


def write_shogi_game_records_jsonl(path: str | Path, records: Sequence[ShogiGameRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        if not record.transitions:
            continue
        lines.append(json.dumps(shogi_game_record_to_json(record), separators=(",", ":"), sort_keys=True))
    if not lines:
        raise ValueError("records must contain at least one non-empty game")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def shogi_game_record_to_json(record: ShogiGameRecord) -> dict[str, object]:
    return {
        "black_actor": shogi_actor_spec_to_json(record.black_actor),
        "white_actor": shogi_actor_spec_to_json(record.white_actor),
        "initial_position_sfen": record.initial_position_sfen,
        "transitions": [shogi_transition_record_to_json(transition) for transition in record.transitions],
        "winner": record.winner,
        "end_reason": record.end_reason,
    }


def shogi_game_record_from_json(payload: dict[str, object]) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=shogi_actor_spec_from_json(_object_dict(payload.get("black_actor"))),
        white_actor=shogi_actor_spec_from_json(_object_dict(payload.get("white_actor"))),
        initial_position_sfen=str(payload["initial_position_sfen"]),
        transitions=tuple(
            shogi_transition_record_from_json(_object_dict(transition))
            for transition in _object_list(payload.get("transitions"))
        ),
        winner=_normalize_winner(payload.get("winner")),
        end_reason=None if payload.get("end_reason") is None else str(payload["end_reason"]),
    )


def shogi_transition_record_to_json(record: ShogiTransitionRecord) -> dict[str, object]:
    return {
        "ply": record.ply,
        "side": record.side,
        "position_sfen": record.position_sfen,
        "legal_moves": list(record.legal_moves),
        "action_usi": record.action_usi,
        "next_position_sfen": record.next_position_sfen,
        "reward": record.reward,
        "done": record.done,
        "policy_targets": record.policy_targets,
        "usi_info_lines": list(record.usi_info_lines),
    }


def shogi_transition_record_from_json(payload: dict[str, object]) -> ShogiTransitionRecord:
    return ShogiTransitionRecord(
        ply=int(payload["ply"]),
        side=_normalize_side(payload["side"]),
        position_sfen=str(payload["position_sfen"]),
        legal_moves=tuple(str(move) for move in _object_list(payload.get("legal_moves"))),
        action_usi=str(payload["action_usi"]),
        next_position_sfen=str(payload["next_position_sfen"]),
        reward=float(payload["reward"]),
        done=bool(payload["done"]),
        policy_targets=_optional_float_dict(payload.get("policy_targets")),
        usi_info_lines=tuple(str(line) for line in _object_list(payload.get("usi_info_lines", []))),
    )


def shogi_actor_spec_to_json(spec: ShogiActorSpec) -> dict[str, object]:
    return {
        "kind": spec.kind,
        "name": spec.name,
        "settings": spec.settings,
    }


def shogi_actor_spec_from_json(payload: dict[str, object]) -> ShogiActorSpec:
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("actor settings must be an object")
    return ShogiActorSpec(
        kind=str(payload["kind"]),
        name=str(payload["name"]),
        settings={
            str(key): _json_scalar(value)
            for key, value in settings.items()
        },
    )


def shogi_winner_to_side_code(winner: str | None) -> str | None:
    if winner == "black":
        return "b"
    if winner == "white":
        return "w"
    return None


def shogi_side_code_to_winner(winner: str | None) -> str | None:
    if winner == "b":
        return "black"
    if winner == "w":
        return "white"
    return None


def shogi_game_transitions_from_usi_moves(
    moves: Sequence[str],
    *,
    winner: str | None = None,
) -> tuple[ShogiTransitionRecord, ...]:
    board = shogi.Board()
    records: list[ShogiTransitionRecord] = []
    normalized_winner = _normalize_winner(winner)
    for ply, move in enumerate(moves):
        side = "black" if board.turn == shogi.BLACK else "white"
        position_sfen = board.sfen()
        legal_moves = tuple(sorted(legal_move.usi() for legal_move in board.legal_moves))
        if move not in legal_moves:
            raise ValueError(f"illegal move at ply {ply}: {move}")
        board.push_usi(move)
        done = ply == len(moves) - 1
        records.append(
            ShogiTransitionRecord(
                ply=ply,
                side=side,
                position_sfen=position_sfen,
                legal_moves=legal_moves,
                action_usi=move,
                next_position_sfen=board.sfen(),
                reward=_transition_reward(side=side, winner=normalized_winner, done=done),
                done=done,
            )
        )
    return tuple(records)


def _transition_reward(*, side: str, winner: str | None, done: bool) -> float:
    if not done or winner is None:
        return 0.0
    return 1.0 if side == winner else -1.0


def _normalize_side(value: object) -> str:
    if value in {"black", "white"}:
        return str(value)
    raise ValueError("side must be black or white")


def _normalize_winner(value: object) -> str | None:
    if value in {"black", "white"}:
        return str(value)
    if value is None:
        return None
    raise ValueError("winner must be black, white, or null")


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return value


def _json_scalar(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError("actor setting values must be JSON scalars")


def _optional_float_dict(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    data = _object_dict(value)
    return {str(key): float(item) for key, item in data.items()}
