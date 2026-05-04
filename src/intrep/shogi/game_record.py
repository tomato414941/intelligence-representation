from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Iterator, Sequence

import shogi.KIF

from intrep.shogi.move_choice import (
    ShogiMoveChoiceExample,
    shogi_move_choice_examples_from_usi_moves,
    shogi_move_choice_examples_from_usi_moves_with_winner,
)


@dataclass(frozen=True)
class PlayerSpec:
    kind: str
    name: str
    settings: dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class ShogiGameRecord:
    black_player: PlayerSpec
    white_player: PlayerSpec
    moves: tuple[str, ...]
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
            if record.moves:
                yield record


def write_shogi_game_records_jsonl(path: str | Path, records: Sequence[ShogiGameRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        if not record.moves:
            continue
        lines.append(json.dumps(shogi_game_record_to_json(record), separators=(",", ":"), sort_keys=True))
    if not lines:
        raise ValueError("records must contain at least one non-empty game")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_shogi_move_choice_examples_from_game_records_jsonl(path: str | Path) -> list[ShogiMoveChoiceExample]:
    examples: list[ShogiMoveChoiceExample] = []
    for game_index, record in enumerate(load_shogi_game_records_jsonl(path)):
        winner = shogi_game_winner_to_legacy_side(record.winner)
        if winner is None:
            game_examples = shogi_move_choice_examples_from_usi_moves(record.moves)
        else:
            game_examples = shogi_move_choice_examples_from_usi_moves_with_winner(record.moves, winner=winner)
        examples.extend(_with_game_metadata(game_examples, game_index=game_index))
    return examples


def _with_game_metadata(
    examples: Sequence[ShogiMoveChoiceExample],
    *,
    game_index: int,
) -> list[ShogiMoveChoiceExample]:
    return [
        replace(example, game_index=game_index, ply_index=ply_index)
        for ply_index, example in enumerate(examples)
    ]


def load_kif_game(path: str | Path, *, encoding: str = "cp932") -> tuple[str, ...]:
    return load_kif_game_record(path, encoding=encoding)[0]


def load_kif_game_record(path: str | Path, *, encoding: str = "cp932") -> tuple[tuple[str, ...], str | None]:
    text = Path(path).read_text(encoding=encoding)
    parsed_games = shogi.KIF.Parser.parse_str(text)
    if not parsed_games:
        raise ValueError("KIF file must contain at least one game")
    game = parsed_games[0]
    winner = game.get("win")
    if winner not in {"b", "w"}:
        winner = None
    return tuple(game["moves"]), winner


def load_shogi_game_record_from_kif_file(path: str | Path) -> ShogiGameRecord:
    moves, winner = load_kif_game_record(path)
    return ShogiGameRecord(
        black_player=PlayerSpec(kind="kif", name="black", settings={}),
        white_player=PlayerSpec(kind="kif", name="white", settings={}),
        moves=moves,
        winner=_legacy_side_to_winner(winner),
    )


def load_shogi_move_choice_examples_from_kif_file(path: str | Path) -> list[ShogiMoveChoiceExample]:
    moves, winner = load_kif_game_record(path)
    if winner is None:
        return shogi_move_choice_examples_from_usi_moves(moves)
    return shogi_move_choice_examples_from_usi_moves_with_winner(moves, winner=winner)


def convert_kif_files_to_game_records_jsonl(
    kif_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    max_games: int | None = None,
) -> int:
    records: list[ShogiGameRecord] = []
    for path in kif_paths:
        if max_games is not None and len(records) >= max_games:
            break
        records.append(load_shogi_game_record_from_kif_file(path))
    write_shogi_game_records_jsonl(output_path, records)
    return len(records)


def shogi_game_record_to_json(record: ShogiGameRecord) -> dict[str, object]:
    return {
        "black_player": player_spec_to_json(record.black_player),
        "white_player": player_spec_to_json(record.white_player),
        "moves": list(record.moves),
        "winner": record.winner,
        "end_reason": record.end_reason,
    }


def shogi_game_record_from_json(payload: dict[str, object]) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_player=player_spec_from_json(_object_dict(payload.get("black_player"))),
        white_player=player_spec_from_json(_object_dict(payload.get("white_player"))),
        moves=tuple(str(move) for move in _object_list(payload.get("moves"))),
        winner=_normalize_winner(payload.get("winner")),
        end_reason=None if payload.get("end_reason") is None else str(payload["end_reason"]),
    )


def player_spec_to_json(spec: PlayerSpec) -> dict[str, object]:
    return {
        "kind": spec.kind,
        "name": spec.name,
        "settings": spec.settings,
    }


def player_spec_from_json(payload: dict[str, object]) -> PlayerSpec:
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("player settings must be an object")
    return PlayerSpec(
        kind=str(payload["kind"]),
        name=str(payload["name"]),
        settings={
            str(key): _json_scalar(value)
            for key, value in settings.items()
        },
    )


def _normalize_winner(value: object) -> str | None:
    if value in {"black", "white"}:
        return str(value)
    if value == "b":
        return "black"
    if value == "w":
        return "white"
    if value is None:
        return None
    raise ValueError("winner must be black, white, b, w, or null")


def shogi_game_winner_to_legacy_side(winner: str | None) -> str | None:
    if winner == "black":
        return "b"
    if winner == "white":
        return "w"
    return None


def _legacy_side_to_winner(winner: str | None) -> str | None:
    if winner == "b":
        return "black"
    if winner == "w":
        return "white"
    return None


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
    raise ValueError("player setting values must be JSON scalars")
