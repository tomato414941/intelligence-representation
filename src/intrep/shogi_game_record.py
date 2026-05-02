from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Iterator, Sequence

import shogi.KIF

from intrep.shogi_move_choice import (
    ShogiMoveChoiceExample,
    shogi_move_choice_examples_from_usi_moves,
    shogi_move_choice_examples_from_usi_moves_with_winner,
)


@dataclass(frozen=True)
class ShogiGameRecord:
    moves: tuple[str, ...]
    winner: str | None = None


def load_usi_move_games(path: str | Path) -> list[tuple[str, ...]]:
    games: list[tuple[str, ...]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        moves = tuple(stripped.split())
        if not moves:
            continue
        games.append(moves)
    if not games:
        raise ValueError("USI move file must contain at least one game")
    return games


def load_shogi_move_choice_examples_from_usi_file(path: str | Path) -> list[ShogiMoveChoiceExample]:
    examples: list[ShogiMoveChoiceExample] = []
    for game in load_usi_move_games(path):
        examples.extend(shogi_move_choice_examples_from_usi_moves(game))
    return examples


def load_shogi_game_records_jsonl(path: str | Path) -> list[ShogiGameRecord]:
    records: list[ShogiGameRecord] = []
    for record in iter_shogi_game_records_jsonl(path):
        records.append(record)
    if not records:
        raise ValueError("shogi game records jsonl must contain at least one game")
    return records


def iter_shogi_game_records_jsonl(path: str | Path) -> Iterator[ShogiGameRecord]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        winner = payload.get("winner")
        if winner not in {"b", "w"}:
            winner = None
        moves = tuple(str(move) for move in payload["moves"])
        if moves:
            yield ShogiGameRecord(moves=moves, winner=winner)


def write_shogi_game_records_jsonl(path: str | Path, records: Sequence[ShogiGameRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        if not record.moves:
            continue
        lines.append(json.dumps({"winner": record.winner, "moves": list(record.moves)}, separators=(",", ":")))
    if not lines:
        raise ValueError("records must contain at least one non-empty game")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_shogi_move_choice_examples_from_game_records_jsonl(path: str | Path) -> list[ShogiMoveChoiceExample]:
    examples: list[ShogiMoveChoiceExample] = []
    for game_index, record in enumerate(load_shogi_game_records_jsonl(path)):
        if record.winner is None:
            game_examples = shogi_move_choice_examples_from_usi_moves(record.moves)
        else:
            game_examples = shogi_move_choice_examples_from_usi_moves_with_winner(record.moves, winner=record.winner)
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
    return ShogiGameRecord(moves=moves, winner=winner)


def load_shogi_move_choice_examples_from_kif_file(path: str | Path) -> list[ShogiMoveChoiceExample]:
    moves, winner = load_kif_game_record(path)
    if winner is None:
        return shogi_move_choice_examples_from_usi_moves(moves)
    return shogi_move_choice_examples_from_usi_moves_with_winner(moves, winner=winner)


def write_usi_move_games(path: str | Path, games: Sequence[Sequence[str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(game) for game in games if game]
    if not lines:
        raise ValueError("games must contain at least one non-empty game")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_kif_files_to_usi_file(
    kif_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    max_games: int | None = None,
) -> int:
    games: list[tuple[str, ...]] = []
    for path in kif_paths:
        if max_games is not None and len(games) >= max_games:
            break
        games.append(load_kif_game(path))
    write_usi_move_games(output_path, games)
    return len(games)


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


def convert_kif_archive_to_usi_files(
    archive_path: str | Path,
    *,
    train_output_path: str | Path,
    eval_output_path: str | Path,
    train_games: int,
    eval_games: int,
) -> tuple[int, int]:
    if train_games <= 0:
        raise ValueError("train_games must be positive")
    if eval_games < 0:
        raise ValueError("eval_games must be non-negative")
    try:
        import py7zr
    except ImportError as error:
        raise RuntimeError("py7zr is required to convert .7z KIF archives") from error

    total_games = train_games + eval_games
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        targets = [name for name in archive.getnames() if name.lower().endswith(".kif")][:total_games]
        if len(targets) < total_games:
            raise ValueError(f"KIF archive contains only {len(targets)} games, but {total_games} were requested")
        with tempfile.TemporaryDirectory() as directory:
            archive.extract(path=directory, targets=targets)
            extracted_paths = [Path(directory) / target for target in targets]
            train_count = convert_kif_files_to_usi_file(
                extracted_paths[:train_games],
                train_output_path,
                max_games=train_games,
            )
            eval_count = 0
            if eval_games > 0:
                eval_count = convert_kif_files_to_usi_file(
                    extracted_paths[train_games:],
                    eval_output_path,
                    max_games=eval_games,
                )
            return train_count, eval_count
