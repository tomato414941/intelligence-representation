from __future__ import annotations

from pathlib import Path

import shogi.KIF

from intrep.shogi_move_choice import ShogiMoveChoiceExample, shogi_move_choice_examples_from_usi_moves


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


def load_kif_game(path: str | Path, *, encoding: str = "cp932") -> tuple[str, ...]:
    text = Path(path).read_text(encoding=encoding)
    parsed_games = shogi.KIF.Parser.parse_str(text)
    if not parsed_games:
        raise ValueError("KIF file must contain at least one game")
    return tuple(parsed_games[0]["moves"])


def load_shogi_move_choice_examples_from_kif_file(path: str | Path) -> list[ShogiMoveChoiceExample]:
    return shogi_move_choice_examples_from_usi_moves(load_kif_game(path))
