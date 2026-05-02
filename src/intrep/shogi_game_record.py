from __future__ import annotations

from pathlib import Path

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
