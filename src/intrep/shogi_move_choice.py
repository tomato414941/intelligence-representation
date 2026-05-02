from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import shogi

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - exercised in lightweight preprocessing environments.
    torch = None
    TorchDataset = object

@dataclass(frozen=True)
class ShogiMoveChoiceExample:
    position_sfen: str
    legal_moves: tuple[str, ...]
    chosen_move: str
    value_target: float | None = None
    game_index: int | None = None
    ply_index: int | None = None

    def __post_init__(self) -> None:
        if not self.position_sfen:
            raise ValueError("position_sfen must not be empty")
        if not self.legal_moves:
            raise ValueError("legal_moves must not be empty")
        if self.chosen_move not in self.legal_moves:
            raise ValueError("chosen_move must be included in legal_moves")
        if self.value_target is not None and not -1.0 <= self.value_target <= 1.0:
            raise ValueError("value_target must be between -1.0 and 1.0")
        if self.game_index is not None and self.game_index < 0:
            raise ValueError("game_index must be non-negative")
        if self.ply_index is not None and self.ply_index < 0:
            raise ValueError("ply_index must be non-negative")


def shogi_move_choice_example_from_board(board: shogi.Board, chosen_move: str) -> ShogiMoveChoiceExample:
    legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
    return ShogiMoveChoiceExample(
        position_sfen=board.sfen(),
        legal_moves=legal_moves,
        chosen_move=chosen_move,
    )


def shogi_move_choice_examples_from_usi_moves(moves: Sequence[str]) -> list[ShogiMoveChoiceExample]:
    board = shogi.Board()
    examples: list[ShogiMoveChoiceExample] = []
    for move in moves:
        examples.append(shogi_move_choice_example_from_board(board, move))
        board.push_usi(move)
    return examples


def shogi_move_choice_examples_from_usi_moves_with_winner(
    moves: Sequence[str],
    *,
    winner: str,
) -> list[ShogiMoveChoiceExample]:
    if winner not in {"b", "w"}:
        raise ValueError("winner must be 'b' or 'w'")
    board = shogi.Board()
    examples: list[ShogiMoveChoiceExample] = []
    for move in moves:
        side_to_move = "b" if board.turn == shogi.BLACK else "w"
        value_target = 1.0 if side_to_move == winner else -1.0
        base_example = shogi_move_choice_example_from_board(board, move)
        examples.append(
            ShogiMoveChoiceExample(
                position_sfen=base_example.position_sfen,
                legal_moves=base_example.legal_moves,
                chosen_move=base_example.chosen_move,
                value_target=value_target,
            )
        )
        board.push_usi(move)
    return examples


class ShogiMoveChoiceDataset(TorchDataset):
    def __init__(self, examples: Sequence[ShogiMoveChoiceExample]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        self.examples = tuple(examples)
        self.max_choice_count = max(len(example.legal_moves) for example in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        if torch is None:
            raise RuntimeError("torch is required to materialize ShogiMoveChoiceDataset items")
        from intrep.shogi_move_encoding import shogi_candidate_move_features
        from intrep.shogi_position_encoding import shogi_position_token_ids_from_sfen

        example = self.examples[index]
        position_token_ids = shogi_position_token_ids_from_sfen(example.position_sfen)
        candidate_move_features = shogi_candidate_move_features(
            example.legal_moves,
            max_choice_count=self.max_choice_count,
        )
        move_index = example.legal_moves.index(example.chosen_move)
        candidate_mask = torch.zeros(self.max_choice_count, dtype=torch.bool)
        candidate_mask[: len(example.legal_moves)] = True
        value_target = float("nan") if example.value_target is None else example.value_target
        return (
            position_token_ids,
            candidate_move_features,
            candidate_mask,
            torch.tensor(move_index, dtype=torch.long),
            torch.tensor(value_target, dtype=torch.float32),
        )


def write_shogi_move_choice_examples_jsonl(
    path: str | Path,
    examples: Sequence[ShogiMoveChoiceExample],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for example in examples:
        lines.append(
            json.dumps(
                {
                    "position_sfen": example.position_sfen,
                    "legal_moves": list(example.legal_moves),
                    "chosen_move": example.chosen_move,
                    "value_target": example.value_target,
                    "game_index": example.game_index,
                    "ply_index": example.ply_index,
                },
                separators=(",", ":"),
            )
        )
    if not lines:
        raise ValueError("examples must not be empty")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_shogi_move_choice_examples_jsonl(path: str | Path) -> list[ShogiMoveChoiceExample]:
    examples: list[ShogiMoveChoiceExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        examples.append(
            ShogiMoveChoiceExample(
                position_sfen=str(payload["position_sfen"]),
                legal_moves=tuple(str(move) for move in payload["legal_moves"]),
                chosen_move=str(payload["chosen_move"]),
                value_target=payload.get("value_target"),
                game_index=payload.get("game_index"),
                ply_index=payload.get("ply_index"),
            )
        )
    if not examples:
        raise ValueError("shogi move choice examples jsonl must contain at least one example")
    return examples
