from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import shogi

from intrep.worlds.shogi.game_record import ShogiDecisionTelemetry, ShogiGameRecord, ShogiMoveRecord


@dataclass(frozen=True)
class ShogiTraceTransition:
    ply: int
    side: str
    position_sfen: str
    legal_moves: tuple[str, ...]
    action_usi: str
    next_position_sfen: str
    reward: float
    done: bool
    decision_usi_info_lines: tuple[str, ...] = ()
    decision_telemetry: ShogiDecisionTelemetry | None = None


@dataclass(frozen=True)
class ShogiGameTrace:
    initial_position_sfen: str
    transitions: tuple[ShogiTraceTransition, ...]
    winner: str | None = None


def trace_shogi_game_record(record: ShogiGameRecord) -> ShogiGameTrace:
    return ShogiGameTrace(
        initial_position_sfen=record.initial_position_sfen,
        transitions=shogi_game_trace_transitions_from_move_records(
            record.moves,
            winner=record.winner,
            initial_position_sfen=record.initial_position_sfen,
        ),
        winner=record.winner,
    )


def shogi_game_trace_transitions_from_usi_moves(
    moves: Sequence[str],
    *,
    winner: str | None = None,
    initial_position_sfen: str | None = None,
) -> tuple[ShogiTraceTransition, ...]:
    return shogi_game_trace_transitions_from_move_records(
        tuple(ShogiMoveRecord(action_usi=str(move)) for move in moves),
        winner=winner,
        initial_position_sfen=initial_position_sfen,
    )


def shogi_game_trace_transitions_from_move_records(
    moves: Sequence[ShogiMoveRecord],
    *,
    winner: str | None = None,
    initial_position_sfen: str | None = None,
) -> tuple[ShogiTraceTransition, ...]:
    if winner not in {None, "black", "white"}:
        raise ValueError("winner must be black, white, or null")
    board = shogi.Board(initial_position_sfen) if initial_position_sfen is not None else shogi.Board()
    transitions: list[ShogiTraceTransition] = []
    for ply, move_record in enumerate(moves):
        side = "black" if board.turn == shogi.BLACK else "white"
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        if move_record.action_usi not in legal_moves:
            raise ValueError(f"illegal move at ply {ply}: {move_record.action_usi}")
        position_sfen = board.sfen()
        board.push_usi(move_record.action_usi)
        done = ply == len(moves) - 1
        transitions.append(
            ShogiTraceTransition(
                ply=ply,
                side=side,
                position_sfen=position_sfen,
                legal_moves=legal_moves,
                action_usi=move_record.action_usi,
                next_position_sfen=board.sfen(),
                reward=_terminal_reward(side=side, winner=winner) if done else 0.0,
                done=done,
                decision_usi_info_lines=move_record.decision_usi_info_lines,
                decision_telemetry=move_record.decision_telemetry,
            )
        )
    return tuple(transitions)


def _terminal_reward(*, side: str, winner: str | None) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == side else -1.0
