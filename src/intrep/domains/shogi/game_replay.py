from __future__ import annotations

from dataclasses import dataclass

import shogi

from intrep.domains.shogi.game_record import ShogiActorSpec, ShogiGameRecord
from intrep.domains.shogi.game_trace import trace_shogi_game_record


@dataclass(frozen=True)
class ShogiGamePly:
    ply_index: int
    position_sfen: str
    side_to_move: str
    move: str
    source_actor: ShogiActorSpec


def replay_shogi_game_record(record: ShogiGameRecord) -> tuple[ShogiGamePly, ...]:
    board = shogi.Board(record.initial_position_sfen)
    trace = trace_shogi_game_record(record)
    plies: list[ShogiGamePly] = []
    for ply_index, transition in enumerate(trace.transitions):
        side_to_move = "black" if board.turn == shogi.BLACK else "white"
        legal_moves = {legal_move.usi() for legal_move in board.legal_moves}
        move = transition.action_usi
        if transition.ply != ply_index:
            raise ValueError(f"wrong ply index at ply {ply_index}: {transition.ply}")
        if transition.side != side_to_move:
            raise ValueError(f"wrong side at ply {ply_index}: {transition.side}")
        if transition.position_sfen != board.sfen():
            raise ValueError(f"wrong position at ply {ply_index}")
        if tuple(sorted(legal_moves)) != transition.legal_moves:
            raise ValueError(f"wrong legal moves at ply {ply_index}")
        if move not in legal_moves:
            raise ValueError(f"illegal move at ply {ply_index}: {move}")
        plies.append(
            ShogiGamePly(
                ply_index=ply_index,
                position_sfen=board.sfen(),
                side_to_move=side_to_move,
                move=move,
                source_actor=record.black_actor if side_to_move == "black" else record.white_actor,
            )
        )
        board.push_usi(move)
        if transition.next_position_sfen != board.sfen():
            raise ValueError(f"wrong next position at ply {ply_index}")
        expected_done = ply_index == len(trace.transitions) - 1
        if transition.done != expected_done:
            raise ValueError(f"wrong done flag at ply {ply_index}")
    return tuple(plies)


def validate_shogi_game_record(record: ShogiGameRecord) -> None:
    replay_shogi_game_record(record)
    if record.winner not in {None, "black", "white"}:
        raise ValueError("winner must be black, white, or null")
    if record.end_reason == "max_plies" and record.winner is not None:
        raise ValueError("max_plies records must not have a winner")
