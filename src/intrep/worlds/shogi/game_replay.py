from __future__ import annotations

from dataclasses import dataclass

import shogi

from intrep.worlds.shogi.game_record import PlayerSpec, ShogiGameRecord


@dataclass(frozen=True)
class ShogiGamePly:
    ply_index: int
    position_sfen: str
    side_to_move: str
    move: str
    source_player: PlayerSpec


def replay_shogi_game_record(record: ShogiGameRecord) -> tuple[ShogiGamePly, ...]:
    board = shogi.Board()
    plies: list[ShogiGamePly] = []
    for ply_index, record_ply in enumerate(record.plies):
        side_to_move = "black" if board.turn == shogi.BLACK else "white"
        legal_moves = {legal_move.usi() for legal_move in board.legal_moves}
        move = record_ply.bestmove
        if record_ply.side != side_to_move:
            raise ValueError(f"wrong side at ply {ply_index}: {record_ply.side}")
        if move not in legal_moves:
            raise ValueError(f"illegal move at ply {ply_index}: {move}")
        plies.append(
            ShogiGamePly(
                ply_index=ply_index,
                position_sfen=board.sfen(),
                side_to_move=side_to_move,
                move=move,
                source_player=record.black_player if side_to_move == "black" else record.white_player,
            )
        )
        board.push_usi(move)
    return tuple(plies)


def validate_shogi_game_record(record: ShogiGameRecord) -> None:
    replay_shogi_game_record(record)
    if record.winner not in {None, "black", "white"}:
        raise ValueError("winner must be black, white, or null")
    if record.end_reason == "max_plies" and record.winner is not None:
        raise ValueError("max_plies records must not have a winner")
