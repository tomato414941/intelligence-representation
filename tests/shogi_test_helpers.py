import shogi

from intrep.tasks.shogi_move_choice.data import shogi_move_choice_examples_from_game_record
from intrep.tasks.shogi_move_choice.examples import ShogiMoveChoiceExample
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    shogi_game_transitions_from_usi_moves,
)


def shogi_move_choice_examples_from_test_moves(
    moves: tuple[str, ...],
    *,
    winner: str | None = None,
) -> list[ShogiMoveChoiceExample]:
    record = ShogiGameRecord(
        black_actor=ShogiActorSpec(kind="test", name="black", settings={}),
        white_actor=ShogiActorSpec(kind="test", name="white", settings={}),
        initial_position_sfen=shogi.Board().sfen(),
        transitions=shogi_game_transitions_from_usi_moves(moves, winner=winner),
        winner=winner,
    )
    return shogi_move_choice_examples_from_game_record(record)
