from intrep.problems.shogi_policy_value.data import (
    shogi_move_choice_examples_from_game_record,
    shogi_move_policy_value_examples_from_game_record,
)
from intrep.problems.shogi_policy_value.examples import ShogiMoveChoiceExample, ShogiMovePolicyValueExample
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    shogi_game_record_from_usi_moves,
)
from intrep.worlds.shogi.game_trace import trace_shogi_game_record


def shogi_move_choice_examples_from_test_moves(
    moves: tuple[str, ...],
    *,
    winner: str | None = None,
) -> list[ShogiMoveChoiceExample]:
    record = _record_from_test_moves(moves, winner=winner)
    return shogi_move_choice_examples_from_game_record(record)


def shogi_move_policy_value_examples_from_test_moves(
    moves: tuple[str, ...],
    *,
    winner: str | None = None,
) -> list[ShogiMovePolicyValueExample]:
    record = _record_from_test_moves(moves, winner=winner)
    return shogi_move_policy_value_examples_from_game_record(record)


def _record_from_test_moves(moves: tuple[str, ...], *, winner: str | None) -> ShogiGameRecord:
    record = shogi_game_record_from_usi_moves(
        moves,
        black_actor=ShogiActorSpec(kind="test", name="black", settings={}),
        white_actor=ShogiActorSpec(kind="test", name="white", settings={}),
        winner=winner,
    )
    return record


def shogi_trace_transitions_from_test_record(record: ShogiGameRecord):
    return trace_shogi_game_record(record).transitions
