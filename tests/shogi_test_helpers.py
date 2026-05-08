import shogi

from intrep.problems.shogi_policy_value.data import (
    shogi_move_choice_examples_from_game_record,
    shogi_policy_value_examples_from_game_record,
)
from intrep.problems.shogi_policy_value.examples import ShogiMoveChoiceExample, ShogiPolicyValueExample
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
    record = _record_from_test_moves(moves, winner=winner)
    return shogi_move_choice_examples_from_game_record(record)


def shogi_policy_value_examples_from_test_moves(
    moves: tuple[str, ...],
    *,
    winner: str | None = None,
) -> list[ShogiPolicyValueExample]:
    record = _record_from_test_moves(moves, winner=winner)
    return shogi_policy_value_examples_from_game_record(record)


def _record_from_test_moves(moves: tuple[str, ...], *, winner: str | None) -> ShogiGameRecord:
    record = ShogiGameRecord(
        black_actor=ShogiActorSpec(kind="test", name="black", settings={}),
        white_actor=ShogiActorSpec(kind="test", name="white", settings={}),
        initial_position_sfen=shogi.Board().sfen(),
        transitions=shogi_game_transitions_from_usi_moves(moves, winner=winner),
        winner=winner,
    )
    return record
