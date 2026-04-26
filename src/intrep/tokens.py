from __future__ import annotations

from intrep.types import Action, Fact


PREDICT_TOKEN = "PREDICT"
UNSUPPORTED_TOKEN = "UNSUPPORTED"


def fact_token(fact: Fact) -> str:
    return f"FACT:{fact.subject}:{fact.predicate}:{fact.object}"


def action_token(action: Action) -> str:
    return f"ACTION:{action.type}:{action.object}:{action.target}"


def state_tokens(state: list[Fact]) -> list[str]:
    return [fact_token(fact) for fact in sorted(state, key=lambda fact: fact.key())]


def model_input_tokens(state: list[Fact], action: Action) -> list[str]:
    return [*state_tokens(state), action_token(action), PREDICT_TOKEN]


def target_token(fact: Fact | None) -> str:
    if fact is None:
        return UNSUPPORTED_TOKEN
    return fact_token(fact)


def fact_from_token(token: str) -> Fact | None:
    if token == UNSUPPORTED_TOKEN:
        return None
    prefix, subject, predicate, object_name = token.split(":", maxsplit=3)
    if prefix != "FACT":
        raise ValueError(f"Expected FACT token, got {token!r}.")
    return Fact(subject=subject, predicate=predicate, object=object_name)
