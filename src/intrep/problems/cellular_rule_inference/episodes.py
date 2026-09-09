from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from intrep.worlds.cellular.arrays import neighborhood_keys, rule_table, step_grids
from intrep.worlds.cellular.world import CellularRule, generate_random_cellular_rule


def rule_id(rule: CellularRule) -> int:
    return sum(int(value) << index for index, value in enumerate(rule_table(rule)))


def sample_rules(count: int, seed: int, *, excluded: tuple[CellularRule, ...] = ()) -> list[CellularRule]:
    seen = {rule_id(rule) for rule in excluded}
    if count < 1 or count + len(seen) > 2 ** 17:
        raise ValueError("rule count must fit the B0-excluded rule family")
    rng = random.Random(seed)
    rules = []
    while len(rules) < count:
        rule = generate_random_cellular_rule(rng.getrandbits(64))
        identity = rule_id(rule)
        if identity not in seen:
            seen.add(identity)
            rules.append(rule)
    return rules


def serialize_rules(rules: list[CellularRule]) -> list[dict[str, list[int]]]:
    return [{"birth": sorted(rule.birth), "survival": sorted(rule.survival)} for rule in rules]


def deserialize_rules(records: list[dict[str, list[int]]]) -> list[CellularRule]:
    return [CellularRule(frozenset(row["birth"]), frozenset(row["survival"])) for row in records]


@dataclass(frozen=True)
class RuleEpisodes:
    support: np.ndarray  # [batch, demonstrations, before/after, height, width]
    query: np.ndarray  # [batch, height, width]
    targets: np.ndarray


def sample_episodes(
    rules: list[CellularRule], rng: np.random.Generator, *, height: int, width: int, context_count: int,
) -> RuleEpisodes:
    if not rules or height < 2 or width < 2 or context_count < 0 or context_count >= 2 ** (height * width):
        raise ValueError("need rules, at least 2x2 cells, and nonnegative context count")
    shape = (len(rules), context_count + 1, height, width)
    density = rng.choice([0.2, 0.5, 0.8], size=(len(rules), context_count + 1, 1, 1))
    states = (rng.random(shape) < density).astype(np.int64)
    # A demonstration must never reveal the exact query's answer.
    for index in range(len(rules)):
        while context_count and np.any(np.all(states[index, :-1] == states[index, -1], axis=(-2, -1))):
            states[index, -1] = rng.random((height, width)) < 0.5
    next_states = step_grids(states, rules)
    return RuleEpisodes(
        support=np.stack((states[:, :-1], next_states[:, :-1]), axis=2),
        query=states[:, -1], targets=next_states[:, -1],
    )


def replace_context_rules(episodes: RuleEpisodes, rules: list[CellularRule]) -> RuleEpisodes:
    support = episodes.support.copy()
    support[:, :, 1] = step_grids(support[:, :, 0], rules)
    return RuleEpisodes(support, episodes.query, step_grids(episodes.query, rules))


def evidence_coverage(support: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Evaluation-only oracle: was this local condition seen in a demonstration?

    This declares knowledge of the rule family, not knowledge available to the model.
    Targets from the query are never used to determine coverage.
    """
    query_keys = neighborhood_keys(query)
    if support.shape[1] == 0:
        return np.zeros_like(query, dtype=bool)
    context_keys = neighborhood_keys(support[:, :, 0]).reshape(len(query), -1)
    return (query_keys[..., None] == context_keys[:, None, None, :]).any(axis=-1)
