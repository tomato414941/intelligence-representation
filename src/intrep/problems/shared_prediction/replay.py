"""Finite, interleaved supervised learning with a fixed experience replay budget."""
from __future__ import annotations

import copy
import random

import torch

from intrep.problems.shared_prediction.population import completed_epochs, limit_epochs
from intrep.problems.shared_prediction.questions import QuestionSource
from intrep.problems.shared_prediction.sources import NativeSource
from intrep.problems.shogi_policy_value.examples import (
    shogi_move_policy_value_example_from_json,
    shogi_move_policy_value_example_to_json,
)


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def pack_batch(source, batch):
    """Keep disk references for media; keep text targets on CPU, never activations."""
    reader = getattr(source, "reader", source)
    records = batch["records"]
    if isinstance(reader, NativeSource):
        records = [[reader.episode_indices[episode.id], index] for episode, index in records]
    elif hasattr(reader, "read_record"):
        records = [record["index"] for record in records]
    elif reader.config["kind"] == "shogi_examples":
        records = [shogi_move_policy_value_example_to_json(record) for record in records]
    return _move({**batch, "records": records}, "cpu")


def unpack_batch(source, batch):
    reader = getattr(source, "reader", source)
    batch = _move(batch, reader.device)
    records = batch["records"]
    if isinstance(reader, NativeSource):
        episodes = {}
        restored = []
        for episode_index, transition in records:
            if episode_index not in episodes:
                episodes[episode_index] = reader.read_episode(episode_index)
            restored.append((episodes[episode_index], transition))
        records = restored
    elif hasattr(reader, "read_record"):
        records = [reader.read_record(index) for index in records]
    elif reader.config["kind"] == "shogi_examples":
        records = [shogi_move_policy_value_example_from_json(record) for record in records]
    return {**batch, "records": records}


def batch_loss(source, batch):
    if isinstance(source, QuestionSource):
        return source.batch_loss(batch)
    return source.record_loss(batch["records"][0])


class ExperienceReplay:
    """Read sources in turn; after N fresh updates, replay one past source batch.

    Replay chooses a source uniformly, then a batch from its reservoir uniformly.
    A batch enters the reservoir only after its successful optimizer update.
    """

    def __init__(self, sources, *, epochs=1, every=1, capacity=128, seed=47):
        if (not sources or type(epochs) is not int or epochs < 1
                or type(every) is not int or every < 0
                or type(capacity) is not int or capacity < 1):
            raise ValueError("replay requires sources, positive epochs/capacity and a nonnegative interval")
        self.sources = sources
        self.epochs, self.every, self.capacity = epochs, every, capacity
        self.generator = random.Random(seed)
        self.cursor = self.since_replay = 0
        self.memory = {name: [] for name in sources}
        self.fresh_updates = dict.fromkeys(sources, 0)
        self.replay_updates = dict.fromkeys(sources, 0)
        for source in sources.values():
            limit_epochs(source, epochs)

    @property
    def replay_due(self):
        return bool(self.every and self.since_replay == self.every and any(self.memory.values()))

    @property
    def complete(self):
        return not self.replay_due and all(completed_epochs(source) >= self.epochs for source in self.sources.values())

    def next(self):
        if self.replay_due:
            name = self.generator.choice([name for name, rows in self.memory.items() if rows])
            batch = self.generator.choice(self.memory[name])
            return name, unpack_batch(self.sources[name], batch), True
        names = list(self.sources)
        for _ in names:
            name = names[self.cursor % len(names)]
            self.cursor += 1
            source = self.sources[name]
            if completed_epochs(source) >= self.epochs:
                continue
            try:
                if isinstance(source, QuestionSource):
                    batch = source.next_batch()
                else:
                    record = source.next_transition() if isinstance(source, NativeSource) else source.next_record()
                    batch = {"records": [record]}
            except StopIteration:
                continue
            return name, batch, False
        raise StopIteration

    def record_update(self, name, batch, replay):
        if replay:
            self.replay_updates[name] += 1
            self.since_replay = 0
            return
        self.fresh_updates[name] += 1
        if self.every:
            self.since_replay += 1
            rows = self.memory[name]
            # Reservoir sampling retains a uniform subset of all learned batches.
            index = len(rows) if len(rows) < self.capacity else self.generator.randrange(self.fresh_updates[name])
            if index < self.capacity:
                packed = pack_batch(self.sources[name], batch)
                if index == len(rows):
                    rows.append(packed)
                else:
                    rows[index] = packed

    def progress(self):
        return {"fresh_epochs": self.epochs, "fresh_updates_per_replay": self.every,
                "capacity_per_source_batches": self.capacity,
                "fresh_selection": "round robin over unfinished sources",
                "replay_selection": "uniform source, then uniform reservoir batch",
                "fresh_updates": dict(self.fresh_updates), "replay_updates": dict(self.replay_updates),
                "retained_batches": {name: len(rows) for name, rows in self.memory.items()}}

    def state_dict(self):
        return {"epochs": self.epochs, "every": self.every, "capacity": self.capacity, "names": list(self.sources),
                "cursor": self.cursor, "since_replay": self.since_replay,
                "fresh_updates": dict(self.fresh_updates), "replay_updates": dict(self.replay_updates),
                "memory": self.memory, "generator": self.generator.getstate()}

    def load_state_dict(self, state, *, extend=False):
        names = list(self.sources)
        previous = state["names"]
        if (state["every"] != self.every or state["capacity"] != self.capacity
                or (names[:len(previous)] != previous if extend else names != previous)):
            raise ValueError("exact replay resume requires the same schedule and previous sources")
        if self.epochs < state["epochs"]:
            raise ValueError("replay continuation cannot reduce the requested fresh epochs")
        if (type(state["cursor"]) is not int or state["cursor"] < 0
                or type(state["since_replay"]) is not int or not 0 <= state["since_replay"] <= self.every):
            raise ValueError("invalid replay schedule cursor")
        for field in ("fresh_updates", "replay_updates", "memory"):
            if set(state[field]) != set(previous):
                raise ValueError("replay state must cover every previous source")
        for name in previous:
            fresh, replay = state["fresh_updates"][name], state["replay_updates"][name]
            if (type(fresh) is not int or type(replay) is not int or min(fresh, replay) < 0
                    or len(state["memory"][name]) != (min(fresh, self.capacity) if self.every else 0)):
                raise ValueError("invalid replay counts or reservoir size")
            self.fresh_updates[name], self.replay_updates[name] = fresh, replay
            self.memory[name] = copy.deepcopy(state["memory"][name])
        fresh, replay = sum(self.fresh_updates.values()), sum(self.replay_updates.values())
        if (self.every and fresh - self.every * replay != state["since_replay"]) or (not self.every and replay):
            raise ValueError("replay counts do not match the fixed schedule")
        self.cursor, self.since_replay = state["cursor"], state["since_replay"]
        self.generator.setstate(state["generator"])
