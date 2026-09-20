from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


class LineStream:
    """Traverse the complete declared file, retaining the byte position on resume."""

    def __init__(self, path: Path, *, start: int = 0, end: int | None = None) -> None:
        size = path.stat().st_size
        end = size if end is None else end
        if not 0 <= start < end <= size:
            raise ValueError("training files must declare a nonempty byte range")
        with path.open("rb") as handle:
            for boundary in (start, end):
                if boundary not in (0, size):
                    handle.seek(boundary - 1)
                    if handle.read(1) != b"\n":
                        raise ValueError("training byte ranges must align with complete lines")
        self.path = path
        self.start, self.end = start, end
        self.offset = start
        self.epochs = 0
        self.records = 0
        self.epoch_limit = None

    def next(self, *, wrap: bool = True) -> str:
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            for _ in range(2):
                while handle.tell() < self.end:
                    raw = handle.readline()
                    self.offset = handle.tell()
                    if raw.strip():
                        self.records += 1
                        return raw.decode("utf-8")
                if not wrap or (self.epoch_limit is not None and self.epochs + 1 >= self.epoch_limit):
                    raise StopIteration
                self.epochs += 1
                self.offset = self.start
                handle.seek(self.start)
        raise ValueError("training file contains no nonempty records")

    def state_dict(self) -> dict:
        return {"offset": self.offset, "epochs": self.epochs, "records": self.records}

    def load_state_dict(self, state: dict) -> None:
        if not self.start <= state["offset"] <= self.end or min(state.values()) < 0:
            raise ValueError("invalid training file cursor")
        self.offset, self.epochs, self.records = state["offset"], state["epochs"], state["records"]


class EpochSampler:
    """Visit every item before repeating; never select a permanent small subset."""

    def __init__(self, count: int, seed: int) -> None:
        if count < 1:
            raise ValueError("a source must contain training examples")
        self.count = count
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(count, generator=self.generator)
        self.cursor = 0
        self.epochs = 0
        self.samples = 0
        self.epoch_limit = None

    def next(self) -> int:
        if self.cursor == self.count:
            if self.epoch_limit is not None and self.epochs + 1 >= self.epoch_limit:
                raise StopIteration
            self.order = torch.randperm(self.count, generator=self.generator)
            self.cursor = 0
            self.epochs += 1
        index = int(self.order[self.cursor])
        self.cursor += 1
        self.samples += 1
        return index

    def state_dict(self) -> dict:
        return {"order": self.order.clone(), "cursor": self.cursor, "epochs": self.epochs,
                "samples": self.samples, "generator": self.generator.get_state()}

    def load_state_dict(self, state: dict) -> None:
        order = state["order"]
        if (order.shape != (self.count,) or not torch.equal(order.sort().values, torch.arange(self.count))
                or not 0 <= state["cursor"] <= self.count or min(state["epochs"], state["samples"]) < 0):
            raise ValueError("source checkpoint has a different population or invalid cursor")
        self.order = order.clone()
        self.cursor, self.epochs, self.samples = state["cursor"], state["epochs"], state["samples"]
        self.generator.set_state(state["generator"])
