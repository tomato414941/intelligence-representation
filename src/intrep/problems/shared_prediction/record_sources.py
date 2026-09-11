"""Recorded speech, inertial sensor windows and passage-grounded questions."""
from __future__ import annotations

import hashlib
import json
import wave

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from intrep.problems.shared_prediction.answers import answer_loss
from intrep.problems.shared_prediction.sources import (
    Source,
    TextSource,
    attach,
    register_source,
)
from intrep.problems.shared_prediction.streams import (
    EpochSampler,
    LineStream,
    file_identity,
)
from intrep.representation.inputs.sequence_heads import (
    FeatureSequenceInput,
    WaveformSequenceInput,
)

ACTIVITIES = ("walking", "walking upstairs", "walking downstairs", "sitting", "standing", "lying down")


def speech_heads(model, config):
    attach(model, "input", "waveform", lambda: WaveformSequenceInput(model.dimension, config["audio_chunk_size"]))
    attach(model, "output", config["name"], lambda: nn.Linear(model.dimension, 10))


def sensor_heads(model, config):
    attach(model, "input", "inertial", lambda: FeatureSequenceInput(9, model.dimension))
    attach(model, "output", config["name"], lambda: nn.Linear(model.dimension, 6))


class IndexedSource(Source):
    def next_record(self):
        return self.read_record(self.sampler.next())

    def loss(self):
        return self.record_loss(self.next_record())

    def record_loss(self, record):
        self.last_group = record["group"]
        hidden = self.model(self.encode_record(record))[:, -1:]
        logits = self.model.decode(self.config["name"], hidden)[:, 0]
        self.last_metrics = {"accuracy": (logits.detach().argmax(-1) == record["label"]).float().mean()}
        return F.cross_entropy(logits, self.ids([record["label"]])[0])

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state):
        self.sampler.load_state_dict(state)

    def progress(self):
        return {"population": self.sampler.count, "samples": self.sampler.samples,
                "cursor": self.sampler.cursor, "epochs": self.sampler.epochs}

    def evaluation_cases(self, examples, generator):
        cursors = generator.sample(range(self.sampler.count), min(examples, self.sampler.count))
        return [{"cursor": cursor, "key": f"record:{int(self.sampler.order[cursor])}"} for cursor in cursors]

    def set_evaluation_case(self, case):
        self.sampler.cursor = case["cursor"]


class SpokenSource(IndexedSource):
    def __init__(self, *args):
        super().__init__(*args)
        self.manifest = self.path("manifest")
        payload = json.loads(self.manifest.read_text())
        assignments = {}
        for row in payload["records"]:
            if row["speaker"] in assignments and assignments[row["speaker"]] != row["split"]:
                raise ValueError("speech speakers cross dataset splits")
            assignments[row["speaker"]] = row["split"]
        self.entries = [row for row in payload["records"] if row["split"] == self.config.get("split", "train")]
        self.labels = np.array([row["label"] for row in self.entries])
        self.sampler = EpochSampler(len(self.entries), self.config["seed"])

    def read_record(self, index):
        entry = self.entries[index]
        path = (self.manifest.parent / entry["path"]).resolve()
        if not path.is_relative_to(self.manifest.parent.resolve()):
            raise ValueError("speech record leaves its dataset directory")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("speech audio digest changed")
        with wave.open(str(path)) as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != entry["sample_rate"]:
                raise ValueError("speech encoding differs from its manifest")
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
        return {"index": index, "audio": torch.tensor(samples, device=self.device, dtype=self.dtype),
                "sample_rate": entry["sample_rate"], "label": entry["label"], "group": entry["speaker"]}

    def encode_record(self, record):
        return self.model.encode("waveform", record["audio"], record["sample_rate"])

    def provenance(self):
        return {"manifest": file_identity(self.manifest), "split": self.config.get("split", "train"),
                "population": len(self.entries), "objective": "spoken digit classification from complete waveforms"}


class SensorSource(IndexedSource):
    def __init__(self, *args):
        super().__init__(*args)
        with np.load(self.path("path"), allow_pickle=False) as arrays:
            self.signals, self.labels, self.subjects = arrays["signals"], arrays["labels"], arrays["subjects"]
        normalization = json.loads(self.path("normalization").read_text())
        self.mean = torch.tensor(normalization["mean"], dtype=self.dtype, device=self.device)
        self.std = torch.tensor(normalization["std"], dtype=self.dtype, device=self.device)
        if self.signals.shape != (len(self.labels), 128, 9) or not np.isfinite(self.signals).all():
            raise ValueError("invalid inertial signal windows")
        self.sampler = EpochSampler(len(self.labels), self.config["seed"])

    def read_record(self, index):
        signal = torch.tensor(self.signals[index], dtype=self.dtype, device=self.device)
        return {"index": index, "sensor": (signal - self.mean) / self.std,
                "label": int(self.labels[index]), "group": f"subject:{int(self.subjects[index])}"}

    def encode_record(self, record):
        return self.model.encode("inertial", record["sensor"].unsqueeze(0))

    def provenance(self):
        return {"file": file_identity(self.path("path")), "normalization": file_identity(self.path("normalization")),
                "population": len(self.labels), "subjects": sorted(set(map(int, self.subjects))),
                "objective": "activity classification from nine measured inertial channels"}


class BoolQSource(TextSource):
    def __init__(self, *args):
        Source.__init__(self, *args)
        self.stream = LineStream(self.path("path"))
        self.tokens = 0
        self.pending = []

    def next_record(self):
        record = json.loads(self.stream.next())
        self.last_group = hashlib.sha256(record["passage"].encode()).hexdigest()
        self.tokens += len(self.text_ids(record["passage"])) + len(self.text_ids(record["question"]))
        return record

    def record_loss(self, record, *, generate=False):
        self.last_group = hashlib.sha256(record["passage"].encode()).hexdigest()
        prompt = f"Passage:\n{record['passage']}\n\nQuestion: {record['question']}\nAnswer only yes or no."
        return answer_loss(self, prompt, "yes" if record["answer"] else "no", generate=generate)

    def provenance(self):
        return {"file": file_identity(self.path("path")), "population": "all official labeled examples; complete passages",
                "objective": "answer only the supplied yes/no question using its passage"}

    def progress(self):
        return {**self.stream.state_dict(), "read_tokens": self.tokens}


register_source("spoken_digits", speech_heads, SpokenSource)
register_source("inertial_activity", sensor_heads, SensorSource)
register_source("boolq", lambda model, config: None, BoolQSource)
