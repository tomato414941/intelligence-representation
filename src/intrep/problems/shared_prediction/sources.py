from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import shogi
import torch
from torch import nn
from torch.nn import functional as F

from intrep.datasets.vision.cifar10 import read_cifar10_images_and_labels
from intrep.datasets.vision.idx import read_idx_images, read_idx_labels
from intrep.experience.multimodal.records import (
    SELECTION_SCHEMA,
    episode_digest,
    load_episode,
)
from intrep.problems.shared_prediction.streams import (
    EpochSampler,
    LineStream,
    file_identity,
)
from intrep.problems.shogi_policy_value.examples import (
    shogi_move_policy_value_example_from_json,
)
from intrep.representation.inputs.multimodal_observation import (
    image_coordinates,
    patches_to_image,
)
from intrep.representation.inputs.sequence_heads import (
    CoordinateQueryInput,
    FeatureSequenceInput,
    ImageSequenceInput,
    WaveformSequenceInput,
)
from intrep.representation.inputs.shogi_minimal_single_global_position import (
    ShogiMinimalSingleGlobalPositionInputLayer,
)
from intrep.representation.inputs.shogi_position_features.position_features import (
    stack_shogi_position_features,
)
from intrep.representation.inputs.shogi_position_features.position_minimal_single_global import (
    shogi_minimal_single_global_position_features_from_sfen,
)
from intrep.representation.outputs.shogi_action_plane_policy_encoding import (
    SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT,
    shogi_action_plane_policy_action_index,
)


def attach(model, group, name, factory):
    heads = model.input_heads if group == "input" else model.output_heads
    if name not in heads:
        (model.attach_input if group == "input" else model.attach_output)(name, factory())


def image_heads(model, config):
    attach(model, "input", "rgb", lambda: ImageSequenceInput(model.dimension, config["patch_size"]))


def classification_heads(model, config):
    image_heads(model, config)
    attach(model, "output", config["name"], lambda: nn.Linear(model.dimension, config.get("classes", 10)))


def shogi_heads(model, config):
    attach(model, "input", "shogi", lambda: ShogiMinimalSingleGlobalPositionInputLayer(embedding_dim=model.dimension))
    attach(model, "output", "shogi_policy", lambda: nn.Linear(model.dimension, SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT))
    attach(model, "output", "shogi_value", lambda: nn.Linear(model.dimension, 1))


def native_heads(model, config):
    image_heads(model, config)
    dim = model.dimension
    attach(model, "input", "waveform", lambda: WaveformSequenceInput(dim, config["audio_chunk_size"]))
    attach(model, "input", "action", lambda: nn.Embedding(config["action_count"], dim))
    attach(model, "input", "feedback", lambda: FeatureSequenceInput(3, dim))
    for name in ("observation_time", "policy_query", "image_query", "audio_query", "feedback_query"):
        attach(model, "input", name, lambda: CoordinateQueryInput(dim))
    for name, count in (("action", config["action_count"]), ("next_image", 3 * config["patch_size"]**2),
                        ("next_audio", config["audio_chunk_size"]), ("next_feedback", 3)):
        attach(model, "output", name, lambda count=count: nn.Linear(dim, count))


class Source:
    def __init__(self, model, tokenizer, config, root):
        self.model, self.tokenizer, self.config, self.root = model, tokenizer, config, root
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.last_metrics = {}
        self.last_group = None

    def path(self, key):
        return (self.root / self.config[key]).resolve()

    def text_ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def ids(self, values):
        return torch.tensor([values], dtype=torch.long, device=self.device)


class TextSource(Source):
    def __init__(self, *args):
        super().__init__(*args)
        self.stream = LineStream(self.path("path"), start=self.config.get("byte_start", 0),
                                 end=self.config.get("byte_end"))
        self.pending = []
        self.tokens = 0

    def next_record(self):
        count = self.config["block_tokens"]
        while len(self.pending) < count + 1:
            epoch = self.stream.epochs
            row = self.stream.next()
            if self.config["kind"] == "conversations":
                messages = json.loads(row)["messages"]
                encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                             return_dict=False)
            else:
                encoded = self.text_ids(row)
            if not encoded:
                raise ValueError("a nonempty text record produced no tokens")
            if self.stream.epochs != epoch:
                self.pending.append(self.tokenizer.eos_token_id)
            self.pending.extend(encoded)
        tokens = self.ids(self.pending[:count + 1])
        self.pending = self.pending[count:]
        self.tokens += count
        return tokens

    def loss(self):
        return self.record_loss(self.next_record())

    def record_loss(self, tokens):
        return self.record_batch_loss([tokens])

    def record_batch_loss(self, records):
        tokens = torch.cat(records, dim=0)
        hidden = self.model(self.model.encode("text", tokens[:, :-1]))
        logits = self.model.decode("text", hidden)
        loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
        self.last_metrics = {"token_accuracy": (logits.detach().argmax(-1) == tokens[:, 1:]).float().mean()}
        return loss

    def state_dict(self):
        return {"stream": self.stream.state_dict(), "pending": list(self.pending), "tokens": self.tokens}

    def load_state_dict(self, state):
        self.stream.load_state_dict(state["stream"])
        self.pending, self.tokens = list(state["pending"]), state["tokens"]

    def provenance(self):
        return {"file": file_identity(self.path("path")), "byte_range": [self.stream.start, self.stream.end],
                "population": "complete nonempty records in the declared training range",
                "objective": "next token over all text, including all conversation roles"}

    def progress(self):
        return {**self.stream.state_dict(), "byte_range": [self.stream.start, self.stream.end],
                "trained_tokens": self.tokens, "buffered_tokens": len(self.pending)}


def classification_batch_loss(source, sequences, labels):
    """Batch equal-length observations without padding, truncation or dropped examples."""
    groups = {}
    for sequence, label in zip(sequences, labels):
        groups.setdefault(sequence.shape[1], []).append((sequence, label))
    losses, correct = [], []
    for rows in groups.values():
        hidden = source.model(torch.cat([row[0] for row in rows], dim=0))[:, -1:]
        logits = source.model.decode(source.config["name"], hidden)[:, 0]
        targets = torch.tensor([row[1] for row in rows], device=source.device)
        losses.append(F.cross_entropy(logits, targets, reduction="sum"))
        correct.append((logits.detach().argmax(-1) == targets).float().sum())
    source.last_metrics = {"accuracy": torch.stack(correct).sum() / len(labels)}
    return torch.stack(losses).sum() / len(labels)


class ClassificationSource(Source):
    def __init__(self, *args):
        super().__init__(*args)
        if self.config["kind"] == "idx":
            self.images = read_idx_images(self.path("images"))
            self.labels = read_idx_labels(self.path("labels"))
            self.files = [self.path("images"), self.path("labels")]
        else:
            self.files = [(self.root / name).resolve() for name in self.config["batches"]]
            self.images, self.labels = read_cifar10_images_and_labels(self.files)
        if len(self.images) != len(self.labels):
            raise ValueError("classification images and labels differ in length")
        if any(type(index) is not int or not 0 <= index < len(self.images)
               for index in self.config.get("evaluation_excluded_indices", [])):
            raise ValueError("evaluation exclusions must be valid raw image indices")
        self.sampler = EpochSampler(len(self.images), self.config["seed"])

    def next_record(self):
        return self.read_record(self.sampler.next())

    def read_record(self, index):
        pixels = np.array(self.images[index], copy=True)
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[:, :, None], 3, axis=-1)
        image = torch.tensor(pixels, dtype=self.dtype, device=self.device) / 255
        return {"index": index, "image": image, "label": int(self.labels[index])}

    def loss(self):
        return self.record_loss(self.next_record())

    def record_loss(self, record):
        return self.record_batch_loss([record])

    def record_batch_loss(self, records):
        return classification_batch_loss(self, [self.model.encode("rgb", row["image"]) for row in records],
                                         [row["label"] for row in records])

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state):
        self.sampler.load_state_dict(state)

    def provenance(self):
        return {"files": [file_identity(path) for path in self.files], "examples": len(self.images),
                "population": "every image in the declared training files; shuffled without replacement"}

    def progress(self):
        return {"population": self.sampler.count, "cursor": self.sampler.cursor,
                "epochs": self.sampler.epochs, "samples": self.sampler.samples}


class ShogiSource(Source):
    def __init__(self, *args):
        super().__init__(*args)
        self.stream = LineStream(self.path("path"))

    def next_record(self):
        return shogi_move_policy_value_example_from_json(json.loads(self.stream.next()))

    def loss(self):
        return self.record_loss(self.next_record())

    def record_loss(self, example):
        self.last_group = f"game:{example.game_index}" if example.game_index is not None else None
        board = shogi.Board(example.position_sfen)
        features = stack_shogi_position_features([
            shogi_minimal_single_global_position_features_from_sfen(example.position_sfen),
        ]).to(self.device)
        hidden = self.model(self.model.encode("shogi", features))[:, -1:]
        logits = self.model.decode("shogi_policy", hidden)[0, 0]
        moves = [move.usi() for move in board.legal_moves]
        indices = [shogi_action_plane_policy_action_index(move, turn=board.turn) for move in moves]
        probabilities = example.policy_targets or {example.chosen_move: 1.0}
        if not probabilities or set(probabilities) - set(moves):
            raise ValueError("shogi targets contain unavailable moves")
        target = logits.new_tensor([probabilities.get(move, 0) for move in moves])
        target = target / target.sum()
        loss = -(target * logits[indices].log_softmax(-1)).sum()
        predicted = logits.detach()[indices].argmax()
        self.last_metrics = {"policy_nll": loss.detach(), "policy_accuracy": (predicted == target.argmax()).float()}
        if example.value_target is not None:
            value = self.model.decode("shogi_value", hidden)[0, 0, 0].tanh()
            value_loss = (value - example.value_target).square()
            self.last_metrics["value_mse"] = value_loss.detach()
            loss = loss + value_loss
        return loss

    def state_dict(self):
        return self.stream.state_dict()

    def load_state_dict(self, state):
        self.stream.load_state_dict(state)

    def provenance(self):
        return {"file": file_identity(self.path("path")), "population": "every example in the declared training file",
                "objective": "recorded policy and value with an actual legal-move mask"}

    def progress(self):
        return {**self.stream.state_dict(), "bytes": self.stream.end}


class NativeSource(Source):
    def __init__(self, *args):
        super().__init__(*args)
        self.selection = self.path("selection")
        payload = json.loads(self.selection.read_text())
        if payload.get("schema_version") != SELECTION_SCHEMA:
            raise ValueError("unsupported native selection")
        worlds, ids = {}, set()
        for row in payload["episodes"]:
            if row["id"] in ids or worlds.get(row["world_id"], row["split"]) != row["split"]:
                raise ValueError("native selection duplicates records or crosses world splits")
            ids.add(row["id"])
            worlds[row["world_id"]] = row["split"]
        split = self.config.get("split", "train")
        self.entries = [row for row in payload["episodes"] if row["split"] == split]
        self.sampler = EpochSampler(len(self.entries), self.config["seed"])
        self.active = None
        self.transition = 0
        self.episode = None
        self.transitions_seen = 0
        self.omitted_inputs = set()

    def _load(self, index):
        entry = self.entries[index]
        path = (self.selection.parent / entry["path"]).resolve()
        if not path.is_relative_to(self.selection.parent.resolve()) or episode_digest(path) != entry["sha256"]:
            raise ValueError("native episode path or digest differs from its selection")
        self.episode = load_episode(path)
        if self.episode.id != entry["id"] or self.episode.world_id != entry["world_id"]:
            raise ValueError("native episode identity differs from its selection")
        self.active = index

    def next_transition(self):
        if self.episode is None or self.transition == len(self.episode.actions):
            self._load(self.sampler.next())
            self.transition = 0
        index = self.transition
        self.transition += 1
        self.transitions_seen += 1
        return self.episode, index

    def _query(self, name, coordinates):
        return self.model.encode(name, coordinates.to(device=self.device, dtype=self.dtype).unsqueeze(0))

    def _observations(self, observations):
        return [self.model.encode(name, *values) for name, values in self.observation_inputs(observations)]

    def observation_inputs(self, observations):
        sequence = []
        for step, observation in enumerate(observations):
            sequence.append(("observation_time", (torch.tensor([[[step]]], device=self.device, dtype=self.dtype),)))
            if observation.text and "text" not in self.omitted_inputs:
                sequence.append(("text", (self.ids(self.text_ids(observation.text)),)))
            if observation.image is not None and "image" not in self.omitted_inputs:
                sequence.append(("rgb", (observation.image.to(device=self.device, dtype=self.dtype),)))
            if observation.audio is not None and "audio" not in self.omitted_inputs:
                sequence.append(("waveform", (observation.audio.to(device=self.device, dtype=self.dtype), observation.sample_rate)))
            if observation.previous_action is not None:
                sequence.append(("action", (self.ids([observation.previous_action]),)))
            if observation.feedback is not None:
                sequence.append(("feedback", (observation.feedback.to(device=self.device, dtype=self.dtype).view(1, 1, 3),)))
        return sequence

    def loss(self):
        return self.record_loss(self.next_transition())

    def record_loss(self, record):
        episode, index = record
        future = episode.observations[index + 1]
        sequence = self._observations(episode.observations[:index + 1])
        offset = sum(part.shape[1] for part in sequence)
        policy_position = offset
        sequence.append(self._query("policy_query", torch.zeros(1, 1)))
        # The policy precedes the executed action; only forecasts can observe it.
        sequence.append(self.model.encode("action", self.ids([episode.actions[index]])))
        offset += 2
        image_count = 0
        patch = self.config["patch_size"]
        if future.image is not None:
            shape = tuple(math.ceil(size / patch) for size in future.image.shape[:2])
            coordinates = image_coordinates(*shape, device=self.device)
            image_count = len(coordinates)
            sequence.append(self._query("image_query", coordinates))
        audio_count = 0
        chunk = self.config["audio_chunk_size"]
        if future.audio is not None:
            audio_count = math.ceil(len(future.audio) / chunk)
            times = torch.arange(audio_count).view(-1, 1) * chunk / future.sample_rate
            sequence.append(self._query("audio_query", times))
        sequence.append(self._query("feedback_query", torch.zeros(1, 1)))
        text_targets = []
        if episode.answers and episode.answers[index] is not None:
            text_targets = self.text_ids(episode.answers[index]) + [self.tokenizer.eos_token_id]
            sequence.append(self.model.encode("text", self.ids([self.tokenizer.bos_token_id, *text_targets[:-1]])))
        hidden = self.model(torch.cat(sequence, dim=1))
        teacher = episode.teacher_actions[index] if episode.teacher_actions else None
        action = episode.actions[index] if teacher is None else teacher
        logits = self.model.decode("action", hidden[:, policy_position:policy_position + 1])[:, 0]
        loss = F.cross_entropy(logits, torch.tensor([action], device=self.device))
        self.last_group = episode.world_id
        self.last_metrics = {"action_nll": loss.detach(),
                             "action_accuracy": (logits.detach().argmax(-1) == action).float().mean()}
        if image_count:
            patches = self.model.decode("next_image", hidden[:, offset:offset + image_count])[0].sigmoid()
            predicted = patches_to_image(patches, tuple(future.image.shape[:2]), patch)
            image_loss = F.mse_loss(predicted, future.image.to(predicted))
            self.last_metrics["image_mse"] = image_loss.detach()
            loss = loss + image_loss
        offset += image_count
        if audio_count:
            predicted = self.model.decode("next_audio", hidden[:, offset:offset + audio_count]).tanh().flatten()[:len(future.audio)]
            audio_loss = F.mse_loss(predicted, future.audio.to(predicted))
            self.last_metrics["audio_mse"] = audio_loss.detach()
            loss = loss + audio_loss
        offset += audio_count
        feedback = self.model.decode("next_feedback", hidden[:, offset:offset + 1])[0, 0]
        target = future.feedback.to(feedback)
        feedback_loss = (feedback[0] - target[0]).square() + F.binary_cross_entropy_with_logits(feedback[1:], target[1:])
        self.last_metrics["feedback_loss"] = feedback_loss.detach()
        loss = loss + feedback_loss
        if text_targets:
            logits = self.model.decode("text", hidden[:, offset + 1:])[0]
            language_loss = F.cross_entropy(logits, self.ids(text_targets)[0])
            self.last_metrics["answer_nll"] = language_loss.detach()
            self.last_metrics["answer_token_accuracy"] = (logits.detach().argmax(-1) == self.ids(text_targets)[0]).float().mean()
            loss = loss + language_loss
        return loss

    def state_dict(self):
        return {"sampler": self.sampler.state_dict(), "active": self.active, "transition": self.transition,
                "transitions_seen": self.transitions_seen}

    def load_state_dict(self, state):
        self.sampler.load_state_dict(state["sampler"])
        self.episode = None
        self.active, self.transition = state["active"], state["transition"]
        self.transitions_seen = state["transitions_seen"]
        if self.active is not None:
            self._load(self.active)
            if not 0 <= self.transition <= len(self.episode.actions):
                raise ValueError("invalid native transition cursor")

    def provenance(self):
        return {"selection": file_identity(self.selection), "split": self.config.get("split", "train"),
                "episodes": len(self.entries), "population": "every transition of every episode in the declared split",
                "objective": "actions, next RGB/audio/feedback and available language targets; actual observations only"}

    def progress(self):
        return {"episodes": self.sampler.count, "episodes_started": self.sampler.samples,
                "epochs": self.sampler.epochs, "transitions": self.transitions_seen,
                "active_episode": self.active, "transition": self.transition}


# Extensions register their own source and head constructors. The shared model
# and joint trainer contain no list of supported modalities or objectives.
SOURCE_FACTORIES = {
    "text": (lambda model, config: None, TextSource),
    "conversations": (lambda model, config: None, TextSource),
    "idx": (classification_heads, ClassificationSource),
    "cifar10": (classification_heads, ClassificationSource),
    "shogi_examples": (shogi_heads, ShogiSource),
    "native": (native_heads, NativeSource),
}


def register_source(kind, configure_heads, source_factory):
    if not kind or kind in SOURCE_FACTORIES:
        raise ValueError("source extensions require a new kind")
    SOURCE_FACTORIES[kind] = (configure_heads, source_factory)


def source_configs(recipe):
    configurations = []
    seen = set()
    for item in recipe["sources"]:
        config = {**recipe.get("defaults", {}), **item}
        name = config["name"]
        if name in seen or not name or config["kind"] not in SOURCE_FACTORIES:
            raise ValueError("source names must be unique and their factories registered")
        seen.add(name)
        configurations.append(config)
    if not configurations:
        raise ValueError("a joint recipe must declare training sources")
    return configurations


def configure_heads(model, recipe):
    for config in source_configs(recipe):
        SOURCE_FACTORIES[config["kind"]][0](model, config)
        if "question_mode" in config:
            from intrep.problems.shared_prediction.questions import (
                configure_question_heads,
            )
            configure_question_heads(model, config)


def build_sources(model, tokenizer, recipe, root: Path):
    configure_heads(model, recipe)
    sources = {}
    for config in source_configs(recipe):
        factory = SOURCE_FACTORIES[config["kind"]][1]
        if config["kind"] == "conversations":
            objective = config.get("conversation_objective", "all_tokens")
            if objective == "assistant":
                from intrep.problems.shared_prediction.conversations import AssistantConversationSource
                factory = AssistantConversationSource
            elif objective != "all_tokens":
                raise ValueError("conversation_objective must be all_tokens or assistant")
        source = factory(model, tokenizer, config, root)
        if "question_mode" in config:
            from intrep.problems.shared_prediction.questions import QuestionSource
            source = QuestionSource(source)
        sources[config["name"]] = source
    return sources
