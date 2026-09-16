"""Build several questions from each record while retaining all source populations."""
from __future__ import annotations

import copy
import hashlib
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import shogi
import torch
from torch import nn
from torch.nn import functional as F

from intrep.problems.shared_prediction.answers import answer_scores, question_prefix
from intrep.problems.shared_prediction.sources import NativeSource, attach
from intrep.problems.shared_prediction.streams import EpochSampler
from intrep.representation.inputs.multimodal_observation import (
    image_coordinates,
    image_patches,
)
from intrep.representation.inputs.sequence_heads import CoordinateQueryInput
from intrep.representation.inputs.shogi_position_features.position_features import (
    stack_shogi_position_features,
)
from intrep.representation.inputs.shogi_position_features.position_minimal_single_global import (
    shogi_minimal_single_global_position_features_from_sfen,
)

FORMS = {
    "text": ("original", "infill", "first_word", "last_word"),
    "conversations": ("original", "infill", "first_word", "last_word"),
    "idx": ("original", "identify", "same", "different", "inpaint"),
    "cifar10": ("original", "identify", "same", "different", "inpaint"),
    "shogi_examples": ("original", "legal", "illegal", "after_move"),
    "native": ("original", "first_action", "last_action", "last_reward"),
    "spoken_digits": ("original", "identify", "same", "different", "sum", "audio_gap"),
    "inertial_activity": ("original", "identify", "same", "different", "sensor_future"),
    "boolq": ("original", "verify_yes", "verify_no"),
}

CLASS_NAMES = {
    "mnist": tuple(map(str, range(10))),
    "spoken_digits": tuple(map(str, range(10))),
    "fashion_mnist": ("T-shirt or top", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"),
    "cifar10": ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"),
    "inertial_activity": ("walking", "walking upstairs", "walking downstairs", "sitting", "standing", "lying down"),
}


@dataclass
class Prediction:
    head: str
    coordinates: torch.Tensor
    target: torch.Tensor
    kind: str
    baseline: torch.Tensor | None = None
    valid: torch.Tensor | None = None


@dataclass
class Question:
    prompt: str
    answer: str | None = None
    inputs: list = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)


def available_forms(config):
    forms = FORMS[config["kind"]]
    if config["name"] == "mnist":
        forms = (*forms, "sum", "greater")
    return forms


def configure_question_heads(model, config):
    attach(model, "input", "answer_query", lambda: CoordinateQueryInput(model.dimension))
    kind, dim = config["kind"], model.dimension
    if kind in ("idx", "cifar10"):
        attach(model, "output", "next_image", lambda: nn.Linear(dim, 3 * config["patch_size"] ** 2))
    elif kind == "shogi_examples":
        attach(model, "output", "board_pieces", lambda: nn.Linear(dim, 29))
        attach(model, "output", "board_hands", lambda: nn.Linear(dim, 19))
    elif kind == "spoken_digits":
        attach(model, "output", "next_audio", lambda: nn.Linear(dim, config["audio_chunk_size"]))
    elif kind == "inertial_activity":
        attach(model, "output", "next_sensor", lambda: nn.Linear(dim, 9))


def stable_seed(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "little")


def text_question(source, tokens, form, wording):
    if isinstance(tokens, dict):
        tokens = tokens["tokens"]
    ids = tokens[0].tolist()
    text = source.tokenizer.decode(ids, skip_special_tokens=True)
    if form == "infill":
        words = list(re.finditer(r"\S+", text))
        first = len(words) // 3
        last = min(len(words) - 1, first + max(0, min(7, len(words) // 4)))
        start, end = words[first].start(), words[last].end()
        prefix, gap, suffix = text[:start], text[start:end], text[end:]
        instruction = ("Fill [GAP] in this excerpt. Return only the missing text.",
                       "What text belongs at [GAP]? Supply that text alone.")[wording]
        return Question(f"{instruction}\n{prefix}[GAP]{suffix}", gap)
    words = text.split()
    index = 0 if form == "first_word" else -1
    location = "first" if index == 0 else "last"
    instruction = (f"Copy only the {location} whitespace-separated word from this excerpt, preserving punctuation.",
                   f"Return the {location} word in the excerpt below, including its punctuation. No explanation.")[wording]
    return Question(f"{instruction}\nExcerpt:\n{text}", words[index])


def categorical_inputs(source, record):
    if "image" in record:
        return [("rgb", (record["image"],))]
    if "audio" in record:
        return [("waveform", (record["audio"], record["sample_rate"]))]
    return [("inertial", (record["sensor"].unsqueeze(0),))]


def categorical_question(source, records, form, wording):
    inputs = []
    for index, record in enumerate(records):
        inputs.append(("text", (source.ids(source.text_ids(f"\nObservation {index + 1}:\n")),)))
        inputs.extend(categorical_inputs(source, record))
    a, b = [record["label"] for record in records]
    if form == "identify":
        prompt = ("Name the class of observation 1. Return only its class name.",
                  "Which class does the first observation belong to? Give the class name alone.")[wording]
        answer = CLASS_NAMES.get(source.config["name"], tuple(map(str, range(10))))[a]
    elif form in ("same", "different"):
        adjective = "the same" if form == "same" else "different"
        prompt = (f"Do the two observations have {adjective} classes? Answer only yes or no.",
                  f"Are their classes {adjective}? Reply yes or no and nothing else.")[wording]
        answer = "yes" if ((a == b) == (form == "same")) else "no"
    elif form == "sum":
        prompt = ("Add the digits in observations 1 and 2. Return only the sum.",
                  "What is the sum of these two digits? Output the number alone.")[wording]
        answer = str(a + b)
    else:
        prompt = ("Is the digit in observation 1 greater than the digit in observation 2? Answer only yes or no.",
                  "Does the first digit exceed the second? Reply only yes or no.")[wording]
        answer = "yes" if a > b else "no"
    return Question(prompt, answer, inputs)


def image_question(source, record, seed, wording):
    image, patch = record["image"], source.config["patch_size"]
    target, shape = image_patches(image, patch)
    rows, columns = shape
    generator = random.Random(seed)
    height, width = max(1, rows // 2), max(1, columns // 2)
    top, left = generator.randrange(rows - height + 1), generator.randrange(columns - width + 1)
    mask = torch.zeros(shape, dtype=torch.bool, device=image.device)
    mask[top:top + height, left:left + width] = True
    corrupted = image.clone()
    corrupted[top * patch:(top + height) * patch, left * patch:(left + width) * patch] = 0
    coordinates = image_coordinates(rows, columns, device=image.device)[mask.flatten()]
    prompt = ("Restore the missing image patch region", "Reconstruct the hidden image region")[wording]
    prompt += f": patch rows {top} to {top + height - 1}, columns {left} to {left + width - 1}."
    # Only visible pixels may determine a completion baseline.
    visible = torch.ones(image.shape[:2], dtype=torch.bool, device=image.device)
    visible[top * patch:(top + height) * patch, left * patch:(left + width) * patch] = False
    if bool(visible.any()):
        baseline = image[visible].mean(0).repeat_interleave(patch ** 2).expand(int(mask.sum()), -1)
    else:
        baseline = torch.zeros_like(target[mask.flatten()])
    return Question(prompt, inputs=[("rgb", (corrupted,))], predictions=[
        Prediction("next_image", coordinates, target[mask.flatten()], "image", baseline),
    ])


def waveform_question(source, record, wording):
    waveform, chunk = record["audio"], source.config["audio_chunk_size"]
    count = math.ceil(len(waveform) / chunk)
    start, stop = count // 3, min(count, count // 3 + max(1, count // 4))
    corrupted = waveform.clone()
    corrupted[start * chunk:stop * chunk] = 0
    target = F.pad(waveform, (0, count * chunk - len(waveform))).view(count, chunk)[start:stop]
    valid = torch.arange(start * chunk, stop * chunk, device=waveform.device).view(-1, chunk) < len(waveform)
    coordinates = torch.arange(start, stop, device=waveform.device).view(-1, 1) * chunk / record["sample_rate"]
    prompt = ("Restore the missing audio interval", "Reconstruct the hidden waveform interval")[wording]
    prompt += f": samples {start * chunk} to {min(len(waveform), stop * chunk) - 1}."
    baseline = torch.zeros_like(target)
    if start:
        baseline = waveform[(start - 1) * chunk:start * chunk].expand_as(target)
    return Question(prompt, inputs=[("waveform", (corrupted, record["sample_rate"]))], predictions=[
        Prediction("next_audio", coordinates, target, "audio", baseline, valid),
    ])


def sensor_question(record, wording):
    history, target = record["sensor"][:96], record["sensor"][96:]
    prompt = ("Predict the next 32 samples of all nine sensor channels.",
              "Continue these nine measured signals for 32 more time steps.")[wording]
    return Question(prompt, inputs=[("inertial", (history.unsqueeze(0),))], predictions=[
        Prediction("next_sensor", torch.arange(96, 128, device=history.device).view(-1, 1) / 50,
                   target, "mse", history[-1:].expand_as(target)),
    ])


def board_labels(board, device):
    pieces = [board.piece_at(square) for square in range(81)]
    occupied = torch.tensor([0 if piece is None else piece.piece_type + 14 * piece.color for piece in pieces], device=device)
    hands = torch.tensor([board.pieces_in_hand[color].get(piece, 0) for color in (0, 1) for piece in range(1, 8)], device=device)
    return occupied, hands


def shogi_question(source, record, form, seed, wording):
    board = shogi.Board(record.position_sfen)
    legal = list(board.legal_moves)
    generator = random.Random(seed)
    if form == "after_move" or seed % 2 == 0:
        candidate = generator.choice(legal)
    else:
        for _ in range(1000):
            candidate = shogi.Move(generator.randrange(81), generator.randrange(81))
            if not board.is_legal(candidate):
                break
        else:
            raise ValueError("could not construct an illegal candidate")
    inputs = [("shogi", (stack_shogi_position_features([
        shogi_minimal_single_global_position_features_from_sfen(record.position_sfen),
    ]).to(source.device),))]
    if form in ("legal", "illegal"):
        positive = board.is_legal(candidate)
        answer = "yes" if (positive == (form == "legal")) else "no"
        prompt = (f"Is move {candidate.usi()} {form} in this shogi position? Answer only yes or no.",
                  f"For the displayed position, would {candidate.usi()} be {form}? Reply yes or no.")[wording]
        return Question(prompt, answer, inputs)
    before = board_labels(board, source.device)
    board.push(candidate)
    after = board_labels(board, source.device)
    prompt = (f"Predict the complete board and hands after legal move {candidate.usi()}.",
              f"Apply {candidate.usi()}. What are all the resulting board pieces and pieces in hand?")[wording]
    return Question(prompt, inputs=inputs, predictions=[
        Prediction("board_pieces", torch.stack((torch.zeros(81), torch.arange(81)), dim=-1), after[0], "class", before[0]),
        Prediction("board_hands", torch.stack((torch.ones(14), torch.arange(14)), dim=-1), after[1], "class", before[1]),
    ])


def history_question(source, record, form, wording):
    episode, index = record
    observations = episode.observations[:index + 1]
    if form in ("first_action", "last_action"):
        location = "first" if form == "first_action" else "most recent"
        action_index = 0 if form == "first_action" else index - 1
        answer = str(episode.actions[action_index]) if index else "none"
        prompt = (f"What was the {location} executed action in the observed history? Return its numeric ID, or none if no action occurred.",
                  f"Report only the ID of the {location} past action. If there are no past actions, answer none.")[wording]
    else:
        feedback = observations[-1].feedback
        value = float(feedback[0]) if feedback is not None else None
        answer = "none" if value is None else "positive" if value > 0 else "negative" if value < 0 else "zero"
        prompt = ("Was the most recently observed reward positive, negative, or zero? Return one of those words, or none if absent.",
                  "Classify the sign of the last observed reward. Output positive, negative, zero, or none.")[wording]
    return Question(prompt, answer, source.observation_inputs(observations))


def boolq_question(record, form, wording):
    proposed = form == "verify_yes"
    value = "yes" if proposed else "no"
    instruction = (f"A proposed answer to the question is {value}. Is that answer correct? Answer only yes or no.",
                   f"Check the suggested response, {value}. Does it correctly answer the question? Reply yes or no.")[wording]
    return Question(f"Passage:\n{record['passage']}\n\nQuestion: {record['question']}\n{instruction}",
                    "yes" if record["answer"] == proposed else "no")


def _metric_means(rows):
    # Transfer detached scalars together instead of waiting on every question's
    # metrics. Keep dtypes separate so conversion preserves each scalar's value.
    groups = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, torch.Tensor):
                groups.setdefault((value.device, value.dtype), []).append((row, key, value.detach().reshape(())))
            else:
                row[key] = float(value)
    for entries in groups.values():
        values = torch.stack([value for _, _, value in entries]).cpu().tolist()
        for (row, key, _), value in zip(entries, values):
            row[key] = float(value)
    # Preserve finite-value rejection and per-key means, including metrics
    # present in only some questions, before the trainer can update parameters.
    from intrep.problems.shared_prediction.evaluation import summarize
    summary = summarize([{"metrics": row} for row in rows])
    return {key: row["mean"] for key, row in summary.items()}


class QuestionSource:
    def __init__(self, reader):
        if reader.config.get("question_mode") not in ("fixed", "varied"):
            raise ValueError("question_mode must be fixed or varied")
        self.reader = reader
        self.step = 0
        self.counts = Counter()
        self.distinct = set()
        self.answer_tokens = 0
        self._forced = None
        self.last_response = None
        self.forms = available_forms(reader.config)
        self.class_indices, self.partners = {}, {}
        self.evaluation_excluded_indices = set(reader.config.get("evaluation_excluded_indices", []))
        if hasattr(reader, "labels"):
            for label in sorted(set(map(int, reader.labels))):
                indices = np.flatnonzero(np.asarray(reader.labels) == label).tolist()
                self.class_indices[label] = indices
                self.partners[label] = EpochSampler(len(indices), reader.config["seed"] + label + 1000)
        self.records_per_update = reader.config.get("records_per_update", 2 if self.partners else 1)
        if (type(self.records_per_update) is not int or self.records_per_update < 1
                or self.partners and (self.records_per_update < 2 or self.records_per_update % 2)):
            raise ValueError("records_per_update must be positive, and paired sources require an even count of at least two")

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def state_dict(self):
        return {"reader": self.reader.state_dict(), "step": self.step, "counts": dict(self.counts),
                "distinct": sorted(self.distinct), "answer_tokens": self.answer_tokens,
                "partners": {str(key): value.state_dict() for key, value in self.partners.items()}, "forced": self._forced}

    def load_state_dict(self, state):
        self.reader.load_state_dict(state["reader"])
        self.step, self.counts = state["step"], Counter(state["counts"])
        self.distinct, self.answer_tokens = set(state["distinct"]), state["answer_tokens"]
        self._forced = copy.deepcopy(state["forced"])
        for key, value in state["partners"].items():
            self.partners[int(key)].load_state_dict(value)

    def provenance(self):
        return {"reader": self.reader.provenance(), "forms": list(self.forms),
                "training_mode": self.config["question_mode"],
                "pair_sampling": "complete primary stream plus class-balanced partners; each class partner pool cycles completely"}

    def progress(self):
        progress = self.reader.progress()
        if "trained_tokens" in progress:
            progress["read_tokens"] = progress.pop("trained_tokens")
        return {"reader": progress, "source_updates": self.step, "questions_by_form": dict(self.counts),
                "distinct_indexed_records_or_transitions": len(self.distinct), "answer_target_tokens": self.answer_tokens}

    def evaluation_cases(self, examples, generator):
        from intrep.problems.shared_prediction.evaluation import make_panel
        name = self.config["name"]
        panel = make_panel({name: self.reader}, examples, seed=generator.randrange(2**32))[name]
        if isinstance(self.reader, NativeSource):
            groups = list(dict.fromkeys(case["group"] for case in panel))
            selected_groups = set(generator.sample(groups, min(len(groups), self.config.get("question_evaluation_worlds", 16))))
            selected = {index for index, case in enumerate(panel) if case["group"] in selected_groups}
        else:
            selected = set(generator.sample(range(len(panel)), min(len(panel), self.config.get("question_evaluation_examples", 32))))
        return [{**case, "record_key": case["key"], "form": form, "wording": wording,
                 "key": f"{case['key']}/{form}/{wording}", "seed": stable_seed(case["key"]),
                 "group": case.get("group", case["key"])}
                for index, case in enumerate(panel) for form in self.forms if form == "original" or index in selected
                for wording in ((0,) if form == "original" else (0, 1))]

    def set_evaluation_case(self, case):
        from intrep.problems.shared_prediction.evaluation import set_case
        set_case(self.reader, case)
        self._forced = case

    def _records(self, seed):
        next_record = self.reader.next_transition if isinstance(self.reader, NativeSource) else self.reader.next_record
        first = next_record()
        if self._forced and isinstance(first, dict) and first.get("index") in self.evaluation_excluded_indices:
            raise ValueError("a reserved holdout image cannot enter development evaluation")
        records = [first]
        if self.partners:
            generator = random.Random(seed)
            # Each added form must see both pair labels, independently of the
            # original/added alternation and its own position in the cycle.
            cycle = self.step // (2 * (len(self.forms) - 1))
            same = seed % 2 == 0 if self._forced else cycle % 2 == 0
            for pair_index in range(1 if self._forced else self.records_per_update // 2):
                if pair_index:
                    first = next_record()
                    records.append(first)
                label = first["label"]
                eligible_labels = [value for value, indices in self.class_indices.items() if value != label
                                   and (not self._forced or any(index not in self.evaluation_excluded_indices for index in indices))]
                if not same and not eligible_labels:
                    raise ValueError("no different-class development partner remains outside the holdout")
                selected = label if same else generator.choice(eligible_labels)
                pool = self.class_indices[selected]
                if self._forced:
                    pool = [index for index in pool if index not in self.evaluation_excluded_indices]
                    if not pool:
                        raise ValueError("no development partner remains outside the holdout for this class")
                index = generator.choice(pool) if self._forced else pool[self.partners[selected].next()]
                while len(pool) > 1 and index == first["index"]:
                    index = generator.choice(pool) if self._forced else pool[self.partners[selected].next()]
                records.append(self.reader.read_record(index))
        elif not self._forced:
            records.extend(next_record() for _ in range(self.records_per_update - 1))
        for record in records:
            if isinstance(record, dict) and "index" in record:
                self.distinct.add(str(record["index"]))
            elif isinstance(self.reader, NativeSource):
                self.distinct.add(f"{record[0].id}:{record[1]}")
        return records

    def _question(self, records, form, seed, wording):
        kind, record = self.config["kind"], records[0]
        if kind in ("text", "conversations"):
            return text_question(self.reader, record, form, wording)
        if kind == "shogi_examples":
            return shogi_question(self.reader, record, form, seed, wording)
        if kind == "native":
            return history_question(self.reader, record, form, wording)
        if kind == "boolq":
            return boolq_question(record, form, wording)
        if form == "inpaint":
            return image_question(self.reader, record, seed, wording)
        if form == "audio_gap":
            return waveform_question(self.reader, record, wording)
        if form == "sensor_future":
            return sensor_question(record, wording)
        return categorical_question(self.reader, records, form, wording)

    def _score(self, question):
        loss, self.last_metrics, self.last_response = self._score_batch([question])[0]
        return loss

    def _score_batch(self, questions):
        # Match prefix/target lengths and output heads without padding or
        # truncation. Results retain input order and each question's loss weight.
        omit = self._forced is not None and self._forced.get("omit_observations", False)
        answers, groups = [], {}
        results = [None] * len(questions)
        for index, question in enumerate(questions):
            observations = [] if omit else [self.model.encode(name, *values) for name, values in question.inputs]
            if question.answer is not None:
                answers.append((index, (question.prompt, question.answer, observations)))
                continue
            prefix = question_prefix(self.reader, question.prompt, observations)
            sequence = [prefix]
            for prediction in question.predictions:
                coordinates = prediction.coordinates.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                sequence.append(self.model.encode("answer_query", coordinates))
            layout = tuple((prediction.head, len(prediction.coordinates)) for prediction in question.predictions)
            groups.setdefault((prefix.shape[1], layout), []).append((index, question, torch.cat(sequence, dim=1)))
        if answers:
            scores = answer_scores(self, [case for _, case in answers],
                                   generate=self._forced is not None and self._forced.get("generate", True))
            for (index, _), score in zip(answers, scores):
                results[index] = score
        for (prefix_length, layout), rows in groups.items():
            hidden = self.model(torch.cat([row[2] for row in rows], dim=0))
            offset, outputs = prefix_length, []
            for head, count in layout:
                outputs.append(self.model.decode(head, hidden[:, offset:offset + count]))
                offset += count
            for row_index, (index, question, _) in enumerate(rows):
                losses, metrics = [], {}
                for prediction, output in zip(question.predictions, outputs):
                    output = output[row_index]
                    target = prediction.target.to(device=self.device)
                    if prediction.kind == "class":
                        loss = F.cross_entropy(output, target)
                        chosen = output.detach().argmax(-1)
                        metrics[f"{prediction.head}_accuracy"] = (chosen == target).float().mean()
                        if prediction.head == "board_pieces" and bool((target != 0).any()):
                            metrics["occupied_square_accuracy"] = (chosen[target != 0] == target[target != 0]).float().mean()
                        metrics[f"{prediction.head}_exact"] = (chosen == target).all().float()
                        metrics[f"{prediction.head}_copy_accuracy"] = (prediction.baseline == target).float().mean()
                    else:
                        output = output.sigmoid() if prediction.kind == "image" else output.tanh() if prediction.kind == "audio" else output
                        valid = prediction.valid if prediction.valid is not None else torch.ones_like(target, dtype=torch.bool)
                        loss = F.mse_loss(output[valid], target[valid].to(output))
                        metrics[f"{prediction.head}_mse"] = loss.detach()
                        metrics[f"{prediction.head}_baseline_mse"] = F.mse_loss(prediction.baseline[valid], target[valid]).detach()
                    losses.append(loss)
                response = {"prompt": question.prompt, "prefix_tokens": prefix_length,
                            "target_values": sum(int(prediction.target.numel()) for prediction in question.predictions)}
                results[index] = torch.stack(losses).mean(), metrics, response
        return results

    def loss(self):
        self.last_metrics, self.last_response = {}, None
        seed = self._forced["seed"] if self._forced else stable_seed((self.config["name"], self.step))
        records = self._records(seed)
        if self._forced:
            form, wording = self._forced["form"], self._forced["wording"]
        else:
            form = "original" if self.config["question_mode"] == "fixed" or self.step % 2 == 0 else self.forms[1:][(self.step // 2) % (len(self.forms) - 1)]
            wording = 0
        self.last_form = form
        first = records[0]
        self.last_group = (first[0].world_id if isinstance(self.reader, NativeSource) else
                           (f"game:{first.game_index}" if first.game_index is not None else None) if self.config["kind"] == "shogi_examples" else
                           hashlib.sha256(first["passage"].encode()).hexdigest() if self.config["kind"] == "boolq" else
                           first.get("group") if isinstance(first, dict) else None)
        paired = form in ("identify", "same", "different", "sum", "greater")
        batches = [records[index:index + 2] for index in range(0, len(records), 2)] if paired else [[record] for record in records]
        batched_original = (form == "original" and not self._forced
                            and self.config["kind"] in {"text", "conversations", "idx", "cifar10", "spoken_digits", "inertial_activity"}
                            and not (self.config["kind"] == "conversations"
                                     and self.config.get("conversation_objective") == "assistant"))
        if batched_original:
            loss = self.reader.record_batch_loss(records)
            scores = [(loss, self.reader.last_metrics.copy(), None)]
        elif form != "original" and not self._forced:
            scores = self._score_batch([self._question(batch, form, seed, wording) for batch in batches])
        else:
            scores = []
            for batch in batches:
                if form == "original":
                    loss = (self.reader.record_loss(batch[0], generate=self._forced is not None and self._forced.get("generate", True))
                            if self.config["kind"] == "boolq" else self.reader.record_loss(batch[0]))
                    self.last_metrics = self.reader.last_metrics.copy()
                    self.last_response = copy.deepcopy(getattr(self.reader, "last_response", None))
                else:
                    loss = self._score(self._question(batch, form, seed, wording))
                scores.append((loss, self.last_metrics.copy(), self.last_response))
        losses, rows, responses = [], [], []
        for loss, metrics, response in scores:
            losses.append(loss)
            rows.append(metrics)
            if response is not None:
                responses.append(response)
                self.answer_tokens += response.get("target_tokens", 0)
        self.last_metrics = _metric_means(rows)
        self.last_response = {"form": form, "responses": responses,
                              "record_indices": [record["index"] for record in records if isinstance(record, dict) and "index" in record]}
        self.counts[form] += len(batches)
        self.step += 1
        self.last_update_info = {"form": form, "records": len(records), "questions": len(batches)}
        return torch.stack(losses).mean()
