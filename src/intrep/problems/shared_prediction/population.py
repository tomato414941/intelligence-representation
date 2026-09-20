"""Finite traversal of the existing text, indexed and episode readers."""
from __future__ import annotations

from intrep.problems.shared_prediction.conversations import AssistantConversationSource
from intrep.problems.shared_prediction.sources import NativeSource, TextSource


def primary_stream(source):
    reader = getattr(source, "reader", source)
    stream = getattr(reader, "stream", None) or getattr(reader, "sampler", None)
    if stream is None:
        raise TypeError(f"source {reader.config['name']!r} needs a finite population reader")
    return stream


def limit_epochs(source, epochs):
    if epochs is not None and (type(epochs) is not int or epochs < 1):
        raise ValueError("epochs must be a positive integer or None")
    primary_stream(source).epoch_limit = epochs


def completed_epochs(source):
    reader = getattr(source, "reader", source)
    stream = primary_stream(source)
    if hasattr(stream, "offset"):
        complete = stream.offset == stream.end
        if isinstance(reader, AssistantConversationSource):
            complete = complete and reader.position >= len(reader.pending)
        elif isinstance(reader, TextSource):
            complete = complete and len(reader.pending) <= 1
    else:
        complete = stream.cursor == stream.count
        if isinstance(reader, NativeSource):
            complete = complete and reader.episode is not None and reader.transition == len(reader.episode.actions)
    return stream.epochs + int(complete)


def rewind_population(source):
    """Start a complete evaluation pass; callers restore the saved reader state."""
    reader = getattr(source, "reader", source)
    stream = primary_stream(source)
    if hasattr(stream, "offset"):
        stream.load_state_dict({"offset": stream.start, "epochs": 0, "records": 0})
        if isinstance(reader, TextSource):
            reader.pending, reader.tokens = [], 0
        if isinstance(reader, AssistantConversationSource):
            reader.pending_mask, reader.position = [], 0
            reader.record_id = reader.group_id = None
            reader.input_tokens = reader.supervised_tokens = reader.windows = reader.skipped = 0
    else:
        stream.cursor = stream.epochs = stream.samples = 0
        if isinstance(reader, NativeSource):
            reader.active = reader.episode = None
            reader.transition = reader.transitions_seen = 0
    limit_epochs(source, 1)


def population_records(source):
    """Yield every primary record/window/transition, retaining its exact identity."""
    reader = getattr(source, "reader", source)
    next_record = reader.next_transition if isinstance(reader, NativeSource) else reader.next_record
    ordinal = 0
    while True:
        stream = primary_stream(source)
        offset = getattr(stream, "offset", None)
        token_start = getattr(reader, "tokens", None)
        try:
            record = next_record()
        except StopIteration:
            return
        if isinstance(record, dict) and record.get("index") in reader.config.get("evaluation_excluded_indices", []):
            continue
        if isinstance(reader, NativeSource):
            episode, transition = record
            case = {"key": f"{episode.id}:{transition}", "group": episode.world_id}
        elif isinstance(reader, AssistantConversationSource):
            case = {"key": f"{record['index']}:{record['start']}:{record['end']}",
                    "group": record["group"], "window": [record["start"], record["end"]]}
        elif isinstance(record, dict) and "index" in record:
            case = {"key": f"record:{record['index']}", "group": record.get("group", f"record:{record['index']}")}
        elif reader.config["kind"] == "text" or reader.config["kind"] == "conversations":
            case = {"key": f"block:{ordinal}", "group": f"block:{ordinal}",
                    "token_start": token_start, "tokens": record.shape[1] - 1}
        else:
            case = {"key": f"byte:{offset}", "group": f"byte:{offset}"}
        yield case, record
        ordinal += 1
