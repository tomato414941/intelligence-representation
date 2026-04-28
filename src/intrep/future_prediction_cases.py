from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from intrep.signals import PayloadRef, Signal, render_payload_text
from intrep.signal_stream import render_signal_stream

FuturePredictionRendering = Literal["signal", "payload", "image-tokens"]


@dataclass(frozen=True)
class FuturePredictionCase:
    prefix_events: tuple[Signal, ...]
    positive_event: Signal
    negative_events: tuple[Signal, ...] = field(default_factory=tuple)
    target_channel: str = "consequence"
    condition: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return str(self.metadata.get("positive_payload", self.positive_event.payload))

    @property
    def prefix(self) -> str:
        return render_future_prediction_prefix(self)

    @property
    def positive(self) -> str:
        return render_future_prediction_positive(self)

    @property
    def negatives(self) -> tuple[str, ...]:
        return render_future_prediction_negatives(self)


def extract_future_prediction_cases(
    events: Sequence[Signal],
    *,
    target_channel: str | None = None,
    condition: str | None = None,
) -> list[FuturePredictionCase]:
    ordered_events = list(events)
    negatives_by_channel = _negative_events_by_channel(ordered_events)
    cases: list[FuturePredictionCase] = []

    channel = target_channel
    if channel in (None, "consequence"):
        cases.extend(_extract_observation_action_consequence_cases(ordered_events, negatives_by_channel, condition))
    if channel in (None, "tool_result"):
        cases.extend(_extract_tool_call_result_cases(ordered_events, negatives_by_channel, condition))
    if channel in (None, "prediction_error"):
        cases.extend(
            _extract_adjacent_channel_pair_cases(
                ordered_events,
                negatives_by_channel,
                prefix_channel="prediction",
                target_channel="prediction_error",
                default_condition="prediction_to_error",
                condition=condition,
            )
        )
    if channel in (None, "label"):
        cases.extend(
            _extract_adjacent_channel_pair_cases(
                ordered_events,
                negatives_by_channel,
                prefix_channel="image",
                target_channel="label",
                default_condition="image_to_label",
                condition=condition,
            )
        )

    if channel not in (None, "consequence", "tool_result", "prediction_error", "label"):
        raise ValueError(f"unsupported target_channel for future prediction cases: {channel}")
    return cases


def render_future_prediction_prefix(case: FuturePredictionCase) -> str:
    return render_signal_stream(case.prefix_events)


def render_future_prediction_positive(case: FuturePredictionCase) -> str:
    return render_signal_stream([case.positive_event])


def render_future_prediction_negatives(case: FuturePredictionCase) -> tuple[str, ...]:
    return tuple(render_signal_stream([event]) for event in case.negative_events)


def render_future_prediction_continuations(case: FuturePredictionCase) -> tuple[str, ...]:
    return (render_future_prediction_positive(case), *render_future_prediction_negatives(case))


def render_future_prediction_texts(
    case: FuturePredictionCase,
    *,
    rendering: FuturePredictionRendering = "signal",
) -> tuple[str, str, tuple[str, ...]]:
    if rendering == "signal":
        return (
            render_future_prediction_prefix(case),
            render_future_prediction_positive(case),
            render_future_prediction_negatives(case),
        )
    if rendering == "payload":
        return (
            _render_event_payloads(case.prefix_events),
            _render_event_payloads((case.positive_event,)),
            tuple(_render_event_payloads((event,)) for event in case.negative_events),
        )
    if rendering == "image-tokens":
        return (
            _render_event_image_token_payloads(case.prefix_events),
            _render_event_image_token_payloads((case.positive_event,)),
            tuple(_render_event_image_token_payloads((event,)) for event in case.negative_events),
        )
    raise ValueError(f"unsupported future prediction rendering: {rendering}")


def _render_event_payloads(events: Sequence[Signal]) -> str:
    return "\n".join(render_payload_text(event) for event in events) + "\n"


def _render_event_image_token_payloads(events: Sequence[Signal]) -> str:
    return "\n".join(_render_event_image_token_payload(event) for event in events) + "\n"


def _render_event_image_token_payload(event: Signal) -> str:
    if event.channel != "image" or not isinstance(event.payload, PayloadRef):
        return render_payload_text(event)

    from intrep.image_tokenizer import ImagePatchTokenizer

    tokenizer = ImagePatchTokenizer(patch_size=1, channel_bins=4)
    token_ids = tokenizer.encode_ref(event.payload)
    return " ".join(str(token_id) for token_id in token_ids)


def _extract_observation_action_consequence_cases(
    episode_events: Sequence[Signal],
    negatives_by_channel: Mapping[str, tuple[Signal, ...]],
    condition: str | None,
) -> list[FuturePredictionCase]:
    cases: list[FuturePredictionCase] = []
    for index in range(len(episode_events) - 2):
        observation, action, event = episode_events[index : index + 3]
        if (
            observation.channel != "observation"
            or action.channel != "action"
            or event.channel != "consequence"
        ):
            continue
        prefix_events = (observation, action)
        negative_events = _negative_events_for(event, negatives_by_channel, prefix_events, condition)
        if not negative_events:
            continue
        cases.append(
            FuturePredictionCase(
                prefix_events=prefix_events,
                positive_event=event,
                negative_events=negative_events,
                target_channel="consequence",
                condition=condition or "observation_action_to_consequence",
                metadata=_case_metadata(event),
            )
        )
    return cases


def _extract_tool_call_result_cases(
    episode_events: Sequence[Signal],
    negatives_by_channel: Mapping[str, tuple[Signal, ...]],
    condition: str | None,
) -> list[FuturePredictionCase]:
    cases: list[FuturePredictionCase] = []
    for index in range(len(episode_events) - 1):
        event, result = episode_events[index : index + 2]
        if event.channel != "tool_call" or result.channel != "tool_result":
            continue
        negative_events = _negative_events_for(result, negatives_by_channel, (event,), condition)
        if not negative_events:
            continue
        cases.append(
            FuturePredictionCase(
                prefix_events=(event,),
                positive_event=result,
                negative_events=negative_events,
                target_channel="tool_result",
                condition=condition or "tool_call_to_tool_result",
                metadata=_case_metadata(result),
            )
        )
    return cases


def _extract_adjacent_channel_pair_cases(
    episode_events: Sequence[Signal],
    negatives_by_channel: Mapping[str, tuple[Signal, ...]],
    *,
    prefix_channel: str,
    target_channel: str,
    default_condition: str,
    condition: str | None,
) -> list[FuturePredictionCase]:
    cases: list[FuturePredictionCase] = []
    for index, event in enumerate(episode_events):
        if event.channel != target_channel:
            continue
        if index == 0 or episode_events[index - 1].channel != prefix_channel:
            continue
        prefix_events = (episode_events[index - 1],)
        negative_events = _negative_events_for(event, negatives_by_channel, prefix_events, condition)
        if not negative_events:
            continue
        cases.append(
            FuturePredictionCase(
                prefix_events=prefix_events,
                positive_event=event,
                negative_events=negative_events,
                target_channel=target_channel,
                condition=condition or default_condition,
                metadata=_case_metadata(event),
            )
        )
    return cases


def _first_following_tool_result(call: Signal, following_events: Sequence[Signal]) -> Signal | None:
    del call
    for event in following_events:
        if event.channel == "tool_result":
            return event
    return None


def _nearest_previous_channel(events: Sequence[Signal], channel: str) -> Signal | None:
    for event in reversed(events):
        if event.channel == channel:
            return event
    return None


def _negative_events_by_channel(events: Sequence[Signal]) -> dict[str, tuple[Signal, ...]]:
    by_channel: dict[str, list[Signal]] = defaultdict(list)
    for event in events:
        by_channel[event.channel].append(event)
    return {channel: tuple(channel_events) for channel, channel_events in by_channel.items()}


def _negative_events_for(
    positive_event: Signal,
    negatives_by_channel: Mapping[str, tuple[Signal, ...]],
    prefix_events: Sequence[Signal],
    condition: str | None,
) -> tuple[Signal, ...]:
    candidates = tuple(
        event
        for event in negatives_by_channel.get(positive_event.channel, ())
        if event != positive_event
        and event.channel == positive_event.channel
        and event.payload != positive_event.payload
    )
    if condition == "same_action_different_context":
        candidates = tuple(event for event in candidates if event != positive_event)
    elif condition == "same_history_different_action":
        candidates = tuple(event for event in candidates if event != positive_event)
    elif condition not in (
        None,
        "same_modality_negative",
        "observation_action_to_consequence",
        "tool_call_to_tool_result",
        "prediction_to_error",
    ):
        raise ValueError(f"unsupported distractor condition: {condition}")
    return candidates


def _case_metadata(event: Signal) -> dict[str, object]:
    return {"positive_payload": event.payload}
