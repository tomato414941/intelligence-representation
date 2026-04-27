from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from intrep.typed_events import EventRole, TypedEvent
from intrep.typed_stream import render_typed_stream


@dataclass(frozen=True)
class FuturePredictionCase:
    prefix_events: tuple[TypedEvent, ...]
    positive_event: TypedEvent
    negative_events: tuple[TypedEvent, ...] = field(default_factory=tuple)
    target_role: EventRole = EventRole.CONSEQUENCE
    condition: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return str(self.metadata.get("positive_event_id", self.positive_event.id))

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
    events: Sequence[TypedEvent],
    *,
    target_role: EventRole | str | None = None,
    condition: str | None = None,
) -> list[FuturePredictionCase]:
    ordered_events = _stable_event_order(events)
    negatives_by_role = _negative_events_by_role(ordered_events)
    cases: list[FuturePredictionCase] = []

    role = EventRole(target_role) if target_role is not None else None
    for episode_events in _events_by_episode(ordered_events).values():
        if role in (None, EventRole.CONSEQUENCE):
            cases.extend(_extract_observation_action_consequence_cases(episode_events, negatives_by_role, condition))
        if role in (None, EventRole.TOOL_RESULT):
            cases.extend(_extract_tool_call_result_cases(episode_events, negatives_by_role, condition))
        if role in (None, EventRole.PREDICTION_ERROR):
            cases.extend(
                _extract_adjacent_role_pair_cases(
                    episode_events,
                    negatives_by_role,
                    prefix_role=EventRole.PREDICTION,
                    target_role=EventRole.PREDICTION_ERROR,
                    default_condition="prediction_to_error",
                    condition=condition,
                )
            )

    if role not in (None, EventRole.CONSEQUENCE, EventRole.TOOL_RESULT, EventRole.PREDICTION_ERROR):
        raise ValueError(f"unsupported target_role for future prediction cases: {role.value}")
    return cases


def render_future_prediction_prefix(case: FuturePredictionCase) -> str:
    return render_typed_stream(case.prefix_events)


def render_future_prediction_positive(case: FuturePredictionCase) -> str:
    return render_typed_stream([case.positive_event])


def render_future_prediction_negatives(case: FuturePredictionCase) -> tuple[str, ...]:
    return tuple(render_typed_stream([event]) for event in case.negative_events)


def render_future_prediction_continuations(case: FuturePredictionCase) -> tuple[str, ...]:
    return (render_future_prediction_positive(case), *render_future_prediction_negatives(case))


def _extract_observation_action_consequence_cases(
    episode_events: Sequence[TypedEvent],
    negatives_by_role: Mapping[EventRole, tuple[TypedEvent, ...]],
    condition: str | None,
) -> list[FuturePredictionCase]:
    cases: list[FuturePredictionCase] = []
    for index, event in enumerate(episode_events):
        if event.role != EventRole.CONSEQUENCE:
            continue
        action = _nearest_previous_role(episode_events[:index], EventRole.ACTION)
        observation = _nearest_previous_role(episode_events[:index], EventRole.OBSERVATION)
        if observation is None or action is None:
            continue
        prefix_events = tuple(
            prefix_event
            for prefix_event in episode_events[:index]
            if prefix_event == observation or prefix_event == action
        )
        if len(prefix_events) != 2:
            continue
        negative_events = _negative_events_for(event, negatives_by_role, prefix_events, condition)
        if not negative_events:
            continue
        cases.append(
            FuturePredictionCase(
                prefix_events=prefix_events,
                positive_event=event,
                negative_events=negative_events,
                target_role=EventRole.CONSEQUENCE,
                condition=condition or "observation_action_to_consequence",
                metadata=_case_metadata(event),
            )
        )
    return cases


def _extract_tool_call_result_cases(
    episode_events: Sequence[TypedEvent],
    negatives_by_role: Mapping[EventRole, tuple[TypedEvent, ...]],
    condition: str | None,
) -> list[FuturePredictionCase]:
    cases: list[FuturePredictionCase] = []
    for index, event in enumerate(episode_events):
        if event.role != EventRole.TOOL_CALL:
            continue
        result = _first_following_tool_result(event, episode_events[index + 1 :])
        if result is None:
            continue
        negative_events = _negative_events_for(result, negatives_by_role, (event,), condition)
        if not negative_events:
            continue
        cases.append(
            FuturePredictionCase(
                prefix_events=(event,),
                positive_event=result,
                negative_events=negative_events,
                target_role=EventRole.TOOL_RESULT,
                condition=condition or "tool_call_to_tool_result",
                metadata=_case_metadata(result),
            )
        )
    return cases


def _extract_adjacent_role_pair_cases(
    episode_events: Sequence[TypedEvent],
    negatives_by_role: Mapping[EventRole, tuple[TypedEvent, ...]],
    *,
    prefix_role: EventRole,
    target_role: EventRole,
    default_condition: str,
    condition: str | None,
) -> list[FuturePredictionCase]:
    cases: list[FuturePredictionCase] = []
    for index, event in enumerate(episode_events):
        if event.role != target_role:
            continue
        if index == 0 or episode_events[index - 1].role != prefix_role:
            continue
        prefix_events = (episode_events[index - 1],)
        negative_events = _negative_events_for(event, negatives_by_role, prefix_events, condition)
        if not negative_events:
            continue
        cases.append(
            FuturePredictionCase(
                prefix_events=prefix_events,
                positive_event=event,
                negative_events=negative_events,
                target_role=target_role,
                condition=condition or default_condition,
                metadata=_case_metadata(event),
            )
        )
    return cases


def _first_following_tool_result(call: TypedEvent, following_events: Sequence[TypedEvent]) -> TypedEvent | None:
    call_id = call.metadata.get("tool_call_id")
    for event in following_events:
        if event.role != EventRole.TOOL_RESULT:
            continue
        if call_id is None or event.metadata.get("tool_call_id") == call_id:
            return event
    return None


def _nearest_previous_role(events: Sequence[TypedEvent], role: EventRole) -> TypedEvent | None:
    for event in reversed(events):
        if event.role == role:
            return event
    return None


def _events_by_episode(events: Iterable[TypedEvent]) -> dict[str, list[TypedEvent]]:
    by_episode: dict[str, list[TypedEvent]] = defaultdict(list)
    for event in events:
        if event.episode_id is None or event.time_index is None:
            continue
        by_episode[event.episode_id].append(event)
    return dict(by_episode)


def _negative_events_by_role(events: Sequence[TypedEvent]) -> dict[EventRole, tuple[TypedEvent, ...]]:
    by_role: dict[EventRole, list[TypedEvent]] = defaultdict(list)
    for event in events:
        by_role[event.role].append(event)
    return {role: tuple(role_events) for role, role_events in by_role.items()}


def _negative_events_for(
    positive_event: TypedEvent,
    negatives_by_role: Mapping[EventRole, tuple[TypedEvent, ...]],
    prefix_events: Sequence[TypedEvent],
    condition: str | None,
) -> tuple[TypedEvent, ...]:
    explicit_negative_ids = _metadata_list(positive_event.metadata.get("negative_event_ids"))
    if explicit_negative_ids:
        by_id = {event.id: event for event in negatives_by_role.get(positive_event.role, ())}
        return tuple(
            by_id[event_id]
            for event_id in explicit_negative_ids
            if event_id in by_id and by_id[event_id].content != positive_event.content
        )
    candidates = tuple(
        event
        for event in negatives_by_role.get(positive_event.role, ())
        if event != positive_event
        and event.modality == positive_event.modality
        and event.content != positive_event.content
    )
    if condition == "same_action_different_context":
        action = _nearest_previous_role(prefix_events, EventRole.ACTION)
        if action is not None:
            candidates = tuple(
                event
                for event in candidates
                if _episode_has_event_content(negatives_by_role, event.episode_id, EventRole.ACTION, action.content)
            )
    elif condition == "same_history_different_action":
        history = tuple(event.content for event in prefix_events if event.role == EventRole.OBSERVATION)
        candidates = tuple(
            event
            for event in candidates
            if _episode_observation_history(negatives_by_role, event.episode_id) == history
            and event.episode_id != positive_event.episode_id
        )
    elif condition not in (
        None,
        "same_modality_negative",
        "observation_action_to_consequence",
        "tool_call_to_tool_result",
        "prediction_to_error",
    ):
        raise ValueError(f"unsupported distractor condition: {condition}")
    return candidates


def _episode_has_event_content(
    events_by_role: Mapping[EventRole, tuple[TypedEvent, ...]],
    episode_id: str | None,
    role: EventRole,
    content: str,
) -> bool:
    return any(
        event.episode_id == episode_id and event.role == role and event.content == content
        for event in events_by_role.get(role, ())
    )


def _episode_observation_history(
    events_by_role: Mapping[EventRole, tuple[TypedEvent, ...]],
    episode_id: str | None,
) -> tuple[str, ...]:
    return tuple(
        event.content
        for event in _stable_event_order(list(events_by_role.get(EventRole.OBSERVATION, ())))
        if event.episode_id == episode_id
    )


def _metadata_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split("|") if part.strip())
    if isinstance(value, Sequence):
        return tuple(str(part) for part in value)
    return ()


def _case_metadata(event: TypedEvent) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if event.episode_id is not None:
        metadata["episode_id"] = event.episode_id
    if event.time_index is not None:
        metadata["time_index"] = event.time_index
    if event.id:
        metadata["positive_event_id"] = event.id
    return metadata


def _stable_event_order(events: Sequence[TypedEvent]) -> list[TypedEvent]:
    return sorted(
        events,
        key=lambda event: (
            event.episode_id or "",
            event.time_index if event.time_index is not None else 10**12,
            event.id,
        ),
    )
