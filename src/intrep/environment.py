from __future__ import annotations

from intrep.types import Action, Fact


class MiniTransitionEnvironment:
    def __init__(self, initial_locations: dict[str, str] | None = None) -> None:
        self.locations = dict(initial_locations or {})

    def apply(self, action: Action) -> Fact | None:
        if action.type == "place":
            self.locations[action.object] = action.target
            return Fact(subject=action.object, predicate="located_at", object=action.target)

        if action.type == "move_container":
            self.locations[action.object] = action.target
            return Fact(subject=action.object, predicate="located_at", object=action.target)

        if action.type == "find":
            location = self.resolve_location(action.object)
            if location is None:
                return None
            return Fact(subject=action.object, predicate="located_at", object=location)

        return None

    def resolve_location(self, object_name: str) -> str | None:
        current = object_name
        seen = {current}
        destination = self.locations.get(current)

        while destination in self.locations and destination not in seen:
            seen.add(destination)
            current = destination
            destination = self.locations.get(current)

        return destination

    def facts(self) -> list[Fact]:
        return [
            Fact(subject=subject, predicate="located_at", object=location)
            for subject, location in sorted(self.locations.items())
        ]

