from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Modality = Literal["text", "image", "audio", "video", "sensor", "tool_output", "action_result"]
ObservationType = Literal["observation", "summary", "decision", "question", "artifact"]
LogType = Literal["retrieve", "build_context", "generated_summary"]


@dataclass(frozen=True)
class Observation:
    id: str
    content: str
    modality: Modality
    source: str
    timestamp: str
    type: ObservationType = "observation"
    links: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UpdateLogEntry:
    id: str
    type: LogType
    timestamp: str
    query: str | None = None
    input_observations: list[str] = field(default_factory=list)
    output_observation: str | None = None


class LoggedObservationMemory:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.update_log: list[UpdateLogEntry] = []
        self._next_observation_id = 1
        self._next_log_id = 1

    def add(
        self,
        content: str,
        *,
        modality: Modality = "text",
        source: str = "manual",
        timestamp: str = "t0",
        observation_type: ObservationType = "observation",
        links: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> Observation:
        observation = Observation(
            id=f"obs_{self._next_observation_id}",
            content=content,
            modality=modality,
            source=source,
            timestamp=timestamp,
            type=observation_type,
            links=list(links or []),
            tags=list(tags or []),
        )
        self._next_observation_id += 1
        self.observations.append(observation)
        return observation

    def retrieve(self, query: str, *, limit: int = 5, timestamp: str = "t0") -> list[Observation]:
        query_terms = self._tokenize(query)
        scored = []
        for observation in self.observations:
            score = self._score(observation, query_terms)
            if score > 0:
                scored.append((score, observation))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        results = [observation for _, observation in scored[:limit]]
        self._log(
            "retrieve",
            timestamp=timestamp,
            query=query,
            input_observations=[observation.id for observation in results],
        )
        return results

    def build_context(self, observations: list[Observation], *, timestamp: str = "t0") -> str:
        self._log(
            "build_context",
            timestamp=timestamp,
            input_observations=[observation.id for observation in observations],
        )
        return "\n".join(f"[{observation.id}] {observation.content}" for observation in observations)

    def store_generated_summary(
        self,
        content: str,
        *,
        query: str,
        links: list[str],
        timestamp: str = "t0",
        tags: list[str] | None = None,
    ) -> Observation:
        observation = self.add(
            content,
            modality="text",
            source="generated",
            timestamp=timestamp,
            observation_type="summary",
            links=links,
            tags=tags or ["summary"],
        )
        self._log(
            "generated_summary",
            timestamp=timestamp,
            query=query,
            input_observations=links,
            output_observation=observation.id,
        )
        return observation

    def _log(
        self,
        log_type: LogType,
        *,
        timestamp: str,
        query: str | None = None,
        input_observations: list[str] | None = None,
        output_observation: str | None = None,
    ) -> UpdateLogEntry:
        entry = UpdateLogEntry(
            id=f"log_{self._next_log_id}",
            type=log_type,
            timestamp=timestamp,
            query=query,
            input_observations=list(input_observations or []),
            output_observation=output_observation,
        )
        self._next_log_id += 1
        self.update_log.append(entry)
        return entry

    def _score(self, observation: Observation, query_terms: set[str]) -> int:
        content_terms = self._tokenize(observation.content)
        tag_terms = {tag.lower() for tag in observation.tags}
        return len(query_terms & content_terms) + (2 * len(query_terms & tag_terms))

    def _tokenize(self, text: str) -> set[str]:
        normalized = text.lower()
        for separator in ["、", "。", ",", ".", ":", ";", "　", "\n", "\t"]:
            normalized = normalized.replace(separator, " ")
        return {term for term in normalized.split(" ") if term}


def run_demo() -> None:
    memory = LoggedObservationMemory()
    memory.add("Transformerは系列モデルというより関係計算器に近い", tags=["transformer", "関係"])
    memory.add("Attentionはトークン間の重み付き関係を作る", tags=["attention", "関係"])
    results = memory.retrieve("Transformer 関係")
    print(memory.build_context(results))
    memory.store_generated_summary(
        "Transformerはトークン間関係を計算する機構として見られる。",
        query="Transformer 関係",
        links=[observation.id for observation in results],
    )
    print()
    for entry in memory.update_log:
        print(f"- {entry.id} {entry.type} input={entry.input_observations} output={entry.output_observation}")


if __name__ == "__main__":
    run_demo()
