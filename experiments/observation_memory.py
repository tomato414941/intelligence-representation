from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Modality = Literal["text", "image", "audio", "video", "sensor", "tool_output", "action_result"]
ObservationType = Literal["observation", "summary", "decision", "question", "artifact"]


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


class ObservationMemory:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self._next_observation_id = 1

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

    def retrieve(self, query: str, *, limit: int = 5) -> list[Observation]:
        query_terms = self._tokenize(query)
        scored = []
        for observation in self.observations:
            score = self._score(observation, query_terms)
            if score > 0:
                scored.append((score, observation))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [observation for _, observation in scored[:limit]]

    def build_context(self, observations: list[Observation]) -> str:
        return "\n".join(f"[{observation.id}] {observation.content}" for observation in observations)

    def store_generated_summary(
        self,
        content: str,
        *,
        links: list[str],
        timestamp: str = "t0",
        tags: list[str] | None = None,
    ) -> Observation:
        return self.add(
            content,
            modality="text",
            source="generated",
            timestamp=timestamp,
            observation_type="summary",
            links=links,
            tags=tags or ["summary"],
        )

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
    memory = ObservationMemory()
    memory.add(
        "Transformerは系列モデルというより関係計算器に近い",
        tags=["transformer", "関係"],
    )
    memory.add(
        "Attentionはトークン間の重み付き関係を作る",
        tags=["attention", "関係"],
    )
    memory.add(
        "State Memoryは真理DBではなくキャッシュとして扱う",
        tags=["state-memory", "cache"],
    )

    results = memory.retrieve("Transformer 関係")
    print(memory.build_context(results))

    summary = memory.store_generated_summary(
        "Transformerはトークン間関係を計算する機構として見られる。",
        links=[observation.id for observation in results],
    )
    print()
    print(f"stored {summary.id} type={summary.type} links={summary.links}")


if __name__ == "__main__":
    run_demo()
