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


class NgramObservationMemory:
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
        scored = []
        for observation in self.observations:
            score = self._score(observation, query)
            if score > 0:
                scored.append((score, observation))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [observation for _, observation in scored[:limit]]

    def build_context(self, observations: list[Observation]) -> str:
        return "\n".join(f"[{observation.id}] {observation.content}" for observation in observations)

    def _score(self, observation: Observation, query: str) -> float:
        query_terms = self._word_terms(query)
        content_terms = self._word_terms(observation.content)
        tag_terms = {tag.lower() for tag in observation.tags}

        term_score = len(query_terms & content_terms)
        tag_score = 3 * len(query_terms & tag_terms)

        query_ngrams = self._char_ngrams(query)
        content_ngrams = self._char_ngrams(observation.content)
        if not query_ngrams:
            ngram_score = 0.0
        else:
            ngram_score = len(query_ngrams & content_ngrams) / len(query_ngrams)

        return float(term_score + tag_score) + ngram_score

    def _word_terms(self, text: str) -> set[str]:
        normalized = text.lower()
        for separator in ["、", "。", ",", ".", ":", ";", "　", "\n", "\t"]:
            normalized = normalized.replace(separator, " ")
        return {term for term in normalized.split(" ") if term}

    def _char_ngrams(self, text: str) -> set[str]:
        normalized = "".join(ch for ch in text.lower() if not ch.isspace())
        grams: set[str] = set()
        for n in (2, 3):
            for index in range(0, max(0, len(normalized) - n + 1)):
                grams.add(normalized[index : index + n])
        return grams


def run_demo() -> None:
    memory = NgramObservationMemory()
    memory.add("Transformerは系列モデルというより関係計算器に近い")
    memory.add("State Memoryは真理DBではなくキャッシュとして扱う")
    results = memory.retrieve("関係計算")
    print(memory.build_context(results))


if __name__ == "__main__":
    run_demo()
