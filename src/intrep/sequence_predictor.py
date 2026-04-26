from __future__ import annotations

from collections import Counter, defaultdict

from intrep.dataset import ActionConditionedExample
from intrep.sequence import SequenceExample, sequence_from_example, sequences_from_examples
from intrep.tokens import fact_from_token, model_input_tokens
from intrep.types import Action, Fact


class SequenceFeaturePredictor:
    def __init__(self) -> None:
        self._exact_targets: dict[tuple[str, ...], str] = {}
        self._feature_targets: dict[str, Counter[str]] = defaultdict(Counter)

    def fit(self, examples: list[ActionConditionedExample]) -> None:
        sequences = sequences_from_examples(examples)
        self._exact_targets = {
            tuple(sequence.input_tokens): sequence.target_token
            for sequence in sequences
        }
        self._feature_targets = defaultdict(Counter)
        for sequence in sequences:
            for feature in _features(sequence):
                self._feature_targets[feature][sequence.target_token] += 1

    def predict(self, state: list[Fact], action: Action) -> Fact | None:
        sequence = SequenceExample(
            id="prediction",
            input_tokens=model_input_tokens(state, action),
            target_token="",
            source="runtime",
        )
        exact = self._exact_targets.get(tuple(sequence.input_tokens))
        if exact is not None:
            return fact_from_token(exact)

        scores: Counter[str] = Counter()
        for feature in _features(sequence):
            scores.update(self._feature_targets.get(feature, Counter()))

        if not scores:
            return None
        target, _ = scores.most_common(1)[0]
        return fact_from_token(target)


def _features(sequence: SequenceExample) -> set[str]:
    action = next((token for token in sequence.input_tokens if token.startswith("ACTION:")), None)
    fact_tokens = [token for token in sequence.input_tokens if token.startswith("FACT:")]
    features = set(fact_tokens)
    if action is not None:
        features.add(action)
        action_parts = action.split(":")
        if len(action_parts) >= 4:
            features.add(f"ACTION_TYPE:{action_parts[1]}")
            features.add(f"ACTION_OBJECT:{action_parts[2]}")
            for fact in fact_tokens:
                if f":{action_parts[2]}:" in fact or fact.endswith(f":{action_parts[2]}"):
                    features.add(f"OBJECT_FACT:{fact}")
    return features


def sequence_prediction_for_example(
    predictor: SequenceFeaturePredictor,
    example: ActionConditionedExample,
) -> Fact | None:
    sequence_from_example(example)
    return predictor.predict(example.state_before, example.action)
