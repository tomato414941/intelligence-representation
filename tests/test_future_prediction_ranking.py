import unittest

from intrep.byte_tokenizer import ByteTokenizer
from intrep.future_prediction_cases import FuturePredictionCase
from intrep.future_prediction_ranking import evaluate_future_prediction_ranking
from intrep.typed_events import TypedEvent


class FuturePredictionRankingTest(unittest.TestCase):
    def test_scores_future_prediction_cases_by_condition(self) -> None:
        case = FuturePredictionCase(
            prefix_events=(
                _event("obs", "observation", "box closed", 0),
                _event("act", "action", "open box", 1),
            ),
            positive_event=_event("cons_pos", "consequence", "see key", 2),
            negative_events=(_event("cons_neg", "consequence", "see coin", 2),),
            condition="same_modality_negative",
        )
        calls: list[tuple[str, str]] = []

        def scorer(model, tokenizer, prefix, continuation):
            calls.append((prefix, continuation))
            return 1.0 if "see key" in continuation else 2.0

        summary = evaluate_future_prediction_ranking(
            [case],
            model=object(),
            tokenizer=ByteTokenizer(),
            score_continuation_loss=scorer,
        )

        self.assertEqual(summary.overall.top1_accuracy, 1.0)
        self.assertEqual(summary.overall.mean_margin, 1.0)
        self.assertEqual(summary.condition_counts, {"same_modality_negative": 1})
        self.assertEqual(len(calls), 2)

    def test_rejects_empty_cases(self) -> None:
        with self.assertRaisesRegex(ValueError, "cases must not be empty"):
            evaluate_future_prediction_ranking([], object(), ByteTokenizer())


def _event(event_id: str, role: str, content: str, time_index: int) -> TypedEvent:
    return TypedEvent(
        id=event_id,
        role=role,
        modality="text",
        content=content,
        episode_id="ep1",
        time_index=time_index,
    )


if __name__ == "__main__":
    unittest.main()
