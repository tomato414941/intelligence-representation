import unittest

from intrep.byte_tokenizer import ByteTokenizer
from intrep.future_prediction_cases import FuturePredictionCase
from intrep.future_prediction_ranking import evaluate_future_prediction_ranking
from intrep.signals import Signal


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

    def test_payload_rendering_scores_only_signal_payloads(self) -> None:
        case = FuturePredictionCase(
            prefix_events=(
                _event("obs", "observation", "box contains key", 0),
                _event("act", "action", "open box", 1),
            ),
            positive_event=_event("cons_pos", "consequence", "see key", 2),
            negative_events=(_event("cons_neg", "consequence", "see coin", 2),),
            condition="same_action_different_context",
        )
        calls: list[tuple[str, str]] = []

        def scorer(model, tokenizer, prefix, continuation):
            calls.append((prefix, continuation))
            return 1.0 if continuation == "see key\n" else 2.0

        summary = evaluate_future_prediction_ranking(
            [case],
            model=object(),
            tokenizer=ByteTokenizer(),
            score_continuation_loss=scorer,
            rendering="payload",
        )

        self.assertEqual(summary.overall.top1_accuracy, 1.0)
        self.assertEqual(
            calls,
            [
                ("box contains key\nopen box\n", "see key\n"),
                ("box contains key\nopen box\n", "see coin\n"),
            ],
        )

    def test_custom_scorer_still_scores_continuations_individually(self) -> None:
        case = FuturePredictionCase(
            prefix_events=(_event("obs", "observation", "box contains key", 0),),
            positive_event=_event("label_pos", "label", "9:Ankle boot", 1),
            negative_events=(
                _event("label_neg_1", "label", "0:T-shirt/top", 1),
                _event("label_neg_2", "label", "1:Trouser", 1),
            ),
            condition="image_to_label",
        )
        calls: list[str] = []

        def scorer(model, tokenizer, prefix, continuation):
            del model, tokenizer, prefix
            calls.append(continuation)
            return 1.0 if continuation == "<SIGNAL channel=\"label\">\n9:Ankle boot\n</SIGNAL>\n" else 2.0

        summary = evaluate_future_prediction_ranking(
            [case],
            model=object(),
            tokenizer=ByteTokenizer(),
            score_continuation_loss=scorer,
        )

        self.assertEqual(summary.overall.top1_accuracy, 1.0)
        self.assertEqual(len(calls), 3)

    def test_rejects_empty_cases(self) -> None:
        with self.assertRaisesRegex(ValueError, "cases must not be empty"):
            evaluate_future_prediction_ranking([], object(), ByteTokenizer())


def _event(event_id: str, channel: str, payload: str, time_index: int) -> Signal:
    del event_id, time_index
    return Signal(
        channel=channel,
        payload=payload,
    )


if __name__ == "__main__":
    unittest.main()
