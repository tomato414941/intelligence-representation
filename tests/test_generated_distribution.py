import unittest

from intrep.transition_data import (
    generated_find_examples,
    split_generated_examples,
    split_strict_generated_examples,
    strict_generated_examples,
)


class GeneratedDistributionTest(unittest.TestCase):
    def test_generated_distribution_has_fixed_train_and_slices(self) -> None:
        train, slices = split_generated_examples(generated_find_examples())

        self.assertEqual(len(train), 12)
        self.assertEqual(len(slices["generated_seen"]), 12)
        self.assertEqual(len(slices["generated_held_out_object"]), 12)
        self.assertEqual(len(slices["generated_held_out_container"]), 12)
        self.assertEqual(len(slices["generated_held_out_location"]), 6)

    def test_generated_slices_do_not_overlap_by_id(self) -> None:
        _, slices = split_generated_examples(generated_find_examples())
        seen_ids: set[str] = set()

        for examples in slices.values():
            ids = {example.id for example in examples}
            self.assertTrue(seen_ids.isdisjoint(ids))
            seen_ids.update(ids)

    def test_strict_generated_distribution_has_fixed_train_and_slices(self) -> None:
        train, slices = split_strict_generated_examples(strict_generated_examples())

        self.assertEqual(len(train), 4)
        self.assertEqual(len(slices["generated_strict_held_out_combination"]), 1)
        self.assertEqual(len(slices["generated_strict_action_sequence"]), 1)
        self.assertEqual(len(slices["generated_strict_partial"]), 1)
        self.assertEqual(len(slices["generated_strict_noisy"]), 1)
        self.assertEqual(len(slices["generated_strict_same_entity_negative"]), 1)

    def test_strict_generated_slices_are_deterministic_and_disjoint(self) -> None:
        first_train, first_slices = split_strict_generated_examples(strict_generated_examples())
        second_train, second_slices = split_strict_generated_examples(strict_generated_examples())

        self.assertEqual([example.id for example in first_train], [example.id for example in second_train])
        self.assertEqual(
            {name: [example.id for example in examples] for name, examples in first_slices.items()},
            {name: [example.id for example in examples] for name, examples in second_slices.items()},
        )

        train_ids = {example.id for example in first_train}
        slice_ids = {
            example.id
            for examples in first_slices.values()
            for example in examples
        }
        self.assertTrue(train_ids.isdisjoint(slice_ids))

    def test_strict_generated_partial_and_same_entity_negative_are_negative(self) -> None:
        _, slices = split_strict_generated_examples(strict_generated_examples())
        partial = slices["generated_strict_partial"][0]
        same_entity = slices["generated_strict_same_entity_negative"][0]

        self.assertIsNone(partial.expected_observation)
        self.assertIsNone(same_entity.expected_observation)
        self.assertEqual(same_entity.state_before[0].subject, same_entity.state_before[0].object)


if __name__ == "__main__":
    unittest.main()
