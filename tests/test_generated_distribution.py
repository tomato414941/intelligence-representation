import unittest

from intrep.transition_data import generated_find_examples, split_generated_examples


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


if __name__ == "__main__":
    unittest.main()
