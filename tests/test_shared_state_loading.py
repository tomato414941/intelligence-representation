import unittest

import torch
from torch import nn

from intrep.core.shared_state_loading import load_compatible_module_state


class SharedStateLoadingTest(unittest.TestCase):
    def test_loads_only_selected_named_modules(self) -> None:
        target = _SmallModel()
        source = _SmallModel()
        with torch.no_grad():
            source.core.weight.fill_(1.0)
            source.head.weight.fill_(2.0)
            target.core.weight.zero_()
            target.head.weight.zero_()

        loaded = load_compatible_module_state(
            target,
            source.state_dict(),
            module_names=("core",),
        )

        self.assertEqual(loaded, ("core.weight",))
        self.assertTrue(torch.equal(target.core.weight, source.core.weight))
        self.assertTrue(torch.equal(target.head.weight, torch.zeros_like(target.head.weight)))

    def test_skips_shape_incompatible_module_state(self) -> None:
        target = _SmallModel()
        source_state = {
            "core.weight": torch.ones((3, 3)),
            "head.weight": torch.ones_like(target.head.weight),
        }

        loaded = load_compatible_module_state(
            target,
            source_state,
            module_names=("core",),
        )

        self.assertEqual(loaded, ())
        self.assertTrue(torch.equal(target.core.weight, torch.zeros_like(target.core.weight)))


class _SmallModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.core.weight.zero_()
            self.head.weight.zero_()


if __name__ == "__main__":
    unittest.main()
