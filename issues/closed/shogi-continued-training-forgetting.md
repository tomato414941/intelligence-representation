# Shogi Continued Training Forgetting

Status: closed.

## Issue

Continuing training from an existing shogi checkpoint can forget earlier useful
data if the next training dataset does not include it.

For example, a checkpoint trained on model-vs-YaneuraOu correction positions may
lose that behavior if it is continued only on YaneuraOu-vs-YaneuraOu games.
This is different from scratch training on a newly defined dataset.

Experience Store and Training Data Bundle now reduce this risk by making training data
explainable through an explicit dataset definition. The issue is not that
continued training is impossible; it is that continued-training runs still need
to make the retained dataset scope and initialization choice explicit.

## Why It Matters

Shogi teacher-policy experiments now use multiple data sources:

- YaneuraOu-vs-YaneuraOu games
- checkpoint-vs-YaneuraOu games with only YaneuraOu moves selected
- future self-play or model-vs-engine correction data

If continued training uses only the newest source, earlier behavior can be
overwritten. Training results should be explainable by the dataset definition,
not only by the checkpoint that was used as initialization.

Current shogi training metrics record `dataset_definition` and
`init_checkpoint_path`, which is useful. They do not yet explicitly classify a
run as scratch or continued training, and manual RL-cycle training still passes
an initial checkpoint.

## Initial Policy

Prefer scratch training from an explicit dataset definition while experiments
are small enough to make this practical.

Use continued training only when the dataset definition deliberately includes
the earlier data that should be retained, or when the experiment is explicitly
testing forgetting.

When continued training is used, treat the dataset definition as the source of
truth for what behavior should be retained. The initial checkpoint should be
recorded as initialization, not as a substitute for dataset scope.

## Acceptance Criteria

This issue can close when shogi training runs have enough metadata to explain
the training data scope and checkpoint initialization choice without adding a
second source of truth.

## Resolution

Closed because existing run metadata is sufficient for the current workflow:

- `dataset_definition` records the training data scope.
- `init_checkpoint_path` records whether the run initialized from a checkpoint.

Do not add a derived scratch/continued field unless result comparison or
automation actually needs it.
