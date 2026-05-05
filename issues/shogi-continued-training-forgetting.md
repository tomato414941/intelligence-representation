# Shogi Continued Training Forgetting

Status: open.

## Issue

Continuing training from an existing shogi checkpoint can forget earlier useful
data if the next training dataset does not include it.

For example, a checkpoint trained on model-vs-YaneuraOu correction positions may
lose that behavior if it is continued only on YaneuraOu-vs-YaneuraOu games.
This is different from scratch training on a newly defined dataset.

## Why It Matters

Shogi teacher-policy experiments now use multiple data sources:

- YaneuraOu-vs-YaneuraOu games
- checkpoint-vs-YaneuraOu games with only YaneuraOu moves selected
- future self-play or model-vs-engine correction data

If continued training uses only the newest source, earlier behavior can be
overwritten. Training results should be explainable by the dataset definition,
not only by the checkpoint that was used as initialization.

## Initial Policy

Prefer scratch training from an explicit dataset definition while experiments
are small enough to make this practical.

Use continued training only when the dataset definition deliberately includes
the earlier data that should be retained, or when the experiment is explicitly
testing forgetting.

## Acceptance Criteria

This issue can close when shogi training runs record whether they are scratch or
continued training, and continued-training runs make the retained dataset scope
explicit.
