# Replay View Naming Boundary

Status: closed.

## Issue

The current implementation is closer to Offline Experience Reuse than to Online
Experience Replay or a reinforcement-learning Replay Buffer.

General RL usage usually treats a Replay Buffer as a training-time component
for Online Experience Replay. It stores transitions and samples minibatches
during training. It may support capacity limits, recency, random sampling,
prioritized replay, and continuous append / sample behavior.

The current shogi implementation instead:

- reads fixed `ShogiGameRecord` JSONL sources
- selects records once before training
- writes a fixed Training View
- lets the normal PyTorch Dataset / training loop consume that fixed view

That is useful, but calling the implementation a Replay Buffer would be
misleading.

## Why It Matters

If the project uses "Replay Buffer" for fixed view creation, future design
discussions can become confused:

- online replay sampling vs offline view construction become mixed
- PyTorch `Dataset` / `Sampler` responsibilities become unclear
- RL terminology diverges from common usage
- future prioritized or online replay may not have a clear name left

## Current Policy

Keep implementation names close to what exists:

- `offline experience reuse`
- `source selection`
- `training view construction`

Reserve `Replay Buffer` for a real training-time replay component if the project
later needs one.

## Acceptance Criteria

- Current code and docs do not describe the fixed view builder itself as a
  Replay Buffer or Online Experience Replay.
- The boundary issue may still discuss Replay Buffer as a possible future
  component.
- Naming makes clear whether a component samples once before training or samples
  repeatedly during training.

## Non-Goals

- implement an online Replay Buffer
- rename every occurrence of "replay"
- introduce a generic replay abstraction

## Resolution

The fixed Training View builder no longer uses `replay` in its code-level name:

- `src/intrep/worlds/shogi/source_selection.py`
- `scripts/create_shogi_training_view_from_sources.py`
- `create_shogi_training_view_from_sources`
- `select_shogi_game_records`
- `shogi_source_selected_training_view_v1`

`Replay Buffer` remains reserved for a future training-time sampling component.
