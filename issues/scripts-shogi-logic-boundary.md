# Scripts Shogi Logic Boundary

Status: open.

## Issue

`scripts/` is still small enough to manage, but shogi data/replay/store logic is
starting to live there instead of in importable package code.

Current shogi-heavy scripts include:

- `scripts/append_shogi_experience_store.py`
- `scripts/create_shogi_training_view.py` now delegates its reusable behavior
  to `src/intrep/worlds/shogi/training_view.py`.
- `scripts/create_shogi_replay_view.py` now delegates its reusable behavior to
  `src/intrep/worlds/shogi/replay.py`.

These files do more than command orchestration. They contain reusable behavior:

- Experience Store append logic
- Training View creation
- replay selection
- manifest creation
- actor-pair summaries
- position-stat summaries

That conflicts with the project rule that CLI modules should be thin and core
behavior should live in importable package code.

## Why It Matters

If shogi learning data logic keeps accumulating under `scripts/`, the project
will get harder to test and reuse:

- replay selection cannot be reused outside one CLI
- future PyTorch `Sampler` or online replay work has to copy script logic
- Training View and Experience Store behavior are harder to import from tests or
  other tools
- script count stays superficially small while script responsibility grows

## Direction

Keep environment and job orchestration in `scripts/`.

Move reusable shogi logic into package modules when these scripts are next
edited substantially. Candidate modules:

- `src/intrep/worlds/shogi/experience_store.py`
- `src/intrep/worlds/shogi/training_view.py`
- `src/intrep/worlds/shogi/replay.py`

After that, scripts should mainly parse CLI arguments, call package functions,
and print JSON results.

## Acceptance Criteria

- Shogi Experience Store append behavior is importable from package code.
- Shogi Training View creation behavior is importable from package code. [x]
- Shogi replay selection / replay-view creation behavior is importable from
  package code. [x]
- The corresponding scripts remain as thin wrappers or are removed if no longer
  needed.

## Non-Goals

- remove RunPod/setup scripts
- create a generic multi-domain replay framework
- move all CLI entrypoints at once
