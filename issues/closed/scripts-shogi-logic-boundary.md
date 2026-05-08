# Scripts Shogi Logic Boundary

Status: closed.

## Issue

`scripts/` is still small enough to manage, but shogi data/training-view/store logic is
starting to live there instead of in importable package code.

Current shogi-heavy scripts include:

- `scripts/append_shogi_experience_store.py` now delegates its reusable
  behavior to `src/intrep/worlds/shogi/experience_store.py`.
- `scripts/create_shogi_training_view.py` now delegates its reusable behavior
  to `src/intrep/worlds/shogi/training_view.py`.
These files do more than command orchestration. They contain reusable behavior:

- Experience Store append logic
- Training View creation
- manifest creation
- actor-pair summaries
- position-stat summaries

That conflicts with the project rule that CLI modules should be thin and core
behavior should live in importable package code.

## Why It Matters

If shogi learning data logic keeps accumulating under `scripts/`, the project
will get harder to test and reuse:

- Training View creation cannot be reused outside one CLI
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

After that, scripts should mainly parse CLI arguments, call package functions,
and print JSON results.

## Acceptance Criteria

- Shogi Experience Store append behavior is importable from package code. [x]
- Shogi Training View creation behavior is importable from package code. [x]
- The corresponding scripts remain as thin wrappers or are removed if no longer
  needed. [x]

## Non-Goals

- remove RunPod/setup scripts
- create a generic multi-domain replay framework
- move all CLI entrypoints at once

## Resolution

Shogi reusable logic was moved out of scripts:

- Experience Store append behavior lives in `src/intrep/worlds/shogi/experience_store.py`.
- Training View creation behavior lives in `src/intrep/worlds/shogi/training_view.py`.

The corresponding scripts remain as thin CLI wrappers.
