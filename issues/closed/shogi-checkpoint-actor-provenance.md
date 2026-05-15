# Shogi Checkpoint Actor Provenance

Status: closed.

## Issue

Shogi Experience Store records can include checkpoint actors, but checkpoint
provenance is still fragile once many model generations and search settings are
mixed.

Current summaries track actor pairs and checkpoint actor counts, including
checkpoint path, policy, and simulations. That is enough for small experiments,
but it is not yet a durable way to reason about model generations.

Current game records do not carry a per-game generated-at timestamp. The append
timestamp exists in Experience Store `history.jsonl`, so generated time can only
be reconstructed by joining game index ranges with history entries. For example,
the current store can explain that games `282..481` were appended at
`2026-05-06T12:40:35Z` from
`runs/shogi/model-reached-g200-n1-mcts8-p120/annotated-mpv3-n30-games.jsonl`,
and used
`runs/shogi/model-reached-g50-n1-mcts2-mpv3-earlystop/best-checkpoint.pt` with
MCTS8. That is useful, but fragile because the record itself only has the
checkpoint actor settings and the source path is still under disposable `runs/`.

## Why It Matters

Checkpoint-generated experience is not all equivalent.

Examples:

- checkpoint A with direct policy
- checkpoint A with MCTS2
- checkpoint A with MCTS8
- checkpoint B with MCTS8
- checkpoint B with MCTS16

These produce different position distributions and different policy/value
signals. If the store cannot clearly explain which checkpoint generation and
search settings produced each slice of experience, later Training Data Bundles may
mix weak, stale, or incompatible data without that being obvious.

The risk grows when local `runs/.../checkpoint.pt` paths are deleted, renamed,
or become hard to interpret.

## Scope

- Decide what checkpoint identity should be recorded for generated shogi
  experience.
- Decide whether a short generation name, checkpoint path, git commit, run name,
  model config, policy, and search settings are enough.
- Decide what should appear in Experience Store manifest/history summaries.
- Decide whether per-game generated/append time should remain reconstructable
  from history ranges, or whether record/manifest metadata needs a simpler
  durable identity.
- Decide what Training Data Bundle needs in order to include, exclude, or cap
  checkpoint-generated experience by generation or search settings.

## Non-Goals

- Do not introduce a broad model registry before a concrete need exists.
- Do not require long-lived storage for every local checkpoint as part of this
  issue.
- Do not solve future source-mix controls here; source-mix observability is
  covered by `closed/shogi-training-data-bundle-source-mix.md`.

## Acceptance Criteria

This issue can close when checkpoint-generated shogi experience has a clear,
durable provenance policy, and Experience Store / Training Data Bundle metadata can
explain which checkpoint generation and search settings a training slice came
from.

## Resolution

Checkpoint actors should not use a checkpoint file path as their only identity.
The runtime player may load a checkpoint from a path, but the recorded actor
provenance should also carry a `checkpoint_id` when the caller can provide one.

The shogi arena generation CLI now accepts per-side checkpoint IDs. Generated
records can therefore distinguish:

- checkpoint identity: `checkpoint_id`, `checkpoint_path`
- runtime move selection: `move_selector`, `move_selection_profile`
- MCTS settings: `mcts_simulations_per_move`, `nn_leaf_eval_batch_limit`,
  `mcts_move_time_limit_sec`, `board_backend`

`intelligence-representation` passes a checkpoint ID when it invokes
`shogi-arena-agent` for generated-data cycles and Online Replay. Experience
Store manifest/history and Training Data Bundle manifests now include
structured checkpoint actor summaries in addition to legacy count strings.

This does not introduce a model registry and does not require keeping every
run-local checkpoint forever. The durable rule is that generated experience
must carry enough actor provenance to understand the model generation and search
settings even if a disposable `runs/` path is later removed.
