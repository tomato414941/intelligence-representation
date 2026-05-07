# Runs Reference Cleanup

Status: closed.

## Issue

`runs/` is now defined as disposable experiment output, but several local files
still refer to paths under `runs/`.

Observed references include:

- `data/shogi/player-registry.json`
  - checkpoint players pointing to `runs/shogi/...`
- `data/shogi/records/.../manifest.json`
  - source paths pointing to generated files under `runs/shogi/...`
- documentation and issues
  - examples or historical notes using `runs/...`
- tests
  - fixture strings using `runs/...`

Some of these are harmless examples. Others can break local workflows if
`runs/` is deleted.

## Why It Matters

The project wants `runs/` to be safely deletable. That cannot be true while
runtime configuration or data manifests depend on files inside `runs/`.

This is separate from choosing a long-lived artifact store. The immediate issue
is to remove operational dependencies on disposable paths.

## Scope

- Identify current `runs/` references.
- Classify each reference as:
  - harmless documentation/example,
  - test fixture string,
  - runtime configuration dependency,
  - data manifest dependency, or
  - actual required artifact.
- Remove or replace operational dependencies on `runs/`.
- Leave historical docs alone if they are clearly historical and not used as
  current instructions.

## Non-Goals

- Do not delete `runs/` as part of this issue unless explicitly requested.
- Do not build a general artifact registry.
- Do not solve the full player-registry boundary issue here.

## Acceptance Criteria

This issue is closed because current operational dependencies on `runs/` have
been removed:

- `data/shogi/player-registry.json` no longer points checkpoint players at
  `runs/shogi/...`.
- The current player registry validates without requiring `runs/`.
- The old copied record collection manifest no longer depends on a `runs/`
  source path; the copied JSONL files in that directory are the usable records.

Remaining `runs/` references are classified as non-operational:

- README and docs examples use `runs/` as disposable output paths.
- legacy docs and compute-cost records are historical.
- tests use `runs/` as fixture strings.
- Experience Store history and raw game-record actor settings may still mention
  `runs/` as provenance for old generated data.

The last category is intentionally not rewritten here. It belongs to the
narrower checkpoint/source provenance problem tracked by
[`shogi-checkpoint-actor-provenance.md`](shogi-checkpoint-actor-provenance.md).

Deleting `runs/` may make old provenance paths non-resolvable, but it should not
break current intended training, evaluation, or registry validation workflows.
