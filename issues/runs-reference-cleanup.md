# Runs Reference Cleanup

Status: open.

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

This issue can close when deleting `runs/` would not break current intended
workflows, excluding historical documentation and explicit test fixture strings.
