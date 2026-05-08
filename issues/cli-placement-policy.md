# CLI Placement Policy

Status: open.

## Problem

The project has several command-line entrypoints for training, data preparation,
inspection, generation, setup, and remote jobs. The count is still manageable,
but top-level `intrep.*` commands can become a catch-all if new source-specific
tools are added there by default.

## Policy

- Keep CLI modules thin; core behavior should live in importable package code.
- Use top-level `intrep.train_*` only for major experiment training entrypoints.
- Put world/source-specific inspection or record operations near the source
  package, for example `intrep.worlds.shogi.*`.
- Put problem-specific dataset/example preparation near the problem package when
  that CLI is actively touched.
- Keep setup, RunPod, Modal, and other environment/job orchestration in
  `scripts/`.
- Do not add new top-level CLI modules unless they are project-level entrypoints.

## Current State

- Shogi policy-value training reads `game_records_jsonl` data-selection sources directly;
  the old shogi policy-value example-preparation CLI has been removed.

## Current Follow-up

- `intrep.prepare_fineweb_edu_text` and `intrep.generate_text` should be
  reconsidered when text CLI ownership is next changed.
