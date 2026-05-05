# CLI Placement Policy

Status: open

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
- Put task-specific dataset/example preparation near the task package when that
  CLI is actively touched.
- Keep setup, RunPod, Modal, and other environment/job orchestration in
  `scripts/`.
- Do not add new top-level CLI modules unless they are project-level entrypoints.

## Current Follow-up

- `intrep.prepare_shogi_move_choice_examples` is task-specific and can move to
  `intrep.tasks.shogi_move_choice.prepare_examples` when touched next.
- `intrep.prepare_fineweb_edu_text` and `intrep.generate_text` should be
  reconsidered when text CLI ownership is next changed.
