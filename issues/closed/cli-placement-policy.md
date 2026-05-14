# CLI Placement Policy

Status: closed.

Resolution: CLI placement now follows responsibility boundaries. Top-level
`intrep.train_*` modules remain as major training entrypoints; problem-specific,
world/source-specific, text data preparation, and orchestration CLIs live under
their owning packages or `scripts/`.

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

- Top-level `intrep.train_*` modules are major training entrypoints.
- Shogi policy-value training reads `game_records_jsonl` data-selection sources directly;
  the old shogi policy-value example-preparation CLI has been removed.
- Language modeling code lives under `intrep.problems.language_modeling`;
  generic text token and corpus utilities remain under `intrep.text`.
- Hugging Face streaming text slice preparation lives under
  `intrep.text.prepare_hf_text_slice`.
- Shogi policy/value checkpoint evaluation lives under
  `intrep.problems.shogi_policy_value.evaluate`.
- World/source-specific shogi inspection and splitting live under
  `intrep.worlds.shogi`.
- Run orchestration and environment/job entrypoints live under `scripts/`.
