# README Scope

Status: closed.

## Issue

`README.md` has started to mix project overview, design framing, environment
setup, RunPod setup, tokenizer workflow, training entrypoints, dataset-specific
commands, and checkpoint notes. This makes it harder to decide where to record
small but important operational facts.

## Current Symptoms

| Area | Symptom |
| --- | --- |
| training commands | Detailed command examples were moved to `docs/training.md`; README keeps the entrypoint list and link. |
| RunPod setup | Detailed RunPod commands were removed from README; README now links to `docs/runpod.md`. |
| tokenizer workflow | Reuse workflow now lives in `docs/training.md`. |
| checkpoint reuse | Compatibility notes now live in `docs/training.md`. |

## Current State

README now keeps only a RunPod pointer and no longer repeats RunPod setup,
torchvision/CUDA wheel commands, outdated shared multimodal shell wording, or
image JSONL form lists. Detailed training commands and shared training-time
conventions live in `docs/training.md`.

## Resolution

README remains the project entry point. Detailed training examples moved to
`docs/training.md`, dataset preparation entrypoints moved to `docs/datasets.md`,
and RunPod details remain in `docs/runpod.md`.

## Non-Goal

Do not reorganize the documentation tree just to make the README smaller. Split
only when it reduces confusion in an active edit.
