# README Scope

## Issue

`README.md` has started to mix project overview, design framing, environment
setup, RunPod setup, tokenizer workflow, training entrypoints, dataset-specific
commands, and checkpoint notes. This makes it harder to decide where to record
small but important operational facts.

## Current Symptoms

| Area | Symptom |
| --- | --- |
| training commands | The README is acting like a CLI manual. |
| RunPod setup | Environment-specific details live next to the project overview. |
| tokenizer workflow | Defaults, examples, and preferred reuse workflow are close together but have different lifetimes. |
| checkpoint reuse | Compatibility notes are mixed into command examples. |

## Candidate Direction

Keep the README as the project entry point. Move detailed training and
environment instructions into focused docs only when editing those sections
again.

Possible split:

| Destination | Content |
| --- | --- |
| `README.md` | project summary, setup pointer, canonical docs, shortest common commands |
| `docs/training.md` | tokenizer workflow, training entrypoints, checkpoint reuse, task command examples |
| `docs/compute-costs.md` | measured and estimated compute cost, including RunPod-specific run notes |

## Non-Goal

Do not reorganize the documentation tree just to make the README smaller. Split
only when it reduces confusion in an active edit.
