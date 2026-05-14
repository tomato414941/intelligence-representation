# Text Tokenizer Artifact Policy

Status: closed.

## Issue

Text tokenizers are learned preprocessing artifacts. The project can keep
multiple tokenizer artifacts with different vocabulary sizes, corpora, or
training settings, but text checkpoints and training runs must preserve which
tokenizer artifact they used.

If this boundary is unclear, a tokenizer vocabulary size can look like a
project-wide default instead of a property of one tokenizer artifact.

## Current State

| Area | Current value or behavior |
| --- | --- |
| CLI default | `--tokenizer-vocab-size 512` |
| training doc example | `--tokenizer-vocab-size 1024` |
| archived experiments | include 2048 and 8192 vocab runs |
| preferred workflow | train a tokenizer once and reuse it with `--tokenizer-path` |
| language-modeling checkpoint | embeds the tokenizer payload in the checkpoint |
| image-text checkpoints | embed the tokenizer payload in the checkpoint |
| non-text paths | image and shogi routes may bypass the text tokenizer entirely |

## Problem

Vocabulary size is a tokenizer artifact parameter, not a shared project
constant. Different tokenizer artifacts can coexist, but a text checkpoint is
only meaningful with the tokenizer it was trained against.

The documentation should make this relationship explicit:

- a tokenizer JSON is an artifact, not only a CLI option
- tokenizer artifacts may have different vocabulary sizes
- text checkpoints must be interpreted with their tokenizer artifact
- run records should preserve the tokenizer artifact or payload that was used

## Policy

Do not choose a universal text vocabulary size for the project.

Use small CLI defaults and examples for smoke-scale runs only. For real
text-consuming runs, prefer an explicit tokenizer artifact via
`--tokenizer-path`, and record the actual tokenizer payload or artifact path with
the run output.

Checkpoint formats that consume text should either embed the tokenizer payload
or record a stable tokenizer artifact reference.

## Documentation Placement

| Topic | Where it belongs |
| --- | --- |
| shortest CLI behavior | README or CLI help |
| tokenizer artifact workflow | `docs/training.md` |
| tokenizer/input boundary | `docs/model-boundaries.md` |
| dataset-specific expectations | `docs/datasets.md` |
| actual vocab size used by a run | run metrics or tokenizer artifact |

## Acceptance Criteria

- `docs/training.md` states that tokenizer vocabulary size belongs to a
  tokenizer artifact, not a project-wide default.
- `docs/training.md` states that text checkpoints and text-consuming runs must
  preserve the tokenizer artifact or payload they used.
- README does not introduce a competing tokenizer vocabulary recommendation.

## Resolution

`docs/training.md` now records tokenizer vocabulary size as a property of the
tokenizer artifact and states that text checkpoints and text-consuming runs must
preserve the tokenizer artifact or payload they used. README keeps only the
training-doc pointer and does not introduce a tokenizer vocabulary
recommendation.

`docs/artifact-layout.md` now defines `tokenizers/<tokenizer-name>/tokenizer.json`
as the long-lived tokenizer location when a tokenizer is reused outside the run
that created it. Tokenizers under `runs/` remain disposable.

## Non-Goal

Do not choose a universal vocabulary size before larger text runs create real
pressure for one.
