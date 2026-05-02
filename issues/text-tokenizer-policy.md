# Text Tokenizer Policy

## Issue

Text tokenizer defaults, example values, saved tokenizer workflow, and past
vocabulary-size experiments are easy to confuse. The code default is currently
small, while README examples and archived runs use different vocabulary sizes.

## Current State

| Area | Current value or behavior |
| --- | --- |
| CLI default | `--tokenizer-vocab-size 512` |
| README example | `--tokenizer-vocab-size 1024` |
| archived experiments | include 2048 and 8192 vocab runs |
| preferred workflow | train a tokenizer once and reuse it with `--tokenizer-path` |
| non-text paths | image and shogi routes may bypass the text tokenizer entirely |

## Problem

Vocabulary size is a tokenizer-training hyperparameter, not a shared project
constant. If the README, docs, and code defaults are read as competing
recommendations, it becomes unclear which tokenizer should be treated as the
baseline.

## Candidate Direction

Keep the code default small for smoke runs unless a concrete training run needs
otherwise. For non-smoke text-consuming tasks, prefer an explicit saved tokenizer
and record the actual tokenizer payload with the run output.

When the README or training docs are edited next, clarify:

| Topic | Where it belongs |
| --- | --- |
| shortest CLI behavior | README or CLI help |
| tokenizer/input boundary | `docs/model-boundaries.md` |
| dataset-specific expectations | `docs/datasets.md` |
| actual vocab size used by a run | run metrics or tokenizer artifact |

## Non-Goal

Do not choose a universal vocabulary size before larger text runs create real
pressure for one.
