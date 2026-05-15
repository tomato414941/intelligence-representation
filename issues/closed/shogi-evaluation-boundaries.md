# Shogi Evaluation Boundaries

Status: closed
Priority: medium

## Problem

Shogi evaluation has multiple meanings in this project:

- training-time eval for loss, early stopping, and best-checkpoint selection
- player-vs-player match evaluation for playing strength
- inference-performance evaluation for latency, throughput, and CPU/GPU behavior
- learning experiment summaries that interpret results

These can share the same underlying run artifacts, but they should not share the
same responsibility or become multiple sources of truth for the same conclusion.

## Desired Shape

Document the shogi evaluation boundaries clearly:

- training metrics are the source for training-time eval
- arena player-match summaries are the source for playing-strength evidence
- inference-performance documents summarize latency and throughput only
- learning experiment docs summarize conclusions and cite evidence instead of
  duplicating raw summaries

The boundary should allow one run to produce several kinds of evidence without
turning those interpretations into a single overloaded "evaluation" concept.

## Close Condition

- The shogi docs define these evaluation roles in one place.
- Existing shogi docs use the terms consistently.
- `shogi-checkpoint-match-evaluation.md` remains focused on player-vs-player
  match evaluation, not all evaluation boundaries.

## Resolution

`docs/shogi/learning-boundaries.md` now records the shogi evaluation roles and
their sources of truth. Playing-strength evidence uses game-record JSONL, while
stdout summaries remain derived convenience output.
