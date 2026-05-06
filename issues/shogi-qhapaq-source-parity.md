# Shogi Qhapaq Source Parity

Status: open.

## Issue

Qhapaq should be treated as one shogi data source in the same Experience Store
and Training View flow, not as a special training path.

Current stale paths still treat Qhapaq specially:

- RunPod shogi training defaults point directly at Qhapaq train/eval JSONL.
- compute-cost records are centered on Qhapaq split-cache runs.
- older cache and artifact notes discuss Qhapaq full-cache artifacts as a
  distinct path.

## Why It Matters

Qhapaq remains useful as external game-record data, but it should be comparable
to other sources such as YaneuraOu self-play, model-vs-YaneuraOu games, and
future model self-play.

Keeping Qhapaq as a special path makes it harder to reason about source mix,
dataset definitions, and evaluation. The training path should consume Training
Views, regardless of whether the underlying records came from Qhapaq, an
engine, or self-play.

## Acceptance Criteria

This issue can close when Qhapaq-derived records can be used through the same
Experience Store / Training View path as other shogi records, and Qhapaq-specific
training defaults are either removed or clearly documented as historical.

Do not add a generic data-source framework for this issue. Keep the change
limited to putting Qhapaq on equal footing with the current shogi record sources.
