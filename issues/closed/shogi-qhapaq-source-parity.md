# Shogi Qhapaq Source Parity

Status: closed.

## Issue

Qhapaq should be treated as one shogi data source in the same Experience Store
and Training Data Bundle flow, not as a special training path.

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

This issue is closed because:

- RunPod shogi training now consumes a caller-selected Dataset Definition
  through `DATASET_DEFINITION` instead of defaulting to Qhapaq train/eval JSONL.
- The RunPod script syncs the Dataset Definition and its referenced
  game-record JSONL sources, so Qhapaq is just one possible Training Data Bundle
  source rather than a special training path.
- Existing compute-cost rows that mention Qhapaq are historical cost evidence,
  not current source-selection policy.

Qhapaq-derived records can still be used by creating a Training Data Bundle or Dataset
Definition that points at Qhapaq game records. Do not reintroduce Qhapaq-specific
training defaults unless the run is explicitly a Qhapaq historical comparison.
