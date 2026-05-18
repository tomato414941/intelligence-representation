# Shogi Generated Experience Retention Policy

Status: closed
Priority: medium

## Problem

Online replay and self-play generation currently write generated shogi game
records under `runs/`.

During a run, those records are useful:

- `iteration-*/generated-games.jsonl`
- `iteration-*/generated-train-games.jsonl`
- `experience-store/games.jsonl`

However, `runs/` is disposable by project policy. That means generated records
are not yet durable learning assets, even if a later checkpoint proves useful.

This is especially important for online replay. Generated records may be
valuable future training material, but `runs/` must remain disposable.

## Desired Shape

Generated experience should have a clear lifecycle:

- temporary run artifact while the experiment is running
- archived to a durable location by an explicit command when it should be kept
- selected for training later through Data Selection or a Training Data Bundle
- safe to delete with `runs/` after durable archive, or if it was not archived

The durable form should preserve enough facts to explain later reuse without
encoding a quality judgement:

- source run path
- generation method
- archive creation time
- game and transition counts

Retention does not mean adoption. Training selection remains a later
Data Selection or Training Data Bundle concern.

## Close Condition

- The project defines where archived generated shogi experience belongs.
- Online replay records can be either explicitly archived or safely discarded.
- Training data selections can refer to archived generated experience without
  depending on disposable `runs/` paths.

## Resolution

Generated shogi records are archived under:

```text
data/shogi/records/generated/<record-set-id>/
```

`scripts/archive_shogi_generated_records.py` copies a generated game-record
JSONL into `games.jsonl` and writes a factual `manifest.json`. The command does
not decide whether the records are good training data.
