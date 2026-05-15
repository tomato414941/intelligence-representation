# Shogi Checkpoint Match Evaluation

Status: closed
Priority: high

## Problem

The project can produce shogi checkpoints through multiple training flows, but
it does not yet have a standard player-vs-player entry point for comparing the
playing strength of two checkpoints.

Training-time eval loss is not a playing-strength measurement. A learning
experiment that claims improvement should be able to cite a checkpoint match.

The 2026-05-15 online replay run recorded fixed eval loss, but did not run a
trained-checkpoint-vs-start-checkpoint match. Fixed eval loss worsened, so the
run is not evidence of strength improvement, but the actual playing-strength
change remains undetermined.

## Desired Shape

Shogi learning experiments that are intended to test improvement should use a
standard checkpoint match evaluation.

The evaluation should be separate from training loss and should record enough
context to compare runs:

- player A
- player B
- search settings
- side assignment policy
- game count
- win/loss/draw result
- wall-clock/runtime context when relevant

The durable evidence is the game-record JSONL. The stdout summary is a derived
convenience view and is not a second source of truth.

## Close Condition

- A standard checkpoint-vs-checkpoint playing evaluation exists for shogi.
- The external CLI uses player-vs-player terms throughout.
- The evaluation can compare any two checkpoint players.
- The game-record JSONL records player identity, side assignment, and actor
  settings needed to recover search settings.
- Learning experiment docs can cite game-record JSONL evidence from this
  evaluator when a match has been run.

## Resolution

Shogi player-vs-player match evaluation now has a standard project-facing
entry point:

- `scripts/run_shogi_player_match.py`
- `shogi-arena-agent/scripts/evaluate_shogi_players.py`

Both use `player_a` / `player_b` terms. The durable evidence is the generated
game-record JSONL; stdout summaries are derived convenience output and are not a
second source of truth.

The 2026-05-15 online replay experiment still needs an actual checkpoint match
before its strength conclusion can change. That is experiment work, not part of
this evaluation-entrypoint issue.
