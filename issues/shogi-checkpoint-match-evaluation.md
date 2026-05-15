# Shogi Checkpoint Match Evaluation

Status: open
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

## Close Condition

- A standard checkpoint-vs-checkpoint playing evaluation exists for shogi.
- The external CLI uses player-vs-player terms throughout.
- The evaluation can compare any two checkpoint players.
- The evaluation records side assignment and search settings.
- `docs/shogi/learning-experiments.md` can cite the strength result instead
  of recording the checkpoint strength as undetermined.
