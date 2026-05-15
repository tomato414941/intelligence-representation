# Shogi Online Replay Strength Evaluation

Status: open
Priority: high

## Problem

Shogi online replay can produce a trained checkpoint, but the project does not
yet have a standard way to decide whether that checkpoint is stronger than the
initial checkpoint.

The 2026-05-15 online replay run recorded fixed eval loss, but did not run an
initial-vs-final checkpoint match. Fixed eval loss worsened, so the run is not
evidence of strength improvement, but the actual playing-strength change remains
undetermined.

## Desired Shape

Online replay runs that are intended to test improvement should have an explicit
strength-evaluation step.

The evaluation should be separate from training loss and should record enough
context to compare runs:

- initial checkpoint
- final or best checkpoint
- opponent or comparison checkpoint
- search settings
- side assignment policy
- game count
- win/loss/draw result
- wall-clock/runtime context when relevant

## Close Condition

- A standard post-training strength evaluation exists for shogi online replay.
- The evaluation can compare initial checkpoint vs trained checkpoint.
- The evaluation records side assignment and search settings.
- `docs/shogi/online-experience-replay.md` can cite the strength result instead
  of recording the checkpoint strength as undetermined.
