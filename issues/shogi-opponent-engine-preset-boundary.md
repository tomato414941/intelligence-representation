# Shogi Opponent Engine Preset Boundary

## Problem

Shogi playing-strength checks need repeatable opponent definitions such as
YaneuraOu Material nodes1000 and Suisho nodes1.

Today those definitions are spread across RunPod shell scripts, USI options,
engine build flags, NNUE archive setup, and ad hoc run names. The result is that
the same evaluation concept is easy to run inconsistently and hard to read from
the command line or run artifact alone.

## Desired Shape

Opponent engine setup should be represented as a small named preset, separate
from the model entry being evaluated.

A preset should make these facts explicit:

- engine family and build target
- evaluation function type, such as material or NNUE
- NNUE artifact source when needed
- USI options such as `EvalDir`, `Hash`, and `Threads`
- go command, such as `go nodes 1` or `go nodes 1000`

The arena should remain responsible for running player-vs-player matches. The
intrep evaluation workflow should choose a model entry and an opponent preset,
not rebuild the opponent definition from shell fragments each time.

## Non-Goals

- Do not put opponent presets under `models/`.
- Do not turn every one-off USI option into a large registry.
- Do not require this before running the next Suisho check if the existing
  script is sufficient.
