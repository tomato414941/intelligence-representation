# Shogi Online Replay Gate Cost Policy

Status: closed
Priority: medium

## Problem

Online replay now runs a checkpoint-vs-checkpoint gate before using the current
checkpoint to generate the next iteration's self-play data.

That gate is useful because it prevents a worse checkpoint from becoming the
next data generator, but it is also expensive. In the 2026-05-17 RunPod online
replay run, each gate used:

- 32 games
- MCTS128
- NN leaf eval batch limit 64
- 4 match worker processes
- max plies 320

This made gate evaluation a significant part of iteration wall time.

## Desired Shape

The gate should have an explicit cost policy instead of growing implicitly with
the rest of the training setup.

The policy should define:

- when a gate is required
- how many games it should use for online replay continuation
- which MCTS/search settings are acceptable for the gate
- whether the gate should be cheaper than final strength evaluation
- what evidence is enough to stop or continue training

The gate should remain a training-control mechanism, not a substitute for
full-strength evaluation.

## Close Condition

- Online replay gate settings are documented or encoded as an intentional
  training-control choice.
- Gate cost is visible in run metrics.
- The project can explain why the chosen gate cost is worth paying.

## Resolution

2026-05-20:

- `docs/shogi/learning-boundaries.md` now defines the generator gate as a
  training-control cost, not final playing-strength evaluation.
- The gate's purpose is to stop clearly worse candidate generators before they
  produce more replay experience.
- Online Replay iteration metrics now record the gate config:
  - games
  - worker processes
  - MCTS simulations
  - NN leaf eval batch limit
  - max plies
- Metrics already record gate wall time and the gate result, so run summaries can
  judge whether the guard cost was worth paying.
