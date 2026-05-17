# Shogi Online Replay Gate Statistical Confidence

Status: open
Priority: medium

## Problem

The online replay continuation gate currently accepts or rejects checkpoints
from a small match result.

In the 2026-05-17 RunPod online replay run:

- iteration 2 gate: 16-15-1
- iteration 3 gate: 16-14-2

Both results allowed continuation, but neither result is strong evidence that
the new checkpoint is meaningfully stronger. A barely positive result may be
noise, especially when search sampling and side assignment are involved.

## Desired Shape

The gate should distinguish between:

- clearly worse checkpoint
- statistically unclear checkpoint
- clearly better checkpoint

For KISS, this does not need a full rating system at first. A simple policy is
acceptable if it is explicit, for example:

- continue only above a minimum win margin
- repeat or mark uncertain near 50%
- stop only on a clear loss

The policy should avoid pretending that a one-game margin over 32 games proves
strength improvement.

## Close Condition

- Gate pass/fail semantics are explicit for close results.
- Metrics preserve enough match detail to revisit the decision.
- Online replay can avoid over-interpreting narrow gate wins.
