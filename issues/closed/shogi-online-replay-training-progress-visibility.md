# Shogi Online Replay Training Progress Visibility

Status: closed
Priority: medium

## Problem

Shogi online replay training can run for several minutes with no stdout progress.

During RunPod training this makes it hard to distinguish normal GPU-bound training from a hang without opening a separate SSH session and checking `nvidia-smi`, process state, or output files.

## Desired Shape

Training should emit lightweight progress that is useful for long-running RunPod jobs:

- current cycle
- current training step or completed steps
- recent train loss
- elapsed time

The output should be sparse enough to avoid noisy logs.

## Close Condition

- Online replay training emits periodic progress during the training phase.
- The progress does not require a second SSH session to tell that training is alive.
- Tests cover the progress emission behavior without requiring CUDA.

## Resolution

Online Replay now connects the policy/value training progress callback and
prints sparse cycle-scoped progress lines during training. RunPod Online Replay
jobs set `PROGRESS_EVERY=100` unless overridden.
