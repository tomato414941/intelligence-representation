# Shogi Full Training Progress Visibility

Status: closed

## Problem

Full shogi policy/value training can remain silent for a long time before the
first training step is logged.

The current training loop evaluates train/eval samples before entering the step
loop. With the Qhapaq full tensor cache, this means millions of samples may be
read and evaluated before any progress line appears. During that period it is
hard to distinguish normal evaluation work from stalled shard loading, slow
Volume I/O, or a dead GPU worker.

This affects Modal and any other long-running remote environment. It is not a
Modal-specific issue.

## Current Evidence

- `qhapaq-full` tensor cache:
  - train samples: 4,951,012
  - eval samples: 262,133
- `scripts/modal_train_shogi_policy_value.py` full run can stay silent during
  the initial full evaluation phase.
- Short smoke runs are visible once the step loop starts, so the missing signal
  is specifically before or inside long evaluation phases.

## Desired Shape

Training and evaluation should report coarse progress during long phases without
making the training loop noisy or Modal-specific.

At minimum:

- print when initial train/eval evaluation starts and ends
- print periodic batch/sample progress inside long evaluation loops
- include elapsed seconds in those messages
- keep the mechanism inside the shogi training/evaluation code, not in the
  Modal runner

## Close Conditions

- A full-cache run emits progress before the first optimizer step.
- Long evaluation phases emit periodic progress.
- Existing short training tests still pass.

## Resolution

The shogi training core now exposes evaluation progress as structured
`ShogiPolicyValuePhaseProgress` events instead of printing directly. The CLI
renders those events to stdout when `--log-every` is set, preserving the visible
`initial_train_eval` / `initial_eval` start, batch, and done lines while keeping
RunPod/Modal-specific output concerns outside the training core.

The CLI combines `--log-every`, checkpoint cadence, and metrics cadence into the
training progress callback cadence, so step progress and periodic artifacts use
one progress path. `scripts/runpod_train_shogi_policy_value.sh` defaults
`LOG_EVERY=100`, passes `--log-every`, and reports `log_every` in the run
configuration line.

Tests cover both layers:

- training core emits phase progress events without writing to stdout
- CLI stdout includes initial train/eval start, batch progress, done, elapsed
  seconds, and optimizer-step progress
