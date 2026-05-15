# Shogi Online Replay Step Budget Policy

Status: open
Priority: high

## Problem

Shogi online replay currently exposes replay and optimization knobs without a
clear policy for how they should be chosen together:

- `replay_capacity`
- `min_replay_size`
- `replay_sample_size`
- `batch_size`
- optimizer steps per cycle

The 2026-05-15 run used `replay_sample_size=8192`, `batch_size=512`, and
`1000` optimizer steps per cycle. That means one replay sample pass was 16
steps, and each cycle trained for 62.5 effective passes over the sampled replay
examples.

This may be a reasonable experiment, but it is not currently named or justified
as a training regime. Without a policy, online replay runs can accidentally
become small-sample repeated training rather than controlled online learning.

## Desired Shape

Online replay should make the per-cycle training budget explicit in terms that
are easy to reason about:

- sampled examples per cycle
- batch size
- optimizer steps per cycle
- effective passes over sampled examples
- replay capacity and minimum replay size

The code and docs should make it hard to miss when a run repeatedly trains over
a small sampled set.

## Close Condition

- Online replay metrics record effective passes per cycle.
- The RunPod wrapper and CLI expose or derive the relevant values consistently.
- The shogi online replay documentation explains how replay sample size and
  optimizer steps interact.
- A future online replay run can be interpreted without recomputing the training
  budget by hand.
