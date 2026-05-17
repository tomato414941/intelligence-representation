# Shogi Online Replay Iteration Prep Latency

Status: open
Priority: medium

## Problem

Online replay can spend several minutes between generated game completion and
the first optimizer step of the next training phase.

In the 2026-05-17 RunPod run, after iteration 2 and iteration 3 self-play
generation completed, the process stayed on CPU-side preparation before GPU
training resumed.

The run was launched before the `max_seed_examples_per_iteration` cap was added,
so it sampled about 500k Qhapaq seed examples per iteration. That explains much
of the cost, but the latency is still a distinct part of the iteration pipeline
that should be measured.

## Desired Shape

Online replay should expose coarse timing for each iteration phase:

- gate
- generated experience creation
- generated train extraction
- replay/seed sampling
- training dataset materialization
- training
- checkpoint save

This should make CPU preparation time visible without turning the loop into a
large tracing framework.

## Close Condition

- Online replay metrics include phase-level wall times around replay sampling
  and training-data preparation.
- A capped-seed run can show whether prep latency is still material.
- If prep latency remains high, a follow-up issue can target the measured
  bottleneck.
