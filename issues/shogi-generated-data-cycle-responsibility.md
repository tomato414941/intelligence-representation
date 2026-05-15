# Shogi Generated Data Cycle Responsibility

Status: open.

## Issue

`src/intrep/problems/shogi_policy_value/generated_data_cycle.py` owns too many
responsibilities:

- self-play / generated-game invocation
- generated train/eval split
- Training Data Bundle-style data selection writing
- checkpoint promotion
- generated-data training loops
- Online Replay orchestration
- Experience Store append
- cycle metrics and artifact paths

The file still works, but it is becoming the coordination point for unrelated
changes. That makes future Online Replay, self-play, evaluation, and arena
integration changes harder to reason about.

## Desired Direction

Split the module by responsibility instead of by the current command shape.

Likely boundaries:

- generated game production / sharding
- generated data cycle artifacts and metrics
- fixed generated-data training cycle
- Online Replay orchestration
- checkpoint promotion helper

The goal is not to create a generic RL framework. Keep the implementation
shogi-local unless another concrete use case requires a shared abstraction.

## Acceptance Criteria

This issue can close when `generated_data_cycle.py` no longer owns the full
generated-data and Online Replay lifecycle in one file, and each remaining
module has a clear single reason to change.

## Non-Goals

- redesign Replay Buffer
- redesign Experience Store
- move Online Replay into a shared framework
- change arena-agent boundaries
