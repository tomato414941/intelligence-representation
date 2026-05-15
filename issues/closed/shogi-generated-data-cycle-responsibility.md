# Shogi Generated Data Cycle Responsibility

Status: closed.

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

Met.

## Resolution

The generated-data lifecycle is now split into shogi-local modules:

- `generated_game_production.py`: arena-agent generated-game invocation
- `generated_data_artifacts.py`: generated-data result artifacts and checkpoint
  promotion
- `generated_data_cycle.py`: fixed generated-data training cycle orchestration
- `online_replay.py`: Online Replay orchestration, replay sampling, Experience
  Store append, and Online Replay metrics

`generated_data_cycle.py` still re-exports the existing public names so script
entrypoints do not need to change.

## Non-Goals

- redesign Replay Buffer
- redesign Experience Store
- move Online Replay into a shared framework
- change arena-agent boundaries
