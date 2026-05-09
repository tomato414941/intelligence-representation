# RL Target Network

Status: open. Priority: medium.

## Issue

Replay Buffer now exists, but the project has not decided whether RL updates
need a target network.

In value-based or bootstrap-style RL, the model being updated can also be used
to produce its own training targets. If those targets move every optimizer step,
training can become unstable. A target network is a delayed copy used to make
bootstrap targets more stable.

This is different from distillation. A target network is useful only when new
feedback enters the update, such as reward, terminal outcome, search result, or
environment transition. Without feedback, copying an older model's outputs is
mostly distillation and should not be treated as RL improvement.

## Desired Direction

Do not add a target network abstraction before a concrete RL update path needs
one.

When the project introduces bootstrap-style RL updates, decide:

- which model is the online network
- which model is the target network
- what feedback enters the target
- how often the target network is updated
- whether updates are hard copies or soft updates
- how this interacts with Replay Buffer sampling

## Acceptance Criteria

- target network usage is tied to an RL update that includes feedback
- target network is not confused with plain distillation
- update cadence is explicit
- shogi or another concrete world has a test or small run showing the target
  network path works
