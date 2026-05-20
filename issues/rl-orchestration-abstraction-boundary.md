# RL Orchestration Abstraction Boundary

Status: open. Priority: low.

## Issue

Shogi Online Replay is starting to expose RL orchestration roles:

- actor experience generation
- durable generated record publication
- Training Data Bundle construction
- replay-buffer seeding and sampling
- learner updates
- training evaluation
- playing evaluation
- checkpoint publication

These roles are not shogi-only concepts, but the current concrete implementation
is shogi-only.

## Current Policy

Do not introduce a generic RL orchestration framework yet.

Keep the implementation shogi-local while only shogi has this full lifecycle.
Use clear shogi-specific names and boundaries instead of generic abstractions
such as `Actor`, `Learner`, `Evaluator`, or `Publisher` until a second concrete
use case exists.

The current shogi Online Replay implementation lives under
`problems/shogi_policy_value/online_replay.py`. It uses the generic
`intrep.learning.ReplayBuffer` utility, but the orchestration itself remains
shogi-specific.

Generated records and Online Replay Buffer are intentionally independent:
generated records are durable source artifacts, while Replay Buffer is
learner-facing dynamic sampling state. Training Data Bundle remains a fixed
training/evaluation input artifact, not the replay buffer.

## Why It Matters

Extracting a generic abstraction from only shogi would likely bake in shogi
assumptions such as SFEN, USI moves, game records, MCTS, and win/loss targets.

At the same time, leaving the roles unnamed can make Online Replay absorb too
many responsibilities.

The near-term target is clear local boundaries, not shared framework code.

Current local boundaries are:

- `shogi-arena-agent`: game generation runtime and raw game record JSONL
- `intrep.worlds.shogi`: shogi source-side records and Training Data Bundle
  behavior
- `intrep.problems.shogi_policy_value`: policy/value sample construction,
  training, evaluation, and Online Replay orchestration
- `intrep.learning`: small reusable learning-time utilities such as
  `ReplayBuffer`

## Acceptance Criteria

This issue can close when one of the following is true:

- a second RL-style lifecycle exists and a minimal shared orchestration boundary
  is extracted from both concrete implementations
- the project decides RL orchestration should remain world/problem-specific and
  documents why
- shogi-local roles are split clearly enough that no generic abstraction is
  needed

## Non-Goals

- introduce generic RL base classes now
- move shogi Online Replay into a shared framework now
- redesign Training Data Bundle or Replay Buffer

## Related

- [`mixed-source-store-continual-learning.md`](mixed-source-store-continual-learning.md)
- [`problem-learning-algorithm-boundary.md`](problem-learning-algorithm-boundary.md)
- [`shogi-rl-loop-orchestration-boundary.md`](shogi-rl-loop-orchestration-boundary.md)
