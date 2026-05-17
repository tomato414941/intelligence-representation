# Problem Learning Algorithm Boundary

Status: open. Priority: low.

## Issue

The project currently keeps many training loops under `problems/*/training.py`.
That is practical while each problem has one main training path, but it can blur
two different concepts:

- Problem: what input/output/target relationship is being learned.
- Learning algorithm: how the model is updated from experience or samples.

The distinction matters more as the project adds reinforcement-learning methods.
The same problem shape can be trained with different algorithms, for example:

- supervised policy/value training
- AlphaZero-style search targets and self-play
- PPO-style actor-critic updates
- DQN-style Bellman updates
- MuZero-style unrolled latent dynamics updates

## Current Policy

Keep existing `problems/*/training.py` files as-is for now.

`src/intrep/learning/` exists, but it currently holds learning-time utilities
such as `ReplayBuffer`, not a generic Learner or algorithm framework.

Do not introduce a generic Learner or algorithm framework until one concrete
problem needs multiple learning algorithms or one learning algorithm is reused
across multiple problems.

The shogi generated-data and online-replay cycle is currently kept under
`problems/shogi_policy_value/` as problem-local orchestration. Move it toward
`learning/` only if it becomes a reusable learning algorithm or if shogi policy
value needs multiple clearly distinct update algorithms.

This remains the current state after the shogi RL artifact boundary cleanup.
`docs/learning-data-boundaries.md` documents general learning-data boundaries,
but that does not imply a generic learning algorithm boundary yet.
The only reusable code under `intrep.learning` is still a small uniform
`ReplayBuffer`.

## Desired Direction

World packages should describe what happens or was recorded.

Problem packages should describe the sample shape, target meaning, model output
meaning, and problem-local metrics.

Learning algorithms should own update rules, rollout/update cadence, bootstrap
logic, and algorithm-specific losses when those responsibilities become reusable
or when one problem has multiple training algorithms.

## Current Trigger For Extraction

Move code from `problems/*/training.py` or problem-local orchestration into
`intrep.learning` only when at least one of these is true:

- one concrete problem needs multiple clearly named learning algorithms
- one learning algorithm is reused across multiple problems
- a problem-local loop starts depending on learning concepts that are not part
  of its sample/target/model meaning

Until then, problem-local training remains acceptable. Shared learning code
should stay small and mechanical, such as replay buffers or optimizer helpers,
not broad role objects like `Learner`, `Actor`, `Evaluator`, or `Publisher`.

## Acceptance Criteria

- decide when `problems/*/training.py` remains acceptable
- decide when an update loop should move under `learning/`
- avoid creating a broad Learner abstraction ahead of need
- keep RL algorithms from forcing supervised or self-supervised paths into the
  same update abstraction
