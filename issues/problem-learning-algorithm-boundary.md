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

Do not introduce a generic Learner or algorithm framework until one concrete
problem needs multiple learning algorithms or one learning algorithm is reused
across multiple problems.

## Desired Direction

World packages should describe what happens or was recorded.

Problem packages should describe the sample shape, target meaning, model output
meaning, and problem-local metrics.

Learning algorithms should own update rules, rollout/update cadence, bootstrap
logic, and algorithm-specific losses when those responsibilities become reusable
or when one problem has multiple training algorithms.

## Acceptance Criteria

- decide when `problems/*/training.py` remains acceptable
- decide when an update loop should move under `learning/`
- avoid creating a broad Learner abstraction ahead of need
- keep RL algorithms from forcing supervised or self-supervised paths into the
  same update abstraction
