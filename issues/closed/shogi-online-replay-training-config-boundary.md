# Shogi Online Replay Training Config Boundary

Status: closed
Priority: high

## Problem

Shogi online replay owned a partial copy of training settings such as `max_steps`, `batch_size`, and `learning_rate`.

This split the source of truth for training behavior between:

- the normal shogi policy-value training path
- the online replay orchestration path

Because of that split, existing training capabilities could fail to reach online replay. One observed case was early stopping: `train_shogi_policy_value_model` and the normal training CLI support `early_stopping_patience`, but the online replay CLI/config/RunPod wrapper did not expose or pass it.

## Desired Shape

Online replay should own replay orchestration concerns:

- experience sources
- generation settings
- replay buffer settings
- cycle count
- checkpoint promotion between cycles

Training behavior should stay in `ShogiPolicyValueTrainingConfig` or an equivalent training-owned settings object:

- max steps
- batch size
- learning rate
- eval cadence
- early stopping
- progress/log cadence
- checkpoint cadence

Online replay should pass a complete training config into the training function instead of re-declaring only selected training fields.

## Resolution

2026-05-15:

- `ShogiOnlineReplayConfig` now carries `training_config: ShogiPolicyValueTrainingConfig`.
- Online replay no longer duplicates training-owned fields on its own config.
- The online replay CLI still exposes the practical training knobs, but it materializes them into `ShogiPolicyValueTrainingConfig` before calling the orchestrator.
- Online replay training keeps checkpoint architecture from the starting checkpoint and takes training behavior from `training_config`.
- RunPod online replay wrapper forwards `eval_every` and `early_stopping_patience`.
- Tests cover CLI parsing and propagation into `train_shogi_policy_value_model`.

## Close Condition

- Online replay uses a training-owned config object for shogi policy-value training behavior.
- `eval_every` and `early_stopping_patience` can be configured through the online replay CLI and RunPod wrapper.
- Tests cover that online replay passes early stopping and eval cadence into `train_shogi_policy_value_model`.
- Online replay metrics continue to include the effective training config and early stopping result fields.
