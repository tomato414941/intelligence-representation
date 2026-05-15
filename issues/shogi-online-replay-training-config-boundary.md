# Shogi Online Replay Training Config Boundary

Status: open
Priority: high

## Problem

Shogi online replay currently owns a partial copy of training settings such as `max_steps`, `batch_size`, and `learning_rate`.

This splits the source of truth for training behavior between:

- the normal shogi policy-value training path
- the online replay orchestration path

Because of that split, existing training capabilities can fail to reach online replay. One observed case is early stopping: `train_shogi_policy_value_model` and the normal training CLI support `early_stopping_patience`, but the online replay CLI/config/RunPod wrapper did not expose or pass it.

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

## Close Condition

- Online replay uses a training-owned config object for shogi policy-value training behavior.
- `eval_every` and `early_stopping_patience` can be configured through the online replay CLI and RunPod wrapper.
- Tests cover that online replay passes early stopping and eval cadence into `train_shogi_policy_value_model`.
- Online replay metrics continue to include the effective training config and early stopping result fields.

## Progress

2026-05-15:

- Online replay CLI accepts the remaining shogi policy-value training controls:
  `weight_decay`, `max_train_eval_examples`, `max_eval_examples`, `log_every`,
  `pin_memory`, `progress_every`, `eval_every`, and
  `early_stopping_patience`.
- RunPod online replay wrapper forwards those controls.
- Tests cover CLI parsing and propagation into `train_shogi_policy_value_model`.

Remaining:

- Replace the duplicated training fields on `ShogiOnlineReplayConfig` with a
  training-owned config object, or explicitly decide that the current
  flattened wrapper config is the intended boundary.
