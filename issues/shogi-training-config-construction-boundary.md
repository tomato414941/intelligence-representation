# Shogi Training Config Construction Boundary

Status: open. Priority: low.

## Issue

`ShogiPolicyValueTrainingConfig` currently carries several concerns in one
object:

- model assembly identity
- model size parameters
- training hyperparameters
- runtime settings
- evaluation limits

Removing the implicit assembly-spec default correctly forced the training path
to name the model assembly explicitly, but it also exposed noisy config
construction in scripts and tests.

## Current Policy

Keep `assembly_spec_id` required.

Do not reintroduce an implicit default assembly spec. Also do not split the
config into a large hierarchy just for architectural neatness.

## Desired Direction

Clarify where config construction is allowed:

- `train_shogi_policy_value` builds a config for new training from explicit CLI
  arguments.
- online replay inherits model identity from the checkpoint and applies only
  training/runtime overrides.
- tests may use a small helper for repeated small-model config boilerplate, but
  that helper must not become a production default.

## Acceptance Criteria

This issue can close when:

- config construction sites are intentional and easy to audit
- tests do not repeat irrelevant small-model boilerplate everywhere
- online replay continues to inherit model identity from the checkpoint
- no hidden default training assembly is introduced

## Non-Goals

- introduce broad `ModelSpec` / `TrainingSpec` / `RuntimeSpec` abstractions
- create a project-wide config framework
- hide the assembly spec behind another default
