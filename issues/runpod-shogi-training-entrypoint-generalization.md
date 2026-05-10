# RunPod Shogi Training Entrypoint Generalization

Status: open. Priority: medium.

## Issue

The current RunPod shogi training entrypoint is policy-value specific:

```text
scripts/runpod_train_shogi_policy_value.sh
```

This is accurate for the current job, but it makes the RunPod documentation and
script naming look narrower than the intended shogi training workflow may
become.

## Current Position

Keep the current script while policy-value training is the only concrete
RunPod shogi training job.

Do not introduce a generic wrapper until there is at least one additional
shogi training entrypoint or a clear shared interface between training jobs.

## Revisit Triggers

Revisit this when one of these becomes true:

- another shogi training job needs RunPod orchestration
- policy-only, value-only, or other training flows share enough setup to justify
  a common entrypoint
- documentation needs to describe shogi training without implying policy-value
  is the only possible training mode
- CLI/config naming starts leaking policy-value assumptions into unrelated
  training jobs

## Acceptance Criteria

This issue can close when either:

- the policy-value-specific script remains intentionally documented as the only
  supported RunPod shogi training entrypoint, or
- a small generic shogi RunPod entrypoint is introduced with policy-value as a
  concrete mode or subcommand.
