# RunPod Shogi Training Wrapper Responsibility

Status: open. Priority: medium.

## Issue

`scripts/runpod_train_shogi_policy_value.sh` is intentionally policy/value
specific. That part is not the problem.

The actual risk is that the RunPod wrapper may be carrying too much of the
training configuration surface. It currently exposes many knobs that belong to
`intrep.train_shogi_policy_value`, including model shape, optimizer settings,
loss weights, eval limits, checkpoint cadence, and early stopping.

That can make the shell wrapper look like a second training interface instead
of a thin remote-execution wrapper around the canonical training CLI.

## Desired Boundary

The RunPod wrapper should own remote execution concerns:

- selecting RunPod runner/template/GPU/runtime limits
- syncing the repository and required input artifacts
- setting up the remote Python environment
- invoking the canonical training CLI
- collecting output artifacts and RunPod timing data

The training CLI should own policy/value training concerns:

- data selection and optional tensor cache
- model family and architecture
- optimizer and loss settings
- eval limits and cadence
- checkpoint and metrics semantics

The wrapper can still pass training arguments. The question is whether those
arguments should be listed as many wrapper-level environment variables, or
whether the wrapper should become thinner by forwarding an explicit training
argument string/config to `intrep.train_shogi_policy_value`.

## Non-Goal

Do not introduce a generic shogi training entrypoint just because the current
script name is policy/value specific. The name is accurate for the current job.

Do not add a generic remote job framework until more than one concrete training
job needs the same abstraction.

## Investigation

Check whether the current wrapper is causing real maintenance or operational
cost:

- Does it duplicate defaults already owned by `intrep.train_shogi_policy_value`?
- Do docs or runs treat wrapper environment variables as the training source of
  truth?
- Have wrapper defaults drifted from docs or the training CLI?
- Are common runs easier to express as a small set of RunPod variables plus an
  explicit training args/config payload?
- Would thinning the wrapper make real full-training commands clearer, or would
  it only hide useful operational defaults?

One known drift candidate is `NUM_WORKERS`: `docs/runpod.md` recommends
`NUM_WORKERS=0` for full-cache shogi runs unless measured, while the wrapper
currently defaults to `NUM_WORKERS=8`.

## Acceptance Criteria

This issue can close when one of the following is true:

- the wrapper is intentionally kept as the supported RunPod policy/value
  training surface, with docs aligned to its defaults, or
- the wrapper is thinned so RunPod-specific settings remain in the wrapper and
  training-specific settings are delegated to the canonical training CLI/config.
