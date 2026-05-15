# Shogi Generated Eval Responsibility

Status: open
Priority: medium

## Problem

Shogi online replay can split newly generated games into train and eval examples,
but a run may also use a fixed eval selection for training evaluation.

In the 2026-05-15 run, generated eval examples were produced, but training eval
used the fixed eval selection. This is valid, but the responsibility of
generated eval examples is not obvious:

- Are they a held-out slice for that cycle?
- Are they diagnostic only when fixed eval is absent?
- Should they be stored for later evaluation?
- Should they be avoided when a fixed eval set is configured?

Without a clear rule, generated eval counts can look like they affected training
evaluation even when they did not.

## Desired Shape

Online replay should distinguish:

- fixed eval data used to compare checkpoints across cycles/runs
- generated eval data produced from the current cycle's generated games
- generated data that is stored but not used for evaluation

The run metrics and docs should state which eval source was actually used.

## Close Condition

- Online replay metrics clearly record the eval source used for training
  evaluation.
- Generated eval examples are either used with a defined purpose or not produced
  when they are not needed.
- Documentation explains the role of generated eval data when fixed eval data is
  configured.
