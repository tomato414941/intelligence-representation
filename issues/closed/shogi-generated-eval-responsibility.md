# Shogi Generated Eval Responsibility

Status: closed
Priority: medium

## Problem

Shogi online replay could split newly generated games into train and eval examples, but a run could also use a fixed eval selection for training evaluation.

In the 2026-05-15 run, generated eval examples were produced, but training eval used the fixed eval selection. This was valid, but the responsibility of generated eval examples was not obvious:

- Were they a held-out slice for that cycle?
- Were they diagnostic only when fixed eval was absent?
- Should they be stored for later evaluation?
- Should they be avoided when a fixed eval set is configured?

Without a clear rule, generated eval counts could look like they affected training evaluation even when they did not.

## Desired Shape

Online replay should distinguish:

- fixed eval data used to compare checkpoints across cycles/runs
- generated eval data produced from the current cycle's generated games
- generated data that is stored but not used for evaluation

The run metrics and docs should state which eval source was actually used.

## Resolution

2026-05-15:

- Online Replay now requires `training_eval_data_selection`.
- Generated games are treated only as learner experience and are added to replay.
- Online Replay no longer splits generated games into generated train/eval files.
- `eval_ratio` was removed from the Online Replay CLI/config/RunPod wrapper.
- Cycle artifacts keep generated replay input in `generated-games.jsonl`.
- Metrics record `training_eval_source: fixed_data_selection`, `training_eval_examples`, and `generated_holdout_examples: 0`.
- Shogi learning boundaries document that playing-strength evaluation is separate from training-time eval.

## Close Condition

- Online replay metrics clearly record the eval source used for training evaluation.
- Generated eval examples are either used with a defined purpose or not produced when they are not needed.
- Documentation explains the role of generated eval data when fixed eval data is configured.
