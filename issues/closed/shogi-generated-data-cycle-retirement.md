# Shogi Generated-Data Cycle Retirement

Status: closed
Priority: medium

## Problem

The project now has two shogi training orchestration paths that both generate games and then train a shogi policy-value checkpoint:

- generated-data cycle training
- online experience replay training

They are not identical methods. Generated-data cycle training learns from the data generated for that cycle, while online experience replay appends generated experience to a replay buffer and samples from it.

The question is whether generated-data cycle training is still worth keeping as a separate supported path, or whether online experience replay should become the single maintained generated-experience training path.

## Why This Matters

Keeping both paths increases maintenance cost:

- duplicated CLI entrypoints
- duplicated cycle orchestration concepts
- duplicated checkpoint promotion behavior
- duplicated game generation settings
- risk that training behavior diverges between paths

Removing generated-data cycle training too early also has a cost:

- it removes a simpler no-replay comparison method
- online replay cannot exactly mean "train only on this cycle's generated data" unless configured as an approximation

## Desired Decision

Make an explicit decision:

- keep generated-data cycle training as a distinct experiment method, and align its training config boundary with `ShogiPolicyValueTrainingConfig`
- or retire it and route future generated-experience training through online experience replay

## Investigation

Before changing code, check:

- current references to `run_shogi_generated_data_training_cycle`
- current references to `run_shogi_generated_data_training_loop`
- whether any current docs or scripts still recommend generated-data cycle training
- whether any tests cover behavior that online replay does not cover
- whether a no-replay comparison remains valuable for near-term shogi experiments

## Close Condition

- A decision is recorded.
- If kept, generated-data cycle training no longer duplicates training-owned settings.
- If retired, the generated-data cycle CLI/API/tests/docs are removed or replaced by online replay equivalents.

## Resolution

Generated-data cycle training is retired.

The project keeps the useful independent pieces:

- generated game production
- generated record archive
- Training Data Bundle / Data Selection construction
- fixed-data policy/value training
- Online Replay for generated-experience training

The retired generated-data cycle path was only an orchestration wrapper around:

```text
generate games -> split generated train/eval -> build examples -> run training
```

That path duplicated generation settings and training settings, used generated
data as its own eval split, and overlapped with Online Replay. No compatibility
wrapper remains. Future generated-experience training should use Online Replay;
no-replay fixed-data training should use generated records through Data
Selection or a Training Data Bundle.
