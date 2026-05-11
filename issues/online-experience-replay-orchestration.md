# Online Experience Replay Orchestration

Status: open. Priority: medium.

## Issue

The project has a generic `ReplayBuffer`, but it does not yet have an Online
Experience Replay path.

Per `docs/glossary.md`, Online Experience Replay means new experience is added
during learning and older experience is sampled again for model updates. This
is different from Offline Experience Reuse, where source records are selected
before training and then learned from like an ordinary fixed dataset.

The project should decide where this dynamic append/sample loop lives before
exposing replay controls through offline training CLIs.

## Current Position

Do not add `replay_capacity` or replay-buffer controls to fixed shogi
policy-value training as a shortcut.

Use `ReplayBuffer` for online RL only when there is a loop that:

- produces new experience during learning
- appends that experience to a learner-facing buffer
- samples model-update batches from the changing buffer
- records the source of generated experience and the checkpoint/search settings
  that produced it

## Current Shogi RL Cycle

`scripts/run_shogi_generated_data_training_cycle.py` is currently a manual
one-cycle pipeline, not an Online Experience Replay loop.

The current flow is:

```text
input checkpoint
  -> invoke ../shogi-arena-agent/scripts/generate_shogi_games.py
  -> write generated-games.jsonl
  -> split generated records into train-games.jsonl and eval-games.jsonl
  -> write fixed data-selection.json
  -> run intrep.train_shogi_policy_value once
  -> write checkpoint.pt, best-checkpoint.pt, metrics.json
```

This is closer to Offline Experience Reuse: generated game records are fixed
before training starts, converted through data selection, and then trained like
an ordinary dataset.

The current artifact boundary is:

- `shogi-arena-agent` owns game generation runtime and writes raw game record
  JSONL.
- `intelligence-representation` owns splitting, data selection construction,
  model training, metrics, and checkpoint output.
- The boundary is a CLI/subprocess plus artifact boundary, not a Python import
  boundary.

There is currently no append/sample point during model updates. The natural
future insertion point is a shogi RL orchestrator that sits above generation and
training, appends newly generated records or derived examples to a learner-facing
buffer, and supplies sampled update batches to a trainer.

## Design Questions

- Does the first Online Experience Replay loop belong in a shogi-specific RL
  orchestrator or a shared learning module?
- Who appends experience: the RL loop, a producer adapter, or the trainer?
- Does the trainer receive a `ReplayBuffer`, a batch iterator, or already-built
  tensors?
- Where are shogi game records converted into policy-value training examples?
- What state is needed to resume a run: buffer contents, source records, random
  seed, checkpoint identity, actor settings, and search settings?
- Should sampling remain uniform initially, or does the first concrete RL update
  need recency or priority?

## V1 Decision

The first Online Experience Replay implementation should stay narrow:

- buffer item: `ShogiPolicyValueExample`
- append timing: after each generated-data cycle loads newly generated game
  records into policy-value examples
- sampling: uniform sampling through `intrep.learning.ReplayBuffer`
- trainer boundary: the RL orchestrator samples examples and calls the trainer;
  fixed offline training CLIs do not receive a replay buffer
- target construction: `chosen_move` policy target and `winner` value target
  for v1
- checkpoint promotion: use the configured generated-data loop policy

This is intentionally not a target-network, prioritized-replay, ply-streaming,
or distributed self-play design.

## Acceptance Criteria

- Online Experience Replay is not confused with Offline Experience Reuse.
- Offline training CLIs do not expose replay-buffer terminology unless they are
  actually running an online replay loop.
- A concrete RL loop owns the dynamic append/sample lifecycle.
- The artifact boundary with `shogi-arena-agent` remains explicit if shogi
  self-play is the first producer.
- The conversion boundary from generated world records to training examples is
  documented before implementation.

## Related

- [`shogi-rl-loop-orchestration-boundary.md`](shogi-rl-loop-orchestration-boundary.md)
- [`rl-target-network.md`](rl-target-network.md)
- [`closed/replay-buffer-boundary.md`](closed/replay-buffer-boundary.md)
- [`../docs/glossary.md`](../docs/glossary.md)
