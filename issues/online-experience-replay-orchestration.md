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
