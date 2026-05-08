# Learning Data Flow Boundary

Status: open.

## Issue

The project needs a clear learning data-flow boundary that works for supervised
learning, self-supervised learning, and reinforcement learning.

Ideal responsibility split:

```text
Source / Experience
  -> Selection / Replay
  -> Sample Construction
  -> Batch
  -> Objective / Learner
  -> Model
  -> Output / Actor / Environment
  -> new Source / Experience
```

This should not be reduced to a single `Dataset` concept. Different learning
styles need different middle layers:

- supervised learning can use Data Selection without replay
- self-supervised learning can use Data Selection plus derived targets
- reinforcement learning needs Replay Buffer when experience is repeatedly
  generated, mixed, sampled, and partially forgotten

## Why It Matters

Without this boundary, responsibilities drift:

- source records may become training examples too early
- target construction may be confused with Data Selection
- Replay Buffer may be confused with Experience Store
- PyTorch `Dataset` may absorb split policy, target policy, or learning intent
- fixed datasets and generated experience may be forced into one schema

The project should be able to add RL without making supervised and
self-supervised paths harder to understand.

## Direction

Keep the concepts separate:

- Source / Experience: stores what happened or what was acquired.
- Data Selection: decides what source records or stored targets are included.
- Replay Buffer: decides what stored/generated experience is reused for
  learning.
- Sample Construction: turns selected material into input/target meaning.
- PyTorch `Dataset` / `Sampler` / `DataLoader`: turns samples into tensor
  batches.
- Objective / Learner: decides what loss or learning update to optimize.
- Actor / Environment: generates new experience when learning is interactive.

Do not introduce a generic framework from this issue alone. Use this as the
boundary map for concrete supervised, self-supervised, and RL implementations.

## Related

- [`replay-buffer-boundary.md`](replay-buffer-boundary.md)
  tracks the concrete Replay Buffer side of this boundary.
- [`shogi-target-policy-boundary.md`](shogi-target-policy-boundary.md) tracks
  the current shogi target-policy compromise.

## Acceptance Criteria

This issue can close when current learning paths can be explained by the
boundary above without major contradictions, or when the project replaces this
map with a clearer one.

## Non-Goals

- implement Replay Buffer directly
- introduce a generic multi-domain data-flow framework
- rename every current class immediately
