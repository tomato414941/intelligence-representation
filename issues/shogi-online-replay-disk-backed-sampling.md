# Shogi Online Replay Disk-Backed Sampling

Status: open. Priority: medium.

## Issue

Shogi Online Replay currently loads replay seed examples into process memory,
tensorizes them, and stores them in `ReplayBuffer`.

This is acceptable for small experiments, but it becomes questionable once the
seed budget is large. The current intended scale is:

- replay capacity: 2,097,152 examples
- sampled examples per iteration: 524,288 examples
- Qhapaq train examples: about 4,951,012 examples

At that scale, a Python object based in-memory replay buffer can become a CPU
RAM, startup time, and orchestration bottleneck before the learning itself is
the bottleneck.

## Scope

This issue is about the learner-facing sampling boundary for Shogi Online
Replay.

The target design should avoid requiring the full replay population to be held
as Python objects in memory. It should sample from durable artifacts such as a
Training Data Bundle, tensor cache, or generated-experience cache, then provide
the sampled examples to the normal training path.

## Current Mitigation

The current implementation samples replay seed examples uniformly before
preloading them, so it no longer keeps only the tail of the seed dataset.

That reduces distribution skew, but it does not solve the in-memory replay
population problem.

## Non-Goals

- Do not solve replay-buffer persistence or resume here.
- Do not couple this to Experience Store persistence.
- Do not introduce prioritized replay as part of this issue.
- Do not redesign all RL orchestration abstractions.

## Acceptance Criteria

- Shogi Online Replay can sample a large replay population without loading the
  full population as Python objects in memory.
- Seed replay sampling remains approximately uniform over the selected seed
  data.
- Generated experience can still enter the replay population for later
  iterations.
- The training code receives a clear sampled training set or sampler without
  making `ReplayBuffer` the source of truth for dataset identity.
- Metrics record enough counts to explain loaded, eligible, and sampled replay
  examples.

## Related

- [`online-replay-buffer-persistence.md`](online-replay-buffer-persistence.md)
- [`training-data-bundle-generalization-boundary.md`](training-data-bundle-generalization-boundary.md)
- [`shogi-full-cache-memory.md`](closed/shogi-full-cache-memory.md)
