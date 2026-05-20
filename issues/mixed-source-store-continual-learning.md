# Mixed Source Store for Continual Learning

## Problem

The project may need a durable source-side store that can hold records from
multiple problem settings and data origins, then feed mixed training without
silently forgetting older capabilities.

The retired shogi Experience Store should not be generalized directly. It was a
shogi-local append log for generated games. The future store, if built, should
be designed around heterogeneous source records and explicit selection into
training data.

## Why This Matters

Training one model on a new input/output dataset can overwrite behavior learned
from earlier datasets. The simple mitigation is to train from a deliberate mix
of old and new data. That requires a place to keep source records with enough
provenance to rebuild such mixes.

## Boundary

This store would be source-side storage. It should not be:

- a tensor cache
- a replay buffer
- a Training Data Bundle
- an online replay loop
- a model-specific dataset API

Training should still consume an explicit selected dataset or bundle, not an
implicit mutable store.

## Design Questions

- What record types are admitted: static datasets, generated trajectories,
  teacher evaluations, human demonstrations, environment interaction logs?
- How are records tagged by world, problem, input schema, output schema,
  teacher, actor, and generation method?
- Does selection produce schema-homogeneous batches, mixed batches, or separate
  tasks inside a multi-task objective?
- How are retention, deduplication, and sampling weights represented without
  deleting source evidence?
- What is the smallest second use case beyond shogi that justifies shared
  implementation?

## Done

- Current shogi Experience Store has been retired.
- The future store has an explicit boundary separate from replay buffers,
  tensor caches, and Training Data Bundles.
- A concrete second use case exists before implementation starts.
