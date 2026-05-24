# Research Notes

This document records external sources that may influence future project design
or experiments.

Keep entries short. A note here is not an implementation decision.
Longer notes for selected sources live next to this file.

Each entry should include:

- source
- topic
- why it matters
- possible project relevance

## Screening Is Enough

- Source: https://arxiv.org/abs/2604.01178
- Topic: attention alternative; absolute query-key relevance; screening.
- Why it matters: softmax attention redistributes weight across available keys
  even when no key is genuinely relevant. Screening instead applies an explicit
  relevance threshold and aggregates only accepted keys.
- Possible project relevance: candidate for future shared-core architecture
  experiments, especially where the model should select relevant source
  elements rather than blend all elements.

## Continuous Latent Diffusion Language Model

- Source: https://arxiv.org/abs/2605.06548
- Topic: continuous latent diffusion language model; hierarchical text
  generation.
- Why it matters: separates global semantic organization from local text
  realization by first mapping text into continuous latent variables, modeling a
  latent prior, and decoding back to text.
- Possible project relevance: supports treating language, perception, and other
  modalities as observations that can be encoded into continuous latent
  representations, then predicted or generated at the representation level
  instead of only at the raw token or pixel level.

## MARBLE

- Source: https://arxiv.org/abs/2605.06507
- Topic: multi-aspect reward balancing for diffusion-model reinforcement
  learning.
- Why it matters: replaces naive weighted-sum reward aggregation with
  per-reward advantage estimates and a gradient-space balancing step, reducing
  conflict between reward dimensions.
- Possible project relevance: useful context for future multi-objective training
  where language, perception, action, memory, preference, or task rewards may
  create competing gradients that should not be collapsed into an arbitrary
  scalar loss too early.

## Skill1

- Source: https://arxiv.org/abs/2605.06130
- Topic: skill-augmented agents; reinforcement learning; skill library
  selection, use, and distillation.
- Why it matters: trains one policy to search for relevant skills, use them
  during task execution, and distill new skills from trajectories under a shared
  task-outcome signal.
- Possible project relevance: useful context for treating generated experience
  as source material for reusable memory, strategy, or skill-like structures
  rather than only as direct checkpoint training examples.

## Sparser, Faster, Lighter Transformer Language Models

- Source: https://arxiv.org/abs/2603.23198
- Topic: sparse Transformer language models; feedforward-layer sparsity;
  efficient kernels.
- Why it matters: argues that unstructured sparsity in Transformer feedforward
  layers can greatly reduce parameters and FLOPs when paired with a sparse
  packing format and GPU kernels that make the sparsity practical.
- Possible project relevance: useful context for future shared-core efficiency
  work, especially if richer representations, longer sequences, or multiple
  domains make dense feedforward layers a major compute bottleneck.

## LeWorldModel

- Source: https://arxiv.org/abs/2603.19312
- Topic: joint-embedding predictive architecture; latent world model from pixels.
- Why it matters: learns action-conditioned dynamics by predicting the next
  observation's latent embedding instead of reconstructing raw pixels or relying
  on task labels.
- Possible project relevance: supports treating predictive representation as
  latent-state prediction, where the target can be another encoded observation
  rather than a human-authored label.
- Project note: [leworldmodel.md](leworldmodel.md)

## Modular Memory Is the Key to Continual Learning Agents

- Source: https://arxiv.org/abs/2603.01761
- Topic: continual learning agents; modular memory; in-weight learning and
  in-context learning.
- Why it matters: continual adaptation should not rely only on updating model
  weights. External memory can accumulate experience quickly, while weight
  updates can absorb stable capabilities more slowly.
- Possible project relevance: supports keeping generated experience as durable
  source material, selecting Training Data Bundles from it, and considering future
  retrieval or memory-based use instead of forcing every experience directly
  into checkpoints.
