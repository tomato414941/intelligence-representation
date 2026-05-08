# Research Notes

This document records external sources that may influence future project design
or experiments.

Keep entries short. A note here is not an implementation decision.

Each entry should include:

- source
- topic
- why it matters
- possible project relevance
- status

## Screening Is Enough

- Source: https://arxiv.org/abs/2604.01178
- Topic: attention alternative; absolute query-key relevance; screening.
- Why it matters: softmax attention redistributes weight across available keys
  even when no key is genuinely relevant. Screening instead applies an explicit
  relevance threshold and aggregates only accepted keys.
- Possible project relevance: candidate for future shared-core architecture
  experiments, especially where the model should select relevant source
  elements rather than blend all elements.
- Status: note only. Do not implement until the current data, evaluation, and
  model-management basics are stable.
