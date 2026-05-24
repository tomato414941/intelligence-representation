# LeWorldModel

Source: https://arxiv.org/abs/2603.19312

This document records the project-facing interpretation of LeWorldModel. It is
not a paper summary and not an implementation decision.

## Core Idea

Predict a future observation's latent representation instead of reconstructing
the future observation itself.

For this project, the relevant question is whether a representation can be more
than an intermediate value for a task head. It may also be useful as a
prediction target.

```text
observation_t + context
  -> representation_{t+1}
```

The important part is the representation-space prediction. Pixel
reconstruction, text generation, and game outcome prediction are different
questions.

## Project Interpretation

The relevant boundary to examine is:

```text
source record
  -> observation
  -> representation
  -> predicted representation
```

This is separate from task-specific objectives such as classification, policy,
value, or next-token prediction.

The target representation may come from an encoder. The target should be
explicit, so failure can be attributed to the problem setting, encoder,
predictor, loss, or data.

## Candidate Problem Setting

A small language latent-prediction task is a useful candidate:

```text
previous sentence + next sentence
  -> missing sentence representation
```

The target is not the missing sentence text. The target is an embedding of the
missing sentence.

A simple evaluation is retrieval:

```text
Given the predicted representation, rank the true missing sentence against
negative candidate sentences.
```

This gives a small test of whether the model can predict a meaningful latent
representation without turning the experiment into full language modeling.

## Outside This Note

This note does not cover:

- shogi transition prediction
- full language-model training
- image or video reconstruction
- large-scale world-model training
- memory systems
- control or planning

Those may become relevant later, but they add too many causes of failure for the
small latent-prediction problem setting described here.
