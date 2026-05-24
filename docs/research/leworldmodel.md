# LeWorldModel

Source: https://arxiv.org/abs/2603.19312

This document records the project-facing interpretation of LeWorldModel. It is
not a paper summary and not an implementation decision.

## Adopted Idea

Predict a future observation's latent representation instead of reconstructing
the future observation itself.

In this project, that means a representation can be more than an intermediate
value for a task head. It can also be a prediction target.

```text
observation_t + context
  -> representation_{t+1}
```

The important part is not pixel reconstruction, text generation, or game
outcome prediction. The important part is learning a representation space where
future observations can be predicted.

## Project Interpretation

The relevant boundary is:

```text
source record
  -> observation
  -> representation
  -> predicted representation
```

This should stay separate from task-specific objectives such as classification,
policy, value, or next-token prediction.

The target representation may come from an encoder. The first experiments should
make that target explicit, so failure can be attributed to the problem setting,
encoder, predictor, loss, or data.

## First Problem Setting

Start outside shogi.

The first problem should be small, cheap, and clearly learnable. A good first
candidate is a language latent-prediction task:

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

## Not First

Do not start with:

- shogi transition prediction
- full language-model training
- image or video reconstruction
- large-scale world-model training
- memory systems
- control or planning

Those may become relevant later, but they add too many causes of failure for the
first latent-prediction experiment.
