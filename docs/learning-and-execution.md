# Learning and Execution

This document defines how datasets and interactions become model inputs,
outputs, and learning objectives. It is not a runtime design, an RL framework,
or a generic raw-data schema.

The core distinction is:

```text
data or interaction record:
  what happened or what the dataset provides

model input:
  what is passed into the model for one step or window

model output:
  what the model produces

objective:
  what is optimized or evaluated
```

## Records

Records should stay close to their source.

Dataset examples can be static:

```text
text corpus item
image with label
image with prompt and answer
audio clip with transcript
```

Interaction records can be sequential:

```text
observation
action or model output
feedback, reward, or tool result
next observation
```

Do not force these into one generic schema. A static dataset example and an
environment interaction are related because both can produce model inputs, not
because they need the same raw fields.

## Model Inputs

Training and inference both construct model inputs from records.

Examples:

```text
language modeling:
  preceding text tokens

image classification:
  image patch embeddings

image-to-text:
  image embeddings plus text prompt tokens

interaction modeling:
  recent observations, actions, outputs, and feedback
```

The shared model boundary is still the input embedding sequence described in
[Model Input Boundaries](model-input-boundaries.md).

## Outputs

Model outputs are task-dependent.

Examples:

```text
next token logits
class logits
candidate scores
generated text
predicted reward or value
action distribution
```

These outputs do not need one shared raw format. They need clear objectives and
evaluators.

## Learning Objectives

Different objectives can be built from similar records.

Examples:

```text
self-supervised:
  predict the next token, masked span, next patch, or future observation

supervised:
  predict a label, answer text, transcript, or chosen candidate

reinforcement learning:
  improve a policy using reward, value, return, or preference feedback
```

Reinforcement learning is different because the target is not just a fixed
label from the record. It optimizes behavior under feedback. That difference
belongs in the objective and execution loop, not in a universal raw schema.

## Recursive Execution

Recursive use means model outputs can become part of later inputs.

```text
external input
  -> model input
  -> model output
  -> environment, tool, or user response
  -> next model input
```

This loop is mainly an execution concern. During training, a recorded or
simulated loop can be sliced into windows and objectives.

## Non-Goals

This document does not introduce:

```text
a universal raw-data envelope
generic cross-task fields
an RL runtime
a required agent loop
a claim that every task is sequence learning
```

The project should add those only when experiments require them.
