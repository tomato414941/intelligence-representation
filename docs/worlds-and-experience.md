# Worlds And Experience

This document defines how action-oriented worlds fit into the project. It is a
boundary document, not a reinforcement-learning algorithm spec.

## World

A world is something that receives actions and returns observations and
consequences.

Examples:

- a small simulation such as grid world
- a web browser session
- a tool environment
- a physical or sensor environment
- an offline dataset replaying recorded interactions

The project follows the Gymnasium-style separation at the concept level:

```text
reset() -> observation
step(action) -> observation, reward, terminated, truncated, info
```

This does not require depending on Gymnasium. The important point is the
boundary: `step()` belongs to the world, not to the model.

## Experience

Experience is recorded interaction with a world.

```text
observation_t
action_t
reward_t
observation_t+1
terminated_t
truncated_t
```

Experience is material. It is not yet a model input format.

## Training Examples

Training examples are cut from experience for a specific objective.

Examples:

```text
next observation prediction:
  input: observation_t, action_t
  target: observation_t+1

reward prediction:
  input: observation_t, action_t
  target: reward_t

action prediction:
  input: observation_t or history
  target: action_t

trajectory continuation:
  input: past observations, actions, and rewards
  target: future observations, actions, or rewards
```

The same experience can support multiple training examples. A field can be a
target in one objective and an input in another.

## Model Input

The model receives tensors, token IDs, or embedding sequences produced from a
training example.

Examples:

```text
text observation -> tokenizer -> token embeddings
image observation -> image input layer -> image embeddings
discrete action -> action embedding
reward or done -> scalar/class target, or a history token when needed
```

Do not force every world or task into one raw schema. Keep the world interface,
experience record, training example, and model input separate.

## First Sim World

Grid world is the first action-oriented simulation because it is small,
deterministic, and easy to render as text, image, tensor, or candidate choices.

The first version uses full observation: the agent, goal, and walls are visible
in the observation. Partial observation can be added later when memory or
history use becomes the thing being tested.

The first useful objective is:

```text
observation + action -> next observation / reward / terminated
```

The first implementation path renders grid-world experience as text
language-modeling examples. This is intentionally a small bridge into the
existing text training path, not a new task-specific trainer.

The first reinforcement-learning-style path uses tensor observations and
discrete action IDs:

```text
grid observation tensor + action id
  -> next agent cell
  -> reward class
  -> terminated flag
```

This keeps the world interface close to common RL practice while still using a
small Transformer core.

Policy learning and online reinforcement learning can be added later if the
prediction tasks show useful pressure.
