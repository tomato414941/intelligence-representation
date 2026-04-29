# Signal and Symbol

## Purpose

This note is a concept review for signal, channel, and symbol.
Here, "signal" means the abstract learning object, not the retired `Signal`
class. This is not an implementation plan and does not define a fixed ontology.

The current goal is to clarify what an intelligence handles before deciding
whether any channel-like boundary should exist in code.

Current review status:

```text
signal:
  use as an abstract name for bounded appearances handled by intelligence

channel:
  treat as a practical tokenizer / encoder / evaluator boundary, not ontology

text:
  keep as the reference success case

observation:
  keep under review; close to input-side signal itself

action:
  keep as a strong channel candidate; covers intervention-like signals

tool_call / tool_result:
  remove from core channel candidates for now; likely special cases of action
  and consequence/observation

consequence / prediction_error / state / belief / memory / reward:
  keep under review
```

## Things We Want To Handle

Before choosing channels, the project should ask what kinds of things an
intelligence should handle. `channel` is a later implementation and evaluation
boundary, not the starting point.

Current candidates:

```text
natural language:
  text, dialogue, instruction, explanation, code, symbolic description
  reference success case for token-based prediction

action:
  intervention, command, tool/API call, motor-like control, executable output
  currently represented as text-like payloads or structured calls

perceptual signals:
  image, audio, video, sensor-like data
  not implemented yet, but conceptually important

future / consequence:
  future signals, results, effects, side effects, state changes
  important for prediction and evaluation, but not necessarily a channel

goals / values:
  desired futures, preferences, rewards, constraints, success conditions
  important for action selection, but not yet a fixed project abstraction

history / memory:
  past signals, episodes, interaction traces, reusable context
  important, but may be a storage/reuse mechanism rather than a channel

state / belief:
  inferred situation, latent predictive state, uncertainty, hypotheses
  important, but risky to freeze as hand-designed schema too early

other agents:
  utterances, actions, reactions, social signals, coordination
  likely represented through language, action, and perceptual signals first
```

This list is intentionally broader than the current channel list. Some items
may become channels, some may become tasks or datasets, and some may remain
implicit in learned representations.

## Intelligence And Its Objects

An intelligence does not directly handle the world-as-such.
It handles structured appearances of the world through perception, language,
action, tools, memory, and prediction.

```text
world / object / situation
  -> structured appearance
  -> raw example or signal-like data
  -> tokenization / encoding
  -> model
  -> prediction / generation / action
```

The underlying object is not identical to any single signal.
The same object can appear through text, image, audio, video, action traces,
tool outputs, or memory-like records.

## Signal As Projection

An abstract signal is not the object itself. It is a bounded projection of
something an intelligence can receive, predict,
generate, remember, or act upon.

```text
signal = bounded structured appearance handled by intelligence
```

Examples:

```text
apple as written word -> text-like signal
apple as photograph -> image-like signal
apple as spoken word -> audio-like signal
apple as object to pick up -> action-related signal
```

This means a signal is not input-only.
A signal can be an input, an output, a prediction target, an imagined future,
or a remembered past. Human bodies constrain what humans can output directly,
but models and tool-using systems can output many signal forms through decoders
or external tools.

## Signal And Symbol

A signal is the observable or generatable structured appearance.
A symbol is a sign-like unit that can stand for something.

```text
signal:
  something received, generated, predicted, or encoded
  may be continuous or discrete
  does not require fixed human-defined meaning

symbol:
  something treated as standing for something else
  usually more discrete and interpretable
  has a learned or assigned reference/use
```

Natural language is both signal and symbol.
As model input it is a tokenizable signal.
As human language it also contains symbols that refer, describe, command,
classify, and explain.

Images and audio are usually less symbol-like at the raw input level, but they
can still give rise to symbol-like internal structure after learning.

## Channel Is Not Ontology

`channel` is not a claim about the true categories of the world.
It is a practical boundary chosen for tokenization, encoding, decoding,
data generation, and evaluation.

```text
channel = practical route for a class of signals
```

A channel is useful when it helps with at least one of these:

```text
tokenizer / encoder selection
decoder / renderer selection
sequence boundary management
data collection or generation
evaluation target selection
prediction target selection
```

Therefore a channel may reflect signal structure, such as `text`, `image`,
`audio`, or `video`. It may also temporarily reflect an experimental role, such
as `action` or `tool_result`, if that boundary helps learning or evaluation.

The project should avoid treating the channel list as a handcrafted ontology.

## Text As Reference Case

Natural language should be treated as the reference success case, not as an
exception.

Text works because it can be tokenized into a sequence with strong learnable
structure and abundant data. This gives the baseline pattern:

```text
structured signal
  -> tokenizer / encoder
  -> tokens
  -> predictive model
```

Other signal forms should be judged by whether they can be converted into
tokens or embeddings with learnable structure, not by whether they resemble
natural language externally.

## Channel Versus Format

`channel` is closer to a broad signal family than to a concrete file format.

```text
channel:
  text
  image
  audio
  video

format / encoding:
  utf-8
  markdown
  png
  jpeg
  wav
  mp3
  mp4
```

Concrete encodings and container formats are lower-level loading concerns.
They should not automatically become channels.

## Observation Is Under Review

`observation` is the most problematic current channel candidate because it is
close to signal itself.

If a signal is a bounded appearance handled by intelligence, then an input-side
signal is already close to an observation.

```text
input signal nearly = observation
```

This makes `channel="observation"` conceptually different from channels such as
`text`, `image`, or `audio`.

```text
text:
  signal form / tokenizer family

image:
  signal form / tokenizer family

observation:
  relation between intelligence and a signal
```

There is a separate unresolved question: whether the project needs an explicit
observation concept at all.

This must distinguish two meanings:

```text
observation as phenomenon:
  input received by an intelligence or model
  unavoidable in any predictive system

Observation as project abstraction:
  an explicit class, field, schema, or channel
  optional and should be justified by implementation or evaluation pressure
```

In other words, input is necessary, but an `Observation` class is not
automatically necessary. Natural language models receive input text without
usually modeling that input as an explicit `Observation` object.

One possible distinction is:

```text
signal:
  any bounded structured appearance handled by intelligence

Observation:
  a signal as received by an intelligence
```

Under this distinction, `observation` is not a signal form. It is closer to a
relation or direction:

```text
input / received signal -> observation
output / generated signal -> expression or action
predicted signal -> prediction target
remembered signal -> memory-like reuse
```

This does not mean the implementation should immediately add a `relation`
field. It only means that `channel="observation"` currently mixes two ideas:

```text
channel as signal form / tokenizer route
channel as relation to intelligence
```

An implementation should add `observation` only when a tokenizer, adapter,
dataset, or evaluator actually needs that boundary.

## Action Is A Strong Candidate

`action` is also not a signal form in the same sense as `text`, `image`,
`audio`, or `video`.

But it is different from `observation` because it is harder to replace with a
more concrete signal form.

`observation` may often be replaced by a more concrete signal form:

```text
text observation -> text
visual observation -> image / video
auditory observation -> audio
tool output observation -> text / structured record
```

`action` does not have an equally obvious replacement.
An action is not merely a sensory form. It is an intervention-side signal: a
structured way for intelligence to affect the world, a body, a tool, or another
system.

```text
observation:
  world -> intelligence

action:
  intelligence -> world
```

Action can be represented through text, motor commands, API calls, tool calls,
or other encodings. But those are representations of action, not obvious
replacements for the action concept itself.

This suggests a useful distinction:

```text
action as phenomenon:
  intelligence affecting the world
  necessary for world interaction

action as signal:
  a bounded representation of such intervention
  learnable, predictable, generatable, and evaluable

Action as project abstraction:
  an explicit boundary, class, or schema
  optional, but harder to replace than Observation
```

The irreducible part of `action` is not its representation format.
It is the fact that the system treats the signal as executable or
intervention-like.

```text
text:
  "move north" as a language sequence

action:
  "move north" as something passed to an environment, tool, body, or controller
```

A text payload can describe an action, command an action, or become an action
when the surrounding system treats it as something to execute. For this reason,
`action` should remain in the boundary review as a strong candidate, even though
it is not a sensory signal form like `text`, `image`, `audio`, or `video`.

For now, `action` remains a stronger boundary candidate than `observation`,
because action-conditioned prediction and action generation require a stable
boundary for intervention-like signals.

## Boundary Review Notes

Any channel-like boundary should be reviewed against the things the project
wants to handle. A boundary should exist only when it is useful for
tokenization, encoding, generation, evaluation, or dataset construction.

Older sketches mixed several axes:

```text
text
observation
action
consequence
prediction
prediction_error
state
belief
memory
reward
tool_call
tool_result
```

Possible review grouping:

```text
signal form / tokenizer-family candidates:
  text
  image
  audio
  video

experimental interaction labels:
  action
  consequence

learning / update / internal-state labels:
  prediction
  prediction_error
  state
  belief
  memory
  reward

under-review broad label:
  observation

not core channel candidates for now:
  tool_call
  tool_result
```

This grouping is a review aid only. It is not a proposed implementation schema.

Current tentative decisions:

```text
keep:
  text
  action

under review:
  observation
  consequence
  prediction
  prediction_error
  state
  belief
  memory
  reward

remove from core for now:
  tool_call
  tool_result
```
