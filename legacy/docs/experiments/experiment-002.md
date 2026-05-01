# Experiment 002: 100x Typed Future Prediction And Rendering Check

## Purpose

This follow-up investigates whether the negative result in
[Experiment 001](experiment-001.md) was mainly a data-size issue, a training
budget issue, or an evaluation/rendering issue.

It keeps the same hard-negative future-prediction task:

```text
same_history_different_action:
  same observation
  different action
  different consequence

same_action_different_context:
  same action
  different observation
  different consequence
```

The main question is narrower than a world-model claim:

```text
If the generated signal environment data is scaled by 100x, does the ranking
metric begin to move, and does the metric actually keep the relevant
observation/action prefix in context?
```

## Data Scale

Experiment 001 used:

```text
train_size = 80
eval_size = 32
```

Experiment 002 generated 100x data:

```text
train_size = 8000
eval_size = 3200
seed = 7
```

Generated files:

```text
runs/exp002_100x/same_history_train.signals.jsonl
runs/exp002_100x/same_history_eval.signals.jsonl
runs/exp002_100x/same_action_train.signals.jsonl
runs/exp002_100x/same_action_eval.signals.jsonl
```

Observed scale:

```text
per condition:
  train_events = 24000
  train_cases = 8000
  eval_events = 9600
  eval_cases = 3200
  explicit_negative_rate = 0.0

two conditions total:
  cases = 22400
  events = 67200
```

The generated hard-negative pair space had to be expanded beyond the original
small object/container fixture set. The expansion uses deterministic synthetic
object/container names after the original fixture combinations are exhausted.

## Evaluation Cost

The 100x data exposed two engineering constraints.

First, the train corpus is large relative to the CPU smoke setup:

```text
same_history_train:
  chars = 13452102
  byte tokens = 13452102
```

Second, continuation scoring was too slow when each continuation byte caused a
separate Transformer forward pass. The scorer was updated to batch windows by
length while preserving the same continuation-loss interface.

PyTorch CPU thread settings also mattered. Reliable local verification used:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m unittest
```

Observed verification:

```text
Ran 294 tests in 43.910s
OK
```

## Rendering Bug

The main finding is that the Experiment 001 ranking setup had a rendering
problem.

The default ranking rendering used full signals:

```text
<SIGNAL channel="...">
payload
</SIGNAL>
```

For hard-negative ranking, this made prefixes and continuations long enough
that the meaningful consequence text was scored after the observation/action
prefix had already fallen out of the `context_length = 64` model window.

Measured example from `same_action_different_context`:

```text
signal rendering:
  prefix_tokens = 844
  positive_tokens = 523
  negative_tokens = 526
  prefix tail in a 64-token window:
    _action_different_context_0000" split="eval">
    open box
    </SIGNAL>

payload rendering:
  prefix_tokens = 27
  positive_tokens = 9
  negative_tokens = 11
  prefix tail in a 64-token window:
    box contains coin
    open box
```

This explains why `same_action_different_context` produced exactly zero margins
under signal rendering: the scorer was not reliably testing whether the
model used the observation context.

## Rendering Modes

Future-prediction ranking now separates the case extraction responsibility from
the renderer responsibility. The current byte-tokenizer renderer consumes text
payloads; structured JSON / action data should be canonicalized to text before
it enters `Signal`. Future renderers may tokenize media references through
channel-specific encoders.

```text
signal:
  full Signal tag rendering, treated as a low-priority experiment

payload:
  prefix = rendered event payloads only
  continuation = rendered target event payload only
  current byte-tokenizer implementation scores text payloads
```

The CLI supports the rendering choice:

```sh
uv run python -m intrep.evaluate_future_prediction \
  --train-path train.signals.jsonl \
  --eval-path eval.signals.jsonl \
  --target-channel consequence \
  --condition same_action_different_context \
  --rendering payload
```

## Probe Results

The first 100x probe used the large train corpus, small held-out eval subsets,
`max_steps = 20`, and `batch_stride = 65536` to keep CPU runtime manageable.

Signal rendering:

```text
same_history_different_action:
  train_cases: 8000
  eval_cases: 4
  train_final_loss: 3.6033
  before_top1_accuracy: 0.5000
  after_top1_accuracy: 0.5000
  delta_margin: +0.00000124

same_action_different_context:
  train_cases: 8000
  eval_cases: 4
  train_final_loss: 3.5522
  before_top1_accuracy: 0.5000
  after_top1_accuracy: 0.5000
  delta_margin: 0.0
```

Payload rendering:

```text
same_history_different_action:
  train_cases: 8000
  eval_cases: 4
  train_final_loss: 3.6033
  before_top1_accuracy: 0.5000
  after_top1_accuracy: 0.5000
  before_margin: -0.0025
  after_margin: 0.0013

same_action_different_context:
  train_cases: 8000
  eval_cases: 4
  train_final_loss: 3.5522
  before_top1_accuracy: 0.5000
  after_top1_accuracy: 0.5000
  before_margin: -0.0032
  after_margin: -0.0246
```

Small train-subset payload probes also kept top-1 accuracy at `0.5000`, but the
margins were no longer exactly zero.

## Reading

This experiment does not show a modeling success.

It does show that the Experiment 001 zero-margin result was partly an evaluation
artifact. Full signal rendering hid the causal prefix from the scorer under
the short context length.

Current reading:

```text
Data scale alone is not enough to produce a clean ranking improvement in the
small CPU probe.

However, payload rendering is a better diagnostic for the current context
length because it preserves the observation/action signal inside the scoring
window.
```

The next experiment should not introduce new semantic schemas. It should keep
the hard-negative evaluation fixed and vary the minimal factors that affect
whether the model can see and use the relevant prefix:

```text
rendering = payload
context_length
max_steps
model_preset / model size
train-set versus held-out ranking
eval case count once scoring cost is acceptable
```

## Non-Claims

This experiment does not show:

```text
robust action-conditioned future prediction
held-out generalization
learned latent world state
Predictive Token Machine capability
```

It only establishes that the previous ranking setup had a rendering/context
confound and that 100x generated hard-negative data can be produced for future
sweeps.
