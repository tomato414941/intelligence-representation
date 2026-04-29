# Evaluation

## Current Line

The current evaluation line is the `src/intrep` prototype, not the retired experiment tree.

The conceptual project line is:

```text
A predictive token machine for language, perception, action, memory, and belief.
```

In that broader frame, world modeling is a core evaluation surface, not the whole project. It measures whether the predictive token machine can use observation/action history to predict future observations or consequences.

The main v1 engineering question is:

```text
Can an untrained decoder-only GPT consume signal streams and improve
target-channel future prediction after a short next-token training run?
```

Average next-token loss reduction remains a smoke question, not a predictive-token-machine or world-model claim. The training objective can be next-token prediction, but the evaluation target for world-model-oriented claims must be action-conditioned future prediction.

Use this distinction:

```text
training objective:
  next-token prediction over Signal streams

training smoke metric:
  average next-token loss reduction

world-model-oriented metric:
  held-out action-conditioned next-observation / future-token prediction
```

In this question, the built-in corpus is only the smoke/demo corpus. It verifies the mixed-GPT mainline without requiring external data. The real corpus growth path is JSONL files passed to the training CLI. The legacy mixed-document schema keeps `id`, `modality`, and `content`; the signal schema keeps `channel` and either text `payload` or non-text `payload_ref`.

Signal JSONL text payload:

```json
{"channel":"action","payload":"{\"name\":\"open_box\",\"arguments\":{\"box\":\"red\"}}"}
```

Signal JSONL reference payload:

```json
{"channel":"image","payload_ref":{"uri":"dataset://images/frame-1.png","media_type":"image/png","sha256":"...","size_bytes":12345}}
```

`payload` and `payload_ref` are mutually exclusive. `payload_ref` may represent non-text signals such as images that should become training targets or conditioning inputs. The current byte-tokenizer training and ranking path consumes text `payload` only because no channel-specific loader or encoder is wired in yet, so it explicitly rejects `payload_ref`.

The active signal evaluation unit is `FuturePredictionCase`:

```text
prefix_events:
  signals such as OBSERVATION + ACTION or TOOL_CALL

positive_event:
  the correct CONSEQUENCE / TOOL_RESULT / PREDICTION_ERROR

negative_events:
  same-channel, same-modality hard negatives
```

The general CLI for this path is:

```sh
uv run python -m intrep.evaluate_future_prediction \
  --train-path train.signals.jsonl \
  --eval-path eval.signals.jsonl \
  --target-channel consequence \
  --condition same_modality_negative
```

The generated environment signal corpus builder creates train/eval `Signal` JSONL for hard-negative consequence ranking:

```sh
uv run python -m intrep.generated_environment_signal_corpus \
  --train-output train.signals.jsonl \
  --eval-output eval.signals.jsonl \
  --eval-slice same_history_different_action \
  --train-size 100 \
  --eval-size 40 \
  --seed 7
```

Use `same_history_different_action` to keep the observation fixed while changing the action, and `same_action_different_context` to keep the action fixed while changing the observation. Both generated slices create paired Signal triples so the intended hard-negative contrast is present in the same-channel consequence distractor set. The older generated slices, such as `generated_strict_noisy`, remain available for compatibility and broader smoke coverage.

Those generated files can then be evaluated directly:

```sh
uv run python -m intrep.evaluate_future_prediction \
  --train-path train.signals.jsonl \
  --eval-path eval.signals.jsonl \
  --target-channel consequence \
  --condition same_history_different_action \
  --rendering payload
```

Future-prediction ranking has two rendering modes:

```text
signal:
  full Signal tag rendering; a low-priority experiment for explicitly testing
  long signal-tag streams

payload:
  event text payloads only; the current text/byte-tokenizer scoring path is the
  preferred diagnostic for short-context
  action/context-conditioned consequence ranking
```

Use `--rendering payload` when testing whether a short-context model can use
the relevant observation/action prefix for hard-negative consequence ranking.
Full signal rendering can make the tags long enough that the consequence
payload is scored after the causal prefix has fallen out of the model window.
That makes it a poor default diagnostic for `context_length = 64` consequence
ranking unless the experiment is explicitly about full signal-tag streams. See
[Experiment 002](experiment-002.md).

Fashion-MNIST image-label smoke data can be converted from local IDX files into
Signal JSONL:

```sh
uv run python -m intrep.fashion_mnist_signal_corpus \
  --images-path train-images-idx3-ubyte.gz \
  --labels-path train-labels-idx1-ubyte.gz \
  --output-path fashion-train.signals.jsonl \
  --image-output-dir fashion-images/train \
  --limit 1000
```

The current image path keeps image handling separate from the text tokenizer:
local `file://` image payload refs are loaded as grayscale tensors, patchified,
embedded, passed through a Transformer encoder, and evaluated with a
classification head:

```sh
uv run python -m intrep.evaluate_fashion_mnist \
  --train-path fashion-train.signals.jsonl \
  --eval-path fashion-eval.signals.jsonl \
  --image-patch-size 4 \
  --max-steps 100
```

The old symbolic benchmark should remain available as a support check. It exposes when a predictor succeeds by memorizing seen patterns, when it must use current state relations, and when unsupported is the correct output. It is not the main corpus and should not drive a broad taxonomy.

Past semantic-memory, retrieval, conflict, and state-taxonomy tests are historical sketches. They now live under `legacy/tests/`.

## Metrics

The current mixed-GPT smoke check tracks the built-in smoke corpus:

```text
token count
training steps
initial loss
final loss
loss history / best loss
train corpus average loss
held-out eval corpus loss
symbolic-to-natural pair ranking accuracy and margin
mixed next-observation ranking accuracy and margin
builtin-grid loss reduction smoke check
```

The support symbolic benchmark tracks:

```text
prediction accuracy
unsupported rate
condition-level accuracy
training size before / after update
update success
```

Human-readable Fact / Action structures are evaluation artifacts. They are not claimed to be the final architecture.

## Current Tests

```text
tests/test_benchmark.py:
  checks rule baseline vs frequency predictor vs state-aware predictor vs Transformer-ready adapter vs tiny Transformer,
  condition-level failures,
  generated distribution slices,
  and prediction-error update success

tests/test_generated_distribution.py:
  checks fixed generated train/test slices and non-overlap

tests/test_tokens.py / tests/test_sequence.py:
  check the world-model token sequence interface

tests/test_sequence_predictor.py:
  checks the dependency-free sequence-feature baseline and its limits

tests/test_tiny_transformer.py:
  checks vocabulary construction, a seen training example, and current held-out limits

tests/test_byte_tokenizer.py:
  checks byte-level round-trip for Japanese, English, code, and logs

tests/test_grid_world.py:
  checks hidden grid state, partial observation, action steps, and next observations

tests/test_gpt_training.py:
  checks language-model batches, decoder-only GPT logits, short-run loss reduction,
  and reusable training artifacts

tests/test_pair_ranking.py:
  checks next-token continuation loss helpers

tests/test_future_prediction_ranking.py:
  checks target-channel future prediction ranking and payload-only rendering

tests/test_learned_transition_predictor.py:
  checks generated action-conditioned examples and learned predictor behavior

tests/test_prediction_error_update_loop.py:
  checks unsupported -> update -> correct behavior

tests/test_demo.py:
  checks the package demo runs the mixed-GPT mainline smoke path

tests/test_intrep_imports.py:
  checks the current package surface imports normally
```

## Acceptance Criteria

A change is useful only if it improves or clarifies at least one of:

```text
mixed corpus construction
non-language grid observation corpus construction
JSONL file corpus loading and growth
decoder-only GPT training behavior
loss reduction in short runs
symbolic-to-natural environment-text correspondence evaluation
mixed next-observation ranking evaluation
support benchmark clarity
```

Avoid adding broad schemas, ontology categories, or new experiment files unless the benchmark exposes a repeated need for them.
