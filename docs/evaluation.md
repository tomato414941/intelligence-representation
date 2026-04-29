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
Can raw text/image/action-conditioned examples be converted into token
sequences that improve continuation or future-token prediction after a short
training run?
```

Average next-token loss reduction remains a smoke question, not a predictive-token-machine or world-model claim. The training objective can be next-token prediction, but the evaluation target for world-model-oriented claims must be action-conditioned future prediction.

Use this distinction:

```text
training objective:
  next-token prediction over TokenSequence

training smoke metric:
  average next-token loss reduction

world-model-oriented metric:
  held-out action-conditioned next-observation / future-token prediction
```

In this question, generated corpora are smoke corpora. New datasets should keep
raw examples close to their source and convert them to `TokenSequence` or hidden
sequences at training or evaluation time.

The old Signal JSONL and `FuturePredictionCase` evaluation line has been
retired from `src/intrep`. Historical notes remain under older experiment docs,
but new evaluation code should not use a generic channel envelope.

Fashion-MNIST image-choice smoke data can be converted from local IDX files into
raw image-choice JSONL:

```sh
uv run python -m intrep.fashion_mnist_image_choice_corpus \
  --images-path train-images-idx3-ubyte.gz \
  --labels-path train-labels-idx1-ubyte.gz \
  --output-path fashion-train.jsonl \
  --image-output-dir fashion-images/train \
  --limit 1000
```

The current image path keeps image handling separate from the text tokenizer:
local image paths are loaded as grayscale tensors, patchified, embedded, passed
through a Transformer encoder, and evaluated with a classification head:

```sh
uv run python -m intrep.evaluate_fashion_mnist \
  --train-path fashion-train.jsonl \
  --eval-path fashion-eval.jsonl \
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
