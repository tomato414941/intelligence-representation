# Evaluation

## Current Line

The current evaluation line is the `src/intrep` prototype, not the retired experiment tree.

The main v1 question is:

```text
Can an untrained decoder-only GPT consume a mixed-world corpus
and reduce next-token loss in a short training run?
```

The old symbolic benchmark should remain available as a support check. It exposes when a predictor succeeds by memorizing seen patterns, when it must use current state relations, and when unsupported is the correct output.

Past semantic-memory, retrieval, conflict, and state-taxonomy tests are historical sketches. They now live under `legacy/tests/`.

## Metrics

The current mixed-GPT smoke check tracks:

```text
token count
training steps
initial loss
final loss
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

tests/test_mixed_corpus.py:
  checks the minimal mixed-world corpus and lightweight rendering tags

tests/test_gpt_training.py:
  checks language-model batches, decoder-only GPT logits, and short-run loss reduction

tests/test_learned_transition_predictor.py:
  checks generated action-conditioned examples and learned predictor behavior

tests/test_prediction_error_update_loop.py:
  checks unsupported -> update -> correct behavior

tests/test_demo.py:
  checks the package demo runs and prints benchmark results

tests/test_intrep_imports.py:
  checks the current package surface imports normally
```

## Acceptance Criteria

A change is useful only if it improves or clarifies at least one of:

```text
mixed corpus construction
decoder-only GPT training behavior
loss reduction in short runs
future evaluation of environment-text correspondences
support benchmark clarity
```

Avoid adding broad schemas, ontology categories, or new experiment files unless the benchmark exposes a repeated need for them.
