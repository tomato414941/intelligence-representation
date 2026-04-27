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
Can an untrained decoder-only GPT consume a mixed-world corpus
and reduce next-token loss in a short training run?
```

That is a smoke question, not a predictive-token-machine or world-model claim. The training objective can be next-token prediction, but the evaluation target for world-model-oriented claims must be action-conditioned future prediction.

Use this distinction:

```text
training objective:
  next-token prediction over typed streams

training smoke metric:
  average next-token loss reduction

world-model-oriented metric:
  held-out action-conditioned next-observation / future-token prediction
```

In this question, the built-in corpus is only the smoke/demo corpus. It verifies the mixed-GPT mainline without requiring external data. The real corpus growth path is JSONL files passed to the training CLI. The legacy schema keeps `id`, `modality`, and `content`; the typed-event schema adds `role`, `episode_id`, `time_index`, and `metadata`, then renders events as typed tags before byte-level training.

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

Next-observation ranking now defaults to `distractor_policy=hard`, which ranks the positive next observation against other cases from the same modality. `distractor_policy=all_other` keeps the earlier behavior of using every other case as a distractor.

Ranking requires at least two compatible evaluation cases: one positive case and at least one candidate distractor from the same evaluation pool or explicit hard-negative metadata. A generated slice with only one eval case is useful as a fixture, but it cannot produce a meaningful ranking metric by itself. Strict generated slices should therefore contain at least two cases, and preferably enough cases to make shortcut-driven distractor wins visible.

The independent next-observation CLI supports the generated environment train/eval split directly:

```sh
uv run python -m intrep.evaluate_next_observation \
  --corpus generated-environment \
  --generated-eval-slice generated_held_out_object
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
  checks the built-in smoke corpus and lightweight rendering tags

tests/test_grid_world.py:
  checks hidden grid state, partial observation, action steps, and next observations

tests/test_grid_corpus.py:
  checks grid episodes render into text/grid/action/next-grid mixed documents

tests/test_gpt_training.py:
  checks language-model batches, decoder-only GPT logits, short-run loss reduction,
  and reusable training artifacts

tests/test_pair_ranking.py:
  checks symbolic-to-natural continuation ranking metrics

tests/test_symbolic_to_natural_evaluation.py:
  checks before/after GPT symbolic-to-natural ranking evaluation,
  including held-out environment pair documents

tests/test_next_observation_cases.py:
  checks environment-symbolic and grid document extraction into next-observation cases

tests/test_next_observation_ranking.py:
  checks mixed next-observation continuation ranking and hard/all-other distractor policies

tests/test_next_observation_evaluation.py:
  checks before/after GPT ranking evaluation, distractor policy propagation,
  and held-out eval document separation

tests/test_evaluate_next_observation_cli.py:
  checks the independent next-observation evaluation CLI, including generated-environment selection,
  without changing train_gpt

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
