# intelligence-representation

This repository explores representation for intelligence through an installable, testable prototype.

The current center is not a hand-designed semantic database. It is a predictive token machine scaffold:

```text
Signal streams
  -> optional signal-tag rendering for low-priority byte-tokenizer experiments
  -> decoder-only GPT-style prediction
  -> next-token / future-token training
  -> FuturePredictionCase ranking for language / world / tool / memory / belief targets
```

The active implementation is `src/intrep`. Historical experiments are kept under `legacy/` for reference only.

## Current Position

The repository still includes the earlier toy symbolic prediction benchmark, but that is now a support surface rather than the main direction.

The conceptual center is:

```text
A predictive token machine for language, perception, action, memory, and belief.
```

The broader hypothesis is that multimodal Signal streams can support a learned general-purpose predictive computation substrate. In that framing, a world model is not the whole project. It is the part of a predictive token machine that predicts observations, actions, and environment transitions.

The main v1 direction is:

```text
Can a small untrained decoder-only GPT learn useful future prediction behavior
from signal streams where text, observations, actions, consequences, tool
results, errors, and belief-like records share one sequence-learning substrate?
```

Natural language modeling and world modeling are not treated as opposing architectures here. A natural language model is one special case of an autoregressive predictor over human text streams. A world-model-like trajectory model is another special case over signal observation/action/consequence streams.

This does not use OpenAI API or a pretrained chat model. It uses the successful GPT/Transformer learning pattern directly: initialize a small decoder-only Transformer and train it on project-owned mixed data.

Next-token loss reduction is a training smoke signal, not evidence that a predictive token machine or world model has been learned. World-model-oriented claims require action-conditioned future prediction checks such as held-out next-observation ranking.

`MixedDocument` remains as a legacy smoke/bridge format so existing demos and tests keep working. The active abstraction for new experiments is `Signal` streams, with `FuturePredictionCase` used for target-position-aware evaluation.

The old benchmark still compares rule, frequency, state-aware, sequence-feature, and tiny Transformer predictors over symbolic world-model tokens. It remains useful as a regression and contrast, but it is no longer the main project path.

## Project Map

```text
src/intrep/
  Active prototype package

tests/
  Default test suite for src/intrep

legacy/experiments/
  Retired experiment code

legacy/tests/
  Retired experiment tests

docs/legacy/
  Historical experiment notes and older architecture sketches
```

Current implementation surface:

```text
src/intrep/types.py
src/intrep/dataset.py
src/intrep/environment.py
src/intrep/predictors.py
src/intrep/evaluation.py
src/intrep/tokens.py
src/intrep/sequence.py
src/intrep/sequence_predictor.py
src/intrep/torch_sequence.py
src/intrep/tiny_transformer.py
src/intrep/byte_tokenizer.py
src/intrep/language_modeling_metrics.py
src/intrep/mixed_corpus.py
src/intrep/mixed_corpus_evaluation.py
src/intrep/generated_environment_corpus.py
src/intrep/generated_environment_signal_corpus.py
src/intrep/corpus_split.py
src/intrep/corpus_coverage.py
src/intrep/signals.py
src/intrep/signal_stream.py
src/intrep/signal_corpus.py
src/intrep/grid_world.py
src/intrep/grid_corpus.py
src/intrep/grid_signal_corpus.py
src/intrep/future_prediction_cases.py
src/intrep/future_prediction_ranking.py
src/intrep/gpt_model.py
src/intrep/gpt_training.py
src/intrep/train_gpt.py
src/intrep/train_ptm.py
src/intrep/symbolic_to_natural_evaluation.py
src/intrep/next_observation_cases.py
src/intrep/next_observation_ranking.py
src/intrep/next_observation_evaluation.py
src/intrep/evaluate_next_observation.py
src/intrep/evaluate_future_prediction.py
src/intrep/pair_ranking.py
src/intrep/current_experiment.py
src/intrep/experiment_pipeline.py
src/intrep/external_corpus.py
src/intrep/public_text_fetch.py
src/intrep/run_summary.py
src/intrep/source_manifest.py
src/intrep/tool_log_corpus.py
src/intrep/benchmark.py
src/intrep/demo.py
src/intrep/transition_data.py
src/intrep/update_loop.py
```

## Canonical Docs

Read these first:

- [Predictive Token Machine](docs/predictive-token-machine.md)
- [World Model Centering](docs/world-model.md)
- [Bitter Lesson Correction](docs/bitter-lesson.md)
- [Evaluation](docs/evaluation.md)
- [Experiment 001](docs/experiment-001.md)
- [Experiment 002](docs/experiment-002.md)
- [Experiment 003](docs/experiment-003.md)
- [Current Results](docs/current-results.md)
- [RunPod Training Notes](docs/runpod.md)

Broad background:

- [Concept Background](docs/concept.md)

Legacy / exploratory notes:

- [Legacy Notes](docs/legacy/)

## Design Constraints

This repository should avoid turning into a handcrafted ontology project.

Prefer:

```text
signal mixed-world sequence data
Signal-tag rendering only as a low-priority byte-tokenizer experiment
small decoder-only GPT training runs
byte/char-level tokenization before tokenizer optimization
natural language as important data, not the whole world
environment episodes in symbolic and natural-language renderings
generated environment train/eval slices selected through the evaluation CLI
generated environment Signal train/eval slices for hard-negative checks
for same-history/different-action and same-action/different-context checks
same-modality hard distractors for next-observation ranking by default
loss curves as smoke signals
action-conditioned next-observation evaluation before architecture expansion
tool / memory / belief streams as future evaluation targets, not hand-built schemas
existing symbolic benchmarks as support, not the main path
```

Avoid adding new broad taxonomies, fixed schemas, or semantic dataclasses unless an experiment repeatedly forces them.

## Run Tests

```sh
uv sync
uv run python -m unittest
```

## Run Demo

```sh
uv run python -m intrep.demo
```

The demo now runs the mixed-GPT mainline on the built-in smoke corpus. The older symbolic benchmark remains available through `intrep.benchmark.run_benchmark()` for regression and contrast, but it is support rather than the main corpus or main direction.

## Train Mixed GPT

```sh
uv run python -m intrep.train_gpt --max-steps 20
```

With no corpus flags, this uses the built-in smoke corpus only.

To train the current byte-tokenizer path from a legacy mixed-document JSONL
corpus with records containing `id`, `modality`, and text `content`:

```sh
uv run python -m intrep.train_gpt --corpus file --corpus-path path/to/corpus.jsonl --loss-summary
```

JSONL file corpora are the intended path for real corpus growth.

For GPU hosts such as RunPod, `intrep.train_gpt` supports `--device auto|cpu|cuda`
and final checkpoint writing with `--checkpoint-path`. See `docs/runpod.md`.

Public or internet-sourced corpora should enter through the same JSONL shape. Keep fetching, licensing review, filtering, and provenance capture outside the training architecture until a repeated experiment proves a new repo-level component is needed.
