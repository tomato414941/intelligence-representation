# intelligence-representation

This repository explores representation for intelligence through an installable, testable prototype.

The current center is not a hand-designed semantic database. It is a predictive token machine scaffold:

```text
raw text / image / audio / label-like data
  -> modality-specific tokenization / encoding
  -> TokenSequence
  -> shared next-token / continuation prediction
```

The active implementation is `src/intrep`. Historical experiments are kept under `legacy/` for reference only.

## Current Position

The repository still includes the earlier toy symbolic prediction benchmark, but that is now a support surface rather than the main direction.

The conceptual center is:

```text
A predictive token machine for language, perception, action, memory, and belief.
```

The broader hypothesis is that text, image, audio, action, and outcome data can
be converted into token or embedding sequences that share a predictive learning
form. In that framing, a world model is not the whole project. It is one
evaluation surface for predicting action-conditioned futures.

The main v1 direction is:

```text
Can text, image, audio, and action-conditioned data be converted into a common
token-sequence prediction form without adding unused schema fields?
```

Natural language modeling and world modeling are not treated as opposing architectures here. A natural language model is one special case of an autoregressive predictor over human text streams. A world-model-like trajectory model is another special case over action-conditioned signal streams.

This does not use OpenAI API or a pretrained chat model. It uses the successful GPT/Transformer learning pattern directly: initialize a small decoder-only Transformer and train it on project-owned mixed data.

Next-token loss reduction is a training smoke signal, not evidence that a predictive token machine or world model has been learned. World-model-oriented claims require action-conditioned future prediction checks such as held-out next-observation ranking.

The active direction for new experiments is raw examples that are easy to
tokenize, then `TokenSequence` training. Signal JSONL and signal-tag rendering
remain transitional surfaces for existing experiments only.

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
src/intrep/text_tokenizer.py
src/intrep/token_sequence.py
src/intrep/language_modeling_metrics.py
src/intrep/generated_environment_signal_corpus.py
src/intrep/signals.py
src/intrep/signal_stream.py
src/intrep/signal_io.py
src/intrep/signal_rendering.py
src/intrep/grid_world.py
src/intrep/future_prediction_cases.py
src/intrep/future_prediction_ranking.py
src/intrep/gpt_model.py
src/intrep/gpt_training.py
src/intrep/train_signal_text.py
src/intrep/evaluate_future_prediction.py
src/intrep/evaluate_fashion_mnist.py
src/intrep/fashion_mnist_signal_corpus.py
src/intrep/fashion_mnist_vit.py
src/intrep/image_io.py
src/intrep/pair_ranking.py
src/intrep/run_summary.py
src/intrep/benchmark.py
src/intrep/transition_data.py
src/intrep/update_loop.py
```

## Canonical Docs

Read these first:

- [Predictive Token Machine](docs/predictive-token-machine.md)
- [Token Sequence Direction](docs/token-sequence-direction.md)
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
tool / memory / belief as future task areas, not current core channels or hand-built schemas
TokenSequence as the common learning input, not Signal JSONL
existing symbolic benchmarks as support, not the main path
```

Avoid adding new broad taxonomies, fixed schemas, or semantic dataclasses unless an experiment repeatedly forces them.

## Run Tests

```sh
uv sync
uv run python -m unittest
```

The older symbolic benchmark remains available through `intrep.benchmark.run_benchmark()` for regression and contrast, but it is support rather than the main corpus or main direction.

## Train Text Smoke

```sh
uv run python -m intrep.train_signal_text --train-path path/to/signals.jsonl --max-steps 20
```

This is a transitional text-payload Signal path kept for existing experiments.
New work should prefer raw examples that can be converted into `TokenSequence`.

For GPU hosts such as RunPod, `intrep.train_signal_text` supports `--device auto|cpu|cuda`
and final checkpoint writing with `--checkpoint-path`. See `docs/runpod.md`.

The default text tokenizer remains byte-level for reproducibility. A small
corpus-trained byte-pair tokenizer is available for GPT-like tokenization smoke
checks without adding an external tokenizer dependency:

```sh
uv run python -m intrep.train_signal_text \
  --train-path path/to/signals.jsonl \
  --tokenizer byte-pair \
  --tokenizer-vocab-size 512
```
