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

Natural language modeling and world modeling are not treated as opposing architectures here. A natural language model is one special case of an autoregressive predictor over human text streams. A world-model-like trajectory model is another special case over action-conditioned raw examples.

This does not use OpenAI API or a pretrained chat model. It uses the successful GPT/Transformer learning pattern directly: initialize a small decoder-only Transformer and train it on project-owned mixed data.

Next-token loss reduction is a training smoke signal, not evidence that a predictive token machine or world model has been learned. World-model-oriented claims require action-conditioned future prediction checks such as held-out next-observation ranking.

The active direction for new experiments is raw examples that are easy to
tokenize, then `TokenSequence` training. The old Signal JSONL path has been
retired from `src/intrep`; the reserved `Signal` class remains only to block
accidental reuse of that abstraction.

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
src/intrep/byte_tokenizer.py
src/intrep/text_tokenizer.py
src/intrep/token_sequence.py
src/intrep/language_modeling_metrics.py
src/intrep/signals.py
src/intrep/grid_world.py
src/intrep/gpt_model.py
src/intrep/gpt_training.py
src/intrep/evaluate_fashion_mnist.py
src/intrep/fashion_mnist_image_choice_corpus.py
src/intrep/fashion_mnist_vit.py
src/intrep/image_io.py
src/intrep/pair_ranking.py
src/intrep/run_summary.py
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

Broad background:

- [Concept Background](docs/concept.md)

Legacy / exploratory notes:

- [Legacy Notes](docs/legacy/)

## Design Constraints

This repository should avoid turning into a handcrafted ontology project.

Prefer:

```text
task-specific raw examples before tokenization
small decoder-only GPT training runs
byte/char-level tokenization before tokenizer optimization
natural language as important data, not the whole world
loss curves as smoke signals
action-conditioned next-observation evaluation before architecture expansion
tool / memory / belief as future task areas, not current core channels or hand-built schemas
TokenSequence as the common learning input, not raw JSONL envelopes
```

Avoid adding new broad taxonomies, fixed schemas, or semantic dataclasses unless an experiment repeatedly forces them.

## Run Tests

```sh
uv sync
uv run python -m unittest
```

## Image Choice Smoke

```sh
uv run python -m intrep.fashion_mnist_image_choice_corpus \
  --images-path train-images-idx3-ubyte.gz \
  --labels-path train-labels-idx1-ubyte.gz \
  --output-path fashion-train.jsonl \
  --image-output-dir fashion-images/train \
  --limit 1000
uv run python -m intrep.evaluate_fashion_mnist \
  --train-path fashion-train.jsonl \
  --image-patch-size 4 \
  --max-steps 100
```
