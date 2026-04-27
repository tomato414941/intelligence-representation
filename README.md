# intelligence-representation

This repository explores representation for intelligence through an installable, testable prototype.

The current center is not a hand-designed semantic database. It is a small mixed-world GPT training foundation:

```text
natural language + environment episodes + code + logs
  -> byte-level tokens
  -> decoder-only GPT
  -> next-token training
  -> loss / simple evaluation
```

The active implementation is `src/intrep`. Historical experiments are kept under `legacy/` for reference only.

## Current Position

The repository still includes the earlier toy symbolic prediction benchmark, but that is now a support surface rather than the main direction.

The main v1 direction is:

```text
Can a small untrained decoder-only GPT learn from a mixed corpus where
natural language, environment observations/actions, code, and logs are all
treated as parts of the same sequence-learning world?
```

This does not use OpenAI API or a pretrained chat model. It uses the successful GPT/Transformer learning pattern directly: initialize a small decoder-only Transformer and train it on project-owned mixed data.

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
src/intrep/mixed_corpus.py
src/intrep/gpt_model.py
src/intrep/gpt_training.py
src/intrep/train_gpt.py
src/intrep/benchmark.py
src/intrep/update_loop.py
```

## Canonical Docs

Read these first:

```text
docs/world-model.md
docs/bitter-lesson.md
docs/evaluation.md
docs/current-results.md
```

Broad background:

```text
docs/concept.md
```

Legacy / exploratory notes:

```text
docs/legacy/
```

## Design Constraints

This repository should avoid turning into a handcrafted ontology project.

Prefer:

```text
mixed-world sequence data
small decoder-only GPT training runs
byte/char-level tokenization before tokenizer optimization
natural language as important data, not the whole world
environment episodes in symbolic and natural-language renderings
loss curves and simple eval before architecture expansion
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

The demo now runs the mixed-world GPT smoke path. The older symbolic benchmark remains available through `intrep.benchmark.run_benchmark()` for regression and contrast.

## Train Mixed GPT

```sh
uv run python -m intrep.train_gpt --max-steps 20
```

To train from a JSONL corpus with records containing `id`, `modality`, and `content`:

```sh
uv run python -m intrep.train_gpt --corpus file --corpus-path path/to/corpus.jsonl --loss-summary
```
