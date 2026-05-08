# Training Example Responsibility Mixing

Status: closed.

## Issue

Current code mostly separates training examples from PyTorch `Dataset` samples,
but several paths still mix Data Selection, Training Example Definition, and
runtime sampling responsibilities.

This is not an immediate bug. It is a design risk while the project is deciding
how source records, targets, training examples, samples, and PyTorch datasets
should relate.

## Findings

Image classification and image-text tasks are mostly clean:

- `ImageClassificationExample` stores `image_path` and label meaning.
- `ImageTextChoiceExample` stores image, choices, and answer index.
- `ImageTextAnswerExample` stores image, prompt, and answer text.
- Their PyTorch datasets mostly turn those examples into runtime tensor samples.

Language modeling is less clear:

- `LanguageModelingExample` is just text, so it is close to a source record.
- `LanguageModelingDataset` slices token IDs into context/target windows.
- That means the PyTorch dataset owns part of Training Example Definition.

Grid step prediction is also mixed:

- `GridExperienceTransition` is an experience/source-side record.
- `GridStepPredictionDataset` turns it into observation/action/next-cell/reward
  tensor targets.
- That combines Training Example Definition with runtime sampling.

Shogi move choice is the clearest pressure point:

- `ShogiPolicyValueDatasetDefinition` contains train/eval sources and `max_games`
  style Data Selection.
- It also contains objective and policy/value target-source settings, which are
  closer to Training Example Definition.
- `ShogiMoveChoiceDataset` then converts `ShogiMoveChoiceExample` into tensor
  samples.

## Progress

- The shogi-specific Data Selection boundary problem has been split into
  `shogi-policy-value-data-selection-boundary.md`.
- Language-modeling windowing is intentionally kept in the PyTorch dataset
  because the next-token windows are deterministic and cheap to rebuild from
  token IDs.
- Grid transition-to-target shaping is intentionally kept in the PyTorch dataset
  because next-cell, reward, and termination targets are deterministic and cheap
  to derive from each transition.

## Why It Matters

If these boundaries stay mixed, future multi-source or multi-objective learning
may push more logic into PyTorch datasets or broad dataset-definition files.
That would make it harder to answer:

- which data was included
- which input/target relationship was used
- which targets were generated or stored
- which logic only exists to materialize runtime samples

## Direction

Do not refactor everything now. Use the current code as concrete evidence while
deciding the project-level responsibilities:

- Data Selection: what data is included for a declared use
- Training Example Definition: how included data becomes objective-specific
  input/target relationships
- Runtime Sampling: how training examples become PyTorch samples

## Acceptance Criteria

- [x] Decide whether language-modeling windowing belongs outside the PyTorch
  dataset.
- [x] Decide whether grid transition-to-target shaping should stay in the dataset
  or move to an explicit training-example layer.
- [x] Keep shogi policy-value Data Selection changes in
  `shogi-policy-value-data-selection-boundary.md`.
- [x] Update docs or code only where the boundary decision is stable.

## Resolution

Close this issue. The general boundary terms are now documented, the two small
runtime-sampling exceptions are documented in code, and the remaining shogi
pressure point has its own issue.

## Non-Goals

- immediate code refactor
- broad generic dataset framework
- renaming all `Example` classes
