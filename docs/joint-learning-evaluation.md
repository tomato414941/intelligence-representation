# Joint LFM Learning Evaluation

2026-09-11. This extends the three-update
[exchangeable-head execution check](shared-prediction.md) into a bounded learning
comparison. Both models start from their pinned official weights with the same
head initialization seed and complete nine-source recipe. The initial state is
measured before any update. The earlier SGD checkpoints remain intact; this is
a new, matched AdamW experiment rather than an implicit optimizer change on
their saved training state.

## Fixed Conditions

- LFM2.5-230M and 350M, one learned body per model, all parameters trainable.
- Every update accumulates every source in `configs/joint-lfm.json` before one
  optimizer step. Complete training populations remain available throughout.
- 1,000 updates per model; AdamW at `1e-5`, no weight decay, joint gradient norm
  clipped at 1, FP32, four CPU threads and one CUDA GPU.
- Source batches retain the execution recipe: 64 target tokens per text source,
  one image per image dataset, one shogi position and one native transition
  with its entire preceding history. There is no permanent dataset sample cap.
- Checkpoint and evaluate at 250-update intervals. Final artifacts include all
  parameters, optimizer state and source cursors. This budget does not traverse
  the complete training populations.

`scripts/run_joint_lfm_experiment.sh OUTPUT [STEPS]` runs the two models
sequentially on the same GPU. Native input heads and all their output objectives
remain active throughout training. Input omissions are evaluation controls only.

## Evaluation Panel

Before training, a deterministic local generator with seed 9047 fixes validation
locations, stored in `evaluation-panel.json`. The same panel is used at every
checkpoint. Validation never advances training cursors or RNG states.

| Source | Panel | Metrics |
| --- | --- | --- |
| Text / conversations | Up to 128 distinct complete-line locations across each validation file/range, 64 next-token targets each | Mean NLL and token accuracy |
| Image classification | 128 distinct images from each existing evaluation split | NLL and top-one accuracy |
| Shogi | 128 complete examples across the existing game-disjoint evaluation file | Policy NLL, recorded-move top-one accuracy, value MSE |
| Native experience | All transitions of the 64 available validation episodes | Action accuracy/NLL, image/audio MSE, feedback loss and answer-token accuracy/NLL |

Text and shogi locations are sampled uniformly from nonempty complete lines
using a streaming reservoir over the full validation range. Text blocks can
continue into following lines. This is a development panel, not a uniform sample
of documents/games or an unbiased token-weighted full-corpus estimate. Shogi
examples retain their game grouping and native transitions their world grouping.
Every individual metric and paired before/after change is retained. The reported
changes are descriptive; no independent-sample confidence claim is made for
correlated tokens, positions or transitions.

The image evaluation splits are the official test sets already used for
development in earlier experiments. They are not untouched final tests. Shogi
recorded-move agreement does not measure playing strength. Native action
agreement is evaluated on recorded observations, not newly collected rollouts.

## Input Use And Language Retention

At the initial and final states, native evaluation additionally uses the same
first 16 panel worlds under complete input and under omission of image, audio
or text observations. Targets and past executed actions remain the recorded
ones. This measures sensitivity to an input form; omission also changes sequence
length and can be outside the training distribution. It is not by itself proof
of causal transfer between separately learned tasks.

`configs/joint-lfm-evaluation-prompts.json` contains 12 fixed English/Japanese
prompts that are not added to training: two free explanations and ten short
arithmetic, text transformation or rule-following questions with exact expected
answers. All generated answers are saved. Strict exact match tests both the
answer and the requested output format. These small probes cannot establish
general language capability or novelty relative to the original pretraining.

Each evaluation is stored under `evaluation/step-NNNNNN.json`. `result.json`
includes paired metrics and actual source progress. A useful result requires
checking individual sources for deterioration as well as improvement; a lower
weighted training loss alone is insufficient. Establishing cross-task transfer
will require a matched intervention beyond this joint-learning comparison.
