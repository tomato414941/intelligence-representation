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

`scripts/summarize_joint_lfm_learning.py --root OUTPUT --data-root .` produces
`comparison.json` and PNG/PDF learning curves (Matplotlib is required). It also
computes action-majority frequencies from the training episodes, per-action
accuracy on validation, and native forecast baselines that repeat the preceding
image/audio or predict silent audio. The baselines use the exact recorded panel
and verify its dataset identity. Native text targets repeat the color named in
the initial instruction; accuracy on these targets measures recovery of that
color, not general question answering or visual captioning.

## Results: 1,000 Updates Per Model

Both models completed the predeclared budget on one A40. All nine source losses
decreased on the fixed development panel, and every body parameter tensor
changed (131/131 for 230M, 147/147 for 350M). This demonstrates learning across
the joint objectives. It does not establish preservation of general language
behavior: the generation probes contain deteriorations despite lower text NLL.

| Measurement | 230M before → after | 350M before → after |
| --- | ---: | ---: |
| TinyStories NLL ↓ | 5.060 → 2.223 | 9.542 → 2.169 |
| WikiText-2 NLL ↓ | 7.395 → 3.849 | 11.020 → 3.814 |
| Shakespeare NLL ↓ | 7.805 → 4.607 | 12.608 → 4.555 |
| Conversations NLL ↓ | 4.791 → 2.898 | 4.727 → 3.056 |
| MNIST accuracy ↑ | 10.9% → 50.0% | 7.8% → 48.4% |
| Fashion-MNIST accuracy ↑ | 6.2% → 57.0% | 15.6% → 47.7% |
| CIFAR-10 accuracy ↑ | 7.8% → 18.0% | 7.0% → 15.6% |
| Shogi recorded-move accuracy ↑ | 2.3% → 11.7% | 3.1% → 7.8% |
| Native teacher-action accuracy ↑ | 22.4% → 47.1% | 29.7% → 50.5% |

Each model saw 64,000 target tokens from each of the four text sources, 1,000
images from each image source, 1,000 shogi positions, and 1,000 native transitions
from 167 started episodes. These are actual consumption counts, not dataset
size limits. None of the complete source populations finished an epoch. The
image/board panels each contain 128 examples; native evaluation contains 384
transitions in 64 worlds. The 128 shogi examples span 123 games; value MSE uses
the 113 positions that have a value target and changes from 2.041 to 2.018 for
230M and 2.068 to 1.855 for 350M.

The larger model does not dominate this short comparison. Its final text NLL
is slightly lower on three corpora and action accuracy is higher, while 230M
has higher image and shogi agreement. Both shogi combined losses rise between
750 and 1,000 updates. One seed and this small development panel are insufficient
to rank model sizes in general.

### Native Prediction And Input Use

| Native measurement | 230M final | 350M final | Simple baseline on the same 384 transitions |
| --- | ---: | ---: | ---: |
| Action accuracy ↑ | 47.1% | 50.5% | 21.6%, training-majority action |
| Mean accuracy across five target actions ↑ | 40.4% | 45.4% | 20.0%, constant action |
| Next-image MSE ↓ | 0.05845 | 0.06080 | 0.05179, preceding image |
| Next-audio MSE ↓ | 0.04401 | 0.04753 | 0.06526, preceding audio; 0.06736, silence |

Both audio forecasts beat the two simple baselines; neither image forecast
beats repeating the preceding image. Action learning is uneven. For target
action IDs 0–4, 230M achieves 100%, 0%, 2%, 0%, 100%; 350M achieves 96.4%, 0%,
16.0%, 14.5%, 100%. The improved aggregate therefore does not demonstrate
reliable control across the action space or successful interaction rollouts.

On the smaller, fixed 16-world input-control panel (96 transitions), omitting
audio lowers action accuracy from 41.7% to 36.5% for 230M and from 46.9% to
41.7% for 350M. Omitting images leaves 230M accuracy at 41.7% and raises 350M
accuracy to 51.0%. Omitting text also raises action accuracy, to 47.9% and 50.0%.
These controls offer limited evidence of useful audio sensitivity, but do not
show useful image or instruction-text dependence for action selection here.
Their sequence-length and distribution changes prevent a stronger causal claim.

Native answer-token accuracy reaches 100% for both models. The target is the
instruction color, with the entire observation history still supplied. Omitting
text lowers this to 73.4% and 76.6%. This supports recovery of the supplied
instruction color; it does not establish general question answering, learned
memory compression or transfer from the image-classification tasks.

### Language Retention

The ten strict-format questions score 2/10 → 1/10 for 230M and 3/10 → 3/10 for
350M. Exact match combines content and format, so the saved answers matter:

- 230M changes `cat` uppercasing from `CAT` to an unrelated sentence about a
  cat being a dog, and changes its English second-list-item answer from a
  sentence correctly naming blue to `Red`.
- 350M improves the English subtraction answer from `2` to `7`, but changes
  `CAT` to `CAT\nUPPERCASE\n`, violating the requested output format.
- 350M initially explains ice melting by heat absorption. After training it
  claims ice does not have a melting point like water. Both final Japanese
  explanations are circular or incorrect. These free explanations are outside
  the ten-question exact-match score.

Thus unchanged aggregate exact match for 350M conceals both improvements and
deteriorations. The observations reject a blanket claim that this configuration
preserves existing responses while adding native skills. They do not quantify
population-level forgetting or identify which source caused each change.
Further work needs language retention as an explicit evaluation constraint,
more coverage of the poorly learned actions and visual dependencies, and matched
comparisons that keep every source and all parameters active.

## Artifacts And Verification

The durable local root is `models/joint-lfm-learning-20260911/`. Its `230m/` and
`350m/` directories retain the final `checkpoint.pt`, tokenizer, full recipe,
data provenance, source progress, fixed panel and all five evaluation reports.
The weight checkpoint contains the final 1,000-update state; earlier periodic
weight saves were atomically replaced. All before/after generations remain in
the evaluation reports. The root also contains `comparison.json`,
`learning-curves.png`, `learning-curves.pdf`, environment/resource records and
`runpod_timings.json`. These generated artifacts are not committed.

Training ran from clean revision `661a54d`; analysis uses `87f8c1f`.
`experiment-manifest.json` records full revisions and SHA-256 digests of both
checkpoints and their evaluation/provenance inputs. Both real checkpoints were
loaded independently on CPU: one body, all parameters trainable and finite,
tied text weights, finite restored AdamW state and all nine source cursors.
Each source then produced a finite next loss. The two checked arithmetic
generations per model matched the saved GPU answers. This verifies loading and
inference, not bitwise equality of resumed training across CPU and CUDA.

The full unit suite passed 530 tests, including evaluation-state restoration
and unchanged optimizer updates when periodic evaluation is enabled. The
comparison script ran against both complete results, verified identical panels
and baseline data, and produced the reviewed plot. Verification records are
retained alongside the checkpoints. Retrieval completed and the disposable pod
was deleted; an independent pod listing showed none remaining. Runtime and
cost details are in [compute-costs.md](compute-costs.md#joint-full-parameter-lfm-sizing-reference).
