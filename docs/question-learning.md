# Learning Several Questions From Each Dataset

2026-09-11. The [previous joint experiment](joint-learning-evaluation.md) used
nine data sources with mostly fixed objectives. Its development losses improved,
but instruction responses deteriorated and image forecasts failed to beat
persistence. This experiment adds multiple questions per source, more consumed
records, and three additional datasets. It compares fixed and varied questions
under the same measured training-time budget.

## Data And Learning Conditions

All nine existing training populations remain in `configs/question-learning.json`.
The added populations are complete FSDD spoken digits, UCI HAR inertial windows,
and the labeled BoolQ reading-comprehension release. Their acquisition, licenses
and group splits are in [datasets.md](datasets.md#additional-data-for-question-learning).
The experiment uses one LFM2.5-350M body per condition, with all parameters
eligible for full-model AdamW updates. No pretrained speech or sensor model is
used. New sensor inputs are nine measured channels, not textual descriptions.

Both conditions start from the pinned official base, not a previous trained
checkpoint. Their initial parameter digests, attached heads, training populations
and evaluation panels must match. FP32 AdamW uses learning rate `1e-5`, no weight
decay, gradient clipping at 1, and equal weights for the twelve data sources.
Every optimizer update receives a loss from every source.

The primary budget is 3,600 measured training seconds per condition on the same
GPU, with a 12,000-update ceiling. CUDA is synchronized around each measurement.
Training time includes record reading, question construction, forward/backward
passes and optimizer updates, and excludes evaluation and checkpoint I/O. This
is a comparison per unit of machine time, not exact FLOP matching. Different
question costs can lead to different final numbers of consumed examples.
Evaluations at common 1,000-update intervals offer a secondary comparison with
the same source sampling prefix. Exact update and consumption counts are saved.

Each text source now reads a block with 128 next-token target positions, twice
the previous block length. Image, speech and sensor sources read an anchor from
their complete shuffled population and a second record from a class-balanced
partner pool. Same-class and different-class pairs alternate across complete
question cycles, so each added form receives both pair labels. Partner pools
cycle through all members of each class; distinct partners are used when possible.
Shogi and native sources read two records/transitions per update. BoolQ reads one
complete passage/question. No source has a permanent small-sample cap. Increased
exposure is measured separately from the number of derived questions.

## Questions

The record readers expose records independently of their loss computation.
`QuestionSource` constructs explicit questions and supervises answers or native
outputs. The fixed condition retains each source's original objective. The varied
condition alternates original objectives with the added forms below, cycling
through the additions. Question wording 0 is used for training; wording 1 is a
held-out paraphrase. This is not evidence of a previously unseen task or novelty
relative to pretraining.

| Source | Added forms |
| --- | --- |
| TinyStories, WikiText-2, Shakespeare, conversations | Restore a missing span; extract the first or last word |
| MNIST | Name the digit; same/different class; sum; greater-than comparison; image completion |
| Fashion-MNIST, CIFAR-10 | Name the class; same/different class; image completion |
| Shogi | Judge a candidate legal/illegal; predict the complete board and hands after a legal move |
| Native experience | Recall the first/last executed action; report the sign of the last observed reward |
| Spoken digits | Name the digit; same/different digit; sum; waveform completion |
| Inertial activity | Name the activity; same/different activity; predict the last 32 samples from the first 96 |
| BoolQ | Check whether a proposed yes/no answer is correct |

The same/different, legal/illegal and proposed-yes/proposed-no pairs share the
same observations while requiring complementary answers. They test whether the
model changes its answer with the question. Forecast/reconstruction targets are
excluded from the inputs; their query coordinates specify only where to predict.
Image/audio completion is scored on the missing region, with visible-image-mean
and preceding-waveform-chunk baselines. Sensor forecasting uses persistence, and
shogi successor prediction compares with copying the unchanged board/hands.

All parameters remain trainable in both conditions. Additional output heads
that have no objective in the fixed condition provide untrained references;
improvement on those heads alone does not demonstrate transfer or reuse without
supervision. Text answers share the original tied text embedding/output weights.

## Evaluation

Original objectives retain 128 fixed locations per source and every transition
of the 64 available native validation worlds. Added questions use 32 randomly
selected locations from each source panel, or all transitions of 16 selected
native worlds. The same records, partners, masks and candidates recur across
forms, wordings, checkpoints and conditions. Losses and metrics are reported
by source, form and wording. A mixed average across classification, reconstruction
and language losses is not a capability score.

Initial/final reports retain greedy generated answers as well as teacher-forced
metrics. Intermediate evaluations omit greedy decoding. The original twelve
language prompts remain, with eighty additional fixed arithmetic, list and
conditional-rule probes in English and Japanese: ninety strict questions and
two free explanations in total. These probes are excluded from training.

Native image/audio/text omissions are retained. Additional controls remove all
observations for selected nontext questions while preserving the question and
target. Their changed sequence lengths and distribution limit causal inference.
Source state, question schedule, partner samplers and RNG states are restored
after evaluation; evaluation must not change subsequent optimizer updates.

BoolQ has no identical question/passage pair across its official train/development
splits, but 720 distinct passages occur in both. Results must distinguish new
passages from new questions about passages appearing in training. Likewise,
image test splits already used for development are not untouched final tests.
Shogi agreement is not playing strength; recorded native action accuracy is not
successful interaction. No new environment rollouts are collected in this run.

## Reproduction

```sh
uv run python scripts/prepare_question_datasets.py --output data/question-learning-20260911
bash scripts/run_question_learning_experiment.sh models/question-learning-20260911 3600
```

The experiment script requires the existing pinned LFM base, the original nine
datasets, the newly prepared datasets and CUDA. It saves full final checkpoints,
optimizer/source states, tokenizer, recipes, provenance, panels, individual
responses and consumption counts outside disposable `runs/`. Periodic weight
checkpoints are atomically replaced; their evaluation reports remain. Both
conditions and all evaluations complete before output retrieval and pod deletion.
