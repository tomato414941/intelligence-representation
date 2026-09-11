# Question Learning: Twelve Sources With Fixed And Varied Objectives

2026-09-11. Varied questions improved first/last text extraction and several
reconstruction tasks. They did not preserve general instruction responses: strict
correctness fell from 47/90 initially to 0/90 with fixed objectives and 4/90 with varied
objectives. None of the seven held-out complementary yes/no pair scores exceeded its
best constant-relation baseline.

The [protocol](question-learning.md) defines the questions, splits, sampling, and
one-hour training budgets. The [preceding experiment](joint-learning-evaluation.md) used
nine sources and 1,000 joint updates. This run adds spoken digits, inertial activity and
reading comprehension, and increases the records actually consumed. There are 55
**dataset–form combinations**, including original objectives; these are not 55
independent question families.

## Conditions And Actual Exposure

Both conditions use one LFM2.5-350M body with 358,228,139 attached parameters, all
trainable, FP32 AdamW at `1e-5`, seed 47 and all twelve sources per update. Both start
from the same pinned official base. Their complete initial evaluation reports and panels
are identical; their parameter digests also match. Fixed uses only original objectives.
Varied alternates original and added objectives, so sharing a sampling prefix does not
mean receiving identical supervision. Training time excludes evaluation and checkpoint
I/O and is not exact FLOP matching.

| Condition | Joint updates | Measured training seconds | Seconds/update | Peak Torch CUDA allocation |
| --- | --- | --- | --- | --- |
| fixed | 3,889 | 3600.611 | 0.926 | 7941.6 MiB |
| varied | 4,047 | 3600.081 | 0.890 | 7942.8 MiB |

Every text stream reads a block containing 128 next-token target positions per update.
Image, speech, sensor, shogi and native sources read two records or transitions per
update; BoolQ reads one complete passage/question. Counts below include repetitions and
partners. They are not counts of unique examples or new questions invented from those
examples.

| Consumed quantity | Fixed | Varied |
| --- | --- | --- |
| Read positions per text source | 497,792 | 518,016 |
| Read positions across four text sources | 1,991,168 | 2,072,064 |
| Records per image, speech, sensor or shogi source | 7,778 | 8,094 |
| Native transitions, including repeats | 7,778 | 8,094 |
| Complete BoolQ cases | 3,889 | 4,047 |

The fixed condition reads 7.778 times the preceding run's positions per text source and
records per original nontext source. The varied condition reads 8.094 times those
preceding-run quantities. Its four text sources receive 259,072 original next-token
targets each, plus 10,833 / 10,712 / 12,635 / 14,022 added answer tokens for TinyStories
/ WikiText-2 / Shakespeare / conversations respectively. For varied text, original
next-token supervision occurs on only half the updates; additional answer-token counts
are recorded separately in the results. All training populations are available without a
permanent sample cap. This budget does not exhaust the large text, image or shogi
populations.

| Training population | Available records or transitions | Distinct used: fixed | Distinct used: varied |
| --- | --- | --- | --- |
| MNIST | 60,000 | 7,508 | 7,806 |
| Fashion-MNIST | 60,000 | 7,539 | 7,834 |
| CIFAR-10 | 50,000 | 7,480 | 7,768 |
| FSDD | 2,000 | 2,000 | 2,000 |
| UCI HAR | 6,234 | 5,340 | 5,452 |
| Native experience | 6,144 | 6,144 | 6,144 |

FSDD uses 2,000 training recordings from four speakers, with 500 recordings from a
different speaker for development and another 500 reserved for testing. UCI HAR uses
6,234 training, 1,118 development and 2,947 test windows with subject separation, each
containing 128 samples of nine preprocessed inertial channels. BoolQ retains all 9,427
labeled training and 3,270 development cases. See [data provenance and
licenses](datasets.md#additional-data-for-question-learning). No model score is computed
on the reserved FSDD or UCI test sets.

## Original Objectives

Classification scores use a fixed development panel of 128 anchors and their partners,
giving 256 correlated predictions. Text and shogi use 128 locations; native scores cover
all 384 transitions of 64 development worlds. These are panel measurements rather than
full benchmark test scores. BoolQ content scores ignore letter case and one final
period; text losses are token cross-entropy.

| Source | Metric | Before | Fixed after | Varied after |
| --- | --- | --- | --- | --- |
| TinyStories | Token NLL (lower) | 9.5712 | 1.7641 | 1.9187 |
| WikiText-2 | Token NLL (lower) | 10.4029 | 3.5045 | 3.6582 |
| Shakespeare | Token NLL (lower) | 11.5395 | 3.8410 | 4.0090 |
| Conversations | Token NLL (lower) | 3.8909 | 2.6018 | 3.0072 |
| MNIST | Classification accuracy (higher) | 14.06% | 94.14% | 89.06% |
| Fashion-MNIST | Classification accuracy (higher) | 9.77% | 81.64% | 73.44% |
| CIFAR-10 | Classification accuracy (higher) | 8.20% | 33.98% | 23.44% |
| Shogi | Move agreement (higher) | 2.34% | 12.50% | 10.94% |
| Native experience | Recorded-action accuracy (higher) | 24.74% | 58.07% | 57.55% |
| FSDD | Classification accuracy (higher) | 14.45% | 8.98% | 12.11% |
| UCI HAR | Classification accuracy (higher) | 11.72% | 88.67% | 86.33% |
| BoolQ | Answer content accuracy (higher) | 67.97% | 64.84% | 60.16% |
| Shogi | Value MSE, 118 labeled locations (lower) | 1.2656 | 1.8597 | 1.2890 |
| Native experience | Macro average of five action accuracies | 21.10% | 51.56% | 53.07% |

All twelve original objective losses decrease relative to initialization, but
original-task performance generally favors fixed objectives. Speech classification
remains weak on the held-out speaker in both conditions. Lower shogi policy loss does
not prevent worse value error: fixed value MSE increases from 1.2656 to 1.8597. Native
action accuracy exceeds the training-majority baseline of 21.61%, but this remains
recorded-data prediction. Common evaluations at 1,000, 2,000 and 3,000 updates retain
the same sampling-prefix comparison; for example, at 3,000 updates MNIST accuracy is
91.80% fixed versus 85.16% varied.

## Answering Different Questions About The Same Input

Only wording 0 is used for training. Wording 1 is a held-out paraphrase of the same
task, not an unseen task or evidence of novelty relative to pretraining. The following
tables use greedy generated answers, not intermediate teacher-forced scores. First/last
extraction preserves punctuation exactly. The paired extraction analysis was added after
the first interim results.

| Same-input first + last word | Before | Fixed after | Varied after | Best constant answer pair |
| --- | --- | --- | --- | --- |
| TinyStories | 1/32 (3.12%) | 0/32 (0.00%) | 26/32 (81.25%) | 1/32 |
| WikiText-2 | 1/32 (3.12%) | 0/32 (0.00%) | 29/32 (90.62%) | 1/32 |
| Shakespeare | 0/32 (0.00%) | 0/32 (0.00%) | 31/32 (96.88%) | 1/32 |
| Conversations | 0/32 (0.00%) | 0/32 (0.00%) | 21/32 (65.62%) | 1/32 |

When the two target words differ, varied answers both correctly on 106/127 pairs; fixed
answers none correctly. This supports a narrow learned distinction between these two
extraction requests on held-out excerpts and paraphrases. All four infilling scores
remain 0/32 exact matches for the held-out wording; the single-reference metric can
reject another plausible missing span.

Complementary questions share observations but have opposite yes/no targets. Each pair
is counted correct only if both answers are correct. The constant baseline picks one
relation for every input while changing its response with the question; it can score
well without using the observations.

| Complementary question pair | Before | Fixed after | Varied after | Best constant relation |
| --- | --- | --- | --- | --- |
| MNIST | 17/32 | 0/32 | 17/32 | 17/32 |
| Fashion-MNIST | 12/32 | 0/32 | 5/32 | 20/32 |
| CIFAR-10 | 5/32 | 0/32 | 14/32 | 18/32 |
| Shogi | 2/32 | 1/32 | 16/32 | 16/32 |
| FSDD | 0/32 | 0/32 | 0/32 | 21/32 |
| UCI HAR | 7/32 | 0/32 | 0/32 | 18/32 |
| BoolQ | 16/32 | 1/32 | 0/32 | 19/32 |

MNIST and shogi reach their constant-relation baselines; every other source falls below
its baseline. On the 16-case training-wording controls, varied image-pair scores are
9/16, 13/16 and 10/16 for MNIST, Fashion-MNIST and CIFAR-10, exactly the corresponding
constant-relation scores. Removing observations reduces each to 0/16, but this does not
establish relation understanding: removal also changes the input distribution and
response behavior. Held-out MNIST greater-than answers score 23/32, exactly the
always-no baseline; digit-sum answers score 1/32 for MNIST and 0/32 for speech.

| History question, 96 locations | Before | Fixed after | Varied after | Best constant answer |
| --- | --- | --- | --- | --- |
| First executed action | 15.62% | 0.00% | 26.04% | 26.04% |
| Most recent executed action | 16.67% | 0.00% | 37.50% | 21.88% |
| Sign of last reward | 63.54% | 0.00% | 69.79% | 63.54% |

Varied answers both first/last action questions correctly on only 11/96 histories, below
the best constant answer pair at 16/96. On the 49 histories where the two action targets
differ, joint correctness is 0/49. This does not demonstrate choosing a past action
according to the question. Reward-sign accuracy rises to 67/96, but the 16-case
observation-removal control scores 9/16 both with and without observations.

## Reconstruction And Forecasts

These numbers are mean squared error on exactly the same hidden targets. Image/audio
completions share output heads with native forecasting; fixed does not train the added
answer-query input. The sensor and shogi-successor output heads have no original
objective and are untrained references in fixed. Lower errors relative to these
references alone do not establish useful prediction. The simple baselines below use
visible pixels, the previous waveform chunk, persistence, or zero. Zero means black
pixels, silence, or training-channel means after sensor normalization. Baseline target
reconstruction was checked against the saved reconstruction evaluation rows.

| Added task, held-out wording | Fixed MSE | Varied MSE | Visible mean / prior chunk / persistence | Zero MSE |
| --- | --- | --- | --- | --- |
| MNIST | 0.241624 | 0.129845 | 0.163584 | 0.199099 |
| Fashion-MNIST | 0.180478 | 0.094454 | 0.134795 | 0.279508 |
| CIFAR-10 | 0.120996 | 0.063902 | 0.061212 | 0.277825 |
| FSDD | 0.297494 | 0.003721 | 0.001303 | 0.000367 |
| UCI HAR | 1.610812 | 0.602488 | 0.789194 | 0.872548 |

Varied beats both listed baselines for MNIST and Fashion-MNIST completion and UCI HAR
forecasting. It fails to beat the visible-mean CIFAR baseline. Speech-gap error is lower
than fixed but remains about ten times the silence baseline, so it is not successful
waveform completion by this comparison. On paired training-wording omission controls,
MNIST MSE changes from 0.121639 with observations to 0.143501 without; Fashion-MNIST
from 0.101081 to 0.113276; sensor forecasting from 0.656582 to 0.950741. These controls
support sensitivity to inputs while retaining the distribution-change limitation.
Activity-name answers additionally reach 24/32 on held-out wording and 12/16 versus 0/16
with/without sensor observations on the training wording.

| Native forecast, 384 transitions | Fixed MSE | Varied MSE | Persistence MSE | Silence MSE |
| --- | --- | --- | --- | --- |
| Next image | 0.054238 | 0.060828 | 0.051786 | — |
| Next audio | 0.031093 | 0.038162 | 0.065259 | 0.067364 |

| Shogi successor, held-out wording | Fixed | Varied | Copy previous state |
| --- | --- | --- | --- |
| All board squares | 2.01% | 60.26% | 97.96% |
| Occupied board squares | 3.49% | 7.97% | — |
| Pieces-in-hand counts | 0.00% | 77.01% | 97.10% |
| Entire board correct | 0.00% | 0.00% | — |

Both native audio forecasts beat persistence and silence. Both native image forecasts
fail to beat copying the current image. Shogi successor prediction also fails to
approach the copy baseline; high empty-square prevalence makes overall square accuracy
insufficient. On the 96-case native omission panel, varied recorded-action accuracy is
54.17% with complete inputs, 54.17% without images, 43.75% without audio and 48.96%
without text. This is evidence of sensitivity to audio/text availability, not a
demonstration of successful interaction or causal modality use in general.

## Language Retention And Reading

The same 92 probes are run before and after training: 90 strict questions and two
unscored explanations. The original ten strict probes remain, alongside eighty
arithmetic, list and conditional-rule questions. These probes are not training examples.
Strict correctness includes following the requested output format; it does not measure
all language ability.

| Probe score | Before | Fixed after | Varied after |
| --- | --- | --- | --- |
| All strict probes | 47/90 | 0/90 | 4/90 |
| Original strict probes | 3/10 | 0/10 | 1/10 |
| Additional strict probes | 44/80 | 0/80 | 3/80 |
| English | 23/45 | 0/45 | 3/45 |
| Japanese | 24/45 | 0/45 | 1/45 |
| Arithmetic: matching final numeral (post-hoc) | 30/44 | 20/44 | 3/44 |

Fixed answers 66/90 strict probes with `yes`. Some other fixed answers contain correct
arithmetic with unwanted explanation, so zero strict matches does not mean every
computed value is wrong: the final signed numeral matches on 20/44 arithmetic prompts.
Varied emits only one `yes`, but mostly replaces it with wrong numbers or class names.
For `Compute 7 + 8. Output only the number.`, fixed answers `yes` and varied answers
`7`; for the unscored request to explain why ice melts, varied answers `sneaker`. The
final-numeral diagnostic matches on only 3/44 varied arithmetic answers. The
general-response failure is therefore broader than output formatting.

BoolQ contains no identical train/development question–passage pair, but some passages
recur. Of the 128 evaluation cases, 93 have passages absent from the training population
and 35 use passages present there. Presence in the available population is distinct from
actually being consumed during this run.

| BoolQ population cohort | Cases | Before | Fixed after | Varied after |
| --- | --- | --- | --- | --- |
| Passage absent from training population | 93 | 65.59% | 65.59% | 56.99% |
| Passage present in training population | 35 | 74.29% | 62.86% | 68.57% |

| Condition | Passage consumption cohort | Cases | Before | After |
| --- | --- | --- | --- | --- |
| fixed | Not consumed | 111 | 66.67% | 65.77% |
| fixed | Consumed during this condition | 17 | 76.47% | 58.82% |
| varied | Not consumed | 109 | 66.06% | 58.72% |
| varied | Consumed during this condition | 19 | 78.95% | 68.42% |

Overall content accuracy declines from 67.97% to 64.84% fixed and 60.16% varied. The
always-yes answer scores 60.94% on this panel. Strict scoring alone would show an
apparent increase from 39.84%, mostly because capitalized/punctuated initial answers
become lowercase. Neither the overall score nor the small overlapping-passage cohorts
establish improved reading comprehension; no passage-removal control is included.

## Limits, Verification And Artifacts

This is one seed and a development comparison. Multiple questions are derived from
existing labels, records or deterministic rules. Observation omission also changes input
length and distribution. Good recorded-action accuracy is not successful interaction;
board agreement is not shogi playing strength. No new environment rollouts are
collected, and history questions receive the complete observed history, so they do not
establish compressed or long-term memory. Infilling uses one recorded target span and
exact agreement can reject another plausible completion.

The implementation passed 540 local unit tests and the remote setup checks. The final
models were loaded separately on CPU with all twelve data-source states and provenance
checked, finite full optimizer states restored, and all twelve next losses computed. Two
generated answers per model were compared with the saved GPU answers. This checks
loading and inference; it does not claim bitwise equivalence of CPU and CUDA training.
Every shared-core parameter tensor changed in both conditions.

Training revision: `3af7ae39201a8a7d6f7786fbb6394e37521ce636`. Remote hashes match 214
source/configuration files and four entrypoint/build files in the clean training
checkout. The analysis code is `scripts/summarize_question_learning.py`.

The A40 job took 9,137.5 seconds (2h32m17s), including setup, evaluations, output
retrieval and deletion. At the observed $0.49/hour pod rate this is about $1.244. Two
attempts stopped before the measured training conditions add 168.9 seconds: the first
lacked a transported setup-test dependency; the second was stopped to correct pair-label
scheduling. The combined GPU-time estimate is $1.267, excluding disk charges, not an
invoice. All three experiment pods were confirmed absent after retrieval. Peak monitored
GPU memory was 8,972 MiB; mean GPU utilization was 65.0% over the monitored remote
workload. Future sizing details are in [compute
costs](compute-costs.md#twelve-source-question-learning-sizing-reference).

Canonical artifacts are under `models/question-learning-20260911/`, outside `runs/`.
Each condition retains its final full checkpoint, optimizer/source states, tokenizer,
recipe, provenance, evaluation panel, individual responses, intermediate evaluation
reports and CPU verification. The root retains the comparison JSON, PNG/PDF figures,
resource/timing records, file hashes and verification evidence. Periodic checkpoint
weights are replaced; their update and evaluation records remain. No pre-existing model
or dataset was removed.

```sh
uv run python scripts/summarize_question_learning.py \
  --root models/question-learning-20260911 --data-root .
```

The retained figures are `original-learning-curves.png` and `question-pairs.png`, with
PDF counterparts. The immediate research boundary is preserving general instruction
responses while learning these added tasks, and requiring relation/history scores to
beat constant-answer controls. These results do not support treating all added question
forms as acquired capabilities.
