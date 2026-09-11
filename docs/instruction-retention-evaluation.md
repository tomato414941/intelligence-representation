# Instruction Retention: Context, Assistant Targets, And Source Weight

2026-09-11. Neither assistant-only targets with conversation context nor an
eightfold conversation source weight preserved instruction responses in this
short pilot. The same initial score of 47/90 fell to 8/90 with the old chunked
method, 7/90 with assistant supervision, and 10/90 with increased conversation
weight. Some extraction and native/sensor objectives improved, but those gains
did not amount to broad capability retention.

The [protocol](instruction-retention.md) follows the
[twelve-source question-learning experiment](question-learning-evaluation.md),
which reduced instruction correctness from 47/90 to 4/90. The previous 90
scored prompts now serve as development data. A new 60-prompt bilingual panel
is evaluated only before and after each run.

## Controlled Comparison And Actual Exposure

Every condition starts from the same pinned LFM2.5-350M and attached-head
initialization: 358,228,139 parameters, all trainable, with one shared body.
FP32 AdamW uses learning rate `1e-5`, gradient clipping at one, seed 47 and
300 joint updates. Each update includes all twelve sources. Original and added
question objectives alternate. There is no teacher model, separate language
model, frozen body, or LoRA adapter in this training assembly.

| Condition | Conversation supervision | Conversation source weight |
| --- | --- | ---: |
| A: `chunked` | 128-token stream; all roles supply next-token targets | 1 |
| B: `assistant` | Conversation context; only assistant spans supply targets | 1 |
| C: `assistant_weighted` | Same reader and targets as B | 8 |

Assistant context windows contain at most 2,048 tokens, with 1,024-token overlap
for long branches. Overlap supplies context without repeating assistant target
positions. Assistant turn endings are supervised. User and system tokens remain
in the computation graph as context. Half of conversation updates instead use
the existing excerpt questions; those updates supervise the excerpt answers.
Consequently, retaining every assistant label in the reader does not mean every
label receives the original conversation objective during one traversal.

All other source weights remain one. Conversation loss receives coefficient
1/12 in A/B and 8/19 in C. The larger coefficient applies to the whole
conversation source, including its excerpt questions. It is not an eightfold
increase solely in general instruction replay or a measured share of all body
gradient magnitudes.

All conditions share 40,636 original OASST training branches plus 5,618 unique
programmatically authored instructions, evenly split between English and
Japanese. All 4,686 original validation branches remain available. Original
training languages, ranks and lengths remain included. The additions cover
arithmetic, letter case, extraction, lookup, conditions, summaries and a few
explanations, using narrow templates. No new model generates those targets.
The [data record](datasets.md) and artifact provenance retain original identities,
licenses, counts and hashes. A uses this same expanded corpus, so it repeats the
old method rather than exactly repeating the preceding run's data mixture.

| Conversation exposure | A | B / C |
| --- | ---: | ---: |
| Branches read | 104 | 298 |
| Stream positions read, excluding overlap | 38,400 | 99,533 |
| Tokens in returned records, including overlap | 38,700 | 101,309 |
| Original conversation prediction targets | 19,200, all roles | 39,782, assistant only |
| Added excerpt-answer targets | 1,651 | 1,587 |

Each other text source reads 38,400 positions and supplies 19,200 original
next-token targets. Added answers contribute 815 / 828 / 974 tokens for
TinyStories / WikiText-2 / Shakespeare. Each image, speech, sensor and shogi
source reads 600 records, including partners and possible repetitions. Native
experience supplies 600 distinct transitions from 100 episodes; BoolQ supplies
300 complete cases. The full source populations remain available, but this
short budget does not traverse them. B/C use identical reader progress for all
twelve sources. A has identical progress for the eleven non-conversation sources.

Thus A/B changes context, masking, and actual conversation exposure together;
it cannot identify a mask-only effect. B/C changes only the conversation source
weight. Updates are matched; tokens, wall time and FLOPs are not.

## Instruction Responses

The primary scores use actual greedy generated answers and strict string
matching after stripping surrounding whitespace. Retention is paired by exact
prompt and target identity; a newly correct answer does not replace a lost
answer in the retention count.

| Measurement | Initial | A | B | C |
| --- | ---: | ---: | ---: | ---: |
| Development correct, out of 90 | 47 | 8 | 7 | 10 |
| Initially correct development answers retained, out of 47 | 47 | 2 | 4 | 6 |
| Newly correct development answers | 0 | 6 | 3 | 4 |
| Development English / Japanese, each out of 45 | 23 / 24 | 2 / 6 | 1 / 6 | 3 / 7 |
| Fresh held-out correct, out of 60 | 17 | 2 | 15 | 10 |
| Initially correct held-out answers retained, out of 17 | 17 | 2 | 7 | 4 |
| Newly correct held-out answers | 0 | 0 | 8 | 6 |
| Held-out English / Japanese, each out of 30 | 10 / 7 | 2 / 0 | 6 / 9 | 5 / 5 |

B's 15/60 is close to the initial 17/60 only as an aggregate: it loses ten
initially correct answers and acquires eight others. C loses thirteen and
acquires six. Both reach 10/10 on fresh extraction prompts, and extraction
accounts for every correct fresh answer in C.

| Fresh family, ten prompts each | Initial | A | B | C |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic | 0 | 0 | 0 | 0 |
| Quantity | 0 | 0 | 0 | 0 |
| Letter case conversion | 5 | 0 | 0 | 0 |
| Conditional response | 4 | 2 | 2 | 0 |
| Extraction | 4 | 0 | 10 | 10 |
| Lookup | 4 | 0 | 3 | 0 |

The fresh panel has a low initial ceiling. Arithmetic and quantity begin at
zero strict matches, so those groups cannot measure further loss of strict
correctness. They were not replaced after seeing the baseline. These prompts
have different wording and values from the authored training templates and no
exact user-turn overlap with training; semantic overlap and pretrained exposure
remain possible. The English/Japanese variants and templates are correlated,
not sixty independent benchmark tasks.

The decline is not just answer formatting. Among 44 numeric development
questions, strict correctness falls from 26 to 2 / 1 / 0. A diagnostic accepting
a matching final numeral falls from 30 to the same 2 / 1 / 0. This diagnostic
does not establish semantic equivalence, but it cannot explain away the loss.
For example, `Compute 7 + 8. Output only the number.` changes from `15`
initially to `7` in all three final models.

| Update | A correct / 90 | B correct / 90 | C correct / 90 |
| --- | ---: | ---: | ---: |
| 0 | 47 | 47 | 47 |
| 50 | 18 | 7 | 12 |
| 100 | 18 | 16 | 13 |
| 150 | 6 | 4 | 5 |
| 200 | 11 | 28 | 14 |
| 250 | 12 | 12 | 1 |
| 300 | 8 | 7 | 10 |

The fluctuations make a selected intermediate peak a poor substitute for the
prespecified final comparison. No intermediate fresh-panel evaluation or early
stopping based on low development scores was used. The standalone
`instruction-retention.png` and `.pdf` plots show the development curves and
separate retained from newly correct fresh answers.

## Other Tasks And Gradient Diagnostics

| Original objective | Metric | Initial | A | B | C |
| --- | --- | ---: | ---: | ---: | ---: |
| TinyStories | Token NLL, lower | 9.2649 | 2.2421 | 2.6234 | 2.8049 |
| WikiText-2 | Token NLL, lower | 10.3478 | 3.8191 | 3.8457 | 4.1854 |
| Shakespeare | Token NLL, lower | 11.5217 | 4.5685 | 4.5253 | 4.5491 |
| MNIST | Classification accuracy | 4.69% | 14.06% | 12.50% | 10.94% |
| Fashion-MNIST | Classification accuracy | 7.81% | 14.06% | 12.50% | 14.06% |
| CIFAR-10 | Classification accuracy | 6.25% | 14.06% | 14.06% | 4.69% |
| Shogi | Move agreement | 0.00% | 3.12% | 3.12% | 0.00% |
| Shogi | Value MSE, lower | 0.9642 | 0.9631 | 0.9609 | 0.9718 |
| Native experience | Recorded-action accuracy | 25.00% | 51.56% | 50.52% | 41.67% |
| FSDD | Classification accuracy | 0.00% | 7.81% | 7.81% | 9.38% |
| UCI HAR | Classification accuracy | 14.06% | 50.00% | 51.56% | 50.00% |
| BoolQ | Answer content accuracy | 65.62% | 65.62% | 34.38% | 65.62% |

Conversation NLL measures different targets across A/B and is excluded from
that common table. It changes from 3.9287 to 3.0959 for A's all-role objective,
and from 2.5516 to 2.3031 / 2.3561 for B/C's assistant objective. Lower text
losses coexist with poor instruction responses. Image/speech classification
and shogi remain weak after this short budget. Native recorded-action scores
exceed the training-majority baseline of 18.23%, while C trails A/B. BoolQ's
65.62% in A/C equals this panel's always-yes baseline.

| Same-excerpt first + last word, held-out wording | Initial | A | B | C | Best constant pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyStories | 0/8 | 3/8 | 5/8 | 4/8 | 1/8 |
| WikiText-2 | 0/8 | 6/8 | 8/8 | 5/8 | 1/8 |
| Shakespeare | 0/8 | 6/8 | 5/8 | 8/8 | 1/8 |

All pairs in this table have different target words. With the matching
assistant conversation reader, B/C score 5/8 and 4/8 on conversation extraction;
A uses different excerpts. These are narrow learned question distinctions.
All seven complementary yes/no pair scores remain 0/8 in all three conditions
(images, shogi, speech, sensor and BoolQ), below their constant-relation
baselines of 4/8 to 6/8. Native first/last action pairs score 0/24, 0/24 and
3/24, below the best constant pair's 4/24. On the fourteen histories with
different first/last actions, C gets one pair correct versus a constant pair's
four; A/B get none.

| Completion MSE, held-out wording, eight cases | A | B | C | Supplied simple baseline | Zero output |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST region | 0.17722 | 0.21273 | 0.20324 | 0.13973 | 0.17334 |
| Fashion-MNIST region | 0.13768 | 0.14451 | 0.13863 | 0.14375 | 0.32407 |
| CIFAR-10 region | 0.06645 | 0.06876 | 0.06233 | 0.04638 | 0.29962 |
| FSDD audio gap | 0.034406 | 0.128692 | 0.078229 | 0.000226 | 0.000108 |
| UCI HAR future | 0.98895 | 0.99273 | 1.04234 | 1.01616 | 0.92557 |

The supplied baseline fills images with the visible mean and uses the
preceding signal for audio/sensors. Target records and supplied baseline errors
match across both wordings. Fashion completion is near or slightly better than
its mean-fill baseline; the other tasks fail at least one simple baseline.
Native image prediction also trails copying the current image: MSE
0.06707 / 0.06709 / 0.06796 versus 0.05225. Native audio MSE is
0.06074 / 0.05999 / 0.08154, versus 0.06526 for persistence and 0.06736
for silence. Thus C's increased conversation weight does not deliver a clear
retention-and-new-task tradeoff advantage.

Original image/speech/sensor classification uses 32 fixed anchors and their
partners, giving 64 correlated predictions per source. Text, shogi and BoolQ
use 32 fixed cases. Native original objectives use 192 transitions from 32
validation worlds. Added questions use eight cases per source or 24 transitions
from four native worlds. Wording 1 is an untrained paraphrase of the same task;
it is not a new task or a guarantee against pretraining exposure.

First/last extraction counts a pair only when both generated answers match.
Complementary yes/no questions share observations and opposite targets. Their
constant-relation baseline changes its answer with the question while choosing
the same relation for every observation. These pilot panels are small and are
not full benchmark scores. There are no observation-removal or swapped-input
controls in this pilot, so these results do not establish causal use of the
observations. No reserved FSDD or UCI HAR test set is evaluated.

Gradient probes cover only the first and last body input projections, sampled
before global clipping at steps 1/2 and around every fiftieth update. Original
conversation and excerpt-question updates are summarized separately. The probes
do not measure whole-body gradient balance or establish a cause of forgetting.

| Mean weighted gradient norm on two selected projections | A | B | C |
| --- | ---: | ---: | ---: |
| Conversation | 0.652 | 0.347 | 1.499 |
| Native experience | 3.672 | 3.948 | 2.516 |
| TinyStories | 1.443 | 1.441 | 0.912 |
| BoolQ | 1.086 | 1.134 | 0.782 |
| MNIST | 0.619 | 0.615 | 0.410 |

This table uses the six probes during original-conversation updates. C raises
the observed conversation contribution, but the native contribution is still
larger on average. Native/conversation cosine means are -0.0003 / -0.0167 /
-0.0026; their sampled ranges are approximately -0.051 to +0.015 overall.
These near-zero cosines do not demonstrate a strongly opposing gradient
mechanism. Early large gradients also influence the means. Full per-step
norms and excerpt-update summaries remain in the artifacts.

This is one seed and 300 updates per condition. It establishes failure to
retain these instruction responses under this recipe, not that assistant
supervision or replay weighting can never work. The next useful controlled
change is a lower learning rate for the pretrained body than for the newly
attached input/output parameters, keeping all parameters trainable and
repeating retention and task checks. Simply extending C is not supported by
these measurements. No fourth training condition was run in this pilot.

## Artifacts And Verification

Training code is pinned to commit `2edb0d611605b2d60bf0c5b2ac71d6242799abbc`.
The artifact verification record compares 18 relevant remote source files to
that revision. All three initial parameter digests, initial instruction outputs,
panel locations and non-conversation initial evaluations match. The summarizer
also checks B/C recipe identity after normalizing the single weight change,
equal training exposure where intended, and scores against actual generated
strings. Fresh prompts appear only at steps 0 and 300.

The local artifact root is `reports/instruction-retention-20260911/`; it holds
raw generations, source panels, step logs, recipes, environment and timing
records, `comparison.json`, and standalone PNG/PDF plots. Model checkpoints
and their tokenizers reside in the project R2 archive under
`shared-prediction/instruction-retention-20260911/<condition>/`. Per-condition
`archive.json` records the checkpoint size, SHA-256 and restore command.

All 147 body parameter tensors changed in each condition. CPU restoration
matched the final evaluated parameter digest, restored 203 AdamW state entries
and all twelve source readers, and reproduced the two checked development
generations exactly. Each archive passed full byte comparison with zero
differences before its disposable working checkpoint was removed. The three
checkpoints total 12,906,737,137 bytes. Existing local models and datasets were
preserved.

The full local unit suite passed 546 tests in 107.2 seconds after the shared
training changes. The final analysis changes passed their two focused tests;
the complete summarizer also passed against all three collected runs. Its
PNG plot was visually checked. Verification records and the test log accompany
the artifacts; generated data, raw metrics and checkpoints are not committed.

The RunPod job completed successfully in 3,539.5 seconds and deleted its pod.
A separate account query confirmed that the experiment pod was absent. At the
observed A40 rate, the job represents approximately $0.482 of GPU time before
disk charges. [Cost notes](compute-costs.md#instruction-retention-and-remote-checkpoint-storage)
separate the 851.0 seconds of optimizer updates from evaluation and archive
overhead and estimate ongoing storage costs.
