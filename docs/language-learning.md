# Language Learning In The Single-Core Agent

This experiment diagnoses repetitive generation and applies established language
model training methods to the existing multimodal core. It does not load a
pretrained language model. Language and native tasks still update the same
5,095,054 parameters.

The current [six-month research review](research/recent-language-world-action-20260911.md)
prioritizes recent work and reusable capabilities. The earlier
[literature and implementation comparison](research/language-training-literature-20260911.md)
remains background.

## Research Used In The Implementation

| Primary source | Method applied here | Boundary of the evidence |
| --- | --- | --- |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | Causal next-token prediction on text before judging instruction behavior. | This prototype does not reproduce GPT-3's scale or capabilities. |
| [TinyStories](https://arxiv.org/abs/2305.07759) | A restricted English story domain to investigate language generation in a small model; inspect unseen-prefix completions. | The paper uses a different architecture and subword vocabulary. Its results are not this model's results. |
| [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) | Separate text pretraining from instruction tuning, and evaluate on held-out prompts. | Instruction examples do not substitute for a learned language foundation. |
| [LLaMA](https://arxiv.org/abs/2302.13971) | AdamW with betas 0.9/0.95, warmup, cosine decay to 10% of the peak rate, gradient clipping at 1.0, and explicit data-mixture accounting. | Peak rate, batch size and schedule length are chosen for this small continued-training run. No LLaMA weights or additional core are used. |
| [Chinchilla](https://arxiv.org/abs/2203.15556) | Record supervised token exposure and match the learning-rate schedule to the intended training horizon. | Its numerical scaling fit is not directly transferable to this 5M-parameter, byte-level, multimodal model. |
| [ByT5](https://aclanthology.org/2022.tacl-1.17/) | Retain raw UTF-8 bytes as a valid modeling choice while accounting for their longer sequences. | This is not a reproduction of ByT5's encoder-decoder architecture. Byte loss is not comparable to subword loss. |
| [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | Compare greedy decoding with nucleus sampling on the same checkpoint and prefixes; retain all generated samples. | Reduced repetition is not evidence of grammatical prose or instruction following. |
| [Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682) | Revisit past experience while updating a shared model, and measure retention. | Replay's usefulness depends on the setting and mixture; a buffer does not guarantee retention. |

The papers motivate procedures and controls. The measurements below, rather
than the papers' reported capabilities, determine what this implementation can
do. No external model is used to grade or supply answers during inference.

## What Replay Means Here

Text pretraining does not require a dynamic replay buffer. A fixed corpus and
minibatch sampler suffice. The text implementation samples contiguous blocks
from NumPy arrays. Conversation batches are sampled directly from the fixed
conversation list; the previous `ReplayBuffer` wrapper for conversations was
removed.

Native experience is separated into retained and actor-generated records for
sampling. The current trainer loads a snapshot of those records at startup.
Its `ReplayBuffer` containers are convenient grouping and sampling objects,
not a mathematical requirement: an equivalent dataset sampler can implement
the same replay. A bounded buffer becomes useful when new experience arrives
continually and storage or sampling priorities must be managed.

Replaying prior observations and actions addresses a different question from
how to read a static language corpus: whether learning new tasks changes old
behavior. This is measured rather than assumed.

## Repetition Diagnostic

Eight short, authored English/Japanese prompt-answer pairs were used only for
a memorization diagnostic. Both conditions began at the same single-core
checkpoint and sampled the same language batches. They used four conversations
per update, a peak rate of 0.0003, 20 warmup steps, AdamW betas 0.9/0.95, weight
decay 0.1, clipping at 1.0, and a cosine schedule with an 800-update horizon.
Evaluation every 100 updates stopped each run at complete reproduction.

| Condition | Updates | Teacher-forced byte accuracy | Exact generated training answers | Native validation decisions |
| --- | ---: | ---: | ---: | ---: |
| Initial checkpoint | 0 | 33.49% | 0/8 | 43/48 |
| Language only | 200 | 100% | 8/8 | 41/48 |
| Language plus native replay | 200 | 100% | 8/8 | 36/48 |

The replay condition included two complete native episodes per update, with
supervised native and forecast losses; this diagnostic did not add TD
bootstraps. It used more compute than the language-only condition. The native
score covers eight worlds with 48 decisions and one training seed. These results
show that the model can learn and generate short sequences through its existing
core. They do not establish general language ability or a general benefit or
harm from replay. In particular, the replay condition did not preserve native
behavior better in this sample.

The diagnostic weights are retained separately and are not used to initialize
the larger text experiment. Its authored examples are not added to the corpus
or instruction replay.

## Training And Generation Context Repair

A separate inspection found a real context mismatch: training retained up to
1,024 prefix bytes plus the current answer chunk, while inference immediately
cropped the whole sequence to 1,024 bytes. Once that limit was reached, the two
paths could condition on different text and position coordinates.

Generation now uses the same chunk boundaries as the answer loss: retain the
last 1,024 preceding bytes at the beginning of each answer chunk, then append
that chunk's bytes. The regression test compares batched teacher-forced logits
with token-by-token inference across boundaries. Short diagnostic answers did
not reach the old context limit, so this mismatch does not explain all of the
previous repetition.

## Text Pretraining And Instruction Tuning

The corpus uses the existing local TinyStories V2-GPT4 text files. These are
synthetic English stories, not human Japanese conversations. Preparation keeps
32,768 unique training documents and 256 validation documents, excluding exact
document duplicates across splits. It records document hashes and the token
archive checksum. The training stream contains 26,407,324 byte/end tokens; the
validation stream contains 199,933. Sources and license are in
[datasets](datasets.md#single-core-text-pretraining).

Pretraining predicts every token in a sampled block. Blocks can span document
boundaries marked by EOS. EOS is not inserted merely because a block ends.
Target text never enters an observation-memory encoder before prediction:
pretraining uses initial memory and causal byte prefixes. The loss averages
over supervised tokens. All computation still uses the original shared core,
text embeddings and output layer.

The measured schedule is:

1. Initialize from the single-core model, without diagnostic memorization weights.
2. Perform 4,000 text-pretraining updates, using 16 blocks of 384 tokens each,
   peak rate 0.0001, 100 warmup steps and cosine decay over 4,000 updates.
   Replay two native episodes every fourth update, including TD bootstraps.
3. Perform 1,000 instruction-tuning updates with two OpenAssistant conversations,
   two native episodes and 16 text blocks per update. Use a text-loss weight of
   0.25, peak rate 0.0001, 50 warmup steps and cosine decay over 1,000 updates.

This supplies 24,576,000 pretraining-token exposures in the first stage and
6,144,000 in the second. They are sampled exposures, not counts of unique tokens
seen. The instruction data retains its original 2,009 English / 39 Japanese
selection, so Japanese language coverage remains very limited.

Every stage evaluates identical held-out text blocks, four unseen story
prefixes, conversation prompts and the same native worlds. Text loss, generated
prose, instruction behavior and native retention are reported separately.
Checkpoint history includes the corpus checksum and training-document hashes.
Exact resume also restores the shared sampling RNG and the fixed schedule
horizon; initialization begins a new optimizer for the next stage.

## Measured Results (2026-09-11)

| Checkpoint | Held-out story byte/end CE | Held-out conversation loss | Native teacher-action matches |
| --- | ---: | ---: | ---: |
| Before continued training | 2.798587 | 2.905507 | 43/48 |
| After text pretraining | 1.449830 | 4.601790 | 43/48 |
| After instruction tuning | 1.542502 | 2.594386 | 42/48 |

Story loss averages eight fixed batches of eight 384-token blocks from the
validation stream (24,576 target tokens, sampled with seed 7001). Conversation
loss uses the first 16 conversations of the 128-record validation split; it is
the existing conversation-loss aggregation, not a pooled corpus-byte average.
Native accuracy compares actions against recorded teacher actions on eight
held-out worlds. It is not an online task-success rate. These are development
validation measurements from one run, not a fresh final benchmark.

The lower prediction losses did **not** produce working general conversation,
explanation, translation or code generation. Inspection of all eight held-out
conversation cases found repetition or malformed text instead of useful
responses. Japanese prompts still produce repeated characters. English story
completions now contain recognizable phrases but still have incorrect grammar,
misspelled words and repeated fragments. Instruction tuning also worsened the
story loss relative to the pretraining checkpoint. Replay did not eliminate
changes to native performance.

The diagnostic establishes that short sequences can be fitted. The larger run
establishes improved held-out next-byte prediction within its English domain.
Neither establishes that language learning is solved. Japanese coverage, token
exposure, model capacity and the current prompt-memory conditioning remain
possible limitations; this experiment does not isolate their causal effects.
A useful next controlled experiment would train the same core in a coherent
language domain to a validated generation baseline before expanding instruction
tasks, rather than inferring capability from loss alone.

The complete local artifacts are retained outside disposable `runs/`:

- `models/language-diagnostic-20260911/`: separate memorization controls.
- `models/language-pretraining-20260911/pretraining/checkpoint.pt`: text stage.
- `models/language-pretraining-20260911/instruction/checkpoint.pt`: final stage.
- `before/`, `after-pretraining/`, `after-instruction/` under that directory:
  unedited evaluation cases and metrics.
- `verification.json`: local checkpoint reload, data provenance, one shared
  transformer, all 100 changed model tensors, exact session restoration and
  actual environment feedback updating the same memory.

The diagnostic used an A40 for about 151 seconds. The staged run used an A40
for about 1,012 seconds including setup, evaluation, retrieval and deletion;
measured training time was 490 seconds plus 410 seconds. Its peak sampled GPU
memory was 1,808 MiB. At the quoted $0.49/hour, total elapsed-time compute is
approximately $0.16 for both runs, excluding storage and billing rounding.
Both pods were deleted and the subsequent pod list was empty.

Validation passed 25 focused tests and the complete 511-test suite, plus Ruff
and whitespace checks. The first final-suite attempt encountered nine
filesystem-full write errors; after clearing stale package-index cache entries,
the full suite passed using a temporary directory with available space. No
model or training-code change was needed for that environment failure.

## Decoding Control

The final checkpoint was also evaluated on the same four unseen 80-character
story prefixes with a 256-byte output budget. Greedy decoding was compared with
nucleus sampling at temperature 0.8 and top-p 0.9, using seeds 9001 through 9004.
All eight outputs are retained in `sampling-comparison.json` beside the final
checkpoints. The byte generator first excludes invalid UTF-8 continuations,
then applies the requested sampling rule. The default remains greedy.

Counting repeated word four-grams within each completion, greedy decoding had
43 repeated occurrences out of 221 (19.46%); nucleus sampling had 0 out of 206.
This tiny comparison demonstrates that decoding contributes to repeated phrases.
It does not fix the model's language: sampled text still includes malformed
words and sentences, for example `The wated the was for dind and toy.`
There is one sample per prefix, and no seed or example was selected for quality.

## Reproduction

```sh
uv run --inexact python scripts/prepare_agent_pretraining.py \
  --train data/tinystories/raw/TinyStoriesV2-GPT4-train.txt \
  --validation data/tinystories/raw/TinyStoriesV2-GPT4-valid.txt \
  --output data/language/single-core-pretraining-20260911

uv run --inexact python scripts/continue_language_training.py \
  --checkpoint models/single-core-agent-20260911/cycle/learning/checkpoint.pt \
  --corpus data/language/single-core-pretraining-20260911 \
  --selection data/multimodal-navigation-20260910/selection.json \
  --conversations data/language/agent-conversations-oasst1-20260911/train.jsonl \
  --validation-conversations data/language/agent-conversations-oasst1-20260911/validation.jsonl \
  --output runs/language-pretraining --device cuda
```

The corpus preparation command creates a new directory; reuse an existing
verified corpus without rerunning it. `scripts/diagnose_language_agent.py`
provides the separate bounded memorization/replay control. Its output directory
must also be new.
