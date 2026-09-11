# Datasets

This document records datasets that are already supported or under discussion
for this project. It is not a complete downloader or ingestion guide.

Local artifact placement rules live in [artifact-layout.md](artifact-layout.md).

## Supported And Candidate Datasets

| Dataset | Modality | Approximate size | Status | Description |
| --- | --- | ---: | --- | --- |
| Tiny Shakespeare | text | about 1 MB | supported | A single small Shakespeare text corpus commonly used for toy language-model examples. |
| OpenAssistant OASST1 conversations | human text conversations | bounded local selection: 2,048 train / 128 validation | supported | Assistant rank-zero, undeleted en/ja chains up to 1,200 characters, partitioned by conversation-tree hash; conversation replay for the [single-core agent](language-agent.md). |
| WikiText-2 | text | about 2M tokens | supported | A small Wikipedia-derived language-modeling corpus with train, validation, and test splits. |
| WikiText-103 | text | about 103M tokens | candidate | A larger Wikipedia-derived language-modeling corpus built from full articles. |
| TinyStories | synthetic English text | over 2M stories | supported | A synthetic corpus of short English stories written with simple vocabulary and grammar. |
| Project Gutenberg | text | main mirror about 2.7 TiB | candidate | A public-domain ebook corpus. Use selected raw texts first; do not mirror the full collection without a concrete need. |
| OpenWebText | text | about 8M documents / 40 GB text | candidate | An open reproduction of GPT-2-style WebText, collected from web pages linked by Reddit posts. |
| FineWeb-Edu | text | about 1.3T tokens | candidate | A filtered educational subset of FineWeb built from Common Crawl web pages. |
| FineWeb-Edu score-2 | text | about 5.4T tokens | candidate | A broader FineWeb-Edu variant using a lower educational-score threshold. |
| MNIST | image | 70k images | supported | A grayscale handwritten digit image dataset with 10 classes. |
| Fashion-MNIST | image | 70k images | supported | A grayscale fashion-product image dataset with 10 classes. |
| CIFAR-10 | image | 60k images | supported | A color natural-image dataset with 10 classes. |
| Mini-ImageNet | image | about 65k images | candidate | A smaller ImageNet-derived image classification dataset commonly organized around 100 classes. |
| Food-101 | image | about 101k images | candidate | A food image classification dataset with 101 classes. |
| Places365 | image | large | candidate | A scene recognition dataset with 365 place categories. |
| ImageNet-1K | image | about 1.28M train images | candidate | A large object image classification dataset with 1000 classes. |
| iNaturalist 2021 | image | large | candidate | A fine-grained species image dataset with many biological categories. |
| Qhapaq computer shogi KIF records | game / shogi | 39,740 games | supported | Computer-shogi game records from Qhapaq Research Lab, downloaded as KIF archives and converted to local source-record JSONL for shogi move-choice experiments. |
| Synthetic cellular rule episodes | binary grids / before-after examples | generated stream | experimental | Multiple outer-totalistic rules, with disjoint training/validation/test rule identities; [generation and evaluation protocol](research/cellular-rule-inference.md), [noisy outputs and chronological rule changes](research/cellular-rule-stress.md). |
| Synthetic cellular control tasks | initial grids / independent goal grids / executed interventions | generated trials | experimental | Nine tasks per unfamiliar rule; disjoint one-flip candidate pools, only executed transitions carried forward; [control protocol](research/cellular-rule-control.md). |
| Multimodal navigation episodes | text / RGB images / PCM audio / actions / feedback | generated episodes | experimental | Text identifies a target, sound identifies control mode, and images show the world; recurrent memory connects successive observations. Native PNG/WAV media, explicit episode/world splits and replay selections; [integrated agent](multimodal-agent.md). |

## Notes

| Area | Note |
| --- | --- |
| Local artifacts | Downloaded datasets, generated samples, run metrics, and checkpoints are usually local artifacts under paths such as `data/` or `runs/`. |
| Large text data | Large datasets can be used through streaming or fixed-size slices before deciding whether full local copies are needed. |
| Evidence level | Tiny or toy datasets are useful for quick checks, but larger or problem-specific datasets are needed for stronger evaluation claims. |
| WikiText-2 raw data | Local raw data is `data/wikitext-2/raw/wiki.train.raw.txt`, `data/wikitext-2/raw/wiki.valid.raw.txt`, and `data/wikitext-2/raw/wiki.test.raw.txt` from `Salesforce/wikitext` on Hugging Face. |
| TinyStories raw data | Local raw data is `data/tinystories/raw/TinyStoriesV2-GPT4-train.txt` and `data/tinystories/raw/TinyStoriesV2-GPT4-valid.txt` from `roneneldan/TinyStories` on Hugging Face. |
| Qhapaq raw data | Local raw data under `data/qhapaq/raw/results/` contains the fetched `kifdownload` result CSVs. Local raw KIF archives under `data/qhapaq/raw/kiffiles/` contain every currently downloadable `.7z` link found on the source page; unavailable links are recorded in the local manifest. Source pages include `https://www.qhapaq.org/shogi/kifdb/` and `https://www.qhapaq.org/shogi/`. |
| Qhapaq processed data | The local source-derived records are `data/qhapaq/processed/qhapaq_games.jsonl`; train/eval splits belong in Data Selection or fixed training data bundles, not in `processed/`. Regenerate them with `scripts/prepare_qhapaq_shogi_records.py` after raw KIF archive refreshes. |

## Conversation Replay

The [instruction retention pilot](instruction-retention.md) uses a separate
corpus under `data/instruction-retention-20260911/`. It retains all 40,636
training and 4,686 validation branches from the complete joint-learning OASST
export, and adds 5,618 unique programmatically authored training instructions
(2,809 English and 2,809 Japanese). All three conditions use the same shuffled
46,254-branch training file and the same validation file. Original tree splits
are preserved. The added examples cover arithmetic, case conversion, lookup,
extraction, conditions, summaries, and a small set of explanations; their
limited template diversity is explicitly recorded. Sixty fresh bilingual
evaluation prompts have separate wording and values and do not occur exactly
among original or generated training user turns. Provenance records file
identities and counts, and `OASST-LICENSE` preserves the original license.

`scripts/prepare_agent_conversations.py` prepares a bounded selection from
[OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1),
revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`, with its Apache-2.0 license
and an archive checksum. Local files are under
`data/language/agent-conversations-oasst1-20260911/`: the source archive,
`train.jsonl`, `validation.jsonl`, `LICENSE` and `provenance.json`.

Selection keeps undeleted rank-zero assistant chains in English or Japanese,
up to 1,200 characters per conversation. Conversation-tree identity determines
the split, so related branches cannot appear on both sides. The stored source
records retain role, content, message identity, tree identity and provenance.
The language-agent checkpoint records the selected file hash and identities.
The single-core correction reuses these human conversations directly; it does
not distill answers from Qwen or load pretrained language weights.

## Preparation Entrypoints

Large Hugging Face text datasets can be sampled into a local text corpus before
training:

```sh
python -m intrep.sources.language.prepare_hf_text_slice \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --output-path data/external/fineweb_edu_sample.txt \
  --max-bytes 1000000
```

CIFAR-10 python batches and IDX image datasets can be converted into local JSONL
records with:

```text
intrep.problems.image_classification.dataset_builders
intrep.problems.image_text_answer.dataset_builders
intrep.problems.image_text_choice.dataset_builders
```

## Single-Core Text Pretraining

`scripts/prepare_agent_pretraining.py` builds a bounded byte stream from the
existing `data/tinystories/raw/TinyStoriesV2-GPT4-{train,valid}.txt` files.
The [official dataset card](https://huggingface.co/datasets/roneneldan/TinyStories)
identifies TinyStories as English and licenses the dataset under
CDLA-Sharing-1.0. Its stories were generated with language models; this work uses
the text as training data and does not load those models' weights.

The current selection under `data/language/single-core-pretraining-20260911/`
contains 32,768 training documents and 256 validation documents. Exact document
hashes are disjoint. `tokens.npz` contains byte values with EOS markers only at
document boundaries; `provenance.json` records hashes, counts and source
filenames. `validation-documents.json` retains the held-out prefixes for
inspection. The stream checksum identifies the exact prepared data even though
the original local download did not record a Hub revision. See
[language learning](language-learning.md) for training and evaluation.

## Complete Populations For Joint LFM Learning

`configs/joint-lfm.json` declares the populations for the
[exchangeable-head joint learner](shared-prediction.md). All nine sources
contribute to every optimizer update. Mini-batches advance through their full
training files; the recipe has no permanent small-sample limits. This does not
mean that a short execution check has already traversed those populations.

| Source | Training population | Development evaluation |
| --- | --- | --- |
| TinyStories | Entire raw GPT4 training file, 2,227,753,162 bytes | Entire raw validation file |
| WikiText-2 | Entire `wiki.train.raw.txt` | `wiki.valid.raw.txt` |
| Tiny Shakespeare | Bytes `[0, 1003856)` of the original file | `[1003856, 1115394)`; a disjoint approximately 10% suffix, aligned to a line boundary |
| OASST1 | 40,636 complete branches covering 78,351 distinct usable messages | 4,686 branches covering 8,934 messages; disjoint tree hashes |
| MNIST / Fashion-MNIST | All 60,000 training images each | Official 10,000-image test files, used here for development measurements |
| CIFAR-10 | All five official training batches, 50,000 images | Official test batch, used here for development measurements |
| Qhapaq | All 4,951,012 examples in `qhapaq-full/train-examples.jsonl` | Its existing game-disjoint `eval-examples.jsonl` |
| Native navigation | Every transition in all 1,024 training episodes of `data/multimodal-navigation-20260910/selection.json` | Its 64 validation episodes; 128 test episodes remain separate |

`scripts/prepare_joint_conversations.py` exports the previously downloaded,
pinned OASST1 archive to `data/language/joint-conversations-oasst1-20260911/`.
It removes the earlier replay selection's count, language, rank and length
limits. Of 88,838 archived messages, 87,285 are usable; 1,553 deleted messages
are excluded. Every usable message is represented in a complete root-to-leaf
branch. Shared ancestors recur in their branches. Deleted/empty ancestors make
a branch unusable, rather than being silently replaced by invented content.
The archive hash, output counts, source revision and license are recorded.

Text training covers all conversation roles. Training does not read validation
or test targets. The image test sets listed above are development data once
used for before/after measurements; they are not an untouched final test.
Archived duplicates and the older bounded text selections are not concatenated
again into this recipe. New data sources are added explicitly while retaining
the existing populations and training cursors.

## Additional Data For Question Learning

`scripts/prepare_question_datasets.py` downloads complete source releases into
`data/question-learning-20260911/`, retaining source archives, provenance and
checksums. `configs/question-learning.json` adds these three sources to the nine
complete populations above. See [question learning](question-learning.md) for
the questions and evaluation controls. Split counts describe available records,
not how many records a particular training run has consumed.

| Dataset | Training | Development | Separate test | Source and license |
| --- | ---: | ---: | ---: | --- |
| Free Spoken Digit Dataset (FSDD) | 2,000 recordings | 500 | 500 | [Official repository](https://github.com/Jakobovski/free-spoken-digit-dataset), CC BY-SA 4.0 |
| UCI Human Activity Recognition (HAR) | 6,234 windows | 1,118 | 2,947 | [Official dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), CC BY 4.0 |
| BoolQ | 9,427 passage/question pairs | 3,270 | Hidden labels | [Official repository](https://github.com/google-research-datasets/boolean-questions), CC BY-SA 3.0 |

FSDD is pinned to commit `26eb9aaf76e81b692f806f9140c2d2777410d7a1`.
All 3,000 full-length 8 kHz PCM recordings are preserved. Four speakers are used
for training and one each for development and test. This custom speaker-disjoint
split differs from the repository's recording-index split.

HAR uses all nine released inertial signal channels and all 128 samples in each
window. These are the source's preprocessed signals, not the 561 engineered
features or unfiltered raw sensor measurements. Three subjects from the official
training partition are held out with seed 9047. The official test partition is
retained. Subject identities are disjoint across all three partitions; channel
normalization is fitted using the resulting 6,234 training windows only.

BoolQ is downloaded from the official [SuperGLUE v2 archive](https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip).
It preserves complete questions and passages, mapping the source `label` to
`answer`. No identical question/passage pair occurs across train and development,
but 720 distinct passages occur in both. Evaluation therefore separates novel
passages from new questions about training passages. The archive's unlabeled
test records are retained in the archive and excluded from supervised use.

```sh
uv run python scripts/prepare_question_datasets.py --output data/question-learning-20260911
```
