# Existing Language-Training Methods And Reusable Artifacts

The current shortlist is the [six-month research review](recent-language-world-action-20260911.md).
This document is retained as background; it is not a survey of the latest work.

Reviewed on 2026-09-11 against implementation commit `8b6afee`. This review
replaces the proposal to simply increase the current prototype's training
budget. The useful research question is how language and native experience can
share computation; basic language-model training should start from established
methods and available implementations.

The [previous experiment](../language-learning.md) remains a valid measurement
of that implementation. It is not a reproduction of a published language model.
No further training or architecture replacement was performed for this review.

## Findings From Primary Sources

| Source | Evidence relevant to this project | Consequence and limit |
| --- | --- | --- |
| [TinyStories, 2023](https://arxiv.org/html/2305.07759v2), sections 1, 3 and 6 | Uses GPT-Neo, a reported 512-token context and 256-token local window; footnote 2 describes keeping the 10K most common tokenizer tokens. Evaluation separates grammar and consistency, with approximately 50 prompts and repeated sampling. | Its small-model results do not validate a plain byte model with 384-byte blocks. Use its released models as reference artifacts and its evaluation dimensions, without assuming equal parameter counts imply comparable conditions. |
| [SmolLM2, 2025](https://arxiv.org/html/2502.02737v1), section 6 | The 135M and 360M models use 2T and 4T training tokens respectively. Unlike the 1.7B version, the smaller models use a single-stage data mixture, WSD scheduling, and simplified instruction data. | Small models have their own published recipes. Do not transfer either the 1.7B recipe or its capabilities to a 5M model. Published exposure counts are context, not a prescription to reproduce trillions of tokens. |
| [SpaceByte, 2024](https://arxiv.org/html/2404.14408v2), sections 1, 4 and 5 | Controls training and inference compute; plain byte Transformers can need roughly ten times the training FLOPs of subword models in the tested settings. Multiscale, boundary-aligned computation improves the comparison. | Tokenization and effective text context are first-order design choices. The reported multiplier is not an estimate for our model. Its experiments cover English books, papers and code; they do not establish Japanese or native-world performance. |
| [ByT5, 2022](https://aclanthology.org/2022.tacl-1.17/) | Demonstrates pretrained byte-to-byte models and studies sequence-length and computational tradeoffs. | It supports the feasibility of bytes, but its encoder-decoder design does not validate our autoregressive architecture or training recipe. |
| [BLT, 2024 / ACL 2025](https://arxiv.org/abs/2412.09871) | Uses dynamic byte patches, allocating latent computation according to next-byte entropy. | This is an architectural solution to byte efficiency, not evidence that replacing a subword vocabulary with bytes is sufficient. Its scale and additional components need separate assessment. |
| [On Layer Normalization, ICML 2020](https://proceedings.mlr.press/v119/xiong20b.html) | Analyzes the effect of Post-LN versus Pre-LN on initialization gradients and warmup requirements. | Our core is Post-LN. Modern decoder learning rates cannot be copied without checking normalization and initialization. This paper does not establish that normalization caused our failure. |
| [Gato, 2022](https://arxiv.org/html/2205.06175v3), sections 2 and appendices C-D | One shared network models text and actions with masked next-token loss. Text uses 32K SentencePiece units; training mixes fixed offline sequences. Image and nontext observation targets are masked. | Joint text/action learning and fixed-dataset training have prior art. Gato does not implement our image/audio forecasts or TD objective, and its reported agent performance must not be transferred to this project. |
| [GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html) and [PCGrad, NeurIPS 2020](https://arxiv.org/abs/2001.06782) | Address gradient magnitude imbalance and conflicting gradient directions respectively in multitask learning. | Loss values alone do not reveal which task dominates shared updates. Measure norms and directions before selecting a balancing method; neither method is established here as the solution. |

## Reusable Implementations And Checkpoints

- **Small language-model recipe:** the authors' [SmolLM pretraining code and
  configurations](https://github.com/huggingface/smollm/tree/main/text/pretraining)
  use Nanotron and include continued-pretraining instructions. The
  [135M configuration](https://github.com/huggingface/smollm/blob/main/text/pretraining/smollm2/config_smollm2_135M.yaml)
  specifies 2,048-token sequences, tied embeddings, GQA, learning rate 0.003,
  2,000 warmup steps, a final 20% linear decay, AdamW betas 0.9/0.95 and weight
  decay 0.01. These settings form a coherent published reference; mixing them
  selectively with our different architecture would still be a new experiment.
- **Byte/subword implementation comparison:** the author's
  [SpaceByte repository](https://github.com/kjslag/spacebyte) includes plain
  Transformer, MegaByte and SpaceByte implementations. Its
  [reproduction instructions](https://github.com/kjslag/spacebyte/blob/main/reproduce/README.md)
  cover data preparation, tokenizers, training and comparison plots. This is a
  better reference than inventing another byte-model training scaffold. The
  full published runs are much larger than our pilot and are not a proposed
  immediate compute expenditure.
- **Released language baselines:** [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M)
  and [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) are author
  releases. The first is a narrow English story baseline; the latter is a
  broader small language model. Availability of weights does not establish
  Japanese suitability or compatibility with the existing native-memory API.
- **Byte-model resources:** [ByT5](https://github.com/google-research/byt5)
  publishes model resources. [BLT's official code](https://github.com/facebookresearch/blt)
  publishes weight-loading and training paths; the documented loading example
  includes a separate entropy model. Therefore it is not an automatic fit for
  a strict single-core implementation.

Released configurations also need inspection. The TinyStories paper's footnote
and the public checkpoint metadata are not interchangeable: the
[1M configuration](https://huggingface.co/roneneldan/TinyStories-1M/blob/main/config.json)
and [33M configuration](https://huggingface.co/roneneldan/TinyStories-33M/blob/main/config.json)
both declare vocabulary size 50,257 and position capacity 2,048. The latter has
hidden width 768 and four layers. Position capacity does not establish the
training context length. Model names are not sufficient to establish total
parameter counts; count the actual tensors before making size comparisons.

## What Differs In This Implementation

The following are code observations, not conclusions from the papers:

| Component | Current implementation | Required comparison with a chosen reference |
| --- | --- | --- |
| Text units | UTF-8 bytes plus special symbols | Vocabulary, bytes represented per sequence, embedding/output parameter cost |
| Pretraining context | 384 target tokens per sampled block, with 32 memory tokens | Effective text span and document-boundary treatment, not token count alone |
| Transformer | Six Post-LN layers, width 256, feed-forward width 1,024, eight heads | Normalization, initialization, positions, attention, output normalization and tied embeddings |
| Conditioning | Initial memory in text pretraining; observation-derived memory plus explicit text prefix for chat | Whether inference and training follow the reference model's conditioning distribution |
| Objectives | Text CE plus separately scaled native action, image, audio, feedback and TD losses | Per-objective gradients, masks, normalization, data proportions and effective update frequency |
| Evaluation | Four story prefixes; 16 conversation-loss cases; eight native worlds | Reference generation protocol, held-out text scoring, and actual environment returns |

Relevant code: [shared core](../../src/intrep/representation/cores/transformer.py),
[text/memory assembly](../../src/intrep/representation/assemblies/multimodal_agent.py),
[language paths](../../src/intrep/representation/assemblies/language_agent.py).

A numerical byte CE cannot be compared directly with a subword CE. A common
raw-text evaluation can report total negative log-likelihood divided by actual
UTF-8 bytes and by ln(2), with boundary and special-token conventions stated.
The existing metric includes end tokens, so simply relabeling it as a published
bits-per-byte metric would be incorrect.

## Revised Next Step

1. Select an established language implementation and its complete configuration
   as the reference. Keep its tokenizer, preprocessing and generation behavior
   together. Where an author checkpoint exists, evaluate it directly instead
   of spending training compute to rediscover its demonstrated capability.
2. Make the project's intended deviation explicit: one shared core for language
   and native experience. Establish the required interfaces and which weights
   can be reused before changing the architecture. A separate pretrained
   dialogue model attached to the old native predictor does not satisfy this.
3. Use Gato as a concrete reference for shared sequence processing and masked
   text/action supervision. Identify additional research work for recurrent
   memory, sensory prediction and learning from actual feedback; those are not
   automatically solved by adopting Gato's recipe.
4. Limit subsequent experiments to these deviations from the reference. Use
   published data/evaluation tooling and matched controls. Measure task-gradient
   interaction if shared training changes behavior, rather than starting with
   an arbitrary replay ratio or an immediate new balancing algorithm.

Using an existing model's weights as the shared core and reusing only its
methods/code are distinct design choices. This review establishes resources
for both; the current core was not replaced during the review. The
paper comparison alone cannot identify the cause of the pilot's malformed
language or guarantee that longer training, a tokenizer change, or replay will
fix it.
