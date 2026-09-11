# Recent Language, World And Action Research

Research window: **2026-03-11 through 2026-09-11**, checked on 2026-09-11.
The [broader shared-learning review](shared-learning-design-space-20260911.md)
is now the entry point for deciding the next project direction, including
architecture, CPU adaptation, memory and evaluation. The
[earlier review](language-training-literature-20260911.md) supplies historical
background and implementation comparisons, not a current-frontier shortlist.

The motivation is to start from demonstrated contemporary capabilities and
reuse existing work. A new paper's reported result, available implementation,
and suitability for this project's shared core are separate questions.

## Papers Inside The Window

Dates below are the first arXiv submission dates shown by the primary records.
An updated arXiv version or a 2026 conference label does not turn an older
initial release into a new paper. This is a targeted review, not an exhaustive
bibliographic survey or a claim that these papers establish universal rankings.

| Paper and initial date | Relevant finding | Reuse and project fit |
| --- | --- | --- |
| [WorldBagel](https://arxiv.org/abs/2607.03461), **2026-07-03** | Adapts pretrained BAGEL for language-conditioned actions and action-conditioned future observations. | Closely matches the joint prediction/control objective, but its Understanding and Generation experts have distinct projections. The paper promises code and checkpoints after acceptance; a runnable release was not verified. |
| [Small LLMs: Pruning vs. Training from Scratch](https://arxiv.org/abs/2606.14150), **2026-06-12** | In token-matched experiments with a Llama-3.1-8B architecture, pruned initialization beats random initialization under the same additional training budget. The advantage depends on pruning granularity, ratio and token accounting. | Directly relevant to avoiding repeated language pretraining. It does not demonstrate compressing modern language ability into our 5M model. [Author code](https://github.com/zlab-princeton/pruning-vs-scratch) supplies experiment scripts and links its pruning/training library. |
| [ARM](https://arxiv.org/abs/2606.11188), **2026-06-09** | Builds unified image understanding, generation and editing on autoregressive discrete representations, initializing the main model from Qwen2.5-7B. | Relevant to extending existing language capability into perception and generation. It does not establish action learning or online adaptation. The complete system also has substantial visual tokenizer/decoder components. The [project repository](https://github.com/wdrink/ARM) currently exposes README/assets, not a verified runnable training implementation. |
| [World-Language-Action Model (WLA)](https://arxiv.org/abs/2606.05979), **2026-06-04** | Jointly predicts textual intentions, future visual states and actions, starting from a pretrained vision-language backbone. | The [official repository](https://github.com/SJTU-DENG-Lab/WLA) contains model/training code, benchmark instructions and checkpoint links. However, it also uses separate World and Action Experts; it is not directly equivalent to the current single transformer core. |
| [Fast Byte Latent Transformer](https://arxiv.org/abs/2605.08044), **2026-05-08** | Adds block diffusion and self-speculation/verification variants to address slow byte-by-byte generation. | Useful if a byte-based design remains a requirement. Its evidence concerns generation efficiency/quality tradeoffs in trained BLT models, not a cure for our prototype's failure to learn coherent language. A release specific to these new variants was not verified. |

All performance claims above are the authors' findings. No model in this table
has been run locally for this review, and none is claimed to solve general
Japanese conversation plus native image/audio prediction and online learning.

## Details That Change The Engineering Decision

**Existing knowledge should be treated as an asset.** The pruning study
separates equal additional-token and equal total-pipeline-token comparisons;
coarse structured pruning can lose its advantage when scratch training gets
more tokens. Its authors provide a controlled test of reuse, not a universal
argument that pretrained initialization always wins. Our inference is that a
limited-budget project should evaluate available capable starting points before
paying to rediscover elementary language generation.
[Paper, sections 3-4 and limitations](https://arxiv.org/html/2606.14150v3).

**Unified does not necessarily mean identical computation for every modality.**
WLA-0 combines a RynnBrain-2B backbone, a SANA-based World Expert and an Action
Expert, totaling 3.4B parameters. Its default fast mode omits the World Expert
at inference. The paper's absence of *embodied* pretraining does not mean it
learns language from random weights. WorldBagel instead fine-tunes BAGEL's two
experts with action and future-frame objectives. These are useful integration
methods to study, but they are architectural differences to resolve rather
than overlook.
[WLA, sections 3-4](https://arxiv.org/html/2606.05979v1),
[WorldBagel, section 3](https://arxiv.org/html/2607.03461v1).

**Reusable implementation must be checked, not inferred from a project link.**
WLA's repository has `models/`, `train.py`, configs and benchmark training
instructions. ARM's repository tree contained only `README.md` and `assets/`
when inspected. The pruning repository links to its implementation library,
[llm-pruning-collection](https://github.com/zlab-princeton/llm-pruning-collection).
This makes their immediate reproduction status materially different. Repository
existence does not establish that all experiments and artifacts are reproducible.

## Date Audit And Exclusions

- Gato, TinyStories, SpaceByte, BLT, SmolLM2 and SmolLM3 remain background sources;
  their initial releases precede the window.
- UniVLA was submitted on **2025-06-24**; its ICLR 2026 label does not satisfy
  this review's recency condition. MobileLLM-R1 was submitted on
  **2025-09-29** and revised in February 2026; it is also outside the window.
  [UniVLA record](https://arxiv.org/abs/2506.19850),
  [MobileLLM-R1 record](https://arxiv.org/abs/2509.24945).
- WorldAgen is a useful date-boundary example. Its
  [arXiv submission](https://arxiv.org/abs/2609.08162) is **2026-09-08**, but its
  [AAAI proceedings record](https://ojs.aaai.org/index.php/AAAI/article/view/38925)
  says **2026-03-14**, and its
  [author repository](https://github.com/mll-lab-nu/WorldAgen) announces acceptance
  in November 2025. Do not present it as a wholly new September result. Its
  original public full-text date before the proceedings release was not
  established, so it is not ranked in the strict initial-release shortlist.
  Its shared state/action backbone and adaptation from real observed transitions
  remain relevant background for online learning; they do not establish a
  generative conversation capability.

## Revised Priority

1. **Inspect WLA's available implementation** for precisely how pretrained
   language capability, predicted futures and actions interact. Specify which
   part is shared and which part violates or extends the project's core design.
2. **Use the pruning study to inform the starting-model decision**, comparing
   capability retention and adaptation cost rather than assuming the 5M pilot
   must be trained into a capable language model.
3. **Compare WorldBagel and ARM as architectural alternatives**, while recording
   unavailable code and their extra components. Use Fast BLT as a separate
   reference for byte efficiency if that representation is retained.
4. Select the reusable base and define the remaining research question before
   another training run. Keep language generation, action success, future-state
   prediction and post-update retention as separate measured outcomes.

These priorities are an inference from the sources and the current project's
requirements. They are not a claim that newer publication dates alone prove
better results, nor a decision to attach an independent language model to the
old predictor. No training or model replacement was performed for this review.
