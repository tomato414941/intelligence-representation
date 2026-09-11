# Instruction retention pilot

The twelve-source question-learning run reduced strict instruction accuracy
from 47/90 before training to 4/90 with varied questions. This pilot tests
whether preserving conversation context and supervising assistant answers,
then increasing their loss weight, limits that decline while the other tasks
continue to learn. It does not assume that either change will solve forgetting.

## Prespecified comparison

All conditions start from the same pinned LFM2.5-350M checkpoint, with the same
new head initialization, AdamW at `1e-5`, float32, seed 47, and 300 optimizer
updates. Every update includes all twelve sources and all attached parameters
remain trainable. The original/added-question alternation is unchanged.

| Condition | Conversation objective | Conversation weight |
| --- | --- | ---: |
| `chunked` | Existing 128-token stream, all roles as next-token targets | 1 |
| `assistant` | Assistant spans with conversation context | 1 |
| `assistant_weighted` | Same assistant spans and context | 8 |

All other source weights are one. Therefore the conversation source contributes
1/12 or 8/19 of the normalized source loss; approximately half its updates use
the original conversation objective and half use excerpt questions. These
fractions are loss coefficients, not fractions of parameter updates or evidence
that gradient magnitudes are balanced.

The first contrast changes context length and target masking together. It
cannot identify their individual effects. The second contrast changes only the
conversation weight. Conditions match optimizer updates and declared
populations, not consumed conversation records, tokens, wall time, or FLOPs.
Token and record counts must be reported with the results.

## Conversation data and targets

`scripts/prepare_instruction_retention.py` retains every original OASST training
and validation branch, including all languages, ranks, and lengths. It adds
unique programmatically authored English/Japanese instruction examples to the
training split, covering arithmetic, case conversion, lookup, extraction,
conditions, summaries, and a small set of explanations. No teacher model
produces these answers. A seeded shuffle is identical across conditions.
The synthetic templates are narrow; OASST supplies the broader conversations.

The new corpus is common to all conditions. Consequently `chunked` is a control
for the old *method*, not an exact rerun of the previous dataset mixture.
Original file identities, branch counts, licenses, generated example counts,
and evaluation prompt hashes are recorded in data provenance.

Assistant masks come from the pinned tokenizer's chat template generation
spans. User/system text supplies context but does not supply target labels.
Assistant end-of-turn tokens are supervised. Conversations fitting 2,048 tokens
remain together. Longer conversations use windows with 1,024 overlapping context
tokens; all assistant targets are retained and each is supervised once per
traversal. Very long histories therefore have bounded recent context. Branches
without assistant targets are counted and skipped rather than given an empty
loss. Checkpoints retain both the file cursor and pending token/mask windows.

## Measurements

- The previous 90 scored instruction probes are now development data. Generate
  their answers at steps 0, 50, 100, 150, 200, 250, and 300, retaining raw outputs.
- A fresh bilingual prompt group uses different wording and values and is
  evaluated only at steps 0 and 300. Check exact prompt separation from all
  original and generated training user turns. This is a small held-out probe,
  not a general language benchmark or a guarantee against semantic overlap.
- Measure the same validation locations for all twelve sources, with 32
  original cases and up to eight question cases per source. Native experience
  uses complete transitions from selected validation worlds, with at most four
  worlds for added questions. Generate added-question answers before and after
  training. Intermediate source panels use teacher forcing.
- Compare paired questions to constant-answer baselines as in the preceding
  experiment. Treat changed conversation losses as different objectives;
  cross-condition loss values are not directly comparable for that source.
- At the first two updates and around every fiftieth update, record per-source
  weighted and unweighted gradient norms and their cosine with the conversation
  gradient on the first and last body input projections. These are selected
  parameter diagnostics before global gradient clipping, not whole-model
  gradient norms or proof of a causal mechanism.

The pilot asks whether instruction responses remain close to their own initial
score while new-task measures improve. A short-run advantage warrants a longer
controlled run; it does not establish long-term retention. All three conditions
receive the fixed budget so final comparisons are not confounded by stopping
only a poorly performing condition early. Abort for nonfinite values or resource
failures, not merely for a disappointing development score.

## Commands

```sh
python scripts/prepare_instruction_retention.py \
  --conversations data/language/joint-conversations-oasst1-20260911 \
  --development-prompts configs/question-learning-prompts.json \
  --output data/instruction-retention-20260911
R2_ENV_FILE=PATH_TO_PROJECT_R2_CONFIG \
  bash scripts/run_instruction_retention_experiment.sh reports/instruction-retention-20260911 300
```

Use the established RunPod LFM setup and install `rclone`. Full checkpoints
are verified on CPU, including exact model parameter hashes and restoration
of AdamW and all source cursors, then copied into a new project R2 prefix:
`shared-prediction/instruction-retention-20260911/<condition>/`.
`rclone check --download` compares the actual stored bytes before the disposable
working checkpoint is removed. The working directory is separate from the
collected output, so a failed archive cannot accidentally fill the local disk.
Keep the GPU on archive failure until recovery is complete.

Small outputs, tokenizer files, verification records, and archive pointers are
retained locally under `reports/instruction-retention-20260911/`. These are
promoted experiment records, not loadable model entries. Restore a full
checkpoint and its tokenizer using `scripts/restore_r2_artifact.sh` and the
recorded prefix. Terminate the disposable GPU after verified archive and
result retrieval. Existing local models and datasets are preserved.
