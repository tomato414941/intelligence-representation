# Text-To-Image Rule Transfer

This temporary experiment asks whether a newly taught textual relation can be
applied to images without teaching that image/relation combination. It measures
reuse after supervised digit grounding: MNIST labels teach the model to answer
with the LLM's existing digit tokens. Internal image states are learned; no
constraint makes them equal to the corresponding token embeddings. This does
not test acquisition of cross-modal correspondence without labels, or establish
that learned representations are better than explicit digit symbols.

Preparation, calibration, intervention training, evaluation and paired-comparison
tools are implemented. Their implementation tests include exact training
continuation and identical background sampling across counterfactual branches.
Scientific calibration and intervention results are separate generated artifacts;
synthetic execution tests do not establish the capability prerequisites.

## Learning Conditions

Each condition uses one shared body with all attached parameters trainable and
all twelve existing data sources contributing to joint updates. The common
starting checkpoint must support digit naming and an already learned image-pair
order. Both image grounding and order answers use the original language output
layer, rather than interpreting the separate MNIST classifier head as a spoken
digit answer.

From the same checkpoint, A and B receive different random strict total orders
over digits 0--9. Only textual observations teach the new order. The control
condition rehearses the old order, whose digit sequence is 0 through 9. All
three text manifests contain the same 90 ordered pairs, excluding equal digits,
in the same sequence, with 45 yes and 45 no answers. A and B differ only in the
answers: 23 of the 45 unordered class-pair relations change, and 22 remain the
same. Training and evaluation use the same question wording; this is transfer
to an untrained modality/rule combination, not to a new operator family.

The text manifests contain `rule`, `digits` and `answer` fields. Construct their
questions with `order_question(..., modality="text")` and supervise the answer
through the existing answer loss. They contain no image inputs or image-rule
answer examples. `development-recipe.json` preserves the existing training
populations and reserves holdout images; it does not itself add the new rule
loss to the training loop. The intervention runner must explicitly consume
the text manifests while preserving the twelve background data streams.

## Calibration And Intervention Training

`scripts/train_rule_transfer.py` adds digit naming, the old image order and the
old text order to every joint update. These supplemental samplers are separate
from the twelve original readers. Digit naming traverses every training image
before repeating; old image pairs cycle through all 90 ordered unequal-digit
pairs with complete, shuffled per-class image pools. No new-order image answer
is taught. Equal-length questions are batched with the same answer-token and EOS
loss as the existing language answer objective.

The initial calibration uses the prior LFM2.5-350M varied-question checkpoint.
Its trained image input provides a starting point, but the separate classifier's
past accuracy does not establish generated digit naming or order competence.
The entire body and all attached heads remain trainable. Calibration resets
AdamW to `1e-5`, preserves every background training configuration and reader
cursor, and uses additional batches of 16 digit names, eight old image pairs and
eight old text pairs. Their loss weights are 8, 8 and 2; each original source
retains weight 1. There is no parameter freezing or reduced training population.

```sh
uv run python scripts/train_rule_transfer.py \
  --initialize models/question-learning-20260911/varied/checkpoint.pt \
  --condition calibration \
  --recipe data/rule-transfer-20260913/development-recipe.json \
  --panel data/rule-transfer-20260913/panel.json \
  --extension intrep.problems.shared_prediction.record_sources \
  --output models/rule-transfer/calibration --device cuda \
  --steps 12000 --training-seconds 10800 --interval 500 --stop-when-calibrated \
  --prompts configs/question-learning-prompts.json
```

Every 500 updates, development generation measures digit names, the old image
order and all 90 old text relations. Calibration stops after passing their
95%, 90% and 95% gates, or reaching 12,000 updates or three measured training
hours. Evaluation, checkpoint I/O and storage verification are additional costs.
New-order image questions are not queried to select the starting checkpoint.
All twelve original tasks and the existing instruction prompt panel are measured
before and after each training invocation to expose deterioration separately.

Use `--common` with the passing calibration checkpoint and `--condition a`, `b`
or `control`, together with the corresponding `--manifest` text file. This forks
the identical model weights and background/supplemental reader states while
resetting AdamW identically in every branch. Each branch adds eight tuition
questions per update with weight 8. A/B receive the new text order; the control
receives old-order rehearsal. The manifest is checked against the prepared order
and must match exactly. Development prerequisite checks add the new text gate
for A/B without querying new-order image answers.

The intervention comparison must choose the same update count for every branch.
Use `--resume` in the original output directory to continue the exact optimizer,
sampling and RNG state. Settings and the complete recipe must remain unchanged.
The checkpoint includes supplemental lesson state; use this experiment's runner
for training continuation. The standard shared-checkpoint evaluator can read it.

Training records retain initial checkpoint and parameter hashes, every update's
background reader-state hash and supplemental input indices/text example IDs.
Verify these traces and equal update counts before interpreting A/B contrasts.
All new-rule supervision and every model-selection query have explicit modality
boundaries; the final image split is reserved throughout calibration.

`scripts/run_rule_transfer_calibration.sh` restores the initial checkpoint on a
disposable CUDA worker, checks its expected SHA-256, runs calibration and calls
`scripts/archive_rule_transfer.py`. Archiving restores the model, optimizer and
all reader states on CPU, verifies the recorded parameter/checkpoint digests,
uploads to a new project R2 prefix and compares remote bytes before removing the
working checkpoint. Only small reports and tokenizer files return locally.

Before interpreting a negative transfer result, generated digit naming and
the new text rule must each reach 95%, and the old image rule must reach 90%
on development data. The evaluator reports each gate separately. Transfer
evidence and degradation of existing abilities must still be considered
separately; passing these gates alone is not proof of transfer. Calibration
must establish that the control's old text rule is already known as well.

## Fixed Image Panels

```sh
uv run python scripts/prepare_rule_transfer.py \
  --history models --history reports \
  --output data/rule-transfer-20260913/panel.json
```

Supply every available historical evaluation location. The audit reads both
primary and partner image indices from explicitly identified MNIST results;
its claim of unused images is relative to those audited records. Check archived
or externally stored prior evaluations before freezing a panel. The manifest
records the historical files, their hashes and the excluded indices.

Development and holdout each contain 450 disjoint image pairs: ten per unordered
digit-class pair, with 900 unique images in each split. Images are excluded if
they occurred in audited prior evaluations, duplicate a training image exactly,
or duplicate another selected image. The manifest stores raw source indices,
labels, per-image hashes and source-file hashes. Images remain in their original
IDX files; preparation stores only metadata and the small text manifests.

The generated development recipe sets `evaluation_excluded_indices` on the
MNIST evaluation source. It includes the holdout indices and every exact pixel
duplicate of those images. Both ordinary evaluation anchors and comparison
partners obey this exclusion. Training still traverses the full original
60,000-image MNIST training population. Use this recipe for calibration and
intervention training so ordinary evaluations cannot expose the final panel.

## Readouts And Scoring

For each pair, evaluate both input orientations. The same observations and
question recur in the two rule branches. The model receives the images and
question; the taught order is held in its weights. Expected answers, true digit
labels and the order table are never passed to the direct readout.

The evaluator measures four routes using the same checkpoint:

- **Direct:** generate the yes/no answer from the two images.
- **Read then apply:** generate each digit name, then use the model's textual
  answer for those predicted digits. Invalid digit output produces a failed
  response, with no fallback to the true class.
- **Cached probabilities:** combine normalized digit-candidate probabilities
  with a cached table of the model's own yes/no probabilities for all 100 digit
  pairs, including equal-digit pairs. This table contains the model's errors.
- **Oracle digits:** use the true image labels to query that same learned text
  table. This route is a diagnostic for rule availability.

Strict scores strip surrounding whitespace and require the requested answer
string. Gates use generated answers, not teacher-forced predictions or a
restricted classifier score. Candidate probabilities require distinct
single-token digit/yes/no strings in the selected tokenizer. The probability
mass assigned to those candidates is recorded separately: normalizing within
the candidate set does not imply confident unrestricted generation.

Both bridge routes can reuse digit readouts and the text table. Their cache
construction costs are retained. Synchronized inference time, body calls and
processed sequence positions are measured; sequence positions are not FLOPs.
Image-file loading and training are outside those inference measurements.
Any efficiency claim must add text-rule tuition, background training, cache
construction, storage and actual query reuse to the comparison.

## Evaluation Commands

Given a trained shared checkpoint for condition A:

```sh
uv run python scripts/evaluate_rule_transfer.py \
  --checkpoint models/rule-transfer/a/checkpoint.pt \
  --extension intrep.problems.shared_prediction.record_sources \
  --panel data/rule-transfer-20260913/panel.json \
  --split development --order a \
  --output reports/rule-transfer/a-development.json --device cuda
```

Use condition B's checkpoint with `--order b`. Development evaluations support
calibration. Select `--split holdout` only after the training procedure and
interpretation criteria are fixed. The evaluator reads only the selected
split, checks source hashes, and refuses changed panel/checkpoint files or
overwriting an existing result.

```sh
uv run python scripts/compare_rule_transfer.py \
  --a reports/rule-transfer/a-holdout.json \
  --b reports/rule-transfer/b-holdout.json \
  --output reports/rule-transfer/comparison.json
```

Comparison separates changed and unchanged relations. Its paired score requires
all four answers to be correct: both orientations under both taught orders.
An unchanged rule or a constant answer cannot pass the changed-relation groups.
The four answers are not independent samples. Image pairs are nested within 45
digit-class pairs, and one A/B order pair is not replication over different rules.

These scores are descriptive until common initial weights, matched background
source traces and the absence of new-rule image supervision are verified from
the training artifacts. If prerequisites pass but direct transfer is weak,
the planned follow-up gives A and the rehearsal control the same 32/128/512
image-rule examples and compares their additional learning costs.
