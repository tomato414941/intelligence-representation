# Training

This document records model-training entrypoints and shared training-time
conventions. Dataset preparation lives in [datasets.md](datasets.md), RunPod
operation lives in [runpod.md](runpod.md), and cost/performance records live in
[compute-costs.md](compute-costs.md).

## Entrypoints

```text
intrep.train_text_tokenizer
intrep.train_language_model
intrep.train_image_classification
intrep.train_image_text_choice
intrep.train_image_text_answer
intrep.train_shogi_policy_value
```

The grid step prediction CLI was removed on 2026-06-10; recorded results
reproduce from git history. Its replacement is planned as separate train and
evaluate commands over generated layouts; see
`issues/grid-next-observation-emergence.md`.

Problem models compose task-specific input layers, the shared Transformer core
where useful, and task-specific output heads. See [model-boundaries.md](model-boundaries.md)
and [learning-data-boundaries.md](learning-data-boundaries.md) for the current design.

## Tokenizer Reuse

Text language modeling can train a tokenizer by default, but the preferred
workflow is to train a text tokenizer once and reuse it across text-consuming
tasks:

```sh
uv run python -m intrep.train_text_tokenizer \
  --corpus-path data/tiny-shakespeare/raw/tiny-shakespeare.txt \
  --tokenizer-path runs/text-tokenizer.json \
  --tokenizer-vocab-size 1024
```

```sh
uv run python -m intrep.train_language_model \
  --corpus-path data/tiny-shakespeare/raw/tiny-shakespeare.txt \
  --tokenizer-path runs/text-tokenizer.json \
  --metrics-path runs/text.json \
  --checkpoint-path runs/text.pt
```

Text-consuming commands accept `--tokenizer-path` to reuse a fixed tokenizer.
If both a checkpoint and a tokenizer path are provided, the explicit tokenizer
path is used.

The tokenizer vocabulary size in these examples is not a project-wide default.
It is a property of the tokenizer artifact being created. Multiple tokenizer
artifacts can coexist, but text checkpoints and text-consuming runs must
preserve the tokenizer artifact or payload they used.

Tokenizers written under `runs/` are temporary run artifacts. Promote a
tokenizer that should survive run cleanup to
`tokenizers/<tokenizer-name>/tokenizer.json` before making later runs depend on
it. A checkpoint that embeds the tokenizer payload does not need a separate
tokenizer artifact for loading.

## Image Classification

Image classification uses image patch embeddings, the shared Transformer core,
and a classification head. This command assumes the image-classification JSONL
already exists:

```sh
uv run python -m intrep.train_image_classification \
  --train-path runs/cifar10-train.jsonl \
  --metrics-path runs/cifar10.json \
  --checkpoint-path runs/cifar10.pt
```

The same training command can read torchvision-style ImageFolder datasets:

```sh
uv run python -m intrep.train_image_classification \
  --train-image-folder data/images/train \
  --eval-image-folder data/images/eval \
  --image-size 224 224 \
  --metrics-path runs/image-folder.json \
  --checkpoint-path runs/image-folder.pt
```

## Image/Text Training

Image-text choice trains a model to score candidate text answers:

```sh
uv run python -m intrep.train_image_text_choice \
  --train-path runs/fashion-choice-train.jsonl \
  --eval-path runs/fashion-choice-eval.jsonl \
  --tokenizer-path runs/text-tokenizer.json \
  --prompt "What is this item?" \
  --metrics-path runs/fashion-choice.json \
  --checkpoint-path runs/fashion-choice.pt
```

Image-text answer trains the token output path from image plus prompt to answer
tokens:

```sh
uv run python -m intrep.train_image_text_answer \
  --train-path runs/fashion-answer-train.jsonl \
  --tokenizer-path runs/text-tokenizer.json \
  --metrics-path runs/fashion-answer.json \
  --checkpoint-path runs/fashion-answer.pt
```

## Checkpoint Initialization

Image/text training commands accept `--init-checkpoint-path` for compatible
checkpoints. Checkpoint initialization loads compatible model weights
independent of the source task name.

Image/text training commands also accept repeated `--freeze-module` options.
Freezing is applied after checkpoint initialization and before optimizer
construction. It sets `requires_grad = False` for the named module, so frozen
parameters are excluded from optimization; it is separate from `train()` /
`eval()` mode. Module names are PyTorch module names on the training model, such
as `core` or `image_input_layer`.

```sh
uv run python -m intrep.train_image_text_choice \
  --train-path runs/fashion-choice-train.jsonl \
  --init-checkpoint-path models/<model-name>/checkpoint.pt \
  --freeze-module core \
  --freeze-module image_input_layer
```

## Shogi Training Data Bundles

Shogi policy/value training consumes a fixed Training Data Bundle through its
`data-selection.json`. A bundle's train/eval files are durable policy/value
training examples. The normal durable input for creating a bundle should be a
stable generated record set, not a long command line of run-local outputs.

Repeated training can use a rebuildable tensor cache derived from the same
`data-selection.json`:

```sh
uv run python scripts/build_shogi_policy_value_tensor_cache.py \
  --data-selection data/shogi/training-data-bundles/current/data-selection.json \
  --out data/shogi/training-data-bundles/current/cache/shogi_policy_value_rich_position_transformer_legal_move_attention \
  --assembly-spec shogi_policy_value_rich_position_transformer_legal_move_attention \
  --shard-examples 100000 \
  --resume
```

```sh
uv run python -m intrep.train_shogi_policy_value \
  --data-selection data/shogi/training-data-bundles/current/data-selection.json \
  --tensor-cache data/shogi/training-data-bundles/current/cache/shogi_policy_value_rich_position_transformer_legal_move_attention \
  --checkpoint-path runs/shogi/checkpoint \
  --metrics-path runs/shogi/metrics.json \
  --assembly-spec shogi_policy_value_rich_position_transformer_legal_move_attention
```

The cache is a sharded directory with a manifest and split-specific shard files.
It is an acceleration artifact, not a source of truth. The training command
rejects a cache whose embedded data selection does not match the requested
`data-selection.json`.

For large shogi bundles, build the action-plane policy cache on a RunPod CPU Pod. This
is tensor-cache construction, not training; the GPU training path still consumes
the completed cache through `intrep.train_shogi_policy_value`.

```sh
ASSEMBLY_SPEC=shogi_policy_value_minimal_split_global_position_transformer_action_plane_policy \
  scripts/runpod_build_shogi_action_plane_policy_tensor_cache.sh
```

The RunPod job syncs the Training Data Bundle, builds the cache under:

```text
data/shogi/training-data-bundles/qhapaq-full/cache/<assembly-spec>
```

The script keeps the Pod after completion because the full cache is large. The
local run output contains small metadata files such as `cache_manifest.json`,
`cache_size.txt`, and `remote_cache_path.txt`. Move or release the completed
cache from the kept Pod, then terminate the Pod manually.

The older Modal tensor-cache builder is not the normal path. Remove it after the
RunPod CPU cache path has produced a full cache successfully.

`scripts/create_shogi_training_data_bundle.py` still accepts repeated
`--train-games` inputs for temporary experiments and explicit source mixes. When
multiple train inputs are used, the bundle manifest records every source path
and the CLI prints a warning. Target construction happens while the bundle is
created; training and tensor-cache building consume the resulting example JSONL
instead of reinterpreting source game records.
