#!/usr/bin/env bash
set -euo pipefail

# Run true shogi Online Experience Replay on disposable RunPod compute.
# The learner lives in this repository; game generation is delegated to
# shogi-arena-agent through the documented CLI/artifact boundary.

cd "$(dirname "$0")/.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
REPO_PARENT=${REPO_PARENT:-"$(cd "$PWD/../.." && pwd)"}
PROJECT_REL=${PROJECT_REL:-projects/intelligence-representation}
ARENA_REL=${ARENA_REL:-projects/shogi-arena-agent}

CHECKPOINT=${CHECKPOINT:-models/d256-h1024-heads8-l6-shogi/checkpoint.pt}
REPLAY_SEED_DATA_SELECTION=${REPLAY_SEED_DATA_SELECTION:-data/shogi/training-data-bundles/qhapaq-full/data-selection.json}
TRAINING_EVAL_DATA_SELECTION=${TRAINING_EVAL_DATA_SELECTION:-data/shogi/training-data-bundles/qhapaq-full/data-selection.json}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/online-experience-replay-runpod-$(date -u +%Y%m%d-%H%M%S)}

ITERATIONS=${ITERATIONS:-4}
EXPERIENCE_SOURCES=${EXPERIENCE_SOURCES:-checkpoint-self:64}
CONCURRENT_GAMES_PER_PROCESS=${CONCURRENT_GAMES_PER_PROCESS:-8}
GENERATION_WORKER_PROCESSES=${GENERATION_WORKER_PROCESSES:-8}
SIMULATIONS=${SIMULATIONS:-128}
NN_LEAF_EVAL_BATCH_LIMIT=${NN_LEAF_EVAL_BATCH_LIMIT:-64}
MAX_PLIES=${MAX_PLIES:-320}
GENERATION_PROGRESS_EVERY_PLIES=${GENERATION_PROGRESS_EVERY_PLIES:-100}
USI_COMMAND=${USI_COMMAND:-}
USI_OPTIONS=${USI_OPTIONS:-}
USI_GO_COMMAND=${USI_GO_COMMAND:-go nodes 1}
USI_READ_TIMEOUT_SECONDS=${USI_READ_TIMEOUT_SECONDS:-30}
YANEURAOU_REPOSITORY_URL=${YANEURAOU_REPOSITORY_URL:-https://github.com/yaneurao/YaneuraOu.git}
CHECKPOINT_MOVE_SELECTION_PROFILE=${CHECKPOINT_MOVE_SELECTION_PROFILE:-visit-sampling}
CHECKPOINT_MOVE_SELECTION_TEMPERATURE=${CHECKPOINT_MOVE_SELECTION_TEMPERATURE:-}
CHECKPOINT_MOVE_SELECTION_TEMPERATURE_PLIES=${CHECKPOINT_MOVE_SELECTION_TEMPERATURE_PLIES:-}

REPLAY_CAPACITY=${REPLAY_CAPACITY:-2097152}
MIN_REPLAY_SIZE=${MIN_REPLAY_SIZE:-8192}
SAMPLED_EXAMPLES_PER_ITERATION=${SAMPLED_EXAMPLES_PER_ITERATION:-524288}
MAX_SEED_EXAMPLES_PER_ITERATION=${MAX_SEED_EXAMPLES_PER_ITERATION:-50000}
TRAINING_BATCH_SIZE=${TRAINING_BATCH_SIZE:-512}
TARGET_SAMPLE_PASSES=${TARGET_SAMPLE_PASSES:-1}
MAX_OPTIMIZER_STEPS_PER_ITERATION=${MAX_OPTIMIZER_STEPS_PER_ITERATION:-}
GENERATOR_GATE_GAMES=${GENERATOR_GATE_GAMES:-32}
GENERATOR_GATE_WORKER_PROCESSES=${GENERATOR_GATE_WORKER_PROCESSES:-4}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-1.0}
VALUE_LOSS_WEIGHT=${VALUE_LOSS_WEIGHT:-1.0}
MAX_TRAIN_EVAL_EXAMPLES=${MAX_TRAIN_EVAL_EXAMPLES:-}
MAX_EVAL_EXAMPLES=${MAX_EVAL_EXAMPLES:-}
LOG_EVERY=${LOG_EVERY:-}
NUM_WORKERS=${NUM_WORKERS:-0}
PIN_MEMORY=${PIN_MEMORY:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}
EVAL_EVERY=${EVAL_EVERY:-}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-}
NEXT_CHECKPOINT=${NEXT_CHECKPOINT:-best}
SEED=${SEED:-7}

GPU_TYPE=${GPU_TYPE:-NVIDIA RTX A5000}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-180}
CONTAINER_DISK_SIZE=${CONTAINER_DISK_SIZE:-80}
VOLUME_SIZE=${VOLUME_SIZE:-0}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}
MIN_VCPU_PER_GPU=${MIN_VCPU_PER_GPU:-}
SECURE_CLOUD=${SECURE_CLOUD:-1}
SYNC_TENSOR_CACHE=${SYNC_TENSOR_CACHE:-1}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$REPLAY_SEED_DATA_SELECTION" ]]; then
  echo "replay seed data selection not found: $REPLAY_SEED_DATA_SELECTION" >&2
  exit 1
fi
if [[ ! -f "$TRAINING_EVAL_DATA_SELECTION" ]]; then
  echo "training eval data selection not found: $TRAINING_EVAL_DATA_SELECTION" >&2
  exit 1
fi
if [[ ! -d "$REPO_PARENT/$ARENA_REL" ]]; then
  echo "shogi-arena-agent not found: $REPO_PARENT/$ARENA_REL" >&2
  exit 1
fi
IFS=',' read -ra EXPERIENCE_SOURCE_ITEMS <<< "$EXPERIENCE_SOURCES"
for experience_source in "${EXPERIENCE_SOURCE_ITEMS[@]}"; do
  if [[ "$experience_source" != checkpoint-self:* && "$experience_source" != checkpoint-black-vs-usi:* && "$experience_source" != usi-black-vs-checkpoint:* && "$experience_source" != checkpoint-vs-usi-balanced:* ]]; then
    echo "EXPERIENCE_SOURCES entries must be checkpoint-self:GAMES, checkpoint-black-vs-usi:GAMES, usi-black-vs-checkpoint:GAMES, or checkpoint-vs-usi-balanced:GAMES: $experience_source" >&2
    exit 1
  fi
done

RUNNER_ARGS=()
if [[ "$SECURE_CLOUD" == "1" ]]; then
  RUNNER_ARGS+=(--secure-cloud)
fi
if [[ -n "$DATA_CENTER_IDS" ]]; then
  RUNNER_ARGS+=(--data-center-ids "$DATA_CENTER_IDS")
fi
if [[ -n "$MIN_VCPU_PER_GPU" ]]; then
  RUNNER_ARGS+=(--min-vcpu-per-gpu "$MIN_VCPU_PER_GPU")
fi
if [[ "$SYNC_TENSOR_CACHE" == "1" ]]; then
  CACHE_ROOT="data/shogi/training-data-bundles/qhapaq-full/cache/shogi-policy-value-tensors"
  RUNNER_ARGS+=(--sync "$PROJECT_REL/$CACHE_ROOT/manifest.json")
  while IFS= read -r cache_path; do
    RUNNER_ARGS+=(--sync "$PROJECT_REL/$CACHE_ROOT/$cache_path")
  done < <(
    CACHE_ROOT="$CACHE_ROOT" ITERATIONS="$ITERATIONS" MAX_SEED_EXAMPLES_PER_ITERATION="$MAX_SEED_EXAMPLES_PER_ITERATION" SEED="$SEED" \
      uv run --extra torch python - <<'PY'
import json
import math
import os
from pathlib import Path

import torch

cache_root = Path(os.environ["CACHE_ROOT"])
manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
iterations = int(os.environ["ITERATIONS"])
sample_count = int(os.environ["MAX_SEED_EXAMPLES_PER_ITERATION"])
seed = int(os.environ["SEED"])
paths: set[str] = set()

for shard in manifest["shards"]:
    if shard["split"] == "eval":
        paths.add(str(shard["path"]))

train_shards = [shard for shard in manifest["shards"] if shard["split"] == "train"]
shard_counts = [int(shard["sample_count"]) for shard in train_shards]
average_shard_count = max(1.0, sum(shard_counts) / len(shard_counts))
selected_shard_count = min(len(shard_counts), max(1, math.ceil(sample_count / average_shard_count) + 1))
weights = torch.tensor(shard_counts, dtype=torch.float64)
for iteration_index in range(1, iterations + 1):
    generator = torch.Generator().manual_seed(seed + iteration_index)
    shard_order = torch.multinomial(weights, len(shard_counts), replacement=False, generator=generator).tolist()
    for shard_index in shard_order[:selected_shard_count]:
        paths.add(str(train_shards[shard_index]["path"]))

for path in sorted(paths):
    print(path)
PY
  )
fi

python3 "$RUNPOD_JOB" \
  --repo-root "$REPO_PARENT" \
  --name intrep-shogi-online-experience-replay \
  --gpu-type "$GPU_TYPE" \
  --container-disk-size "$CONTAINER_DISK_SIZE" \
  --volume-size "$VOLUME_SIZE" \
  "${RUNNER_ARGS[@]}" \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --wait-seconds 600 \
  --ssh-wait-seconds 180 \
  --allow-existing-pods \
  --sync "$PROJECT_REL/src" \
  --sync "$PROJECT_REL/scripts/run_shogi_online_replay.py" \
  --sync "$PROJECT_REL/scripts/setup_runpod.sh" \
  --sync "$PROJECT_REL/pyproject.toml" \
  --sync "$PROJECT_REL/uv.lock" \
  --sync "$PROJECT_REL/README.md" \
  --sync "$PROJECT_REL/AGENTS.md" \
  --sync "$PROJECT_REL/$CHECKPOINT" \
  --sync "$PROJECT_REL/data/shogi/training-data-bundles/qhapaq-full/data-selection.json" \
  --sync "$PROJECT_REL/data/shogi/training-data-bundles/qhapaq-full/manifest.json" \
  --sync "$PROJECT_REL/data/shogi/training-data-bundles/qhapaq-full/train-examples.jsonl" \
  --sync "$PROJECT_REL/data/shogi/training-data-bundles/qhapaq-full/eval-examples.jsonl" \
  --sync "$ARENA_REL/src" \
  --sync "$ARENA_REL/scripts/generate_shogi_games.py" \
  --sync "$ARENA_REL/scripts/evaluate_shogi_players.py" \
  --sync "$ARENA_REL/pyproject.toml" \
  --sync "$ARENA_REL/AGENTS.md" \
  --setup-command "cd \"\$REMOTE_DIR/$PROJECT_REL\"; bash scripts/setup_runpod.sh; .venv/bin/python -m pip install -e \"\$REMOTE_DIR/$ARENA_REL\"" \
  --output "$PROJECT_REL/$OUTPUT_DIR" \
  --timings-output "$PROJECT_REL/$OUTPUT_DIR/runpod_timings.json" \
  --remote "set -euo pipefail; cd \"\$REMOTE_DIR/$PROJECT_REL\"; mkdir -p \"$OUTPUT_DIR\"
USI_COMMAND_REMOTE=\"$USI_COMMAND\"
NEEDS_USI=0
IFS=',' read -ra EXPERIENCE_SOURCE_ITEMS <<< \"$EXPERIENCE_SOURCES\"
for experience_source in \"\${EXPERIENCE_SOURCE_ITEMS[@]}\"; do
  if [[ \"\$experience_source\" == checkpoint-black-vs-usi:* || \"\$experience_source\" == usi-black-vs-checkpoint:* || \"\$experience_source\" == checkpoint-vs-usi-balanced:* ]]; then
    NEEDS_USI=1
  fi
done
if [[ \"\$NEEDS_USI\" == \"1\" && -z \"\$USI_COMMAND_REMOTE\" ]]; then
  apt-get update >/dev/null
  DEBIAN_FRONTEND=noninteractive apt-get install -y git build-essential >/dev/null
  rm -rf /root/YaneuraOu
  GIT_TERMINAL_PROMPT=0 git clone --depth 1 \"$YANEURAOU_REPOSITORY_URL\" /root/YaneuraOu
  make -s -C /root/YaneuraOu/source -f Makefile -j\"\$(nproc)\" normal TARGET_CPU=AVX2 YANEURAOU_EDITION=YANEURAOU_ENGINE_MATERIAL COMPILER=g++ TARGET=YaneuraOu-runpod
  USI_COMMAND_REMOTE=/root/YaneuraOu/source/YaneuraOu-runpod
fi
ONLINE_REPLAY_ARGS=(
  --usi-go-command \"$USI_GO_COMMAND\"
  --usi-read-timeout-seconds \"$USI_READ_TIMEOUT_SECONDS\"
  --checkpoint-move-selection-profile \"$CHECKPOINT_MOVE_SELECTION_PROFILE\"
)
if [[ -n \"$CHECKPOINT_MOVE_SELECTION_TEMPERATURE\" ]]; then
  ONLINE_REPLAY_ARGS+=(--checkpoint-move-selection-temperature \"$CHECKPOINT_MOVE_SELECTION_TEMPERATURE\")
fi
if [[ -n \"$CHECKPOINT_MOVE_SELECTION_TEMPERATURE_PLIES\" ]]; then
  ONLINE_REPLAY_ARGS+=(--checkpoint-move-selection-temperature-plies \"$CHECKPOINT_MOVE_SELECTION_TEMPERATURE_PLIES\")
fi
IFS=',' read -ra EXPERIENCE_SOURCE_ITEMS <<< \"$EXPERIENCE_SOURCES\"
for experience_source in \"\${EXPERIENCE_SOURCE_ITEMS[@]}\"; do
  ONLINE_REPLAY_ARGS+=(--experience-source \"\$experience_source\")
done
if [[ \"\$NEEDS_USI\" == \"1\" ]]; then
  ONLINE_REPLAY_ARGS+=(--usi-command \"\$USI_COMMAND_REMOTE\")
  IFS=';' read -ra USI_OPTION_ITEMS <<< \"$USI_OPTIONS\"
  for usi_option in \"\${USI_OPTION_ITEMS[@]}\"; do
    if [[ -n \"\$usi_option\" ]]; then
      ONLINE_REPLAY_ARGS+=(--usi-option \"\$usi_option\")
    fi
  done
fi
TRAINING_ARGS=(
  --weight-decay \"$WEIGHT_DECAY\"
)
if [[ -n \"$MAX_TRAIN_EVAL_EXAMPLES\" ]]; then
  TRAINING_ARGS+=(--max-train-eval-examples \"$MAX_TRAIN_EVAL_EXAMPLES\")
fi
if [[ -n \"$MAX_EVAL_EXAMPLES\" ]]; then
  TRAINING_ARGS+=(--max-eval-examples \"$MAX_EVAL_EXAMPLES\")
fi
if [[ -n \"$LOG_EVERY\" ]]; then
  TRAINING_ARGS+=(--log-every \"$LOG_EVERY\")
fi
if [[ \"$PIN_MEMORY\" == \"1\" ]]; then
  TRAINING_ARGS+=(--pin-memory)
fi
if [[ -n \"$PROGRESS_EVERY\" ]]; then
  TRAINING_ARGS+=(--progress-every \"$PROGRESS_EVERY\")
fi
if [[ -n \"$EVAL_EVERY\" ]]; then
  TRAINING_ARGS+=(--eval-every \"$EVAL_EVERY\")
fi
if [[ -n \"$EARLY_STOPPING_PATIENCE\" ]]; then
  TRAINING_ARGS+=(--early-stopping-patience \"$EARLY_STOPPING_PATIENCE\")
fi
if [[ -n \"$MAX_OPTIMIZER_STEPS_PER_ITERATION\" ]]; then
  TRAINING_ARGS+=(--max-optimizer-steps-per-iteration \"$MAX_OPTIMIZER_STEPS_PER_ITERATION\")
fi
echo \"online_experience_replay_config iterations=$ITERATIONS experience_sources=$EXPERIENCE_SOURCES checkpoint_move_selection_profile=$CHECKPOINT_MOVE_SELECTION_PROFILE checkpoint_move_selection_temperature=$CHECKPOINT_MOVE_SELECTION_TEMPERATURE checkpoint_move_selection_temperature_plies=$CHECKPOINT_MOVE_SELECTION_TEMPERATURE_PLIES concurrent_games_per_process=$CONCURRENT_GAMES_PER_PROCESS generation_worker_processes=$GENERATION_WORKER_PROCESSES simulations=$SIMULATIONS nn_leaf_eval_batch_limit=$NN_LEAF_EVAL_BATCH_LIMIT max_plies=$MAX_PLIES generator_gate_games=$GENERATOR_GATE_GAMES generator_gate_worker_processes=$GENERATOR_GATE_WORKER_PROCESSES usi_go_command=$USI_GO_COMMAND usi_read_timeout_seconds=$USI_READ_TIMEOUT_SECONDS replay_capacity=$REPLAY_CAPACITY sampled_examples_per_iteration=$SAMPLED_EXAMPLES_PER_ITERATION max_seed_examples_per_iteration=$MAX_SEED_EXAMPLES_PER_ITERATION min_replay_size=$MIN_REPLAY_SIZE training_batch_size=$TRAINING_BATCH_SIZE target_sample_passes=$TARGET_SAMPLE_PASSES max_optimizer_steps_per_iteration=$MAX_OPTIMIZER_STEPS_PER_ITERATION learning_rate=$LEARNING_RATE weight_decay=$WEIGHT_DECAY policy_loss_weight=$POLICY_LOSS_WEIGHT value_loss_weight=$VALUE_LOSS_WEIGHT max_train_eval_examples=$MAX_TRAIN_EVAL_EXAMPLES max_eval_examples=$MAX_EVAL_EXAMPLES log_every=$LOG_EVERY num_workers=$NUM_WORKERS pin_memory=$PIN_MEMORY progress_every=$PROGRESS_EVERY eval_every=$EVAL_EVERY early_stopping_patience=$EARLY_STOPPING_PATIENCE next_checkpoint=$NEXT_CHECKPOINT seed=$SEED\"
.venv/bin/python -u scripts/run_shogi_online_replay.py \
  --checkpoint \"$CHECKPOINT\" \
  --run-dir \"$OUTPUT_DIR\" \
  --iterations \"$ITERATIONS\" \
  --replay-capacity \"$REPLAY_CAPACITY\" \
  --min-replay-size \"$MIN_REPLAY_SIZE\" \
  --sampled-examples-per-iteration \"$SAMPLED_EXAMPLES_PER_ITERATION\" \
  --max-seed-examples-per-iteration \"$MAX_SEED_EXAMPLES_PER_ITERATION\" \
  --training-batch-size \"$TRAINING_BATCH_SIZE\" \
  --target-sample-passes \"$TARGET_SAMPLE_PASSES\" \
  --generator-gate-games \"$GENERATOR_GATE_GAMES\" \
  --generator-gate-worker-processes \"$GENERATOR_GATE_WORKER_PROCESSES\" \
  --experience-store-dir \"$OUTPUT_DIR/experience-store\" \
  --replay-seed-data-selection \"$REPLAY_SEED_DATA_SELECTION\" \
  --training-eval-data-selection \"$TRAINING_EVAL_DATA_SELECTION\" \
  --next-checkpoint \"$NEXT_CHECKPOINT\" \
  --arena-repo \"\$REMOTE_DIR/$ARENA_REL\" \
  \"\${ONLINE_REPLAY_ARGS[@]}\" \
  --concurrent-games-per-process \"$CONCURRENT_GAMES_PER_PROCESS\" \
  --generation-worker-processes \"$GENERATION_WORKER_PROCESSES\" \
  --generation-progress-every-plies \"$GENERATION_PROGRESS_EVERY_PLIES\" \
  --board-backend cshogi \
  --max-plies \"$MAX_PLIES\" \
  --simulations \"$SIMULATIONS\" \
  --nn-leaf-eval-batch-limit \"$NN_LEAF_EVAL_BATCH_LIMIT\" \
  --learning-rate \"$LEARNING_RATE\" \
  --policy-loss-weight \"$POLICY_LOSS_WEIGHT\" \
  --value-loss-weight \"$VALUE_LOSS_WEIGHT\" \
  --device cuda \
  --num-workers \"$NUM_WORKERS\" \
  \"\${TRAINING_ARGS[@]}\" \
  --seed \"$SEED\"" \
  "$@"
