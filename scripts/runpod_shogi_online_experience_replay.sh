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
REPLAY_SEED_DATA_SELECTION=${REPLAY_SEED_DATA_SELECTION:-data/shogi/training-data-bundles/online-replay-seed-20260512/data-selection.json}
TRAINING_EVAL_DATA_SELECTION=${TRAINING_EVAL_DATA_SELECTION:-data/shogi/training-data-bundles/online-replay-seed-20260512/data-selection.json}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/online-experience-replay-runpod-$(date -u +%Y%m%d-%H%M%S)}

CYCLES=${CYCLES:-4}
EXPERIENCE_SOURCES=${EXPERIENCE_SOURCES:-self:64}
CONCURRENT_GAMES_PER_PROCESS=${CONCURRENT_GAMES_PER_PROCESS:-8}
GENERATION_WORKER_PROCESSES=${GENERATION_WORKER_PROCESSES:-8}
SIMULATIONS=${SIMULATIONS:-16}
NN_LEAF_EVAL_BATCH_LIMIT=${NN_LEAF_EVAL_BATCH_LIMIT:-32}
MAX_PLIES=${MAX_PLIES:-320}
GENERATION_PROGRESS_EVERY_PLIES=${GENERATION_PROGRESS_EVERY_PLIES:-100}
USI_COMMAND=${USI_COMMAND:-}
USI_OPTIONS=${USI_OPTIONS:-}
USI_GO_COMMAND=${USI_GO_COMMAND:-go nodes 1}
USI_READ_TIMEOUT_SECONDS=${USI_READ_TIMEOUT_SECONDS:-30}
YANEURAOU_REPOSITORY_URL=${YANEURAOU_REPOSITORY_URL:-https://github.com/yaneurao/YaneuraOu.git}

REPLAY_CAPACITY=${REPLAY_CAPACITY:-131072}
REPLAY_SAMPLE_SIZE=${REPLAY_SAMPLE_SIZE:-8192}
MIN_REPLAY_SIZE=${MIN_REPLAY_SIZE:-8192}
EVAL_RATIO=${EVAL_RATIO:-0.05}
MAX_STEPS=${MAX_STEPS:-1000}
BATCH_SIZE=${BATCH_SIZE:-512}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-1.0}
VALUE_LOSS_WEIGHT=${VALUE_LOSS_WEIGHT:-1.0}
MAX_TRAIN_EVAL_EXAMPLES=${MAX_TRAIN_EVAL_EXAMPLES:-}
MAX_EVAL_EXAMPLES=${MAX_EVAL_EXAMPLES:-}
LOG_EVERY=${LOG_EVERY:-}
NUM_WORKERS=${NUM_WORKERS:-0}
PIN_MEMORY=${PIN_MEMORY:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-}
EVAL_EVERY=${EVAL_EVERY:-}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-}
NEXT_CHECKPOINT=${NEXT_CHECKPOINT:-best}
SEED=${SEED:-7}

GPU_TYPE=${GPU_TYPE:-NVIDIA RTX A5000}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-180}
VOLUME_SIZE=${VOLUME_SIZE:-0}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}
MIN_VCPU_PER_GPU=${MIN_VCPU_PER_GPU:-}
SECURE_CLOUD=${SECURE_CLOUD:-1}

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
  if [[ "$experience_source" != self:* && "$experience_source" != usi:* ]]; then
    echo "EXPERIENCE_SOURCES entries must be self:GAMES or usi:GAMES: $experience_source" >&2
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

python3 "$RUNPOD_JOB" \
  --repo-root "$REPO_PARENT" \
  --name intrep-shogi-online-experience-replay \
  --gpu-type "$GPU_TYPE" \
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
  --sync "$PROJECT_REL/data/shogi/training-data-bundles/online-replay-seed-20260512" \
  --sync "$ARENA_REL/src" \
  --sync "$ARENA_REL/scripts/generate_shogi_games.py" \
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
  if [[ \"\$experience_source\" == usi:* ]]; then
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
)
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
echo \"online_experience_replay_config cycles=$CYCLES experience_sources=$EXPERIENCE_SOURCES concurrent_games_per_process=$CONCURRENT_GAMES_PER_PROCESS generation_worker_processes=$GENERATION_WORKER_PROCESSES simulations=$SIMULATIONS nn_leaf_eval_batch_limit=$NN_LEAF_EVAL_BATCH_LIMIT max_plies=$MAX_PLIES usi_go_command=$USI_GO_COMMAND usi_read_timeout_seconds=$USI_READ_TIMEOUT_SECONDS replay_capacity=$REPLAY_CAPACITY replay_sample_size=$REPLAY_SAMPLE_SIZE min_replay_size=$MIN_REPLAY_SIZE eval_ratio=$EVAL_RATIO max_steps=$MAX_STEPS batch_size=$BATCH_SIZE learning_rate=$LEARNING_RATE weight_decay=$WEIGHT_DECAY policy_loss_weight=$POLICY_LOSS_WEIGHT value_loss_weight=$VALUE_LOSS_WEIGHT max_train_eval_examples=$MAX_TRAIN_EVAL_EXAMPLES max_eval_examples=$MAX_EVAL_EXAMPLES log_every=$LOG_EVERY num_workers=$NUM_WORKERS pin_memory=$PIN_MEMORY progress_every=$PROGRESS_EVERY eval_every=$EVAL_EVERY early_stopping_patience=$EARLY_STOPPING_PATIENCE next_checkpoint=$NEXT_CHECKPOINT seed=$SEED\"
.venv/bin/python -u scripts/run_shogi_online_replay.py \
  --checkpoint \"$CHECKPOINT\" \
  --run-dir \"$OUTPUT_DIR\" \
  --cycles \"$CYCLES\" \
  --replay-capacity \"$REPLAY_CAPACITY\" \
  --replay-sample-size \"$REPLAY_SAMPLE_SIZE\" \
  --min-replay-size \"$MIN_REPLAY_SIZE\" \
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
  --evaluation-batch-size \"$NN_LEAF_EVAL_BATCH_LIMIT\" \
  --eval-ratio \"$EVAL_RATIO\" \
  --max-steps \"$MAX_STEPS\" \
  --batch-size \"$BATCH_SIZE\" \
  --learning-rate \"$LEARNING_RATE\" \
  --policy-loss-weight \"$POLICY_LOSS_WEIGHT\" \
  --value-loss-weight \"$VALUE_LOSS_WEIGHT\" \
  --device cuda \
  --num-workers \"$NUM_WORKERS\" \
  \"\${TRAINING_ARGS[@]}\" \
  --seed \"$SEED\"" \
  "$@"
