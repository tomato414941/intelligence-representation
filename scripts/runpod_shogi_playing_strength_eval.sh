#!/usr/bin/env bash
set -euo pipefail

# Run the formal shogi playing-strength protocol on a RunPod GPU Pod.
# This wrapper is intentionally checkpoint-vs-checkpoint first; engine matches
# should use the same protocol entry point when added here.

cd "$(dirname "$0")/.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
RUNPOD_JOB_NAME=${RUNPOD_JOB_NAME:-intrep-shogi-playing-strength-eval}
GPU_TYPE=${GPU_TYPE:-NVIDIA RTX 4000 Ada Generation}
CONTAINER_DISK_SIZE=${CONTAINER_DISK_SIZE:-30}
VOLUME_SIZE=${VOLUME_SIZE:-0}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-240}
WAIT_SECONDS=${WAIT_SECONDS:-600}
SSH_WAIT_SECONDS=${SSH_WAIT_SECONDS:-180}
REMOTE_POLL_SECONDS=${REMOTE_POLL_SECONDS:-30}
SECURE_CLOUD=${SECURE_CLOUD:-0}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}
KEEP_POD=${KEEP_POD:-0}
DRY_RUN=${DRY_RUN:-0}

ARENA_ROOT=${ARENA_ROOT:-"$PWD/../shogi-arena-agent"}
ARENA_REPOSITORY_URL=${ARENA_REPOSITORY_URL:-}
ARENA_REF=${ARENA_REF:-main}

: "${CANDIDATE_CHECKPOINT:?Set CANDIDATE_CHECKPOINT to the candidate shogi checkpoint under this repository}"
: "${BASELINE_CHECKPOINT:?Set BASELINE_CHECKPOINT to the baseline shogi checkpoint under this repository}"
: "${START_POSITION_SEED:?Set START_POSITION_SEED for reproducible random-opening paired evaluation}"

OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/playing-strength-$RUN_ID}
GAMES=${GAMES:-64}
OPENING_PLIES=${OPENING_PLIES:-12}
MAX_PLIES=${MAX_PLIES:-320}
SIMULATIONS=${SIMULATIONS:-128}
NN_LEAF_EVAL_BATCH_LIMIT=${NN_LEAF_EVAL_BATCH_LIMIT:-64}
MCTS_MOVE_TIME_LIMIT_SEC=${MCTS_MOVE_TIME_LIMIT_SEC:-9.0}
BOARD_BACKEND=${BOARD_BACKEND:-cshogi}
PROGRESS_EVERY_GAMES=${PROGRESS_EVERY_GAMES:-1}
CANDIDATE_CHECKPOINT_ID=${CANDIDATE_CHECKPOINT_ID:-}
BASELINE_CHECKPOINT_ID=${BASELINE_CHECKPOINT_ID:-}

if [[ ! -f "$RUNPOD_JOB" ]]; then
  echo "RunPod runner not found: $RUNPOD_JOB" >&2
  exit 1
fi

if [[ -z "$ARENA_REPOSITORY_URL" ]]; then
  if [[ ! -d "$ARENA_ROOT" ]]; then
    echo "ARENA_ROOT not found and ARENA_REPOSITORY_URL is unset: $ARENA_ROOT" >&2
    exit 1
  fi
  ARENA_REPOSITORY_URL=$(git -C "$ARENA_ROOT" config --get remote.origin.url || true)
fi
if [[ -z "$ARENA_REPOSITORY_URL" ]]; then
  echo "ARENA_REPOSITORY_URL is required when arena origin remote is unset" >&2
  exit 1
fi

checkpoint_relpath() {
  local path=$1
  if [[ ! -e "$path" ]]; then
    echo "checkpoint not found: $path" >&2
    exit 1
  fi
  local checkpoint_abs
  local repo_abs
  checkpoint_abs=$(realpath "$path")
  repo_abs=$(realpath "$PWD")
  case "$checkpoint_abs" in
    "$repo_abs"/*) printf '%s\n' "${checkpoint_abs#"$repo_abs"/}" ;;
    *)
      echo "checkpoint must be under the repository root so it can be synced: $checkpoint_abs" >&2
      exit 1
      ;;
  esac
}

CANDIDATE_CHECKPOINT_REMOTE=$(checkpoint_relpath "$CANDIDATE_CHECKPOINT")
BASELINE_CHECKPOINT_REMOTE=$(checkpoint_relpath "$BASELINE_CHECKPOINT")

CLOUD_ARGS=()
if [[ "$SECURE_CLOUD" == "1" ]]; then
  CLOUD_ARGS+=(--secure-cloud)
fi

DATA_CENTER_ARGS=()
if [[ -n "$DATA_CENTER_IDS" ]]; then
  DATA_CENTER_ARGS+=(--data-center-ids "$DATA_CENTER_IDS")
fi

RETENTION_ARGS=()
if [[ "$KEEP_POD" == "1" ]]; then
  RETENTION_ARGS+=(--keep-pod)
fi

RUNNER_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  RUNNER_ARGS+=(--dry-run)
fi

CANDIDATE_ID_ARGS=()
if [[ -n "$CANDIDATE_CHECKPOINT_ID" ]]; then
  CANDIDATE_ID_ARGS+=(--player-a-checkpoint-id "$CANDIDATE_CHECKPOINT_ID")
fi
BASELINE_ID_ARGS=()
if [[ -n "$BASELINE_CHECKPOINT_ID" ]]; then
  BASELINE_ID_ARGS+=(--player-b-checkpoint-id "$BASELINE_CHECKPOINT_ID")
fi

python3 "$RUNPOD_JOB" \
  --repo-root "$PWD" \
  --name "$RUNPOD_JOB_NAME" \
  --pod-name "$RUNPOD_JOB_NAME-$RUN_ID" \
  --gpu-type "$GPU_TYPE" \
  --container-disk-size "$CONTAINER_DISK_SIZE" \
  --volume-size "$VOLUME_SIZE" \
  "${CLOUD_ARGS[@]}" \
  "${DATA_CENTER_ARGS[@]}" \
  "${RUNNER_ARGS[@]}" \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --wait-seconds "$WAIT_SECONDS" \
  --ssh-wait-seconds "$SSH_WAIT_SECONDS" \
  --detached-remote \
  --remote-poll-seconds "$REMOTE_POLL_SECONDS" \
  --allow-existing-pods \
  "${RETENTION_ARGS[@]}" \
  --sync src \
  --sync pyproject.toml \
  --sync uv.lock \
  --sync README.md \
  --sync AGENTS.md \
  --sync scripts/setup_runpod.sh \
  --sync scripts/run_shogi_playing_strength_eval.py \
  --sync "$CANDIDATE_CHECKPOINT_REMOTE" \
  --sync "$BASELINE_CHECKPOINT_REMOTE" \
  --setup-command 'cd "$REMOTE_DIR"; bash scripts/setup_runpod.sh' \
  --output "$OUTPUT_DIR" \
  --timings-output "$OUTPUT_DIR/runpod_timings.json" \
  --remote "set -euo pipefail
cd \"\$REMOTE_DIR\"
apt-get update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y git >/dev/null
rm -rf /root/shogi-arena-agent
GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch \"$ARENA_REF\" \"$ARENA_REPOSITORY_URL\" /root/shogi-arena-agent
.venv/bin/python -m pip install -e /root/shogi-arena-agent
mkdir -p \"$OUTPUT_DIR\"
echo \"playing_strength_eval_config candidate=$CANDIDATE_CHECKPOINT_REMOTE baseline=$BASELINE_CHECKPOINT_REMOTE games=$GAMES seed=$START_POSITION_SEED opening_plies=$OPENING_PLIES simulations=$SIMULATIONS nn_leaf_eval_batch_limit=$NN_LEAF_EVAL_BATCH_LIMIT\"
.venv/bin/python -u scripts/run_shogi_playing_strength_eval.py \
  --arena-repo /root/shogi-arena-agent \
  --output-dir \"$OUTPUT_DIR\" \
  --player-a-checkpoint \"$CANDIDATE_CHECKPOINT_REMOTE\" \
  ${CANDIDATE_ID_ARGS[*]} \
  --player-b-checkpoint \"$BASELINE_CHECKPOINT_REMOTE\" \
  ${BASELINE_ID_ARGS[*]} \
  --games \"$GAMES\" \
  --start-position-seed \"$START_POSITION_SEED\" \
  --opening-plies \"$OPENING_PLIES\" \
  --max-plies \"$MAX_PLIES\" \
  --simulations \"$SIMULATIONS\" \
  --nn-leaf-eval-batch-limit \"$NN_LEAF_EVAL_BATCH_LIMIT\" \
  --mcts-move-time-limit-sec \"$MCTS_MOVE_TIME_LIMIT_SEC\" \
  --device cuda \
  --board-backend \"$BOARD_BACKEND\" \
  --progress-every-games \"$PROGRESS_EVERY_GAMES\"
.venv/bin/python - <<'PY' > \"$OUTPUT_DIR/cuda.txt\"
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))
PY" \
  "$@"
