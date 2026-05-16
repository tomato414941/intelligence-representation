#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
REPO_PARENT=${REPO_PARENT:-"$(cd "$PWD/../.." && pwd)"}
PROJECT_REL=${PROJECT_REL:-projects/intelligence-representation}
ARENA_REL=${ARENA_REL:-projects/shogi-arena-agent}

PLAYER_A_CHECKPOINT=${PLAYER_A_CHECKPOINT:-models/d256-h1024-heads8-l6-shogi/checkpoint.pt}
PLAYER_B_CHECKPOINT=${PLAYER_B_CHECKPOINT:-}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/player-matches-runpod-$(date -u +%Y%m%d-%H%M%S)}

GAMES=${GAMES:-16}
MAX_PLIES=${MAX_PLIES:-320}
SIMULATIONS=${SIMULATIONS:-128}
NN_LEAF_EVAL_BATCH_LIMIT=${NN_LEAF_EVAL_BATCH_LIMIT:-64}
MOVE_SELECTION_PROFILE=${MOVE_SELECTION_PROFILE:-self-play}
USI_GO_COMMAND=${USI_GO_COMMAND:-go nodes 1}
YANEURAOU_REPOSITORY_URL=${YANEURAOU_REPOSITORY_URL:-https://github.com/yaneurao/YaneuraOu.git}

GPU_TYPE=${GPU_TYPE:-NVIDIA RTX A5000}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-120}
CONTAINER_DISK_SIZE=${CONTAINER_DISK_SIZE:-80}
VOLUME_SIZE=${VOLUME_SIZE:-0}
SECURE_CLOUD=${SECURE_CLOUD:-1}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}

if [[ ! -f "$PLAYER_A_CHECKPOINT" ]]; then
  echo "player A checkpoint not found: $PLAYER_A_CHECKPOINT" >&2
  exit 1
fi
if [[ -n "$PLAYER_B_CHECKPOINT" && ! -f "$PLAYER_B_CHECKPOINT" ]]; then
  echo "player B checkpoint not found: $PLAYER_B_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$REPO_PARENT/$ARENA_REL" ]]; then
  echo "shogi-arena-agent not found: $REPO_PARENT/$ARENA_REL" >&2
  exit 1
fi

RUNNER_ARGS=()
if [[ "$SECURE_CLOUD" == "1" ]]; then
  RUNNER_ARGS+=(--secure-cloud)
fi
if [[ -n "$DATA_CENTER_IDS" ]]; then
  RUNNER_ARGS+=(--data-center-ids "$DATA_CENTER_IDS")
fi
SYNC_ARGS=(--sync "$PROJECT_REL/$PLAYER_A_CHECKPOINT")
if [[ -n "$PLAYER_B_CHECKPOINT" ]]; then
  SYNC_ARGS+=(--sync "$PROJECT_REL/$PLAYER_B_CHECKPOINT")
fi

python3 "$RUNPOD_JOB" \
  --repo-root "$REPO_PARENT" \
  --name intrep-shogi-player-matches \
  --template-id runpod-torch-v280 \
  --gpu-type "$GPU_TYPE" \
  --container-disk-size "$CONTAINER_DISK_SIZE" \
  --volume-size "$VOLUME_SIZE" \
  "${RUNNER_ARGS[@]}" \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --wait-seconds 600 \
  --ssh-wait-seconds 180 \
  --allow-existing-pods \
  --sync "$PROJECT_REL/src" \
  --sync "$PROJECT_REL/scripts/run_shogi_player_match.py" \
  --sync "$PROJECT_REL/scripts/setup_runpod.sh" \
  --sync "$PROJECT_REL/pyproject.toml" \
  --sync "$PROJECT_REL/uv.lock" \
  --sync "$PROJECT_REL/AGENTS.md" \
  "${SYNC_ARGS[@]}" \
  --sync "$ARENA_REL/src" \
  --sync "$ARENA_REL/scripts/evaluate_shogi_players.py" \
  --sync "$ARENA_REL/pyproject.toml" \
  --sync "$ARENA_REL/uv.lock" \
  --sync "$ARENA_REL/AGENTS.md" \
  --setup-command "cd \"\$REMOTE_DIR/$PROJECT_REL\"; bash scripts/setup_runpod.sh; .venv/bin/python -m pip install -e \"\$REMOTE_DIR/$ARENA_REL\"" \
  --output "$PROJECT_REL/$OUTPUT_DIR" \
  --timings-output "$PROJECT_REL/$OUTPUT_DIR/runpod_timings.json" \
  --remote "set -euo pipefail
cd \"\$REMOTE_DIR/$PROJECT_REL\"
export SHOGI_ARENA_PYTHON=\"\$REMOTE_DIR/$PROJECT_REL/.venv/bin/python\"
mkdir -p \"$OUTPUT_DIR/sampled-vs-checkpoint\" \"$OUTPUT_DIR/sampled-vs-yaneuraou\"

if [[ -n \"$PLAYER_B_CHECKPOINT\" ]]; then
.venv/bin/python -u scripts/run_shogi_player_match.py \
  --arena-repo \"\$REMOTE_DIR/$ARENA_REL\" \
  --player-a-kind checkpoint \
  --player-a-checkpoint \"$PLAYER_A_CHECKPOINT\" \
  --player-b-kind checkpoint \
  --player-b-checkpoint \"$PLAYER_B_CHECKPOINT\" \
  --out \"$OUTPUT_DIR/sampled-vs-checkpoint/games.jsonl\" \
  --games \"$GAMES\" \
  --max-plies \"$MAX_PLIES\" \
  --simulations \"$SIMULATIONS\" \
  --evaluation-batch-size \"$NN_LEAF_EVAL_BATCH_LIMIT\" \
  --move-selection-profile \"$MOVE_SELECTION_PROFILE\" \
  --device cuda \
  --board-backend cshogi | tee \"$OUTPUT_DIR/sampled-vs-checkpoint/summary.json\"
fi

apt-get update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y git build-essential >/dev/null
rm -rf /root/YaneuraOu
GIT_TERMINAL_PROMPT=0 git clone --depth 1 \"$YANEURAOU_REPOSITORY_URL\" /root/YaneuraOu
make -s -C /root/YaneuraOu/source -f Makefile -j\"\$(nproc)\" normal TARGET_CPU=AVX2 YANEURAOU_EDITION=YANEURAOU_ENGINE_MATERIAL COMPILER=g++ TARGET=YaneuraOu-runpod

.venv/bin/python -u scripts/run_shogi_player_match.py \
  --arena-repo \"\$REMOTE_DIR/$ARENA_REL\" \
  --player-a-kind checkpoint \
  --player-a-checkpoint \"$PLAYER_A_CHECKPOINT\" \
  --player-b-kind usi \
  --player-b-usi-command /root/YaneuraOu/source/YaneuraOu-runpod \
  --player-b-usi-go-command \"$USI_GO_COMMAND\" \
  --out \"$OUTPUT_DIR/sampled-vs-yaneuraou/games.jsonl\" \
  --games \"$GAMES\" \
  --max-plies \"$MAX_PLIES\" \
  --simulations \"$SIMULATIONS\" \
  --evaluation-batch-size \"$NN_LEAF_EVAL_BATCH_LIMIT\" \
  --move-selection-profile \"$MOVE_SELECTION_PROFILE\" \
  --device cuda \
  --board-backend cshogi | tee \"$OUTPUT_DIR/sampled-vs-yaneuraou/summary.json\"" \
  "$@"
