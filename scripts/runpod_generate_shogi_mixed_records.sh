#!/usr/bin/env bash
set -euo pipefail

# Generate fixed shogi game records for later Training Example materialization.
# The game runtime belongs to shogi-arena-agent; this script only orchestrates
# disposable RunPod compute and writes durable source records under data/shogi.

cd "$(dirname "$0")/.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
REPO_PARENT=${REPO_PARENT:-"$(cd "$PWD/../.." && pwd)"}
PROJECT_REL=${PROJECT_REL:-projects/intelligence-representation}
ARENA_REL=${ARENA_REL:-projects/shogi-arena-agent}

CHECKPOINT=${CHECKPOINT:-models/d256-h1024-heads8-l6-shogi/checkpoint.pt}
OUTPUT_NAME=${OUTPUT_NAME:-generated-mix-$(date -u +%Y%m%d-%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/shogi/records/$OUTPUT_NAME}

GAMES_PER_SOURCE=${GAMES_PER_SOURCE:-1024}
TRAIN_RATIO=${TRAIN_RATIO:-0.95}
MAX_PLIES=${MAX_PLIES:-320}
BOARD_BACKEND=${BOARD_BACKEND:-cshogi}
SEED=${SEED:-7}
PROGRESS_EVERY_PLIES=${PROGRESS_EVERY_PLIES:-100}

CHECKPOINT_SIMULATIONS=${CHECKPOINT_SIMULATIONS:-128}
CHECKPOINT_NN_LEAF_EVAL_BATCH_LIMIT=${CHECKPOINT_NN_LEAF_EVAL_BATCH_LIMIT:-64}
CHECKPOINT_WORKER_PROCESSES=${CHECKPOINT_WORKER_PROCESSES:-8}
CHECKPOINT_CONCURRENT_GAMES_PER_PROCESS=${CHECKPOINT_CONCURRENT_GAMES_PER_PROCESS:-8}
ENGINE_WORKER_PROCESSES=${ENGINE_WORKER_PROCESSES:-8}
ENGINE_CONCURRENT_GAMES_PER_PROCESS=${ENGINE_CONCURRENT_GAMES_PER_PROCESS:-1}

USI_GO_COMMAND=${USI_GO_COMMAND:-go nodes 1000}
USI_READ_TIMEOUT_SECONDS=${USI_READ_TIMEOUT_SECONDS:-30}
USI_THREADS=${USI_THREADS:-1}
USI_HASH_MB=${USI_HASH_MB:-128}

YANEURAOU_REPOSITORY_URL=${YANEURAOU_REPOSITORY_URL:-https://github.com/yaneurao/YaneuraOu.git}
YANEURAOU_REF=${YANEURAOU_REF:-master}
YANEURAOU_EDITION=${YANEURAOU_EDITION:-YANEURAOU_ENGINE_NNUE}
YANEURAOU_TARGET_CPU=${YANEURAOU_TARGET_CPU:-AVX2}
YANEURAOU_COMPILER=${YANEURAOU_COMPILER:-g++}
YANEURAOU_BUILD_JOBS=${YANEURAOU_BUILD_JOBS:-8}
SUISHO5_ARCHIVE_URL=${SUISHO5_ARCHIVE_URL:-https://github.com/yaneurao/YaneuraOu/releases/download/suisho5/Suisho5.7z}
TANUKI_ARCHIVE_URL=${TANUKI_ARCHIVE_URL:-https://github.com/nodchip/tanuki-/releases/download/tanuki-.halfkp_256x2-32-32.2023-05-08/tanuki-.halfkp_256x2-32-32.2023-05-08.7z}
SUISHO5_FV_SCALE=${SUISHO5_FV_SCALE:-24}
TANUKI_FV_SCALE=${TANUKI_FV_SCALE:-20}

GPU_TYPE=${GPU_TYPE:-NVIDIA A100-SXM4-80GB}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-720}
CONTAINER_DISK_SIZE=${CONTAINER_DISK_SIZE:-80}
VOLUME_SIZE=${VOLUME_SIZE:-0}
SECURE_CLOUD=${SECURE_CLOUD:-0}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}
MIN_VCPU_PER_GPU=${MIN_VCPU_PER_GPU:-}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$RUNPOD_JOB" ]]; then
  echo "RunPod runner not found: $RUNPOD_JOB" >&2
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
if [[ -n "$MIN_VCPU_PER_GPU" ]]; then
  RUNNER_ARGS+=(--min-vcpu-per-gpu "$MIN_VCPU_PER_GPU")
fi

python3 "$RUNPOD_JOB" \
  --repo-root "$REPO_PARENT" \
  --name intrep-shogi-mixed-records \
  --template-id runpod-torch-v280 \
  --gpu-type "$GPU_TYPE" \
  --gpu-count 1 \
  --container-disk-size "$CONTAINER_DISK_SIZE" \
  --volume-size "$VOLUME_SIZE" \
  "${RUNNER_ARGS[@]}" \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --wait-seconds 900 \
  --ssh-wait-seconds 240 \
  --allow-existing-pods \
  --sync "$PROJECT_REL/src" \
  --sync "$PROJECT_REL/scripts/setup_runpod.sh" \
  --sync "$PROJECT_REL/pyproject.toml" \
  --sync "$PROJECT_REL/uv.lock" \
  --sync "$PROJECT_REL/AGENTS.md" \
  --sync "$PROJECT_REL/$CHECKPOINT" \
  --sync "$ARENA_REL/src" \
  --sync "$ARENA_REL/scripts/generate_shogi_games.py" \
  --sync "$ARENA_REL/pyproject.toml" \
  --sync "$ARENA_REL/uv.lock" \
  --sync "$ARENA_REL/AGENTS.md" \
  --setup-command "cd \"\$REMOTE_DIR/$PROJECT_REL\"; bash scripts/setup_runpod.sh; .venv/bin/python -m pip install -e \"\$REMOTE_DIR/$ARENA_REL\"" \
  --output "$PROJECT_REL/$OUTPUT_ROOT" \
  --timings-output "$PROJECT_REL/$OUTPUT_ROOT/runpod_timings.json" \
  --remote "set -euo pipefail
cd \"\$REMOTE_DIR/$PROJECT_REL\"
OUT=\"$OUTPUT_ROOT\"
GAMES_PER_SOURCE=\"$GAMES_PER_SOURCE\"
TRAIN_RATIO=\"$TRAIN_RATIO\"
SEED=\"$SEED\"
export OUT GAMES_PER_SOURCE TRAIN_RATIO SEED
mkdir -p \"\$OUT/setup-logs\"
apt-get update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y git build-essential curl p7zip-full unzip >/dev/null
rm -rf /root/YaneuraOu /root/shogi-evals
GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch \"$YANEURAOU_REF\" \"$YANEURAOU_REPOSITORY_URL\" /root/YaneuraOu
if ! make -s -C /root/YaneuraOu/source -f Makefile -j\"$YANEURAOU_BUILD_JOBS\" normal TARGET_CPU=\"$YANEURAOU_TARGET_CPU\" YANEURAOU_EDITION=\"$YANEURAOU_EDITION\" COMPILER=\"$YANEURAOU_COMPILER\" TARGET=YaneuraOu-nnue-runpod >\"\$OUT/setup-logs/yaneuraou-build.log\" 2>&1; then
  tail -200 \"\$OUT/setup-logs/yaneuraou-build.log\" >&2
  exit 1
fi
mkdir -p /root/shogi-evals/suisho5/eval /root/shogi-evals/tanuki-hao/eval
curl -L --fail --retry 3 \"$SUISHO5_ARCHIVE_URL\" -o /root/suisho5.7z
7z x -y /root/suisho5.7z -o/root/shogi-evals/suisho5/eval >/dev/null
curl -L --fail --retry 3 \"$TANUKI_ARCHIVE_URL\" -o /root/tanuki.7z
7z x -y /root/tanuki.7z -o/root/shogi-evals/tanuki-hao/extracted >/dev/null
find /root/shogi-evals/tanuki-hao/extracted -type f \\( -name 'nn.bin' -o -name '*.nnue' -o -name '*.bin' \\) | head -n 1 | xargs -r -I{} cp {} /root/shogi-evals/tanuki-hao/eval/nn.bin
test -f /root/shogi-evals/suisho5/eval/nn.bin
test -f /root/shogi-evals/tanuki-hao/eval/nn.bin

PYTHON=\"\$REMOTE_DIR/$PROJECT_REL/.venv/bin/python\"
ARENA=\"\$REMOTE_DIR/$ARENA_REL\"
ENGINE=/root/YaneuraOu/source/YaneuraOu-nnue-runpod
CHECKPOINT_PATH=\"$CHECKPOINT\"

run_generation() {
  local name=\"\$1\"
  local games=\"\$2\"
  shift 2
  mkdir -p \"\$OUT/\$name\"
  echo \"generate_source name=\$name games=\$games\" | tee \"\$OUT/\$name/start.txt\"
  \"\$PYTHON\" -u \"\$ARENA/scripts/generate_shogi_games.py\" \
    --out \"\$OUT/\$name/games.jsonl\" \
    --games \"\$games\" \
    --max-plies \"$MAX_PLIES\" \
    --board-backend \"$BOARD_BACKEND\" \
    --progress-every-plies \"$PROGRESS_EVERY_PLIES\" \
    \"\$@\" | tee \"\$OUT/\$name/summary.json\"
}

checkpoint_args() {
  local side=\"\$1\"
  printf '%s\n' \
    --\"\$side\"-kind checkpoint \
    --\"\$side\"-checkpoint \"\$CHECKPOINT_PATH\" \
    --\"\$side\"-checkpoint-id current-promoted \
    --\"\$side\"-move-selection-profile visit-sampling \
    --\"\$side\"-move-selector mcts \
    --\"\$side\"-mcts-simulations \"$CHECKPOINT_SIMULATIONS\" \
    --\"\$side\"-mcts-nn-leaf-eval-batch-limit \"$CHECKPOINT_NN_LEAF_EVAL_BATCH_LIMIT\" \
    --\"\$side\"-device cuda \
    --\"\$side\"-board-backend \"$BOARD_BACKEND\"
}

engine_args() {
  local side=\"\$1\"
  local eval_dir=\"\$2\"
  local fv_scale=\"\$3\"
  printf '%s\n' \
    --\"\$side\"-kind usi_engine \
    --\"\$side\"-usi-command \"\$ENGINE\" \
    --\"\$side\"-usi-option EvalDir=\"\$eval_dir\" \
    --\"\$side\"-usi-option FV_SCALE=\"\$fv_scale\" \
    --\"\$side\"-usi-option Threads=\"$USI_THREADS\" \
    --\"\$side\"-usi-option Hash=\"$USI_HASH_MB\" \
    --\"\$side\"-usi-option BookFile=no_book \
    --\"\$side\"-usi-go-command \"$USI_GO_COMMAND\" \
    --\"\$side\"-usi-read-timeout-seconds \"$USI_READ_TIMEOUT_SECONDS\"
}

mapfile -t BLACK_CHECKPOINT < <(checkpoint_args black)
mapfile -t WHITE_CHECKPOINT < <(checkpoint_args white)
mapfile -t BLACK_SUISHO < <(engine_args black /root/shogi-evals/suisho5/eval \"$SUISHO5_FV_SCALE\")
mapfile -t WHITE_SUISHO < <(engine_args white /root/shogi-evals/suisho5/eval \"$SUISHO5_FV_SCALE\")
mapfile -t BLACK_TANUKI < <(engine_args black /root/shogi-evals/tanuki-hao/eval \"$TANUKI_FV_SCALE\")
mapfile -t WHITE_TANUKI < <(engine_args white /root/shogi-evals/tanuki-hao/eval \"$TANUKI_FV_SCALE\")

HALF_ENGINE_GAMES=\$((GAMES_PER_SOURCE / 2))
EXTRA_ENGINE_GAME=\$((GAMES_PER_SOURCE % 2))

run_generation checkpoint-self \"$GAMES_PER_SOURCE\" \
  --concurrent-games-per-process \"$CHECKPOINT_CONCURRENT_GAMES_PER_PROCESS\" \
  --generation-worker-processes \"$CHECKPOINT_WORKER_PROCESSES\" \
  --seed \"$SEED\" \
  \"\${BLACK_CHECKPOINT[@]}\" \
  \"\${WHITE_CHECKPOINT[@]}\"

run_generation checkpoint-black-vs-yaneuraou-suisho5 \"\$((HALF_ENGINE_GAMES + EXTRA_ENGINE_GAME))\" \
  --concurrent-games-per-process \"$ENGINE_CONCURRENT_GAMES_PER_PROCESS\" \
  --generation-worker-processes \"$ENGINE_WORKER_PROCESSES\" \
  --seed \"$((SEED + 1000))\" \
  \"\${BLACK_CHECKPOINT[@]}\" \
  \"\${WHITE_SUISHO[@]}\"
run_generation yaneuraou-suisho5-black-vs-checkpoint \"\$HALF_ENGINE_GAMES\" \
  --concurrent-games-per-process \"$ENGINE_CONCURRENT_GAMES_PER_PROCESS\" \
  --generation-worker-processes \"$ENGINE_WORKER_PROCESSES\" \
  --seed \"$((SEED + 2000))\" \
  \"\${BLACK_SUISHO[@]}\" \
  \"\${WHITE_CHECKPOINT[@]}\"

run_generation checkpoint-black-vs-tanuki-hao \"\$((HALF_ENGINE_GAMES + EXTRA_ENGINE_GAME))\" \
  --concurrent-games-per-process \"$ENGINE_CONCURRENT_GAMES_PER_PROCESS\" \
  --generation-worker-processes \"$ENGINE_WORKER_PROCESSES\" \
  --seed \"$((SEED + 3000))\" \
  \"\${BLACK_CHECKPOINT[@]}\" \
  \"\${WHITE_TANUKI[@]}\"
run_generation tanuki-hao-black-vs-checkpoint \"\$HALF_ENGINE_GAMES\" \
  --concurrent-games-per-process \"$ENGINE_CONCURRENT_GAMES_PER_PROCESS\" \
  --generation-worker-processes \"$ENGINE_WORKER_PROCESSES\" \
  --seed \"$((SEED + 4000))\" \
  \"\${BLACK_TANUKI[@]}\" \
  \"\${WHITE_CHECKPOINT[@]}\"

run_generation yaneuraou-suisho5-black-vs-tanuki-hao \"\$((HALF_ENGINE_GAMES + EXTRA_ENGINE_GAME))\" \
  --concurrent-games-per-process \"$ENGINE_CONCURRENT_GAMES_PER_PROCESS\" \
  --generation-worker-processes \"$ENGINE_WORKER_PROCESSES\" \
  --seed \"$((SEED + 5000))\" \
  \"\${BLACK_SUISHO[@]}\" \
  \"\${WHITE_TANUKI[@]}\"
run_generation tanuki-hao-black-vs-yaneuraou-suisho5 \"\$HALF_ENGINE_GAMES\" \
  --concurrent-games-per-process \"$ENGINE_CONCURRENT_GAMES_PER_PROCESS\" \
  --generation-worker-processes \"$ENGINE_WORKER_PROCESSES\" \
  --seed \"$((SEED + 6000))\" \
  \"\${BLACK_TANUKI[@]}\" \
  \"\${WHITE_SUISHO[@]}\"

\"\$PYTHON\" - <<'PY'
from __future__ import annotations

import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path

out = Path(os.environ['OUT'])
train_ratio = float(os.environ['TRAIN_RATIO'])
seed = int(os.environ['SEED'])
source_dirs = [path for path in sorted(out.iterdir()) if (path / 'games.jsonl').is_file()]
records = []
source_counts = {}
for source_dir in source_dirs:
    count = 0
    with (source_dir / 'games.jsonl').open(encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            records.append((source_dir.name, line))
            count += 1
    source_counts[source_dir.name] = count
random.Random(seed).shuffle(records)
train_count = int(len(records) * train_ratio)
splits = {
    'train-games.jsonl': records[:train_count],
    'eval-games.jsonl': records[train_count:],
}
for filename, split_records in splits.items():
    with (out / filename).open('w', encoding='utf-8') as file:
        for _source, line in split_records:
            file.write(line)
manifest = {
    'schema_version': 'intrep.shogi_generated_record_mix.v1',
    'name': out.name,
    'created_at': datetime.now(UTC).isoformat(),
    'source_counts': source_counts,
    'game_count': len(records),
    'train_games': len(splits['train-games.jsonl']),
    'eval_games': len(splits['eval-games.jsonl']),
    'train_ratio': train_ratio,
    'seed': seed,
    'files': {
        'train_games': 'train-games.jsonl',
        'eval_games': 'eval-games.jsonl',
        'sources': {source: f'{source}/games.jsonl' for source in source_counts},
    },
}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(manifest, indent=2))
PY" \
  "$@"
