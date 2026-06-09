#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
PROJECTS_ROOT=${PROJECTS_ROOT:-"$(dirname "$PWD")"}
ARENA_REPO=${ARENA_REPO:-"$PROJECTS_ROOT/shogi-arena-agent"}
OUTPUT_DIR=${OUTPUT_DIR:-"intelligence-representation/runs/shogi/checkpoint-self-play-central-evaluator-$(date -u +%Y%m%d-%H%M%S)"}
CHECKPOINT_REL=${CHECKPOINT_REL:-models/shogi-minimal-split-global-position-action-plane-mcts256-full}
GPU_TYPE=${GPU_TYPE:-"NVIDIA RTX 4000 Ada Generation"}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-20}
CASE_SET=${CASE_SET:-"mcts128"}
INFERENCE_PRECISION=${INFERENCE_PRECISION:-bf16}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-""}
MIN_VCPU_PER_GPU=${MIN_VCPU_PER_GPU:-""}
SECURE_CLOUD=${SECURE_CLOUD:-""}

if [[ ! -d "$ARENA_REPO" ]]; then
  echo "shogi-arena-agent repo not found: $ARENA_REPO" >&2
  exit 1
fi

EXTRA_RUNPOD_ARGS=()
if [[ -n "$DATA_CENTER_IDS" ]]; then
  EXTRA_RUNPOD_ARGS+=(--data-center-ids "$DATA_CENTER_IDS")
fi
if [[ -n "$MIN_VCPU_PER_GPU" ]]; then
  EXTRA_RUNPOD_ARGS+=(--min-vcpu-per-gpu "$MIN_VCPU_PER_GPU")
fi
if [[ -n "$SECURE_CLOUD" ]]; then
  EXTRA_RUNPOD_ARGS+=(--secure-cloud)
fi

python3 "$RUNPOD_JOB" \
  --repo-root "$PROJECTS_ROOT" \
  --name intrep-shogi-checkpoint-self-play-central-evaluator \
  --template-id runpod-torch-v280 \
  --gpu-type "$GPU_TYPE" \
  --gpu-count 1 \
  --container-disk-size 20 \
  --volume-size 0 \
  "${EXTRA_RUNPOD_ARGS[@]}" \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --wait-seconds 900 \
  --ssh-wait-seconds 240 \
  --allow-existing-pods \
  --sync intelligence-representation/src \
  --sync intelligence-representation/scripts/setup_runpod.sh \
  --sync intelligence-representation/pyproject.toml \
  --sync intelligence-representation/README.md \
  --sync intelligence-representation/AGENTS.md \
  --sync "intelligence-representation/$CHECKPOINT_REL" \
  --sync shogi-arena-agent/src \
  --sync shogi-arena-agent/scripts/generate_checkpoint_self_play_games.py \
  --sync shogi-arena-agent/pyproject.toml \
  --setup-command 'set -euo pipefail; cd "$REMOTE_DIR/intelligence-representation"; bash scripts/setup_runpod.sh; .venv/bin/python -m pip install -e "$REMOTE_DIR/shogi-arena-agent"' \
  --output "$OUTPUT_DIR" \
  --timings-output "$PROJECTS_ROOT/$OUTPUT_DIR/runpod_timings.json" \
  --remote "set -euo pipefail
cd \"\$REMOTE_DIR\"
OUT=\"$OUTPUT_DIR\"
mkdir -p \"\$OUT\"
cat > /tmp/measure_checkpoint_self_play_central.py <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time

REMOTE = Path(os.environ['REMOTE_DIR'])
INTREP = REMOTE / 'intelligence-representation'
ARENA = REMOTE / 'shogi-arena-agent'
PYTHON = INTREP / '.venv/bin/python'
OUT = REMOTE / os.environ['MEASURE_OUT']
CASE_SET = os.environ.get('MEASURE_CASE_SET', 'mcts128')
INFERENCE_PRECISION = os.environ.get('INFERENCE_PRECISION', 'bf16')
CHECKPOINT = INTREP / os.environ['MEASURE_CHECKPOINT_REL']

CASE_SETS = {
    'mcts256': [
        (f'w16_c1_s256_b64_g16_{INFERENCE_PRECISION}', 16, 1, 16, 256, 64),
    ],
    'mcts128': [
        (f'w16_c1_s128_b64_g16_{INFERENCE_PRECISION}', 16, 1, 16, 128, 64),
    ],
    'mcts64': [
        (f'w16_c1_s64_b64_g16_{INFERENCE_PRECISION}', 16, 1, 16, 64, 64),
    ],
    'mcts16-scaling': [
        ('w4_c4_s16_b32_g64', 4, 4, 64, 16, 32),
        ('w8_c4_s16_b32_g64', 8, 4, 64, 16, 32),
        ('w16_c4_s16_b32_g64', 16, 4, 64, 16, 32),
    ],
}
try:
    CASES = CASE_SETS[CASE_SET]
except KeyError as exc:
    raise ValueError(f'unknown MEASURE_CASE_SET: {CASE_SET}') from exc


def gpu_sample() -> tuple[float | None, str | None]:
    command = [
        'nvidia-smi',
        '--query-gpu=utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits',
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip().splitlines()[0]
    except Exception:
        return None, None
    parts = [part.strip() for part in output.split(',')]
    if len(parts) < 3:
        return None, None
    return float(parts[0]), f'{parts[1]} MiB / {parts[2]} MiB'


def cpu_sample(pid: int) -> float | None:
    try:
        raw = subprocess.check_output(['ps', '-p', str(pid), '-o', 'pcpu='], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None
    return float(raw) if raw else None


def rss_sample(pid: int) -> int | None:
    try:
        raw = subprocess.check_output(['ps', '-p', str(pid), '-o', 'rss='], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None
    return int(raw) if raw else None


def load_summary(path: Path) -> dict[str, object]:
    text = path.read_text(encoding='utf-8')
    matches = list(re.finditer(r'^\{', text, flags=re.MULTILINE))
    for match in reversed(matches):
        try:
            return json.loads(text[match.start():])
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f'no JSON summary found in {path}')


def run_case(
    name: str,
    worker_processes: int,
    concurrent_games_per_worker: int,
    games: int,
    simulations: int,
    batch_limit: int,
) -> dict[str, object]:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = case_dir / 'generate_stdout.json'
    stderr_path = case_dir / 'generate_stderr.log'
    records_path = case_dir / 'games.jsonl'
    command = [
        str(PYTHON), '-u', str(ARENA / 'scripts/generate_checkpoint_self_play_games.py'),
        '--checkpoint', str(CHECKPOINT),
        '--checkpoint-id', 'shogi-minimal-split-global-action-plane',
        '--out', str(records_path),
        '--games', str(games),
        '--self-play-worker-processes', str(worker_processes),
        '--concurrent-games-per-process', str(concurrent_games_per_worker),
        '--mcts-simulations', str(simulations),
        '--mcts-nn-leaf-eval-batch-limit', str(batch_limit),
        '--central-evaluator-batch-size-limit', str(batch_limit),
        '--central-evaluator-flush-timeout-sec', '0.002',
        '--inference-precision', INFERENCE_PRECISION,
        '--max-plies', '320',
        '--device', 'cuda',
        '--board-backend', 'cshogi',
    ]
    samples: list[dict[str, object]] = []
    started = time.monotonic()
    with stdout_path.open('w', encoding='utf-8') as stdout, stderr_path.open('w', encoding='utf-8') as stderr:
        proc = subprocess.Popen(command, cwd=ARENA, stdout=stdout, stderr=stderr, text=True)
        while proc.poll() is None:
            gpu_util, gpu_memory = gpu_sample()
            samples.append(
                {
                    'elapsed_sec': time.monotonic() - started,
                    'gpu_util': gpu_util,
                    'gpu_memory': gpu_memory,
                    'generator_cpu': cpu_sample(proc.pid),
                    'generator_rss': rss_sample(proc.pid),
                }
            )
            time.sleep(2.0)
        return_code = proc.wait()
    wall = time.monotonic() - started
    if return_code != 0:
        raise RuntimeError(f'case {name} failed with exit code {return_code}; see {stderr_path}')
    summary = load_summary(stdout_path)
    central = summary.get('central_evaluator_performance', {})
    gpu_values = [float(sample['gpu_util']) for sample in samples if isinstance(sample.get('gpu_util'), int | float)]
    cpu_values = [float(sample['generator_cpu']) for sample in samples if isinstance(sample.get('generator_cpu'), int | float)]
    rss_values = [int(sample['generator_rss']) for sample in samples if isinstance(sample.get('generator_rss'), int)]
    memory_values = [str(sample['gpu_memory']) for sample in samples if sample.get('gpu_memory')]
    result = {
        'case': name,
        'total_games': games,
        'self_play_worker_processes': worker_processes,
        'concurrent_games_per_worker': concurrent_games_per_worker,
        'mcts_simulations_per_move': simulations,
        'nn_leaf_eval_batch_limit': batch_limit,
        'inference_precision': INFERENCE_PRECISION,
        'average_plies': summary.get('average_plies'),
        'wall_sec': summary.get('generation_wall_time_sec', wall),
        'plies_per_sec': summary.get('plies_per_sec'),
        'central_model_call_count': central.get('model_call_count'),
        'central_model_wall_time_sec': central.get('model_wall_time_sec'),
        'central_request_count': central.get('request_count'),
        'central_batch_first_wait_sec': central.get('batch_first_wait_sec'),
        'central_batch_fill_wait_sec': central.get('batch_fill_wait_sec'),
        'central_response_send_wall_time_sec': central.get('response_send_wall_time_sec'),
        'central_request_queue_wait_sec_avg': central.get('request_queue_wait_sec_avg'),
        'central_request_queue_wait_sec_max': central.get('request_queue_wait_sec_max'),
        'central_batch_avg': central.get('actual_nn_leaf_eval_batch_size_avg'),
        'central_batch_max': central.get('actual_nn_leaf_eval_batch_size_max'),
        'central_batch_fill': central.get('actual_nn_leaf_eval_batch_size_fill_ratio_avg'),
        'gpu_util_avg': sum(gpu_values) / len(gpu_values) if gpu_values else None,
        'gpu_util_max': max(gpu_values) if gpu_values else None,
        'gpu_memory_used': max(memory_values, key=lambda value: int(value.split()[0])) if memory_values else None,
        'generator_cpu_avg': sum(cpu_values) / len(cpu_values) if cpu_values else None,
        'generator_cpu_max': max(cpu_values) if cpu_values else None,
        'generator_rss': f'{max(rss_values) // 1024} MiB' if rss_values else None,
        'sample_count': len(samples),
        'summary': summary,
    }
    result.update({f'central_{key}': value for key, value in central.items() if key.startswith('backend_')})
    (case_dir / 'measurement.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    environment = subprocess.check_output(
        [
            str(PYTHON),
            '-c',
            'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))',
        ],
        text=True,
    )
    (OUT / 'environment.txt').write_text(environment, encoding='utf-8')
    results = [run_case(*case) for case in CASES]
    (OUT / 'measurement-summary.json').write_text(
        json.dumps({'results': results}, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
PY
REMOTE_DIR=\"\$REMOTE_DIR\" MEASURE_OUT=\"\$OUT\" MEASURE_CASE_SET=\"$CASE_SET\" MEASURE_CHECKPOINT_REL=\"$CHECKPOINT_REL\" INFERENCE_PRECISION=\"$INFERENCE_PRECISION\" \"intelligence-representation/.venv/bin/python\" /tmp/measure_checkpoint_self_play_central.py" \
  "$@"
