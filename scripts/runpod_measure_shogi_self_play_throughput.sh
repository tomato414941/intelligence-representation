#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
PROJECTS_ROOT=${PROJECTS_ROOT:-"$(dirname "$PWD")"}
ARENA_REPO=${ARENA_REPO:-"$PROJECTS_ROOT/shogi-arena-agent"}
OUTPUT_DIR=${OUTPUT_DIR:-"intelligence-representation/runs/shogi/self-play-throughput-rtx4000ada-$(date -u +%Y%m%d-%H%M%S)"}
GPU_TYPE=${GPU_TYPE:-"NVIDIA RTX 4000 Ada Generation"}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-60}
CASE_SET=${CASE_SET:-"worker-scaling"}
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
  --name intrep-shogi-self-play-throughput \
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
  --sync intelligence-representation/models/d256-h1024-heads8-l6-shogi \
  --sync shogi-arena-agent/src \
  --sync shogi-arena-agent/scripts/generate_shogi_games.py \
  --sync shogi-arena-agent/pyproject.toml \
  --setup-command 'set -euo pipefail; cd "$REMOTE_DIR/intelligence-representation"; bash scripts/setup_runpod.sh; .venv/bin/python -m pip install -e "$REMOTE_DIR/shogi-arena-agent"' \
  --output "$OUTPUT_DIR" \
  --timings-output "$PROJECTS_ROOT/$OUTPUT_DIR/runpod_timings.json" \
  --remote "set -euo pipefail
cd \"\$REMOTE_DIR\"
OUT=\"$OUTPUT_DIR\"
mkdir -p \"\$OUT\"
cat > /tmp/measure_self_play.py <<'PY'
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
CHECKPOINT = INTREP / 'models/d256-h1024-heads8-l6-shogi/checkpoint.pt'
OUT = REMOTE / os.environ['MEASURE_OUT']
CASE_SET = os.environ.get('MEASURE_CASE_SET', 'worker-scaling')
CASE_SETS = {
    'worker-scaling': [
        ('w1_c16_s16_b32', 16, 16, 1, 16, 32),
        ('w2_c8_s16_b32', 16, 8, 2, 16, 32),
        ('w4_c8_s16_b32', 32, 8, 4, 16, 32),
        ('w6_c8_s16_b32', 48, 8, 6, 16, 32),
    ],
    'worker8': [
        ('w8_c8_s16_b32', 64, 8, 8, 16, 32),
    ],
    'worker10': [
        ('w10_c8_s16_b32', 80, 8, 10, 16, 32),
    ],
    'worker12': [
        ('w12_c8_s16_b32', 96, 8, 12, 16, 32),
    ],
    'worker6': [
        ('w6_c8_s16_b32', 48, 8, 6, 16, 32),
    ],
    'worker6-batch64': [
        ('w6_c8_s16_b64', 48, 8, 6, 16, 64),
    ],
    'worker8-batch64': [
        ('w8_c8_s16_b64', 64, 8, 8, 16, 64),
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


def child_pids(pid: int) -> set[int]:
    try:
        output = subprocess.check_output(['pgrep', '-P', str(pid)], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return set()
    result = {int(line) for line in output.splitlines() if line.strip().isdigit()}
    for child in list(result):
        result.update(child_pids(child))
    return result


def cpu_sample(root_pid: int) -> float | None:
    pids = {root_pid, *child_pids(root_pid)}
    total = 0.0
    seen = False
    for pid in pids:
        try:
            raw = subprocess.check_output(
                ['ps', '-p', str(pid), '-o', 'pcpu='],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except subprocess.CalledProcessError:
            continue
        if not raw:
            continue
        total += float(raw)
        seen = True
    return total if seen else None


def rss_sample(root_pid: int) -> int | None:
    pids = {root_pid, *child_pids(root_pid)}
    total_kib = 0
    seen = False
    for pid in pids:
        try:
            raw = subprocess.check_output(
                ['ps', '-p', str(pid), '-o', 'rss='],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except subprocess.CalledProcessError:
            continue
        if not raw:
            continue
        total_kib += int(raw)
        seen = True
    return total_kib if seen else None


def system_ram_sample() -> str | None:
    try:
        values: dict[str, int] = {}
        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            key, value = line.split(':', 1)
            values[key] = int(value.strip().split()[0])
    except Exception:
        return None
    total = values.get('MemTotal')
    available = values.get('MemAvailable')
    if total is None or available is None:
        return None
    used = total - available
    return f'{used // 1024} MiB / {total // 1024} MiB'


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
    games: int,
    concurrent_games_per_process: int,
    worker_processes: int,
    simulations: int,
    batch: int,
) -> dict[str, object]:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = case_dir / 'generate_stdout.json'
    stderr_path = case_dir / 'generate_stderr.log'
    records_path = case_dir / 'games.jsonl'
    command = [
        str(PYTHON), '-u', str(ARENA / 'scripts/generate_shogi_games.py'),
        '--out', str(records_path),
        '--games', str(games),
        '--concurrent-games-per-process', str(concurrent_games_per_process),
        '--generation-worker-processes', str(worker_processes),
        '--max-plies', '320',
        '--board-backend', 'cshogi',
        '--progress-every-plies', '50',
        '--black-kind', 'checkpoint',
        '--black-checkpoint', str(CHECKPOINT),
        '--black-move-selection-profile', 'visit-sampling',
        '--black-move-selector', 'mcts',
        '--black-mcts-simulations', str(simulations),
        '--black-mcts-nn-leaf-eval-batch-limit', str(batch),
        '--black-device', 'cuda',
        '--black-board-backend', 'cshogi',
        '--white-kind', 'checkpoint',
        '--white-checkpoint', str(CHECKPOINT),
        '--white-move-selection-profile', 'visit-sampling',
        '--white-move-selector', 'mcts',
        '--white-mcts-simulations', str(simulations),
        '--white-mcts-nn-leaf-eval-batch-limit', str(batch),
        '--white-device', 'cuda',
        '--white-board-backend', 'cshogi',
    ]
    samples: list[dict[str, object]] = []
    started = time.monotonic()
    with stdout_path.open('w', encoding='utf-8') as stdout, stderr_path.open('w', encoding='utf-8') as stderr:
        proc = subprocess.Popen(command, cwd=ARENA, stdout=stdout, stderr=stderr, text=True)
        while proc.poll() is None:
            gpu_util, gpu_memory = gpu_sample()
            generator_cpu = cpu_sample(proc.pid)
            generator_rss = rss_sample(proc.pid)
            samples.append(
                {
                    'elapsed_sec': time.monotonic() - started,
                    'gpu_util': gpu_util,
                    'gpu_memory': gpu_memory,
                    'generator_cpu': generator_cpu,
                    'generator_rss': generator_rss,
                    'system_ram': system_ram_sample(),
                }
            )
            time.sleep(2.0)
        return_code = proc.wait()
    wall = time.monotonic() - started
    if return_code != 0:
        raise RuntimeError(f'case {name} failed with exit code {return_code}; see {stderr_path}')
    summary = load_summary(stdout_path)
    gpu_values = [float(sample['gpu_util']) for sample in samples if isinstance(sample.get('gpu_util'), int | float)]
    cpu_values = [
        float(sample['generator_cpu'])
        for sample in samples
        if isinstance(sample.get('generator_cpu'), int | float)
    ]
    memory_values = [str(sample['gpu_memory']) for sample in samples if sample.get('gpu_memory')]
    rss_values = [int(sample['generator_rss']) for sample in samples if isinstance(sample.get('generator_rss'), int)]
    system_ram_values = [str(sample['system_ram']) for sample in samples if sample.get('system_ram')]
    result = {
        'case': name,
        'total_games': games,
        'concurrent_games_per_process': concurrent_games_per_process,
        'generation_worker_processes': worker_processes,
        'mcts_simulations_per_move': simulations,
        'nn_leaf_eval_batch_limit': batch,
        'average_plies': summary.get('average_plies'),
        'wall_sec': summary.get('generation_wall_time_sec', wall),
        'plies_per_sec': summary.get('plies_per_sec'),
        'gpu_util_avg': sum(gpu_values) / len(gpu_values) if gpu_values else None,
        'gpu_util_max': max(gpu_values) if gpu_values else None,
        'gpu_memory_used': max(memory_values, key=lambda value: int(value.split()[0])) if memory_values else None,
        'generator_cpu_avg': sum(cpu_values) / len(cpu_values) if cpu_values else None,
        'generator_cpu_max': max(cpu_values) if cpu_values else None,
        'system_ram_used': max(system_ram_values, key=lambda value: int(value.split()[0])) if system_ram_values else None,
        'generator_rss': f'{max(rss_values) // 1024} MiB' if rss_values else None,
        'sample_count': len(samples),
        'summary': summary,
    }
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
REMOTE_DIR=\"\$REMOTE_DIR\" MEASURE_OUT=\"\$OUT\" MEASURE_CASE_SET=\"$CASE_SET\" \"intelligence-representation/.venv/bin/python\" /tmp/measure_self_play.py" \
  "$@"
