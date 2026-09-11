#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
retention_output=${1:?output directory is required}
retention_steps=${2:-300}
retention_work=runs/instruction-retention-work-20260911
retention_archive=shared-prediction/instruction-retention-20260911
: "${R2_ENV_FILE:?R2_ENV_FILE is required for durable checkpoint storage}"
set -a
. "$R2_ENV_FILE"
set +a
export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Other
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2_ENDPOINT="$R2_ENDPOINT"
mkdir -p "$retention_output"

.venv/bin/python - "$retention_output" "$retention_steps" <<'PY'
import copy
import json
import sys
from pathlib import Path

import torch
import transformers

if not torch.cuda.is_available():
    raise SystemExit("The retention comparison requires CUDA.")
root = Path(sys.argv[1])
recipe = json.loads(Path("configs/question-learning.json").read_text())
recipe["defaults"].update(question_evaluation_examples=8, question_evaluation_worlds=4)
for condition, objective, weight in (("chunked", "all_tokens", 1.), ("assistant", "assistant", 1.),
                                     ("assistant_weighted", "assistant", 8.)):
    config = copy.deepcopy(recipe)
    conversation = next(source for source in config["sources"] if source["name"] == "conversations")
    conversation.update(path="data/instruction-retention-20260911/train.jsonl",
                        evaluation={"path": "data/instruction-retention-20260911/validation.jsonl"},
                        conversation_objective=objective, weight=weight)
    if objective == "assistant":
        conversation.update(conversation_tokens=2048, conversation_overlap=1024)
    (root / f"{condition}-recipe.json").write_text(json.dumps(config, indent=2) + "\n")
(root / "environment.json").write_text(json.dumps({
    "torch": str(torch.__version__), "transformers": transformers.__version__,
    "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
    "model": "LFM2.5-350M", "base": "models/lfm2.5-350m",
    "precision": "float32", "optimizer": "AdamW", "learning_rate": 1e-5,
    "steps_per_condition": int(sys.argv[2]), "generation_interval": 50,
    "all_sources_per_update": True, "all_parameters_trainable": True,
    "comparison": "same initial weights and populations; fixed optimizer updates, not matched token counts, time or FLOPs",
    "conversation_control": "chunked uses the old 128-token all-role objective with the same expanded bilingual corpus as the other conditions",
    "contrast": "chunked vs assistant changes context and label masking together; assistant vs assistant_weighted changes only source weight",
}, indent=2) + "\n")
PY

for retention_condition in chunked assistant assistant_weighted; do
  HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python -u scripts/train_shared_prediction.py \
    --base models/lfm2.5-350m --recipe "$retention_output/$retention_condition-recipe.json" \
    --extension intrep.problems.shared_prediction.record_sources \
    --data-root . --output "$retention_work/$retention_condition" \
    --steps "$retention_steps" --device cuda --threads 4 \
    --optimizer adamw --learning-rate 0.00001 --max-grad-norm 1 \
    --checkpoint-interval 10000 --evaluation-examples 32 --evaluation-interval 100 \
    --generation-interval 50 --gradient-probe-interval 50 \
    --prompts configs/question-learning-prompts.json \
    --holdout-prompts data/instruction-retention-20260911/holdout-prompts.json
  .venv/bin/python -u scripts/archive_instruction_retention.py \
    --directory "$retention_work/$retention_condition" --data-root . \
    --prefix "$retention_archive/$retention_condition" \
    --local-output "$retention_output/$retention_condition"
done

.venv/bin/python - "$retention_output" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
names = ("chunked", "assistant", "assistant_weighted")
results = [json.loads((root / name / "result.json").read_text()) for name in names]
assert len({result["initial_parameters_sha256"] for result in results}) == 1
assert len({result["completed_steps"] for result in results}) == 1
assert all(len(result["source_progress"]) == 12 for result in results)
assert all(result["parameters"] == result["trainable_parameters"] for result in results)
panels = [json.loads((root / name / "evaluation-panel.json").read_text()) for name in names]
assert panels[0] == panels[1] == panels[2]
generations = [(root / name / "generations/step-000000.json").read_bytes() for name in names]
assert generations[0] == generations[1] == generations[2]
print(json.dumps({"stage": "retention_comparison_complete", "conditions": names,
                  "steps": [result["completed_steps"] for result in results],
                  "training_seconds": [result["training_seconds"] for result in results]}), flush=True)
PY
