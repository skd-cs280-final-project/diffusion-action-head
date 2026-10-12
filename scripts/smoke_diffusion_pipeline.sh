#!/usr/bin/env bash
# Real-label integration check: train split -> validation -> checkpoint reload.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/diffusion_smoke_real}"

# Fail before loading the large dataset/model if the execution environment
# does not expose CUDA. The smoke checkpoint is not a trained navigation policy.
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable for real-model smoke check"'
bash scripts/train_diffusion_head.sh \
    --output-dir "$SMOKE_OUTPUT_DIR" --max-samples 4 --max-val-samples 8 \
    --epochs 1 --max-steps 2 --batch-size 2 --gradient-accumulation-steps 1 \
    --eval-batch-size 2 --seed 42 --eval-seed 42 --num-workers 0 --save-steps 1
bash scripts/eval_diffusion_head.sh \
    --checkpoint "$SMOKE_OUTPUT_DIR" --max-samples 8 --batch-size 2 \
    --seed 42 --num-workers 0 --output "$SMOKE_OUTPUT_DIR/offline_eval.json"
"$PYTHON" - "$SMOKE_OUTPUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
last_validation = json.loads((root / "validation_metrics.jsonl").read_text().splitlines()[-1])
report = json.loads((root / "offline_eval.json").read_text())
metadata = json.loads((root / "diffusion_config.json").read_text())["training_metadata"]
assert metadata["global_step"] == 2
assert metadata["train_split"]["split"] == "train"
assert metadata["train_split"]["num_samples"] == 4
assert metadata["val_split"]["split"] == "val"
assert metadata["val_split"]["num_samples"] == 8
assert report["split_provenance"]["status"] == "verified_diffusion_head_holdout"
assert report["metrics"]["num_samples"] == 8
for key in ("noise_mse", "normalized_continuous_mse"):
    assert math.isclose(last_validation[key], report["metrics"][key], rel_tol=1e-5, abs_tol=1e-6), key
assert last_validation["land"] == report["metrics"]["land"]
result = {"status": "passed", "train_samples": metadata["train_split"]["num_samples"],
          "validation_samples": report["metrics"]["num_samples"],
          "optimizer_steps": metadata["global_step"], "reload_metrics_match": True,
          "scope": "Real-label pipeline check only; no convergence or navigation claim"}
(root / "pipeline_check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
