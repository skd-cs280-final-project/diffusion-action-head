#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHON="${PYTHON:-/data/xiyang/.conda/envs/aero_vla/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/joint_smoke_20261103}"
bash scripts/train_diffusion_joint.sh \
    --checkpoint ./checkpoints/best --model-path ./openvla-7b --adapter-path ./checkpoints/aero_vla \
    --output-dir "$SMOKE_OUTPUT_DIR" --epochs 1 --max-steps 2 \
    --max-samples 8 --max-val-samples 4 --batch-size 1 --eval-batch-size 1 \
    --gradient-accumulation-steps 2 --num-workers 0 --save-steps 2
# Independent single-process inference must restore both adapter and action head.
NPROC_PER_NODE=1 bash scripts/eval_diffusion_head.sh \
    --checkpoint "$SMOKE_OUTPUT_DIR" --max-samples 4 --batch-size 1 \
    --seed 42 --num-workers 0 --output "$SMOKE_OUTPUT_DIR/offline_eval.json"
"$PYTHON" - "$SMOKE_OUTPUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path
root = Path(sys.argv[1])
training = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
validation = json.loads((root / 'validation_metrics.jsonl').read_text().splitlines()[-1])
evaluation = json.loads((root / 'offline_eval.json').read_text())
config = json.loads((root / 'diffusion_config.json').read_text())
assert len(training) == 2 and training[-1]['step'] == 2
assert config['format_version'] == 2 and (root / 'joint_adapter.pt').is_file()
assert all(training[0]['first_step_parameter_updates'].values())
for key in ('noise_mse', 'normalized_continuous_mse'):
    assert math.isclose(validation[key], evaluation['metrics'][key], rel_tol=1e-5, abs_tol=1e-6), key
assert validation['land'] == evaluation['metrics']['land']
result = {'status': 'passed', 'optimizer_steps': 2, 'world_size': training[0]['world_size'],
          'train_samples': 8, 'validation_samples': 4,
          'four_parameter_groups_updated': training[0]['first_step_parameter_updates'],
          'reload_metrics_match': True,
          'scope': 'Pipeline and gradient validation, not convergence or navigation improvement'}
(root / 'pipeline_check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
PY
