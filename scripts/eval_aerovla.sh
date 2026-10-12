#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
    cat <<'EOF'
Usage: bash scripts/eval_aerovla.sh [--dry-run] [-- evaluator arguments ...]

Environment overrides:
  PYTHON, MODEL_DIR, MODEL_BASE, TASK_ID, GPU_ID, SIMULATOR_GPU_ID,
  SIMULATOR_PORT, DDP_MASTER_PORT, TEST_JSON, SAVE_DIR, DATASET_PATH,
  BATCH_SIZE, MAX_WAYPOINTS, MAP_SPAWN_AREA_JSON_PATH, OBJECT_NAME_JSON_PATH.

GPU_ID controls model CUDA visibility. SIMULATOR_GPU_ID is the physical GPU
requested from the simulator service (defaults to GPU_ID).
SIMULATOR_PORT defaults to 30000 + SIMULATOR_GPU_ID * 5000.
--dry-run prints the command without loading a model or contacting AirSim.
Relative paths are resolved from the AeroVLA project root.
EOF
}

dry_run=false
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) echo "Unknown launcher option: $1 (use -- before evaluator arguments)" >&2; exit 2 ;;
    esac
done

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/checkpoints/aero_vla}"
EXP_NAME="$(basename -- "$MODEL_DIR")"
TASK_ID="${TASK_ID:-seen_valset/NYCEnvironmentMegapa}"

# TASK_ID examples: seen_valset/Carla_Town01,
# unseen_object_valset/Carla_Town04, unseen_map_valset/ModularPark.
if [[ "$TASK_ID" != */* || "$TASK_ID" == /* || "$TASK_ID" == */ || "${TASK_ID#*/}" == */* ]]; then
    echo "TASK_ID must have the form category/map_name: $TASK_ID" >&2
    exit 2
fi
CATEGORY="${TASK_ID%%/*}"
MAP_NAME="${TASK_ID#*/}"
TEST_JSON="${TEST_JSON:-$PROJECT_ROOT/data/uav_dataset/${CATEGORY}_splits/${MAP_NAME}.json}"
SAVE_DIR="${SAVE_DIR:-$PROJECT_ROOT/eval_results/${EXP_NAME}/${CATEGORY}/${MAP_NAME}}"

GPU_ID="${GPU_ID:-0}"
SIMULATOR_GPU_ID="${SIMULATOR_GPU_ID:-$GPU_ID}"
if [[ ! "$SIMULATOR_GPU_ID" =~ ^[0-9]{1,5}$ ]]; then
    echo "SIMULATOR_GPU_ID must be a non-negative integer: $SIMULATOR_GPU_ID" >&2
    exit 2
fi
SIMULATOR_GPU_ID=$((10#$SIMULATOR_GPU_ID))
SIMULATOR_PORT="${SIMULATOR_PORT:-$((30000 + SIMULATOR_GPU_ID * 5000))}"
DDP_MASTER_PORT="${DDP_MASTER_PORT:-20001}"
for port_name in SIMULATOR_PORT DDP_MASTER_PORT; do
    port_value="${!port_name}"
    if [[ ! "$port_value" =~ ^[0-9]{1,5}$ ]] || ((10#$port_value < 1 || 10#$port_value > 65535)); then
        echo "$port_name must be an integer between 1 and 65535: $port_value" >&2
        exit 2
    fi
done

command=(
    "${PYTHON:-python}" -u "$PROJECT_ROOT/src/vlnce_src/eval_aerovla.py"
    --project_prefix "$PROJECT_ROOT"
    --run_type eval
    --name AerialVLA_Eval
    --gpu_id "$SIMULATOR_GPU_ID"
    --simulator_tool_port "$SIMULATOR_PORT"
    --DDP_MASTER_PORT "$DDP_MASTER_PORT"
    --batchSize "${BATCH_SIZE:-1}"
    --maxWaypoints "${MAX_WAYPOINTS:-200}"
    --dataset_path "${DATASET_PATH:-$PROJECT_ROOT/dataset_raw/}"
    --eval_save_path "$SAVE_DIR"
    --model_path "$MODEL_DIR"
    --eval_json_path "$TEST_JSON"
    --map_spawn_area_json_path "${MAP_SPAWN_AREA_JSON_PATH:-$PROJECT_ROOT/data/meta/map_spawnarea_info.json}"
    --object_name_json_path "${OBJECT_NAME_JSON_PATH:-$PROJECT_ROOT/data/meta/object_description.json}"
)
if [[ -n "${MODEL_BASE:-}" ]]; then
    command+=(--model_base "$MODEL_BASE")
fi
command+=("$@")

printf '%s\n' "Category: $CATEGORY" "Map Name: $MAP_NAME" \
    "Model GPU: $GPU_ID" "Simulator GPU: $SIMULATOR_GPU_ID" \
    "Simulator Port: $SIMULATOR_PORT" "DDP Master Port: $DDP_MASTER_PORT" \
    "Config File: $TEST_JSON" "Save Dir: $SAVE_DIR"
if "$dry_run"; then
    printf 'Command:'
    printf ' %q' env "CUDA_VISIBLE_DEVICES=$GPU_ID" "${command[@]}"
    printf '\n'
    exit 0
fi
exec env "CUDA_VISIBLE_DEVICES=$GPU_ID" "${command[@]}"
