# 成员 B 可复制命令

[个人任务](member_b.md) · [资源准备](resources.md)

以下均从仓库根目录执行，只调用现有脚本。先完成环境和资源检查；此页面记录命令，不表示已经运行。控制台日志会覆盖同名文件，首次执行用这里的全新目录；续跑须保留旧日志并记录 resume，不混入独立重复结果。正式结果不可与预检混合。

## 终端参数

```bash
set -euo pipefail
export PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python
export GPU_ID=1 SIMULATOR_GPU_ID=1 SIMULATOR_PORT=35000 DDP_MASTER_PORT=20002
export MODEL_BASE="$PWD/openvla-7b" DATASET_PATH="$PWD/dataset_raw"
export BATCH_SIZE=1 MAX_WAYPOINTS=200 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

先按个人任务页启动场景服务。建议一次执行一个小节，核对完整结果后再继续。已有日志路径的运行应先处理续跑记录，勿直接覆盖证据。

## 仅打印命令的核对

```bash
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=seen_valset/TropicalIsland \
  TEST_JSON=./configs/course/manifests/b/preflight/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/preflight/diffusion_head_best/seen_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh --dry-run
```

## 预检：seen_valset/TropicalIsland

每个模型预计 3 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/b/preflight/aerovla_original/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=seen_valset/TropicalIsland TEST_JSON=./configs/course/manifests/b/preflight/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/preflight/aerovla_original/seen_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/preflight/aerovla_original/seen_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/preflight/aerovla_original \
  --analysis_list seen_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/preflight/aerovla_original/seen_valset/TropicalIsland.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/b/preflight/diffusion_head_best/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=seen_valset/TropicalIsland TEST_JSON=./configs/course/manifests/b/preflight/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/preflight/diffusion_head_best/seen_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/preflight/diffusion_head_best/seen_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/preflight/diffusion_head_best \
  --analysis_list seen_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/preflight/diffusion_head_best/seen_valset/TropicalIsland.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/b/preflight/joint_init/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=seen_valset/TropicalIsland TEST_JSON=./configs/course/manifests/b/preflight/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/preflight/joint_init/seen_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/preflight/joint_init/seen_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/preflight/joint_init \
  --analysis_list seen_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/preflight/joint_init/seen_valset/TropicalIsland.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/b/preflight/joint_step_001000/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=seen_valset/TropicalIsland TEST_JSON=./configs/course/manifests/b/preflight/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/preflight/joint_step_001000/seen_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/preflight/joint_step_001000/seen_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/preflight/joint_step_001000 \
  --analysis_list seen_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/preflight/joint_step_001000/seen_valset/TropicalIsland.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/b/preflight/joint_final/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=seen_valset/TropicalIsland TEST_JSON=./configs/course/manifests/b/preflight/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/preflight/joint_final/seen_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/preflight/joint_final/seen_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/preflight/joint_final \
  --analysis_list seen_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/preflight/joint_final/seen_valset/TropicalIsland.metrics.log 2>&1
```

## 正式：seen_valset/TropicalIsland

每个模型预计 165 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/b/formal/aerovla_original/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=seen_valset/TropicalIsland TEST_JSON=./data/uav_dataset/seen_valset_splits/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/aerovla_original/seen_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/aerovla_original/seen_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/aerovla_original \
  --analysis_list seen_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/formal/aerovla_original/seen_valset/TropicalIsland.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/b/formal/diffusion_head_best/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=seen_valset/TropicalIsland TEST_JSON=./data/uav_dataset/seen_valset_splits/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/diffusion_head_best/seen_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/diffusion_head_best/seen_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/diffusion_head_best \
  --analysis_list seen_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/formal/diffusion_head_best/seen_valset/TropicalIsland.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/b/formal/joint_init/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=seen_valset/TropicalIsland TEST_JSON=./data/uav_dataset/seen_valset_splits/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_init/seen_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_init/seen_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_init \
  --analysis_list seen_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/formal/joint_init/seen_valset/TropicalIsland.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/b/formal/joint_step_001000/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=seen_valset/TropicalIsland TEST_JSON=./data/uav_dataset/seen_valset_splits/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_step_001000/seen_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_step_001000/seen_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_step_001000 \
  --analysis_list seen_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/formal/joint_step_001000/seen_valset/TropicalIsland.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/b/formal/joint_final/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=seen_valset/TropicalIsland TEST_JSON=./data/uav_dataset/seen_valset_splits/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_final/seen_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_final/seen_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_final \
  --analysis_list seen_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/formal/joint_final/seen_valset/TropicalIsland.metrics.log 2>&1
```

## 正式：unseen_object_valset/TropicalIsland

每个模型预计 266 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/b/formal/aerovla_original/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=unseen_object_valset/TropicalIsland TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/aerovla_original/unseen_object_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/aerovla_original/unseen_object_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/aerovla_original \
  --analysis_list unseen_object_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/formal/aerovla_original/unseen_object_valset/TropicalIsland.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/b/formal/diffusion_head_best/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=unseen_object_valset/TropicalIsland TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/diffusion_head_best/unseen_object_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/diffusion_head_best/unseen_object_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/diffusion_head_best \
  --analysis_list unseen_object_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/formal/diffusion_head_best/unseen_object_valset/TropicalIsland.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/b/formal/joint_init/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=unseen_object_valset/TropicalIsland TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_init/unseen_object_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_init/unseen_object_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_init \
  --analysis_list unseen_object_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/formal/joint_init/unseen_object_valset/TropicalIsland.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/b/formal/joint_step_001000/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=unseen_object_valset/TropicalIsland TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_step_001000/unseen_object_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_step_001000/unseen_object_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_step_001000 \
  --analysis_list unseen_object_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/formal/joint_step_001000/unseen_object_valset/TropicalIsland.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/b/formal/joint_final/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=unseen_object_valset/TropicalIsland TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/TropicalIsland.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_final/unseen_object_valset/TropicalIsland \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_final/unseen_object_valset/TropicalIsland.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_final \
  --analysis_list unseen_object_valset/TropicalIsland --path_type_list full easy hard \
  --map_filter TropicalIsland > logs/course_inference/b/formal/joint_final/unseen_object_valset/TropicalIsland.metrics.log 2>&1
```

## 正式：seen_valset/Carla_Town04

每个模型预计 106 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/b/formal/aerovla_original/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=seen_valset/Carla_Town04 TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town04.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/aerovla_original/seen_valset/Carla_Town04 \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/aerovla_original/seen_valset/Carla_Town04.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/aerovla_original \
  --analysis_list seen_valset/Carla_Town04 --path_type_list full easy hard \
  --map_filter Carla_Town04 > logs/course_inference/b/formal/aerovla_original/seen_valset/Carla_Town04.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/b/formal/diffusion_head_best/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=seen_valset/Carla_Town04 TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town04.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/diffusion_head_best/seen_valset/Carla_Town04 \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/diffusion_head_best/seen_valset/Carla_Town04.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/diffusion_head_best \
  --analysis_list seen_valset/Carla_Town04 --path_type_list full easy hard \
  --map_filter Carla_Town04 > logs/course_inference/b/formal/diffusion_head_best/seen_valset/Carla_Town04.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/b/formal/joint_init/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=seen_valset/Carla_Town04 TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town04.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_init/seen_valset/Carla_Town04 \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_init/seen_valset/Carla_Town04.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_init \
  --analysis_list seen_valset/Carla_Town04 --path_type_list full easy hard \
  --map_filter Carla_Town04 > logs/course_inference/b/formal/joint_init/seen_valset/Carla_Town04.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/b/formal/joint_step_001000/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=seen_valset/Carla_Town04 TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town04.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_step_001000/seen_valset/Carla_Town04 \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_step_001000/seen_valset/Carla_Town04.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_step_001000 \
  --analysis_list seen_valset/Carla_Town04 --path_type_list full easy hard \
  --map_filter Carla_Town04 > logs/course_inference/b/formal/joint_step_001000/seen_valset/Carla_Town04.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/b/formal/joint_final/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=seen_valset/Carla_Town04 TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town04.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_final/seen_valset/Carla_Town04 \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_final/seen_valset/Carla_Town04.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_final \
  --analysis_list seen_valset/Carla_Town04 --path_type_list full easy hard \
  --map_filter Carla_Town04 > logs/course_inference/b/formal/joint_final/seen_valset/Carla_Town04.metrics.log 2>&1
```

## 正式：unseen_object_valset/Carla_Town04

每个模型预计 14 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/b/formal/aerovla_original/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=unseen_object_valset/Carla_Town04 TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/Carla_Town04.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/aerovla_original/unseen_object_valset/Carla_Town04 \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/aerovla_original/unseen_object_valset/Carla_Town04.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/aerovla_original \
  --analysis_list unseen_object_valset/Carla_Town04 --path_type_list full easy hard \
  --map_filter Carla_Town04 > logs/course_inference/b/formal/aerovla_original/unseen_object_valset/Carla_Town04.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/b/formal/diffusion_head_best/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=unseen_object_valset/Carla_Town04 TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/Carla_Town04.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/diffusion_head_best/unseen_object_valset/Carla_Town04 \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/diffusion_head_best/unseen_object_valset/Carla_Town04.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/diffusion_head_best \
  --analysis_list unseen_object_valset/Carla_Town04 --path_type_list full easy hard \
  --map_filter Carla_Town04 > logs/course_inference/b/formal/diffusion_head_best/unseen_object_valset/Carla_Town04.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/b/formal/joint_init/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=unseen_object_valset/Carla_Town04 TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/Carla_Town04.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_init/unseen_object_valset/Carla_Town04 \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_init/unseen_object_valset/Carla_Town04.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_init \
  --analysis_list unseen_object_valset/Carla_Town04 --path_type_list full easy hard \
  --map_filter Carla_Town04 > logs/course_inference/b/formal/joint_init/unseen_object_valset/Carla_Town04.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/b/formal/joint_step_001000/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=unseen_object_valset/Carla_Town04 TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/Carla_Town04.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_step_001000/unseen_object_valset/Carla_Town04 \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_step_001000/unseen_object_valset/Carla_Town04.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_step_001000 \
  --analysis_list unseen_object_valset/Carla_Town04 --path_type_list full easy hard \
  --map_filter Carla_Town04 > logs/course_inference/b/formal/joint_step_001000/unseen_object_valset/Carla_Town04.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/b/formal/joint_final/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=unseen_object_valset/Carla_Town04 TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/Carla_Town04.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_final/unseen_object_valset/Carla_Town04 \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_final/unseen_object_valset/Carla_Town04.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_final \
  --analysis_list unseen_object_valset/Carla_Town04 --path_type_list full easy hard \
  --map_filter Carla_Town04 > logs/course_inference/b/formal/joint_final/unseen_object_valset/Carla_Town04.metrics.log 2>&1
```

## 正式：seen_valset/Carla_Town10HD

每个模型预计 141 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/b/formal/aerovla_original/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=seen_valset/Carla_Town10HD TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town10HD.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/aerovla_original/seen_valset/Carla_Town10HD \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/aerovla_original/seen_valset/Carla_Town10HD.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/aerovla_original \
  --analysis_list seen_valset/Carla_Town10HD --path_type_list full easy hard \
  --map_filter Carla_Town10HD > logs/course_inference/b/formal/aerovla_original/seen_valset/Carla_Town10HD.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/b/formal/diffusion_head_best/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=seen_valset/Carla_Town10HD TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town10HD.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/diffusion_head_best/seen_valset/Carla_Town10HD \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/diffusion_head_best/seen_valset/Carla_Town10HD.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/diffusion_head_best \
  --analysis_list seen_valset/Carla_Town10HD --path_type_list full easy hard \
  --map_filter Carla_Town10HD > logs/course_inference/b/formal/diffusion_head_best/seen_valset/Carla_Town10HD.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/b/formal/joint_init/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=seen_valset/Carla_Town10HD TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town10HD.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_init/seen_valset/Carla_Town10HD \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_init/seen_valset/Carla_Town10HD.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_init \
  --analysis_list seen_valset/Carla_Town10HD --path_type_list full easy hard \
  --map_filter Carla_Town10HD > logs/course_inference/b/formal/joint_init/seen_valset/Carla_Town10HD.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/b/formal/joint_step_001000/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=seen_valset/Carla_Town10HD TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town10HD.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_step_001000/seen_valset/Carla_Town10HD \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_step_001000/seen_valset/Carla_Town10HD.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_step_001000 \
  --analysis_list seen_valset/Carla_Town10HD --path_type_list full easy hard \
  --map_filter Carla_Town10HD > logs/course_inference/b/formal/joint_step_001000/seen_valset/Carla_Town10HD.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/b/formal/joint_final/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=seen_valset/Carla_Town10HD TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town10HD.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_final/seen_valset/Carla_Town10HD \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_final/seen_valset/Carla_Town10HD.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_final \
  --analysis_list seen_valset/Carla_Town10HD --path_type_list full easy hard \
  --map_filter Carla_Town10HD > logs/course_inference/b/formal/joint_final/seen_valset/Carla_Town10HD.metrics.log 2>&1
```

## 正式：unseen_map_valset/ModularPark

每个模型预计 310 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/b/formal/aerovla_original/unseen_map_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=unseen_map_valset/ModularPark TEST_JSON=./configs/course/manifests/b/unseen_map_valset/ModularPark.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/aerovla_original/unseen_map_valset/ModularPark \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/aerovla_original/unseen_map_valset/ModularPark.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/aerovla_original \
  --analysis_list unseen_map_valset/ModularPark --path_type_list full easy hard \
  --map_filter ModularPark > logs/course_inference/b/formal/aerovla_original/unseen_map_valset/ModularPark.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/b/formal/diffusion_head_best/unseen_map_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=unseen_map_valset/ModularPark TEST_JSON=./configs/course/manifests/b/unseen_map_valset/ModularPark.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/diffusion_head_best/unseen_map_valset/ModularPark \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/diffusion_head_best/unseen_map_valset/ModularPark.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/diffusion_head_best \
  --analysis_list unseen_map_valset/ModularPark --path_type_list full easy hard \
  --map_filter ModularPark > logs/course_inference/b/formal/diffusion_head_best/unseen_map_valset/ModularPark.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/b/formal/joint_init/unseen_map_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=unseen_map_valset/ModularPark TEST_JSON=./configs/course/manifests/b/unseen_map_valset/ModularPark.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_init/unseen_map_valset/ModularPark \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_init/unseen_map_valset/ModularPark.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_init \
  --analysis_list unseen_map_valset/ModularPark --path_type_list full easy hard \
  --map_filter ModularPark > logs/course_inference/b/formal/joint_init/unseen_map_valset/ModularPark.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/b/formal/joint_step_001000/unseen_map_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=unseen_map_valset/ModularPark TEST_JSON=./configs/course/manifests/b/unseen_map_valset/ModularPark.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_step_001000/unseen_map_valset/ModularPark \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_step_001000/unseen_map_valset/ModularPark.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_step_001000 \
  --analysis_list unseen_map_valset/ModularPark --path_type_list full easy hard \
  --map_filter ModularPark > logs/course_inference/b/formal/joint_step_001000/unseen_map_valset/ModularPark.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/b/formal/joint_final/unseen_map_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=unseen_map_valset/ModularPark TEST_JSON=./configs/course/manifests/b/unseen_map_valset/ModularPark.json \
  SAVE_DIR=./eval_results/course_inference/b/formal/joint_final/unseen_map_valset/ModularPark \
  bash scripts/eval_aerovla.sh > logs/course_inference/b/formal/joint_final/unseen_map_valset/ModularPark.log 2>&1

"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/b/formal/joint_final \
  --analysis_list unseen_map_valset/ModularPark --path_type_list full easy hard \
  --map_filter ModularPark > logs/course_inference/b/formal/joint_final/unseen_map_valset/ModularPark.metrics.log 2>&1
```

## 全量离线命令

与自己的闭环错开执行，避免同时占同一 GPU。结果为独立离线 JSON，原 token 不运行此入口。

### diffusion_head_best

```bash
mkdir -p logs/course_inference/b/offline eval_results/course_inference/b/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/diffusion_head_best \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/b/offline/diffusion_head_best_seed42.json \
  > logs/course_inference/b/offline/diffusion_head_best.log 2>&1
```

### joint_step_002000

```bash
mkdir -p logs/course_inference/b/offline eval_results/course_inference/b/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_002000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/b/offline/joint_step_002000_seed42.json \
  > logs/course_inference/b/offline/joint_step_002000.log 2>&1
```

### joint_step_005000

```bash
mkdir -p logs/course_inference/b/offline eval_results/course_inference/b/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_005000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/b/offline/joint_step_005000_seed42.json \
  > logs/course_inference/b/offline/joint_step_005000.log 2>&1
```

### joint_step_008000

```bash
mkdir -p logs/course_inference/b/offline eval_results/course_inference/b/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_008000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/b/offline/joint_step_008000_seed42.json \
  > logs/course_inference/b/offline/joint_step_008000.log 2>&1
```

### joint_step_011000

```bash
mkdir -p logs/course_inference/b/offline eval_results/course_inference/b/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_011000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/b/offline/joint_step_011000_seed42.json \
  > logs/course_inference/b/offline/joint_step_011000.log 2>&1
```

### joint_step_014000

```bash
mkdir -p logs/course_inference/b/offline eval_results/course_inference/b/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_014000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/b/offline/joint_step_014000_seed42.json \
  > logs/course_inference/b/offline/joint_step_014000.log 2>&1
```

### joint_step_017000

```bash
mkdir -p logs/course_inference/b/offline eval_results/course_inference/b/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_017000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/b/offline/joint_step_017000_seed42.json \
  > logs/course_inference/b/offline/joint_step_017000.log 2>&1
```

### joint_step_020000

```bash
mkdir -p logs/course_inference/b/offline eval_results/course_inference/b/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_020000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/b/offline/joint_step_020000_seed42.json \
  > logs/course_inference/b/offline/joint_step_020000.log 2>&1
```

### joint_step_023000

```bash
mkdir -p logs/course_inference/b/offline eval_results/course_inference/b/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_023000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/b/offline/joint_step_023000_seed42.json \
  > logs/course_inference/b/offline/joint_step_023000.log 2>&1
```
