# 成员 A 可复制命令

[执行与完整性检查](../guides/inference_test_handbook.md) · [可视化要求](../guides/inference_visualization.md)。每个推理块结束后先核验任务完整性，再复制其后的指标块。

[个人任务](member_a.md) · [资源准备](resources.md)

以下均从仓库根目录执行，只调用现有脚本。先完成环境和资源检查；此页面记录命令，不表示已经运行。控制台日志会覆盖同名文件，首次执行用这里的全新目录；续跑须保留旧日志并记录 resume，不混入独立重复结果。正式结果不可与预检混合。

## 终端参数

```bash
set -euo pipefail
export PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python
export GPU_ID=0 SIMULATOR_GPU_ID=0 SIMULATOR_PORT=30000 DDP_MASTER_PORT=20001
export MODEL_BASE="$PWD/openvla-7b" DATASET_PATH="$PWD/dataset_raw"
export BATCH_SIZE=1 MAX_WAYPOINTS=200 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

先按个人任务页启动场景服务。建议一次执行一个小节，核对完整结果后再继续。已有日志路径的运行应先处理续跑记录，勿直接覆盖证据。

## 仅打印命令的核对

```bash
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=seen_valset/NYCEnvironmentMegapa \
  TEST_JSON=./configs/course/manifests/a/preflight/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/preflight/diffusion_head_best/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh --dry-run
```

## 预检：seen_valset/NYCEnvironmentMegapa

每个模型预计 3 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/a/preflight/aerovla_original/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=seen_valset/NYCEnvironmentMegapa TEST_JSON=./configs/course/manifests/a/preflight/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/preflight/aerovla_original/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/preflight/aerovla_original/seen_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/preflight/aerovla_original \
  --analysis_list seen_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/preflight/aerovla_original/seen_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/a/preflight/diffusion_head_best/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=seen_valset/NYCEnvironmentMegapa TEST_JSON=./configs/course/manifests/a/preflight/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/preflight/diffusion_head_best/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/preflight/diffusion_head_best/seen_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/preflight/diffusion_head_best \
  --analysis_list seen_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/preflight/diffusion_head_best/seen_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/a/preflight/joint_init/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=seen_valset/NYCEnvironmentMegapa TEST_JSON=./configs/course/manifests/a/preflight/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/preflight/joint_init/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/preflight/joint_init/seen_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/preflight/joint_init \
  --analysis_list seen_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/preflight/joint_init/seen_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/a/preflight/joint_step_001000/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=seen_valset/NYCEnvironmentMegapa TEST_JSON=./configs/course/manifests/a/preflight/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/preflight/joint_step_001000/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/preflight/joint_step_001000/seen_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/preflight/joint_step_001000 \
  --analysis_list seen_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/preflight/joint_step_001000/seen_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/a/preflight/joint_final/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=seen_valset/NYCEnvironmentMegapa TEST_JSON=./configs/course/manifests/a/preflight/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/preflight/joint_final/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/preflight/joint_final/seen_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/preflight/joint_final \
  --analysis_list seen_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/preflight/joint_final/seen_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

## 正式：seen_valset/NYCEnvironmentMegapa

每个模型预计 281 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/a/formal/aerovla_original/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=seen_valset/NYCEnvironmentMegapa TEST_JSON=./data/uav_dataset/seen_valset_splits/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/aerovla_original/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/aerovla_original/seen_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/aerovla_original \
  --analysis_list seen_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/formal/aerovla_original/seen_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/a/formal/diffusion_head_best/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=seen_valset/NYCEnvironmentMegapa TEST_JSON=./data/uav_dataset/seen_valset_splits/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/diffusion_head_best/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/diffusion_head_best/seen_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/diffusion_head_best \
  --analysis_list seen_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/formal/diffusion_head_best/seen_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/a/formal/joint_init/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=seen_valset/NYCEnvironmentMegapa TEST_JSON=./data/uav_dataset/seen_valset_splits/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_init/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_init/seen_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_init \
  --analysis_list seen_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/formal/joint_init/seen_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/a/formal/joint_step_001000/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=seen_valset/NYCEnvironmentMegapa TEST_JSON=./data/uav_dataset/seen_valset_splits/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_step_001000/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_step_001000/seen_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_step_001000 \
  --analysis_list seen_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/formal/joint_step_001000/seen_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/a/formal/joint_final/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=seen_valset/NYCEnvironmentMegapa TEST_JSON=./data/uav_dataset/seen_valset_splits/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_final/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_final/seen_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_final \
  --analysis_list seen_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/formal/joint_final/seen_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

## 正式：unseen_object_valset/NYCEnvironmentMegapa

每个模型预计 189 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/a/formal/aerovla_original/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=unseen_object_valset/NYCEnvironmentMegapa TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/aerovla_original/unseen_object_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/aerovla_original/unseen_object_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/aerovla_original \
  --analysis_list unseen_object_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/formal/aerovla_original/unseen_object_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/a/formal/diffusion_head_best/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=unseen_object_valset/NYCEnvironmentMegapa TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/diffusion_head_best/unseen_object_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/diffusion_head_best/unseen_object_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/diffusion_head_best \
  --analysis_list unseen_object_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/formal/diffusion_head_best/unseen_object_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/a/formal/joint_init/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=unseen_object_valset/NYCEnvironmentMegapa TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_init/unseen_object_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_init/unseen_object_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_init \
  --analysis_list unseen_object_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/formal/joint_init/unseen_object_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/a/formal/joint_step_001000/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=unseen_object_valset/NYCEnvironmentMegapa TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_step_001000/unseen_object_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_step_001000/unseen_object_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_step_001000 \
  --analysis_list unseen_object_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/formal/joint_step_001000/unseen_object_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/a/formal/joint_final/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=unseen_object_valset/NYCEnvironmentMegapa TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/NYCEnvironmentMegapa.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_final/unseen_object_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_final/unseen_object_valset/NYCEnvironmentMegapa.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_final \
  --analysis_list unseen_object_valset/NYCEnvironmentMegapa --path_type_list full easy hard \
  --map_filter NYCEnvironmentMegapa > logs/course_inference/a/formal/joint_final/unseen_object_valset/NYCEnvironmentMegapa.metrics.log 2>&1
```

## 正式：seen_valset/Carla_Town07

每个模型预计 60 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/a/formal/aerovla_original/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=seen_valset/Carla_Town07 TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town07.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/aerovla_original/seen_valset/Carla_Town07 \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/aerovla_original/seen_valset/Carla_Town07.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/aerovla_original \
  --analysis_list seen_valset/Carla_Town07 --path_type_list full easy hard \
  --map_filter Carla_Town07 > logs/course_inference/a/formal/aerovla_original/seen_valset/Carla_Town07.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/a/formal/diffusion_head_best/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=seen_valset/Carla_Town07 TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town07.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/diffusion_head_best/seen_valset/Carla_Town07 \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/diffusion_head_best/seen_valset/Carla_Town07.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/diffusion_head_best \
  --analysis_list seen_valset/Carla_Town07 --path_type_list full easy hard \
  --map_filter Carla_Town07 > logs/course_inference/a/formal/diffusion_head_best/seen_valset/Carla_Town07.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/a/formal/joint_init/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=seen_valset/Carla_Town07 TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town07.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_init/seen_valset/Carla_Town07 \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_init/seen_valset/Carla_Town07.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_init \
  --analysis_list seen_valset/Carla_Town07 --path_type_list full easy hard \
  --map_filter Carla_Town07 > logs/course_inference/a/formal/joint_init/seen_valset/Carla_Town07.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/a/formal/joint_step_001000/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=seen_valset/Carla_Town07 TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town07.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_step_001000/seen_valset/Carla_Town07 \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_step_001000/seen_valset/Carla_Town07.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_step_001000 \
  --analysis_list seen_valset/Carla_Town07 --path_type_list full easy hard \
  --map_filter Carla_Town07 > logs/course_inference/a/formal/joint_step_001000/seen_valset/Carla_Town07.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/a/formal/joint_final/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=seen_valset/Carla_Town07 TEST_JSON=./data/uav_dataset/seen_valset_splits/Carla_Town07.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_final/seen_valset/Carla_Town07 \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_final/seen_valset/Carla_Town07.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_final \
  --analysis_list seen_valset/Carla_Town07 --path_type_list full easy hard \
  --map_filter Carla_Town07 > logs/course_inference/a/formal/joint_final/seen_valset/Carla_Town07.metrics.log 2>&1
```

## 正式：unseen_object_valset/Carla_Town07

每个模型预计 11 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/a/formal/aerovla_original/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=unseen_object_valset/Carla_Town07 TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/Carla_Town07.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/aerovla_original/unseen_object_valset/Carla_Town07 \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/aerovla_original/unseen_object_valset/Carla_Town07.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/aerovla_original \
  --analysis_list unseen_object_valset/Carla_Town07 --path_type_list full easy hard \
  --map_filter Carla_Town07 > logs/course_inference/a/formal/aerovla_original/unseen_object_valset/Carla_Town07.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/a/formal/diffusion_head_best/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=unseen_object_valset/Carla_Town07 TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/Carla_Town07.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/diffusion_head_best/unseen_object_valset/Carla_Town07 \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/diffusion_head_best/unseen_object_valset/Carla_Town07.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/diffusion_head_best \
  --analysis_list unseen_object_valset/Carla_Town07 --path_type_list full easy hard \
  --map_filter Carla_Town07 > logs/course_inference/a/formal/diffusion_head_best/unseen_object_valset/Carla_Town07.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/a/formal/joint_init/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=unseen_object_valset/Carla_Town07 TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/Carla_Town07.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_init/unseen_object_valset/Carla_Town07 \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_init/unseen_object_valset/Carla_Town07.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_init \
  --analysis_list unseen_object_valset/Carla_Town07 --path_type_list full easy hard \
  --map_filter Carla_Town07 > logs/course_inference/a/formal/joint_init/unseen_object_valset/Carla_Town07.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/a/formal/joint_step_001000/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=unseen_object_valset/Carla_Town07 TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/Carla_Town07.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_step_001000/unseen_object_valset/Carla_Town07 \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_step_001000/unseen_object_valset/Carla_Town07.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_step_001000 \
  --analysis_list unseen_object_valset/Carla_Town07 --path_type_list full easy hard \
  --map_filter Carla_Town07 > logs/course_inference/a/formal/joint_step_001000/unseen_object_valset/Carla_Town07.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/a/formal/joint_final/unseen_object_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=unseen_object_valset/Carla_Town07 TEST_JSON=./data/uav_dataset/unseen_object_valset_splits/Carla_Town07.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_final/unseen_object_valset/Carla_Town07 \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_final/unseen_object_valset/Carla_Town07.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_final \
  --analysis_list unseen_object_valset/Carla_Town07 --path_type_list full easy hard \
  --map_filter Carla_Town07 > logs/course_inference/a/formal/joint_final/unseen_object_valset/Carla_Town07.metrics.log 2>&1
```

## 正式：seen_valset/ModernCityMap

每个模型预计 124 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/a/formal/aerovla_original/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=seen_valset/ModernCityMap TEST_JSON=./data/uav_dataset/seen_valset_splits/ModernCityMap.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/aerovla_original/seen_valset/ModernCityMap \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/aerovla_original/seen_valset/ModernCityMap.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/aerovla_original \
  --analysis_list seen_valset/ModernCityMap --path_type_list full easy hard \
  --map_filter ModernCityMap > logs/course_inference/a/formal/aerovla_original/seen_valset/ModernCityMap.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/a/formal/diffusion_head_best/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=seen_valset/ModernCityMap TEST_JSON=./data/uav_dataset/seen_valset_splits/ModernCityMap.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/diffusion_head_best/seen_valset/ModernCityMap \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/diffusion_head_best/seen_valset/ModernCityMap.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/diffusion_head_best \
  --analysis_list seen_valset/ModernCityMap --path_type_list full easy hard \
  --map_filter ModernCityMap > logs/course_inference/a/formal/diffusion_head_best/seen_valset/ModernCityMap.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/a/formal/joint_init/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=seen_valset/ModernCityMap TEST_JSON=./data/uav_dataset/seen_valset_splits/ModernCityMap.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_init/seen_valset/ModernCityMap \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_init/seen_valset/ModernCityMap.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_init \
  --analysis_list seen_valset/ModernCityMap --path_type_list full easy hard \
  --map_filter ModernCityMap > logs/course_inference/a/formal/joint_init/seen_valset/ModernCityMap.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/a/formal/joint_step_001000/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=seen_valset/ModernCityMap TEST_JSON=./data/uav_dataset/seen_valset_splits/ModernCityMap.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_step_001000/seen_valset/ModernCityMap \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_step_001000/seen_valset/ModernCityMap.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_step_001000 \
  --analysis_list seen_valset/ModernCityMap --path_type_list full easy hard \
  --map_filter ModernCityMap > logs/course_inference/a/formal/joint_step_001000/seen_valset/ModernCityMap.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/a/formal/joint_final/seen_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=seen_valset/ModernCityMap TEST_JSON=./data/uav_dataset/seen_valset_splits/ModernCityMap.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_final/seen_valset/ModernCityMap \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_final/seen_valset/ModernCityMap.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_final \
  --analysis_list seen_valset/ModernCityMap --path_type_list full easy hard \
  --map_filter ModernCityMap > logs/course_inference/a/formal/joint_final/seen_valset/ModernCityMap.metrics.log 2>&1
```

## 正式：unseen_map_valset/ModularPark

每个模型预计 337 条任务。先逐个运行，核对完成数，再读取指标。

### 原 AeroVLA

```bash
mkdir -p logs/course_inference/a/formal/aerovla_original/unseen_map_valset
MODEL_DIR=./eval_results/course_inference/resources/models/aerovla_original \
  TASK_ID=unseen_map_valset/ModularPark TEST_JSON=./configs/course/manifests/a/unseen_map_valset/ModularPark.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/aerovla_original/unseen_map_valset/ModularPark \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/aerovla_original/unseen_map_valset/ModularPark.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/aerovla_original \
  --analysis_list unseen_map_valset/ModularPark --path_type_list full easy hard \
  --map_filter ModularPark > logs/course_inference/a/formal/aerovla_original/unseen_map_valset/ModularPark.metrics.log 2>&1
```

### 扩散头 best

```bash
mkdir -p logs/course_inference/a/formal/diffusion_head_best/unseen_map_valset
MODEL_DIR=./eval_results/course_inference/resources/models/diffusion_head_best \
  TASK_ID=unseen_map_valset/ModularPark TEST_JSON=./configs/course/manifests/a/unseen_map_valset/ModularPark.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/diffusion_head_best/unseen_map_valset/ModularPark \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/diffusion_head_best/unseen_map_valset/ModularPark.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/diffusion_head_best \
  --analysis_list unseen_map_valset/ModularPark --path_type_list full easy hard \
  --map_filter ModularPark > logs/course_inference/a/formal/diffusion_head_best/unseen_map_valset/ModularPark.metrics.log 2>&1
```

### 联合 step 0

```bash
mkdir -p logs/course_inference/a/formal/joint_init/unseen_map_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_init \
  TASK_ID=unseen_map_valset/ModularPark TEST_JSON=./configs/course/manifests/a/unseen_map_valset/ModularPark.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_init/unseen_map_valset/ModularPark \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_init/unseen_map_valset/ModularPark.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_init \
  --analysis_list unseen_map_valset/ModularPark --path_type_list full easy hard \
  --map_filter ModularPark > logs/course_inference/a/formal/joint_init/unseen_map_valset/ModularPark.metrics.log 2>&1
```

### 联合 step 1000

```bash
mkdir -p logs/course_inference/a/formal/joint_step_001000/unseen_map_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_step_001000 \
  TASK_ID=unseen_map_valset/ModularPark TEST_JSON=./configs/course/manifests/a/unseen_map_valset/ModularPark.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_step_001000/unseen_map_valset/ModularPark \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_step_001000/unseen_map_valset/ModularPark.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_step_001000 \
  --analysis_list unseen_map_valset/ModularPark --path_type_list full easy hard \
  --map_filter ModularPark > logs/course_inference/a/formal/joint_step_001000/unseen_map_valset/ModularPark.metrics.log 2>&1
```

### 联合 final

```bash
mkdir -p logs/course_inference/a/formal/joint_final/unseen_map_valset
MODEL_DIR=./eval_results/course_inference/resources/models/joint_final \
  TASK_ID=unseen_map_valset/ModularPark TEST_JSON=./configs/course/manifests/a/unseen_map_valset/ModularPark.json \
  SAVE_DIR=./eval_results/course_inference/a/formal/joint_final/unseen_map_valset/ModularPark \
  bash scripts/eval_aerovla.sh > logs/course_inference/a/formal/joint_final/unseen_map_valset/ModularPark.log 2>&1
```

先按手册核验本包的任务覆盖、保存完整性和异常，再执行统计；不完整的结果只能标 partial。

```bash
"$PYTHON" utils/metric.py \
  --root_dir ./eval_results/course_inference/a/formal/joint_final \
  --analysis_list unseen_map_valset/ModularPark --path_type_list full easy hard \
  --map_filter ModularPark > logs/course_inference/a/formal/joint_final/unseen_map_valset/ModularPark.metrics.log 2>&1
```

## 全量离线命令

与自己的闭环错开执行，避免同时占同一 GPU。结果为独立离线 JSON，原 token 不运行此入口。

### joint_step_001000

```bash
mkdir -p logs/course_inference/a/offline eval_results/course_inference/a/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_001000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/a/offline/joint_step_001000_seed42.json \
  > logs/course_inference/a/offline/joint_step_001000.log 2>&1
```

### joint_step_004000

```bash
mkdir -p logs/course_inference/a/offline eval_results/course_inference/a/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_004000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/a/offline/joint_step_004000_seed42.json \
  > logs/course_inference/a/offline/joint_step_004000.log 2>&1
```

### joint_step_007000

```bash
mkdir -p logs/course_inference/a/offline eval_results/course_inference/a/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_007000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/a/offline/joint_step_007000_seed42.json \
  > logs/course_inference/a/offline/joint_step_007000.log 2>&1
```

### joint_step_010000

```bash
mkdir -p logs/course_inference/a/offline eval_results/course_inference/a/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_010000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/a/offline/joint_step_010000_seed42.json \
  > logs/course_inference/a/offline/joint_step_010000.log 2>&1
```

### joint_step_013000

```bash
mkdir -p logs/course_inference/a/offline eval_results/course_inference/a/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_013000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/a/offline/joint_step_013000_seed42.json \
  > logs/course_inference/a/offline/joint_step_013000.log 2>&1
```

### joint_step_016000

```bash
mkdir -p logs/course_inference/a/offline eval_results/course_inference/a/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_016000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/a/offline/joint_step_016000_seed42.json \
  > logs/course_inference/a/offline/joint_step_016000.log 2>&1
```

### joint_step_019000

```bash
mkdir -p logs/course_inference/a/offline eval_results/course_inference/a/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_019000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/a/offline/joint_step_019000_seed42.json \
  > logs/course_inference/a/offline/joint_step_019000.log 2>&1
```

### joint_step_022000

```bash
mkdir -p logs/course_inference/a/offline eval_results/course_inference/a/offline
PYTHON="$PYTHON" CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./eval_results/course_inference/resources/models/joint_step_022000 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json --batch-size 2 \
  --num-workers 4 --max-length 2048 --seed 42 --num-inference-steps 10 \
  --backbone-dtype bfloat16 \
  --output ./eval_results/course_inference/a/offline/joint_step_022000_seed42.json \
  > logs/course_inference/a/offline/joint_step_022000.log 2>&1
```
