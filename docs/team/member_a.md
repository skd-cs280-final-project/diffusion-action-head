# 成员 A：详细评测任务

**执行前必读：[测试过程与指标记录](../guides/inference_test_handbook.md) · [图表与案例要求](../guides/inference_visualization.md)。**

课题：**《基于预训练扩散动作头的无人机视觉语言导航推理评测与分析》**。

[团队总览](README.md) · [全部可复制命令](member_a_commands.md) · [机器可读任务单](../../configs/course/assignments/a.json)

## 任务量和场景

负责 **1002 个正式任务 × 五模型 = 5010 次闭环**，拆为 30 个运行包；另有三任务 × 五模型的 15 次预检。全部场景/划分和精确清单如下，每一行都跑 M0–M4，不能只挑其中一个模型：

| 顺序 | 场景/划分 | 任务数 | easy / hard | TEST_JSON |
| --- | --- | --- | --- | --- |
| 1 | seen_valset/NYCEnvironmentMegapa | 281 | 81 / 200 | `data/uav_dataset/seen_valset_splits/NYCEnvironmentMegapa.json` |
| 2 | unseen_object_valset/NYCEnvironmentMegapa | 189 | 38 / 151 | `data/uav_dataset/unseen_object_valset_splits/NYCEnvironmentMegapa.json` |
| 3 | seen_valset/Carla_Town07 | 60 | 50 / 10 | `data/uav_dataset/seen_valset_splits/Carla_Town07.json` |
| 4 | unseen_object_valset/Carla_Town07 | 11 | 7 / 4 | `data/uav_dataset/unseen_object_valset_splits/Carla_Town07.json` |
| 5 | seen_valset/ModernCityMap | 124 | 124 / 0 | `data/uav_dataset/seen_valset_splits/ModernCityMap.json` |
| 6 | unseen_map_valset/ModularPark | 337 | 220 / 117 | `configs/course/manifests/a/unseen_map_valset/ModularPark.json` |

ModularPark 使用自己的冻结分片，不能误用原始 958 任务全量文件。各个人分片不重叠；清单已有实际任务 UUID，可直接传给现有 TEST_JSON。

## 五个模型与参数

| 编号 | 模型 | 原始权重目录 |
| --- | --- | --- |
| M0 | 原 AeroVLA | `checkpoints/aero_vla` |
| M1 | 扩散头 best | `checkpoints/best` |
| M2 | 联合 step 0 | `checkpoints/joint_lora_stage2_20261103/best` |
| M3 | 联合 step 1000 | `checkpoints/joint_lora_stage2_20261103/checkpoint-1000` |
| M4 | 联合 final | `checkpoints/joint_lora_stage2_20261103` |

运行时 MODEL_DIR 使用 `eval_results/course_inference/resources/models/<model_id>`，这是本机已经准备的资源视图：权重链接到原文件，配置副本仅修正 base_model_path/adapter_path。原权重与原配置不改，不需要训练。原始目录见上表；换机器需按[资源说明](resources.md)重新准备。

| 参数 | 三人统一值 |
| --- | --- |
| 闭环 batch / MAX_WAYPOINTS | 1 / 200 |
| 扩散设置 | DDPM；训练噪声日程 100；推理去噪 10；固定不比较步数 |
| 动作 / 输入 | H=1、E=1、history=0；原双视角和原指令 |
| 精度 / 候选 | 现有加载：基座 bfloat16、扩散头 float32；原单次预测 |
| 随机种子 | 沿用 100 + rank + dagger_it；单进程为 100，不设置新 seed 参数 |
| 离线 | 完整 val、batch=2、seed42、num_workers=4、max_length=2048 |
| CPU 线程 | OMP_NUM_THREADS=4，OPENBLAS_NUM_THREADS=4 |
| 运行方式 | 现有 python/bash 入口；单进程，不用 torchrun；每张卡同一时刻只跑一项 |
| 统计 | 现有 SR/OSR/SPL/NE；full/easy/hard，其中参考轨迹长度 <=250m 为 easy |

你的模型 GPU=0、场景 GPU=0、管理端口=30000、首个场景 RPC=30001、DDP 端口=20001。所有命令从仓库根目录执行，每个终端先设置：

```bash
export PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python
export GPU_ID=0 SIMULATOR_GPU_ID=0 SIMULATOR_PORT=30000 DDP_MASTER_PORT=20001
export MODEL_BASE="$PWD/openvla-7b" DATASET_PATH="$PWD/dataset_raw"
export BATCH_SIZE=1 MAX_WAYPOINTS=200 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

场景服务使用独立终端（确认端口和 GPU 没被其他工作使用）：

```bash
mkdir -p logs/course_inference/a
"$PYTHON" -u airsim_plugin/AirVLNSimulatorServerTool.py \
  --gpus 0 --port 30000 --root_path "$PWD/envs" \
  > logs/course_inference/a/simulator.log 2>&1
```

## 先预检，再正式运行

预检场景为 `NYCEnvironmentMegapa`，TEST_JSON=`configs/course/manifests/a/preflight/NYCEnvironmentMegapa.json`，三条任务按参考帧数的短/中/长位置预先确定，与模型结果无关。五模型各跑三条；预检结果不计入正式成绩。检查模型加载、场景渲染、图片读取、完整轨迹与指标输出，并记录耗时/峰值显存观察；不改代码做新插桩。

[命令页](member_a_commands.md)已列出 dry-run 示例、全部预检、全部正式运行与指标命令，以及你的全量离线命令。先确认预检通过，再逐段执行正式场景；不要把全页一次性提交而跳过预检核对。现有 loader 会按轨迹长度组织运行，所有模型保持同一份清单和默认逻辑。

## 离线模型

| 模型 ID | 原始目录 |
| --- | --- |
| joint_step_001000 | `checkpoints/joint_lora_stage2_20261103/checkpoint-1000` |
| joint_step_004000 | `checkpoints/joint_lora_stage2_20261103/checkpoint-4000` |
| joint_step_007000 | `checkpoints/joint_lora_stage2_20261103/checkpoint-7000` |
| joint_step_010000 | `checkpoints/joint_lora_stage2_20261103/checkpoint-10000` |
| joint_step_013000 | `checkpoints/joint_lora_stage2_20261103/checkpoint-13000` |
| joint_step_016000 | `checkpoints/joint_lora_stage2_20261103/checkpoint-16000` |
| joint_step_019000 | `checkpoints/joint_lora_stage2_20261103/checkpoint-19000` |
| joint_step_022000 | `checkpoints/joint_lora_stage2_20261103/checkpoint-22000` |

以上共 8 个扩散状态，均使用完整 val、batch=2、seed42；没有额外采样实验。另外核对原 AeroVLA 参考模型；它只参加已有闭环，离线指标为不适用。

## 你的交付与报告

[三个联合检查点的离线评估报告](member-a/offline-three-checkpoints/report.md)

1. 将每个正式包的 requested/completed/missing、SR/OSR/SPL/NE 和 full/easy/hard 统计填入个人结果表；预检单独记录。
2. 核对归属于自己的全部 1002 个正式 UUID，在五个模型中均有完整结果；任务失败与程序中断要区分。
3. 整理每个场景至少一个成功和一个失败案例；没有某类时写明实际情况。对同一任务比较五模型，不能只挑好的模型。
4. 输出离线表、闭环表、场景结果图（可用表格软件）、运行环境/命令/日志路径，以及对应场景的分析和局限。

结果目录：`eval_results/course_inference/a/formal/<model_id>/<category>/<scene>/`。日志在 `logs/course_inference/a/`。指标日志不要写进结果轨迹目录。使用自己的资源与端口，完成自己的全部结果整理，不交由其他成员代跑。
