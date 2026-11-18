# 安装、训练与评估指南

[项目首页](../../README.md) · [实验协议](../experiments/protocol.md)

所有命令从仓库根目录执行。以下命令运行已有单步基线 A0；新回归/MLP/序列实验还需 A/B 完成实现。GPU 编号按实际机器修改，输出目录必须不存在或为空。

## 1. 安装环境

现有工作环境使用 Linux、Python 3.10、PyTorch 2.1.2+cu121。依赖固定在 [requirements.txt](../../requirements.txt)。新机器仍需匹配 GPU 驱动、CUDA 编译工具链与 AirSim 渲染环境。

```bash
conda create -n aero_vla python=3.10 -y
conda activate aero_vla
export PYTHON=python
python -m pip install torch==2.1.2 torchvision==0.16.2   --index-url https://download.pytorch.org/whl/cu121
python -m pip install packaging ninja wheel numpy==1.26.3
python -m pip install --no-build-isolation -r requirements.txt
# numba 用于闭环工具；pytest 用于测试，matplotlib 用于监控图表
python -m pip install numba pytest matplotlib
python -m pip check
python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())'
```

flash-attn 构建可能需要 nvcc 和 C++ 编译器，PyTorch wheel 的 CUDA 运行时不等于完整 Toolkit。旧 AirSim/RPC 依赖发生通信问题时参考[专项排障](troubleshooting.md)。完整新环境安装不由现有工作区测试结果代替。

## 2. 准备外部资源

| 资源 | 默认目录 | 内容 |
| --- | --- | --- |
| 基座 | `openvla-7b/` | 权重分片、索引、config、tokenizer、processor、自定义模型代码 |
| 原 adapter | `checkpoints/aero_vla/` | adapter_config 与 adapter_model.safetensors |
| 原始数据 | `dataset_raw/` | TravelUAV 图像、轨迹与元信息 |
| 帧级监督 | `data/aerovla_train_dataset.json` | 逐帧图文和动作标签，需另行准备 |
| 轨迹划分 | `data/continuous_split_seed42.json` | Git 已保留，与监督清单匹配 |
| 闭环资源 | `envs/`、`data/uav_dataset/`、`data/meta/` | 外部场景、已提供的任务清单和元数据 |

数据布局：

```text
dataset_raw/<map>/<trajectory>/
├── frontcamera/000000.png
├── downcamera/000000.png
├── log/000000.json          可选历史输入需要
└── merged_data.json         参考轨迹和闭环指标需要
```

若 `dataset_raw` 尚不存在，可以链接到数据位置：

```bash
ln -s /absolute/path/to/TravelUAV dataset_raw
```

普通训练/离线入口也支持 `--data-root`；闭环使用 `DATASET_PATH`。smoke 脚本使用默认布局。已有链接先核对，不直接覆盖。

训练标注示例：

```json
[{"traj_rel_dir":"NYCEnvironmentMegapa/trajectory-id","img_name":"000000.png","instruction":"<image>\nFly forward and find the target.","label":{"fwd":2.5,"down":0.0,"yaw":0.0},"is_last_step":false,"is_penultimate":false}]
```

训练清单、轨迹 split 和闭环 episode 是不同格式，不能互换。当前全量是 train 369,274 帧/6,850 轨迹，val 41,176 帧/761 轨迹；程序校验覆盖、轨迹交叉与重复记录。资源齐全后可设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

## 3. 数据与两步检查

```bash
PYTHON=python bash scripts/train_diffusion_head.sh   --check-data --history-length 0 --num-workers 0

PYTHON=python CUDA_VISIBLE_DEVICES=0 HISTORY_LENGTH=0 SMOKE_OUTPUT_DIR=./checkpoints/course_smoke bash scripts/smoke_diffusion_pipeline.sh
```

`--check-data` 不加载 7B 权重，但仍需要本地 tokenizer/processor，实际仅预处理首批图像，不是所有图片的完整解码校验。smoke 使用 4 个训练帧、8 个验证帧、2 次更新，检查保存和独立重载指标一致；它不证明导航效果。该脚本默认离线，要求可用 CUDA。

## 4. A0 正式训练

```bash
PYTHON=python CUDA_VISIBLE_DEVICES=0 bash scripts/train_diffusion_head.sh   --model-path ./openvla-7b --adapter-path ./checkpoints/aero_vla   --data-root ./dataset_raw --train-json ./data/aerovla_train_dataset.json   --split-json ./data/continuous_split_seed42.json   --output-dir ./checkpoints/course_A0_seed42   --history-length 0 --hidden-dim 256 --num-layers 4 --num-heads 8   --num-train-timesteps 100 --num-inference-steps 10   --epochs 5 --batch-size 2 --gradient-accumulation-steps 8   --eval-batch-size 2 --learning-rate 1e-4 --seed 42 --eval-seed 42
```

单进程单 GPU，不用 torchrun；骨干默认 bfloat16，动作头 float32，有效 batch 16。默认完整 train/val，每轮及最后更新后验证，定期保存；`--eval-steps N` 可增加验证频率。

`--max-samples N`、`--max-val-samples N`、`--max-steps N` 用于开发限量，必须正数；省略表示全量。样本抽取在完整 split 校验后完成。所有对照若缩减预算，应同步变更并记录，不能把小样本实验与全量基准作为严格对照。

输出含 `run_config.json`、`metrics.jsonl`、`validation_metrics.jsonl`、`best_metrics.json`、最终 `diffusion_head.pt` / `diffusion_config.json`、processor/tokenizer、split 副本、`best/` 与 `checkpoint-N/`。根目录是最终权重，best 按 noise MSE 选择；课程主对照使用同预算最终权重。

## 5. 离线评估和监控

```bash
PYTHON=python CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh   --checkpoint ./checkpoints/course_A0_seed42   --data-root ./dataset_raw   --data-json ./data/aerovla_train_dataset.json   --split-json ./data/continuous_split_seed42.json   --batch-size 2 --seed 42 --num-inference-steps 10   --output ./eval_results/course_A0_seed42/offline.json

python src/monitor_diffusion_training.py   --run-dir ./checkpoints/course_A0_seed42   --log-dir ./logs/course_A0_seed42 --once
```

离线默认全 val，`--max-samples` 可做小检查。输出含 noise MSE、前三维归一化 MSE、逐轴物理误差、LAND 混淆统计和来源核验。没有 LAND 正例时不能用 F1 判断停止能力。此入口目前针对扩散，A1 的通用评估由 C 接入。

监控不加载模型；去掉 `--once` 可持续输出。没有 `log-dir/status.json` 的 completed/failed 标记时不会自动判定训练结束。

监督审计：

```bash
python scripts/audit_diffusion_supervision.py   --data-json ./data/aerovla_train_dataset.json   --split-json ./data/continuous_split_seed42.json   --run-config ./checkpoints/course_A0_seed42/run_config.json   --output-dir ./checkpoints/course_A0_seed42/supervision
```

审计需与真实训练样本范围一致，不能将全量审计冒充子集训练来源。

## 6. AirSim 闭环

场景需符合服务端 `env_exec_path_dict`，例如 `envs/closeloop_envs/NYCEnvironmentMegapa.sh`。在独立终端启动场景管理服务：

```bash
python -u airsim_plugin/AirVLNSimulatorServerTool.py   --gpus 0 --port 30000 --root_path "$PWD/envs"
```

另一个终端先打印最终命令：

```bash
PYTHON=python MODEL_DIR=./checkpoints/course_A0_seed42 GPU_ID=1 SIMULATOR_GPU_ID=0 SIMULATOR_PORT=30000 TASK_ID=seen_valset/NYCEnvironmentMegapa SAVE_DIR=./eval_results/course_A0_seed42/seen_valset/NYCEnvironmentMegapa bash scripts/eval_aerovla.sh --dry-run
```

去掉 `--dry-run` 才执行。管理服务端口区别于场景 RPC；`GPU_ID` 控制模型，`SIMULATOR_GPU_ID` 控制场景。单 GPU 可均设 0，但显存与渲染共享。不要同时向同一管理服务提交多组任务。

`TEST_JSON` 可指定 C 冻结的任务清单，`DATASET_PATH` 可覆盖原始数据目录，`DDP_MASTER_PORT` 默认 20001；并发运行需独立端口和服务。显式指定 `SAVE_DIR` 避免多个 best 目录同名覆盖。原 AeroVLA 参考基线用 `MODEL_DIR=./checkpoints/aero_vla` 和独立结果目录。

`diffusion_config.json` 决定是否加载扩散头；当前单步 wrapper 不等于已支持 B 的动作块，新模型先通过接口检查再做正式评估。

```bash
python utils/metric.py   --root_dir ./eval_results/course_A0_seed42   --analysis_list seen_valset/NYCEnvironmentMegapa   --path_type_list full easy hard --map_filter NYCEnvironmentMegapa
```

root 是包含 seen_valset 的结果根。指标通过 ori_info 引用原始参考轨迹；迁移时保留路径。原协议 NE 是参考终点距离，SPL 的参考长度减 20，不能直接与其他定义混用。`scripts/metric.sh` 的模型路径是脚本内常量，不自动读取 MODEL_DIR。

## 7. 检查点迁移与常见问题

复制整个检查点，而不是单个 `.pt`；另行保留基座和原 adapter。配置可能记录旧绝对路径。离线 CLI 没有 model-path/adapter-path 覆盖参数，可在复制后的配置中仅修正资源路径并保留备份，或使用模型 API 路径覆盖；不修改数据哈希。在线 MODEL_BASE 只覆盖基座。权重不包含精确 resume 状态。

| 现象 | 处理 |
| --- | --- |
| 图像缺失 | 核对 data-root、符号链接和小写相机目录 |
| split/hash 不匹配 | 核对标注、划分、checkpoint 来自同一版本，不绕过校验 |
| 输出非空 | 选择新目录，不是自动续训 |
| 显存不足 | 降低 train/eval batch，记录累积与有效 batch；先跑 smoke |
| CPU dtype 错误 | `--device cpu --backbone-dtype float32`，7B 仍需大量内存 |
| 离线加载缺资源 | 核对权重、处理器和模型 Python 文件 |
| 空图或 RPC 取图慢 | 核对场景和渲染；当前客户端对无效图像重试，见专项排障 |
| 历史模型或联合模型 | 使用[扩展指南](extensions.md)，不混入主实验 |

## 8. 回归检查

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="" HF_HOME="$PWD/.cache/huggingface" HF_MODULES_CACHE="$PWD/.cache/huggingface/modules" OMP_NUM_THREADS=2 python -m pytest tests -q
python scripts/check_course_repository.py
```

单元测试不代表真实 7B 或 AirSim 评测通过；保存/梯度/processor 修改应另跑真实 smoke。仓库检查脚本核对当前文档链接、实验规格和示例语法，归档中的本地历史产物链接另行标记。

<a id="joint-lora"></a>
<a id="uav-history"></a>

历史链接兼容入口：联合 LoRA 和 UAV 数值历史的命令已集中到[可选扩展](extensions.md)。
