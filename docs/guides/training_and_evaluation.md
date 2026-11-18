# 已有模型推理与评估指南

[项目首页](../../README.md) · [实验协议](../experiments/protocol.md) · [三人分工](../team/README.md)

本阶段只加载已训练权重做推理，不启动训练或含参数更新的 smoke。以下命令从仓库根目录执行，GPU 编号按机器修改。新增去噪记录、输入扰动、多候选选择及统一分片入口仍需按分工实现；JSON 是实验规格，不能当作已实现的批量运行器。

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

## 3. 核对模型与加载路径

全部模型及各人负责的 9 项见 [models.json](../../configs/course/models.json)。登记时只检查元数据和必需文件存在，尚不代表全部权重已逐一加载或完成新评测。运行前补齐文件哈希和实际重载检查。

| 代表模型 | 目录 | 解释 |
| --- | --- | --- |
| 原 AeroVLA | `checkpoints/aero_vla` | token 动作参考 |
| 扩散头 best | `checkpoints/best` | v1，仅动作头训练后的权重 |
| 联合 step 0 | `checkpoints/joint_lora_stage2_20261103/best` | v2 初始化对照，不能写成联合训练后的最优模型 |
| 联合 step 1,000 | `checkpoints/joint_lora_stage2_20261103/checkpoint-1000` | 预先指定的中间阶段对照 |
| 联合 final | `checkpoints/joint_lora_stage2_20261103` | v2，23,080 次联合更新后的最终模型 |

其余每 1,000 步的联合检查点也全部参加 S0 离线评估。联合 v2 必须一起加载 `joint_adapter.pt`、动作头和对应配置，不能只复制动作头而漏掉 LoRA/projector。它们仍依赖原基座与原 adapter；共同条件特征不能跨不同联合权重复用。

旧配置可能记录 `/data/xiyang/project_orig/AeroVLA/` 等绝对路径。离线 CLI 暂无 model-path/adapter-path 覆盖选项；先复制检查点到独立评估目录，再只修正副本的资源路径并记录修改，或使用模型 API 显式覆盖。保留原始权重和数据哈希。在线 `MODEL_BASE` 只覆盖基座。需保留 processor/tokenizer 等完整文件，不能只搬 `.pt`。

## 4. 共同离线评估 S0

资源和路径检查通过后可直接使用现有扩散 v1/v2 入口：

```bash
PYTHON=python CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./checkpoints/best \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json \
  --batch-size 2 --seed 42 --num-inference-steps 10 \
  --output ./eval_results/inference/diffusion_head_best/offline_seed42.json

PYTHON=python CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./checkpoints/joint_lora_stage2_20261103 \
  --data-root ./dataset_raw --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json \
  --batch-size 2 --seed 42 --num-inference-steps 10 \
  --output ./eval_results/inference/joint_final/offline_seed42.json
```

默认使用全部 val。相同清单、batch 和顺序用于各 checkpoint，记录前三维动作误差与 LAND 混淆矩阵。扩散 noise MSE 只在扩散模型间使用；原 AeroVLA 的共同动作解码/离线指标适配需补充，无 noise MSE 应记 null。解码失败必须计数，不能静默删去样本。

小规模检查可以加 `--max-samples`，但该入口的 `--seed` 同时影响子集抽取和采样随机性。A 的多 seed 比较不能直接通过变更 seed 同时更换子集：应使用全 val，或先固定观测清单，再由新入口独立控制采样 seed。不同 seed 的推理不是独立训练重复。

固定 DDPM、100 步噪声日程、10 步采样，H=1、E=1、history_length=0；不运行扩散步数对比。三人分别运行登记到自己的 9 个状态，按共同 schema 合并。

## 5. 共同闭环 S1 与各自变体

三人先冻结开发/最终清单，再把 S1 任务按 UUID 分成互不重叠的三份。每人对自己那份任务运行全部五个代表模型；合并后每个模型的任务集合应完全相同。各人另外负责本研究线的正式变体闭环。

场景需符合服务端 `env_exec_path_dict`，例如 `envs/closeloop_envs/NYCEnvironmentMegapa.sh`。在独立终端启动场景管理服务：

```bash
python -u airsim_plugin/AirVLNSimulatorServerTool.py \
  --gpus 0 --port 30000 --root_path "$PWD/envs"
```

先打印现有标准策略命令：

```bash
PYTHON=python MODEL_DIR=./checkpoints/best GPU_ID=1 \
  SIMULATOR_GPU_ID=0 SIMULATOR_PORT=30000 \
  TASK_ID=seen_valset/NYCEnvironmentMegapa \
  SAVE_DIR=./eval_results/inference/diffusion_head_best/seen_valset/NYCEnvironmentMegapa \
  bash scripts/eval_aerovla.sh --dry-run
```

去掉 `--dry-run` 才执行。模型与场景 GPU 可同为 0，但共享显存。管理端口区别于场景 RPC；并发任务需独立管理服务、RPC/DDP 端口与输出目录。不要同时向同一管理服务提交多组任务。

`TEST_JSON` 指向冻结的任务或分片清单，`DATASET_PATH` 覆盖原始轨迹目录。原 AeroVLA 设置 `MODEL_DIR=./checkpoints/aero_vla`。当前在线入口以硬编码 `100 + rank + dagger_it` 设置随机种子（默认单进程评估为 100）；A3 多 seed 闭环需要先显式接入种子控制，不能宣称仅设置环境变量就已生效。输入扰动与候选选择也需要对应适配，不存在可直接套用的新 CLI 开关。

```bash
python utils/metric.py \
  --root_dir ./eval_results/inference/diffusion_head_best \
  --analysis_list seen_valset/NYCEnvironmentMegapa \
  --path_type_list full easy hard --map_filter NYCEnvironmentMegapa
```

root 是包含 seen_valset 的结果根。分片结果先按 UUID 查重、查漏并合并再统计；不要简单平均不同大小分片的百分比。指标经 ori_info 引用参考轨迹，迁移需保留路径。原协议 NE 是参考终点距离，SPL 的参考长度减 20；所有模型沿用同一口径。`scripts/metric.sh` 的模型路径是脚本内常量，不自动读取 MODEL_DIR。

## 6. 记录与验收

每个运行使用独立输出目录，保存权重哈希、代码版本、输入清单、seed、变体参数和逐任务结果。模板见 [run_manifest](../../evaluation/course/run_manifest.example.json)。先在开发任务固定扰动强度/选择规则，再跑最终任务，不按最终效果选 checkpoint 或删失败任务。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \
  HF_HOME="$PWD/.cache/huggingface" HF_MODULES_CACHE="$PWD/.cache/huggingface/modules" \
  OMP_NUM_THREADS=2 python -m pytest tests -q
python scripts/check_course_repository.py
```

仓库检查和单元测试不代表真实 7B 或 AirSim 评测通过。三人完成各自入口后应验证默认条件、单候选行为与旧流程一致。常见加载、显存与 RPC 问题见[排障](troubleshooting.md)。

<a id="joint-lora"></a>
<a id="uav-history"></a>

已有联合训练与历史输入实现的来源见[训练来源说明](extensions.md)；不需要为当前实验重训。
