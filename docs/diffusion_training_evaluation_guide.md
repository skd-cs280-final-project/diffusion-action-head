# 扩散头训练与评估运行指南

日期：2026-10-10。本次承接[架构报告](diffusion_action_head_report.md)与[数据就绪检查](diffusion_data_readiness_report.md)，补齐轨迹划分、训练中验证、离线动作评估及闭环启动配置。当前代码的运行说明以本文为准。

2026-10-11 的正式训练进度、两轮全量验证、监督数据分析和图表见[项目周报](weekly_report_20261011.md)。下文两步流程检查是历史开发验证，不是正式训练效果。

## 当前数据与训练行为

默认读取 `data/aerovla_train_dataset.json` 和 `data/continuous_split_seed42.json`：

| 集合 | 帧数 | 轨迹数 | 用途 |
| --- | --- | --- | --- |
| train | 369,274 | 6,850 | 仅扩散头参数训练 |
| val | 41,176 | 761 | 验证、best 检查点选择、离线动作指标 |

加载数据时校验划分版本、轨迹 SHA256、train/val 不相交、重复轨迹/帧及完整覆盖。错误或缺失 split 会报错，不会回退到全量训练。`--max-samples` 等限量参数在完成全量划分校验之后，用固定种子随机选帧，并保留原顺序；不会只取数据开头的同一张地图。

模型仍然是冻结的视觉骨干、projector、Llama 和已合并的 AeroVLA LoRA，仅训练动作扩散头；LMHead 已移除。输入、动作范围、LAND 标签和单步动作定义保持前序实现。

## 先检查数据接口

```bash
cd /data/xiyang/project_orig/AeroVLA
PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python \
  bash scripts/train_diffusion_head.sh --check-data
```

此命令校验完整 split，并实际处理 train/val 的首个批次，打印划分摘要和张量形状；不会加载 7B 模型、更新权重或创建训练目录。它不是所有图片的解码完整性检查。前序已全量检查 820,900 个图片路径无缺失。

## 两步真实数据流程检查

```bash
PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python \
CUDA_VISIBLE_DEVICES=1 bash scripts/smoke_diffusion_pipeline.sh
```

该脚本执行 train 中随机 4 帧的两次更新，在 val 中固定 8 帧上计算指标，再通过独立评估进程重载最终检查点。最后核对重载前后的噪声 loss、归一化动作 MSE 和 LAND 混淆统计一致。

默认输出 `checkpoints/diffusion_smoke_real/`。已有输出时会拒绝覆盖；复跑可指定 `SMOKE_OUTPUT_DIR=./checkpoints/diffusion_smoke_real_2`。脚本默认离线读取本地基座和 adapter，默认选择物理 GPU 1，可通过环境变量覆盖。

这个检查点仅证明程序流程可用，并非训练收敛的导航策略。

## 正式训练

```bash
PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python \
CUDA_VISIBLE_DEVICES=1 bash scripts/train_diffusion_head.sh \
  --model-path ./openvla-7b \
  --adapter-path ./checkpoints/aero_vla \
  --data-root ./dataset_raw \
  --train-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json \
  --output-dir ./checkpoints/aero_vla_diffusion_train \
  --epochs 5 --batch-size 2 --gradient-accumulation-steps 8 \
  --eval-batch-size 2 --learning-rate 1e-4
```

默认单进程单 GPU，骨干 bfloat16、头 float32；实际默认有效 batch 为 16。验证默认覆盖全部 41,176 帧，每个 epoch 结束及最后一次更新后运行。`--eval-steps N` 可额外每 N 次 optimizer 更新验证；重合的触发点只验证一次。

验证使用独立固定种子，恢复模型训练模式和随机状态，不参与反向传播。按验证集 `noise_mse` 的最小值保存 `best/`。随机动作采样的指标在相同模型、数据顺序、batch 大小、精度、设备和种子下可复现；改变这些条件时不承诺数值逐位一致。

| 参数 | 作用 |
| --- | --- |
| `--max-samples N` | 仅训练固定随机 N 帧；省略即全部 train |
| `--max-val-samples N` | 使用固定随机 N 帧验证；省略即全部 val |
| `--max-steps N` | 达到 N 次更新后提前结束，并执行最终验证 |
| `--eval-seed 42` | 验证子集和噪声/采样随机种子 |
| `--save-steps 1000` | 定期保存头检查点 |
| `--adapter-path none` | 不加载原 AeroVLA adapter，仅冻结 OpenVLA 基座 |
| `--device cpu --backbone-dtype float32` | CPU 执行，需足够内存且速度较慢 |

数据与划分文件在运行期间应保持不变。输出目录必须为空；本版检查点不包含 optimizer/scheduler 状态，不支持精确断点续训。

## 训练产物

```text
aero_vla_diffusion_train/
├── run_config.json              # 参数、数据/划分/抽样摘要及可训练参数数量
├── metrics.jsonl                # 每次更新的训练 loss、梯度范数、学习率
├── validation_metrics.jsonl     # 每次验证的完整离线指标
├── best_metrics.json            # best 对应的验证结果
├── diffusion_head.pt            # 最终头权重
├── diffusion_config.json        # 基座/adapter 引用、动作 schema、训练/划分元信息
├── data_split.json              # 使用的轨迹划分副本
├── ...                         # tokenizer 和 processor 文件
├── best/                       # 最小验证 noise_mse 对应的完整头检查点目录
└── checkpoint-N/                # 定期头检查点
```

冻结的 7B 基座和 adapter 不复制到输出目录，重载时仍需保留原权重。检查点记录原文件绝对路径；模型 API 可覆盖迁移后的基座/adapter 路径。

## 训练监控和监督标签检查

当前 GPU 1 全量运行已经启用独立 CPU 监控。常用入口见[全量训练记录](diffusion_full_training_20261010.md)，包含实时 Markdown、CSV、曲线图、检查点清单及监督标签样例。

对其他运行，可用下列命令持续整理现有训练日志：

```bash
/data/xiyang/.conda/envs/aero_vla/bin/python src/monitor_diffusion_training.py \
  --run-dir ./checkpoints/aero_vla_diffusion_train \
  --log-dir ./logs/aero_vla_diffusion_train --interval 60
```

`--once` 只导出一次。该脚本不加载模型，输出位于 `run-dir/monitoring/`；CSV 保留全部记录，图中 loss 同时展示原始值与最近 100 次更新均值。安装 matplotlib 时可生成 PNG，无绘图依赖时仍保留 CSV/JSON/Markdown 并记录告警。`log-dir/status.json` 使用 `{"status":"running"}`、`{"status":"completed"}` 或 `{"status":"failed"}`；若自行启动训练时没有这个状态文件，监控会持续等待，不会自动认定训练已经结束。

对当前完整 train/val 划分进行静态监督审计：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data/xiyang/.conda/envs/aero_vla/bin/python scripts/audit_diffusion_supervision.py \
  --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json \
  --run-config ./checkpoints/aero_vla_diffusion_train/run_config.json \
  --output-dir ./checkpoints/aero_vla_diffusion_train/supervision
```

审计核对全量来源与 run_config 的 train/val 计数、选中索引和哈希，因此若训练使用了 `--max-samples` 等子集限制，会拒绝将完整清单冒充该次训练使用的数据。输出全量标签统计及固定抽查样本；固定样本不是训练实际批次记录，也不含预测。

## 独立离线评估

```bash
PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python \
CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./checkpoints/aero_vla_diffusion_train/best \
  --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json \
  --batch-size 2 --seed 42 \
  --output ./eval_results/diffusion/offline_val.json
```

默认评估全部 val。可用 `--max-samples 256` 做快速检查，或 `--num-inference-steps N` 改变去噪步数。不使用 AirSim。输出 JSON 包含：

- `noise_mse`：固定随机噪声和时间步的 epsilon 预测均方误差。
- `normalized_continuous_mse`：归一化后 fwd/down/yaw 三维动作均方误差，不包含 LAND。
- `physical`：分别报告 fwd、down、yaw 的 MAE/RMSE，不将不同量纲合成一个物理误差。
- `land`：TP/TN/FP/FN、precision/recall/F1、accuracy、正负样本数量。分母为零的 precision/recall/F1 定义为 0，正样本数为零时不能据此判断降落能力。
- `split_provenance`：用检查点记录的数据和划分文件 SHA256 校验扩散头 train/val 隔离；旧检查点无这些信息时标记 `unknown`，已知不一致直接报错。

动作指标对照与训练一致的裁剪后标签，不是未裁剪的原始极值。此次 split 是同地图内的轨迹留出，不能当作未见地图测试；冻结的旧 LoRA 是否见过验证轨迹仍未知。离线动作指标也不等于闭环飞行 SR/SPL。

## AirSim 闭环入口

模型显卡与模拟器显卡、服务端口现在独立配置。先检查最终命令：

```bash
PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python \
MODEL_DIR=./checkpoints/aero_vla_diffusion_train/best \
GPU_ID=1 SIMULATOR_GPU_ID=0 SIMULATOR_PORT=30000 \
TASK_ID=seen_valset/NYCEnvironmentMegapa \
bash scripts/eval_aerovla.sh --dry-run
```

确认对应服务可用且可供本次评估使用后，去掉 `--dry-run` 执行。此例由物理 GPU 1 运行模型，向 30000 端口的模拟器服务请求在物理 GPU 0 上运行场景。`DDP_MASTER_PORT` 默认已修正为合法端口 20001，可环境覆盖。脚本也支持 `MODEL_BASE`、`TEST_JSON`、`SAVE_DIR` 等覆盖项，使用 `--help` 查看。

闭环仍使用原项目 episode 清单，不能把 `continuous_split_seed42.json` 传作 `TEST_JSON`。本次没有调用、重启已有模拟器，也没有取得闭环导航指标。

## 验证与变更记录

**本次已跑通真实标注的训练→验证→保存→独立进程重载→离线评估，退出码 0；全套 66 项测试通过。**

| 验证项 | 结果 |
| --- | --- |
| 真实数据全量划分与 `--check-data` | train 369,274 / val 41,176，处理器批次通过 |
| 实际模型 | 本地 OpenVLA 7B + 原 AeroVLA adapter，GPU 1 |
| 冻结 / 可训练参数 | 7,409,644,992 / 5,265,668 |
| 短程使用数据 | train 随机 4 帧、val 随机 8 帧，真实标注，seed=42 |
| optimizer 更新 | 2 次，仅扩散头 |
| train loss | 第 1 步 1.2125136852264404；第 2 步 1.2360974550247192 |
| val noise_mse | 1.50709730386734 |
| val normalized_continuous_mse | 0.7305838082736672 |
| 独立重载一致性 | noise_mse、动作 MSE、LAND 统计与训练末尾验证一致 |
| 划分来源检查 | `verified_diffusion_head_holdout`，数据和 split SHA256 匹配 |
| 输出完整性 | 最终、best、checkpoint-1、checkpoint-2 及 split 副本均已核对 |
| 静态检查 | Python compileall、四个 shell 的 bash -n、git diff --check 通过 |

全套测试覆盖：前序 36 项、新增 split 10 项、训练验证触发 2 项、离线指标/状态恢复/CLI 8 项、闭环启动参数 10 项。测试只有依赖的弃用提示，没有失败。

本次真实输出可直接查看：

- [流程核验结果](../checkpoints/diffusion_smoke_real/pipeline_check.json)
- [离线评估完整报告](../checkpoints/diffusion_smoke_real/offline_eval.json)
- [训练配置与数据摘要](../checkpoints/diffusion_smoke_real/run_config.json)
- [训练记录](../checkpoints/diffusion_smoke_real/metrics.jsonl)
- [验证记录](../checkpoints/diffusion_smoke_real/validation_metrics.jsonl)

这 8 帧验证样本没有 LAND 正例，且仅训练两步，不能用它们判断降落能力或导航性能。随后已按用户要求在 GPU 1 启动 369,274 帧、5 轮正式训练，详情见[本次全量训练记录](diffusion_full_training_20261010.md)。截至 2026-10-11，训练仍在进行，已取得两轮全量验证结果，见[周报](weekly_report_20261011.md)；AirSim 闭环评测仍未完成。上述表格仅属于短程流程检查。

完整测试命令（测试环境需要 pytest）：

```bash
HF_HOME="$PWD/.cache/huggingface" \
HF_MODULES_CACHE="$PWD/.cache/huggingface/modules" OMP_NUM_THREADS=2 \
/data/xiyang/.conda/envs/aero_vla/bin/python -m pytest tests -q
```

每批修改记录分别见：[训练集成](diffusion_train_eval_worklog.md)、[数据划分](diffusion_split_worklog.md)、[离线评估](diffusion_eval_worklog.md)、[闭环启动](diffusion_closed_loop_worklog.md)。
