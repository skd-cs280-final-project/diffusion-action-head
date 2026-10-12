# AeroVLA 动作扩散头改造报告

日期：2026-10-10。

**后续实现更新：** 轨迹 split、训练中验证和独立离线动作评估现已接入，最新运行方式见[训练与评估指南](diffusion_training_evaluation_guide.md)。本报告保留首次架构交付的设计和历史验证结果；下文关于数据缺失、人工标签 smoke 和旧命令的描述不代表当前状态。

## 已实现的架构

```text
前视图 + 下视图 → 原 OpenVLA 视觉骨干 → 原 projector
                                            ↓
导航指令 + Action: → token embedding → 多模态 Llama decoder
                                            ↓ 冻结的最终层隐状态
高斯噪声动作 + 扩散时间步 → Transformer 动作扩散头 → 连续动作 + LAND
```

加载本地 OpenVLA 权重，可合并已有 AeroVLA LoRA 和其 `modules_to_save` projector；随后删除 `language_model.lm_head`，直接调用 Llama decoder 取得最终层隐状态。新训练和推理路径不生成词表 logits，不调用 `generate()`，不把动作标签输入语言模型。

视觉骨干、projector、token embedding、Llama 及合并后的 LoRA **全部冻结**。`train()` 重载始终把这些模块保持在 `eval()`，特征提取使用 `torch.no_grad()`，AdamW 仅包含 `diffusion_head` 参数，训练时另检查上游无梯度。扩散头中的条件投影属于可训练头的一部分。

## NavDP 参考与适配

参考本地 `NavDP/baselines/navdp/policy_network.py` 的 `NavDP_Policy`、`predict_noise` 与去噪循环，以及 `policy_backbone.py` 的正弦时间编码。

| 项目 | 本次实现 |
| --- | --- |
| 去噪网络 | 动作线性投影、可学习动作位置、正弦时间步、TransformerDecoder、LayerNorm、epsilon 输出 |
| 条件 | AeroVLA 多模态 Llama 的最终层全部 token，正确插入视觉 token mask 并屏蔽文本 padding |
| 训练目标 | `MSE(epsilon_theta(x_t,t,condition), epsilon)`，随机时间步和高斯噪声 |
| 调度器 | `diffusers==0.30.3` 的 DDPM，`squaredcos_cap_v2`，epsilon prediction，clip_sample |
| 默认头参数 | hidden_dim=256，4 层，8 heads，dropout=0，训练 100 时间步，推理 10 步 |
| 跳步采样 | 从末时间步开始，显式注册时间步序列，确保如 100/30 的不可整除配置使用真正相邻时间步 |
| 动作语义 | 保持 AeroVLA 的 fwd/down/yaw/LAND，不使用 NavDP 的 xyz 路径增量和累加 |
| critic | 未移植；当前 AeroVLA 标签未提供 NavDP critic 所需监督 |

NavDP 原实现默认 10 个训练/推理步；这里的 100/10 是可配置初始值，尚未经过真实数据效果调优。可用 `--num-train-timesteps 10 --num-inference-steps 10` 对齐原步数。

## 数据合同

动作顺序固定为 `[fwd, down, yaw, land]`，每个样本输出 `[1,4]`，batch 为 `[B,1,4]`。

| 维度 | 物理/原标签范围 | 归一化 |
| --- | --- | --- |
| fwd | [0,5] | [-1,1] |
| down | [-5,5] | [-1,1] |
| yaw | [-1.1,1.1] | [-1,1] |
| land | 0/1 | -1/+1 |

前三维沿用原代码裁剪范围，取消 99 桶量化。LAND 标签为 `is_last_step OR is_penultimate`；推理以归一化第四维 `>=0` 判断降落。推理返回原接口的 `{'fwd', 'down', 'yaw'}` 和停止标志列表。

图片处理复用原双视角拼图：前视在上、下视在下。训练文本为 `instruction.strip() + '\nAction: '`，不添加目标数字、LAND 或 EOS。超长 prompt 明确报错；模型还会检查加上视觉 token 后的总上下文长度。

`instruction` 应是已准备好的最终导航提示，可保留原格式的 `<image>` 前缀，但不要包含 `Action:` 或动作答案。数据集不会从评测 episode 的原始描述自动生成方向提示。

当前训练标注每条只定义一步动作，因此数据集只允许 horizon=1；扩散头模块支持更长 horizon，但需要先提供显式、可靠的未来动作序列数据管线。

训练 JSON 格式示例（数值仅示意）：

```json
[
  {
    "traj_rel_dir": "Japanese_Street/trajectory-id",
    "img_name": "000000.png",
    "instruction": "<image>\nFly forward and find the target.",
    "label": {"fwd": 2.5, "down": 0.0, "yaw": 0.0},
    "is_last_step": false,
    "is_penultimate": false
  }
]
```

## 启动训练

正式训练前需补齐 `data/aerovla_train_dataset.json`。当前本地有基座、AeroVLA adapter 和 TravelUAV 原始图片，但缺少此正式逐帧监督文件；评测 episode JSON 不能替代它。

```bash
cd /data/xiyang/project_orig/AeroVLA
PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python \
CUDA_VISIBLE_DEVICES=1 bash scripts/train_diffusion_head.sh \
  --model-path ./openvla-7b \
  --adapter-path ./checkpoints/aero_vla \
  --data-root ./dataset_raw \
  --train-json ./data/aerovla_train_dataset.json \
  --output-dir ./checkpoints/aero_vla_diffusion \
  --batch-size 2 --gradient-accumulation-steps 8 \
  --epochs 5 --learning-rate 1e-4
```

这是单进程、单 GPU 脚本，有效 batch 默认 16。仅基座使用 bfloat16，扩散头以 float32 训练。梯度累积按真实样本数归一化，正确处理最后不足一批或不足一个累积窗口的情况。

- `--adapter-path none`：只冻结 OpenVLA 基座，不加载已有 AeroVLA adapter。
- `--max-steps 2 --max-samples 3 --num-workers 0`：真实标注准备好后可快速检查训练接口。
- `--device cpu --backbone-dtype float32`：CPU 路径，需要足够内存。
- `--save-steps 1000`：每 N 次 optimizer 更新保存一次；训练结束另在输出目录保存最终头。
- `--help`：查看全部超参数。输出目录已有 `run_config.json` 时拒绝覆盖，应使用新的实验目录。

训练输出包含 `run_config.json`、`metrics.jsonl`、`diffusion_head.pt`、`diffusion_config.json`、tokenizer 和 processor。检查点只保存头权重及基座/adapter引用、动作 schema 和训练元数据，**不保存优化器状态，也不支持精确断点续训**。加载仍需要原基座和原 adapter；本地引用保存为绝对路径，迁移机器时可用模型加载 API 的路径参数覆盖。

## 推理与原评测入口

`AerialVLAWrapper` 根据 `diffusion_config.json` 自动识别扩散检查点。准备好原项目要求的模拟器服务、场景和评测数据后，可使用现有 AirSim 入口：

```bash
PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python \
MODEL_DIR=./checkpoints/aero_vla_diffusion GPU_ID=1 bash scripts/eval_aerovla.sh
```

模型 API 也支持直接调用：

```python
import torch
from src.model_wrapper.aerovla_diffusion import AeroVLADiffusionModel
from src.diffusion_dataset import denormalize_actions

model = AeroVLADiffusionModel.from_pretrained(
    "./checkpoints/aero_vla_diffusion", device="cuda", torch_dtype=torch.bfloat16
)
# inputs 仅含 input_ids、attention_mask、pixel_values；与训练使用相同处理器。
normalized = model.predict_action(**inputs)  # [B,1,4]
physical = denormalize_actions(normalized)
should_land = normalized[..., 3] >= 0
```

旧 `train_aerovla.py`、`scripts/train.sh` 保留原 LM/LoRA 行为；训练扩散头请使用新入口。旧 adapter 评测仍可正常选择。NavDP、原始基座权重和已有 adapter 文件未修改。

## 验证结果

- 完整测试命令：`OMP_NUM_THREADS=2 /data/xiyang/.conda/envs/aero_vla/bin/python -m pytest tests -q`。
- 当前全套 **36 项通过**：扩散头 11、数据 13、模型 7、训练及推理接口 5。
- 模型测试使用真实小型 Llama 与实际 PEFT adapter 保存/恢复/合并；逐张量确认一次 optimizer 更新后全部 backbone 权重不变，且头权重发生变化。
- 检查 DDPM 加噪和 epsilon loss、条件 padding、因果动作 mask、随机种子复现、任意合法推理步数、head checkpoint 严格往返、LoRA/projector 保留、无目标泄漏、梯度累积及 LAND 转换。
- 本地真实 tokenizer、processor、TravelUAV 双视角图片处理通过，得到 `[1,6,224,224]` 视觉输入。
- 训练 CLI `--help`、Python compileall、shell `bash -n`、`git diff --check` 通过。
- 真实 7B + 原 AeroVLA adapter 在 GPU 1 上完成两次更新，全部上游参数冻结，默认规格扩散头成功训练并保存逐步/最终检查点。输入为真实 TravelUAV 图片与人工测试标签，不能证明真实数据收敛或导航效果。

| 7B 短程指标 | 结果 |
| --- | --- |
| 冻结参数 | 7,409,644,992 |
| 可训练头参数 | 5,265,668 |
| 第 1 步（2 样本）loss | 1.2759649753570557 |
| 第 2 步（1 样本）loss | 0.8945425152778625 |
| 保存头权重大小 | 21,088,629 bytes，约 20.1 MiB |
| 检查点目录 | `/tmp/aerovla_diffusion_smoke_run`，仅作接口验证 |

检查点保存的 tokenizer、AutoImageProcessor 和 AutoProcessor 已离线重新加载成功。随后在 CPU float32 下完整重载 7B 基座、原 adapter 和保存的头，默认 10 步去噪通过：输出 `[1,1,4]`、全部有限且在 `[-1,1]`，确认不存在 LMHead，所有 backbone 参数仍冻结。CPU 与 GPU 骨干精度不同，此项验证加载和推理接口，不声称二者数值逐位一致。

短程训练实际命令（人工标签临时文件在验证前创建；不能直接作为正式数据使用）：

```bash
HF_HOME="$PWD/.cache/huggingface" \
HF_MODULES_CACHE="$PWD/.cache/huggingface/modules" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/data/xiyang/.conda/envs/aero_vla/bin/python -u src/train_diffusion_head.py \
  --train-json /tmp/aerovla_diffusion_smoke_train.json \
  --output-dir /tmp/aerovla_diffusion_smoke_run \
  --epochs 1 --max-steps 2 --batch-size 2 \
  --gradient-accumulation-steps 1 --num-workers 0 --save-steps 1
```

此次 GPU 访问先在沙箱内明确报 CUDA 不可用，经命令权限批准后在 GPU 1 完成，未占用已有任务所在的 GPU 0。测试环境为 PyTorch 2.1.2+cu121、transformers 4.42.4、diffusers 0.30.3。OpenVLA 自带代码提示期望 transformers 4.40.1；本次使用仓库 requirements 的 4.42.4，实际训练和重载推理已通过。

未开展完整正式训练或 AirSim 闭环指标评测；不保证未训练充分的头输出具有飞行控制质量。临时 smoke 检查点和人工标签应只用于开发验证。

## 逐次改动记录

| 记录 | 范围 |
| --- | --- |
| [扩散头工作记录](diffusion_head_worklog.md) | 去噪器、调度器、时间步修正及测试 |
| [模型工作记录](diffusion_model_worklog.md) | LMHead 移除、冻结、PEFT 合并与检查点 |
| [数据工作记录](diffusion_dataset_worklog.md) | 连续动作、图像/文本处理、数据验证 |
| [集成工作记录](diffusion_integration_worklog.md) | 训练入口、评测入口、依赖、集成修正和最终检查 |

本次以文件修改和上述日志交付，未自行创建 git commit。
