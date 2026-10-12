# AeroVLA：无人机导航动作扩散头

本项目在 [AeroVLA](https://github.com/XuPeng23/AeroVLA) 基础上，将原有 LMHead 替换为动作扩散头，用于根据视觉观测和导航指令生成无人机动作。项目重点是**冻结视觉语言骨干，只训练扩散头**，并建立训练、评估与实验分析流程。

## 模型结构

```text
前视图 + 下视图 + 导航指令
          ↓
冻结的视觉编码器、projector 与 Llama
          ↓
多模态隐藏特征 + 带噪动作 + 扩散时间步
          ↓
可训练的 Transformer 动作扩散头
          ↓
前进 / 升降 / 转向 + LAND 停止信号
```

扩散头默认采用 hidden 256、4 层、8 个注意力头，以噪声预测 MSE 训练，通过 DDPM 去噪生成动作。当前数据管线与闭环策略为**预测 1 步、执行 1 步**。

## 已实现功能

- **扩散头训练**：冻结 OpenVLA 骨干及合并后的 AeroVLA LoRA，仅优化动作头，保存独立头权重。
- **数据与验证**：按轨迹隔离训练/验证集，支持每轮验证、最佳检查点选择和独立离线评估。
- **闭环评估**：接入 AirSim / TravelUAV，支持动作执行、结果保存、基线对比和轨迹可视化。
- **监督与监控**：提供标签统计、样本审计、训练曲线和 Markdown 实验记录。

历史轨迹与当前位姿输入、两阶段规划器和碰撞改进机制目前处于调研与方案设计阶段，详见[研究文档](docs/README.md#下一步研究尚未实施的方案)。

## 使用入口

先准备 `aero_vla` 环境及项目依赖（见 [requirements.txt](requirements.txt)），并放置 OpenVLA-7B（`openvla-7b/`）、原 AeroVLA LoRA（`checkpoints/aero_vla/`）、TravelUAV 图像（`dataset_raw/`）及训练标注（`data/aerovla_train_dataset.json`）。训练使用仓库中的 `data/continuous_split_seed42.json` 划分。大型数据、模型权重和仿真环境需另行准备。

在项目根目录执行，GPU 编号可按实际环境修改：

```bash
conda activate aero_vla

# 检查数据与划分
bash scripts/train_diffusion_head.sh --check-data

# 全量训练扩散头（默认 5 轮），使用新的输出目录
CUDA_VISIBLE_DEVICES=1 bash scripts/train_diffusion_head.sh \
  --output-dir ./checkpoints/aero_vla_diffusion_train

# 对训练生成的最佳检查点做离线评估
CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./checkpoints/aero_vla_diffusion_train/best \
  --output ./eval_results/diffusion/offline_val.json
```

闭环评估还需启动 AirSim 场景并配置服务端口，完整参数与步骤见[训练与评估指南](docs/guides/training_and_evaluation.md)。

## 项目文档

| 内容 | 入口 |
| --- | --- |
| 全部文档与阅读顺序 | [文档导航](docs/README.md) |
| 架构、动作定义与损失 | [扩散头设计](docs/design/diffusion_action_head.md) |
| 全量训练与曲线 | [训练记录](docs/experiments/full_training_20261010.md) |
| NYC 10 任务评估、基线对比与失败分析 | [闭环实验报告](docs/experiments/nyc10_evaluation_20261011.md) |
| 逐次改动与验证记录 | [开发日志](docs/development/changelog.md) |

## 来源与许可

本项目基于 Peng Xu 等作者的 [AeroVLA](https://github.com/XuPeng23/AeroVLA)，扩散头设计参考 [NavDP](https://github.com/InternRobotics/NavDP)。感谢上游项目及其贡献者；原 AeroVLA 论文、演示和实验成果归原作者所有，引用上游工作请参阅其仓库说明。

代码沿用 [Apache License 2.0](LICENSE)，保留原作者版权与许可证。
