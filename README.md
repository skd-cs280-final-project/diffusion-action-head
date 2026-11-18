# Diffusion Head

面向无人机导航的 **预训练 Diffusion Policy 推理研究**：使用全部已保存的正式模型及联合阶段检查点，分析动作生成机制、条件敏感性、采样选择与闭环表现。

**本阶段不训练、不更新参数、不替换网络结构。** 扩散模型固定 DDPM、原 100 步训练噪声日程、每次采样 10 步去噪；H=1、E=1、无数值历史。不开展扩散步数或 DDPM/DDIM 对比。

## 模型范围

本地登记了 27 个模型/状态：原 AeroVLA、仅头训练 best，以及联合阶段 step 0、每 1,000 步的 23 个检查点和 final。step 0 是初始化对照；23 个中间检查点来自同一次联合训练，不能当作独立训练重复。两步 smoke 权重不参加性能比较。

全部 27 个都做共同离线评估。正式闭环和细化变体使用预先指定的五个代表模型：原 AeroVLA、扩散头 best、联合 step 0、联合 step 1,000、联合 final。中间检查点用于完整训练轨迹的推理分析，不把全部检查点与所有扰动做无差别全组合。清单见 [models.json](configs/course/models.json)，权重需本地准备，不随 Git 分发。

## 三人均衡分工

| 成员 | 主问题与代码 | 自己完成的评估 |
| --- | --- | --- |
| [A](docs/team/member_a.md) | 固定 10 步去噪过程、采样分布与随机性 | 分配的 9 个模型离线评估，共同闭环约 1/3，自己的重复推理闭环 |
| [B](docs/team/member_b.md) | 视觉/语言条件依赖与鲁棒性 | 分配的 9 个模型离线评估，共同闭环约 1/3，自己的扰动闭环 |
| [C](docs/team/member_c.md) | 多候选动作选择与运行成本 | 分配的 9 个模型离线评估，共同闭环约 1/3，自己的选择策略闭环 |

每人都做代码、实验、统计、图表和方法章节；C 不再承担全队闭环。先用小规模试跑的 GPU/仿真时间均衡任务，再扩大到正式清单。[完整分工与交付](docs/team/README.md)

## 文档入口

| 内容 | 入口 |
| --- | --- |
| 加载已有权重、离线/在线评估 | [推理指南](docs/guides/training_and_evaluation.md) |
| 扩散动作头与当前研究问题 | [方法设计](docs/design/diffusion_action_head.md) |
| 模型清单、固定条件和 S/A/B/C 实验 | [实验协议 v2](docs/experiments/protocol.md) |
| 输入变换、采样、记录接口 | [接口约定](docs/design/interfaces.md) |
| 所有文档与旧实验 | [文档导航](docs/README.md) |

## 快速离线评估

准备环境、7B 基座、原 adapter、训练标注与图像，以及选定的已训练模型。检查 checkpoint 配置中的旧绝对路径：离线 CLI 不提供基座/adapter 路径覆盖，先修正复制出的配置或使用模型 API。

```bash
conda activate aero_vla
export PYTHON=python
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh   --checkpoint ./checkpoints/best   --data-root ./dataset_raw   --data-json ./data/aerovla_train_dataset.json   --split-json ./data/continuous_split_seed42.json   --batch-size 2 --seed 42 --num-inference-steps 10   --output ./eval_results/inference/diffusion_head_best/offline_seed42.json
```

现有离线 CLI 直接支持扩散 v1/v2；原 AeroVLA 的共同离线指标适配、输入扰动、去噪记录、候选选择和统一分片入口仍需实现。规格文件不等于实验已运行。

## 代码结构

```text
src/model_wrapper/                 已有模型、扩散头、在线 wrapper
src/vlnce_src/                     已有闭环评估与环境
scripts/                           已有入口和仓库检查
scripts/course_infer/a|b|c/         各自的推理实验入口区域
configs/course/                    固定协议、全部模型、实验规格
evaluation/course/generation/      A：去噪轨迹与采样分析
evaluation/course/conditions/      B：图像/语言变换
evaluation/course/selection/       C：候选选择与计时
evaluation/course/shared/          公共指标、任务清单与结果合同
data/                             划分、闭环任务和场景元数据
airsim_plugin/                    模拟器服务与客户端
tools/visualization/              视频、点云、轨迹工具
tests/                            现有模型与接口回归测试
docs/                             当前推理方案和历史证据
```

原训练、历史输入、联合 LoRA 实现保留用于检查点来源解释与兼容，不在当前任务中启动训练。全部权重已训练过并不表示它们的额外训练预算相同；头训练与联合训练对比须注明训练阶段差异。

## 来源与许可

基于 [AeroVLA](https://github.com/XuPeng23/AeroVLA)，参考 [NavDP](https://github.com/InternRobotics/NavDP) 与 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)。保留上游历史、署名和 [Apache License 2.0](LICENSE)。模型/数据遵循各自来源条款，报告区分已有训练与本次新增推理分析。
