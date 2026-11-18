# Diffusion Head

面向无人机导航的 Diffusion Policy 与扩散动作头项目。当前课程工作只使用已经训练好的模型，通过现有入口进行离线与闭环推理测试，整理结果和报告。

不再训练、不修改代码或框架、不开发新接口。固定 DDPM、100 步训练噪声日程、10 步推理去噪，H=1、E=1、history=0；不开展扩散步数实验。

## 模型与测试

[清单](configs/course/models.json)共 27 项：原 AeroVLA、头 best、联合 step0、每 1000 步的 23 个中间检查点和 final。step0 是初始化对照，两步 smoke 排除。

全部 26 个扩散状态用已有离线入口测试；原 AeroVLA 没有对应扩散离线接口，只参加闭环。正式闭环使用原 AeroVLA、头 best、联合 step0、step1000、final 五个预定代表。中间 checkpoint 属于同一次训练，不是独立训练重复；权重不随 Git 分发。

## 三人分工

| 成员 | 离线与资源核对 | 标准闭环与交付 |
| --- | --- | --- |
| A | 8 个扩散状态，另核对原 AeroVLA 资源 | 约 1/3 任务跑全部五个模型，整理指标、案例与自己的报告 |
| B | 9 个扩散状态 | 同上 |
| C | 9 个扩散状态 | 同上 |

三人不再按算法方向分工，也不安排输入扰动、多候选策略、去噪 hook 或适配器开发。各自完成测试和写作，按试跑耗时微调任务量。[完整分工](docs/team/README.md)

## 文档

- [推理指南](docs/guides/training_and_evaluation.md)：已有命令、资源与路径。
- [测试协议](docs/experiments/protocol.md)：固定设置、模型和指标。
- [已有动作头设计](docs/design/diffusion_action_head.md)：模型原理与代码位置。
- [文档导航](docs/README.md)：当前说明与历史实验。

## 快速离线评估

准备环境、7B 基座、原 adapter、训练标注与图像，以及选定的已训练模型。检查 checkpoint 配置中的旧绝对路径：离线 CLI 不提供基座/adapter 路径覆盖，先修正复制出的配置或使用模型 API。

```bash
conda activate aero_vla
export PYTHON=python
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh   --checkpoint ./checkpoints/best   --data-root ./dataset_raw   --data-json ./data/aerovla_train_dataset.json   --split-json ./data/continuous_split_seed42.json   --batch-size 2 --seed 42 --num-inference-steps 10   --output ./eval_results/inference/diffusion_head_best/offline_seed42.json
```

现有扩散离线入口支持 v1/v2，标准闭环入口支持原 AeroVLA 和扩散模型。只准备任务清单、运行参数和结果表，不新增运行器或改变模型行为。新测试尚未启动。

## 来源与许可

基于 [AeroVLA](https://github.com/XuPeng23/AeroVLA)，参考 [NavDP](https://github.com/InternRobotics/NavDP) 与 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)。保留上游历史、署名和 [Apache License 2.0](LICENSE)。模型/数据遵循各自来源条款，报告区分已有训练与本次新增推理分析。
