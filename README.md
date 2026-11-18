# Diffusion Head

**课题：《基于预训练扩散动作头的无人机视觉语言导航推理评测与分析》**

[课题简介与报告提纲](docs/course_project.md) · [详细分工](docs/team/README.md)

**[模型权重与仿真环境下载](docs/guides/resource_download.md)**：通过 [GitHub Release](https://github.com/skd-cs280-final-project/diffusion-action-head/releases/tag/inference-assets-v1) 按成员下载，包含基座、模型权重、12 个场景和闭环任务元数据。`git clone` 不会下载这些大文件。

面向无人机导航的 Diffusion Policy 与扩散动作头项目。当前课程工作只使用已经训练好的模型，通过现有入口进行离线与闭环推理测试，整理结果和报告。

不再训练、不修改代码或框架、不开发新接口。固定 DDPM、100 步训练噪声日程、10 步推理去噪，H=1、E=1、history=0；不开展扩散步数实验。

## 模型与测试

[清单](configs/course/models.json)共 27 项：原 AeroVLA、头 best、联合 step0、每 1000 步的 23 个中间检查点和 final。step0 是初始化对照，两步 smoke 排除。

全部 26 个扩散状态用已有离线入口测试；原 AeroVLA 没有对应扩散离线接口，只参加闭环。正式闭环使用原 AeroVLA、头 best、联合 step0、step1000、final 五个预定代表。中间 checkpoint 属于同一次训练，不是独立训练重复；权重不随 Git 分发。

## 三人分工

| 成员 | 完整负责场景 | ModularPark 分片 | 正式任务 × 五模型 |
| --- | --- | --- | --- |
| [A](docs/team/member_a.md) | NYCEnvironmentMegapa、Carla_Town07、ModernCityMap | 337 | 1002 × 5 = 5010 |
| [B](docs/team/member_b.md) | TropicalIsland、Carla_Town04、Carla_Town10HD | 310 | 1002 × 5 = 5010 |
| [C](docs/team/member_c.md) | NewYorkCity、Carla_Town01/05/06/15 | 311 | 1001 × 5 = 5005 |

覆盖 12 个场景、20 组场景/划分、3,005 个任务。每人负责所列场景的全部现有 seen/unseen-object 划分，以及自己的 unseen-map ModularPark 分片。参数统一：闭环 batch=1、maxWaypoints=200，原单次预测、默认 seed100；不开发新算法。

[详细计划](configs/course/closed_loop_plan.json)、三人任务清单、冻结分片和个人全部命令已准备。A/B/C 分配 GPU 0/1/2 与独立端口，每人先做 15 次推理预检，再开始正式评测。离线另分配 8/9/9 个扩散状态。每人完成自己的统计、案例和报告，当前未启动正式推理。

## 文档

优先阅读[推理测试执行与记录手册](docs/guides/inference_test_handbook.md)和[可视化要求](docs/guides/inference_visualization.md)，再按个人命令页执行。

- [推理指南](docs/guides/training_and_evaluation.md)：已有命令、资源与路径。
- [测试协议](docs/experiments/protocol.md)：固定设置、模型和指标。
- [已有动作头设计](docs/design/diffusion_action_head.md)：模型原理与代码位置。
- [文档导航](docs/README.md)：当前说明与历史实验。

## 快速离线评估

准备环境、7B 基座、原 adapter、原帧级标注、验证集前视/下视图像，以及选定的已训练模型。Release 不含验证集图像；闭环测试只需任务元数据。检查 checkpoint 配置中的旧绝对路径：离线 CLI 不提供基座/adapter 路径覆盖，本机已准备仅修正资源路径的副本，详见[资源说明](docs/team/resources.md)。

```bash
conda activate aero_vla
export PYTHON=python
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_diffusion_head.sh   --checkpoint ./eval_results/course_inference/resources/models/diffusion_head_best   --data-root ./dataset_raw   --data-json ./data/aerovla_train_dataset.json   --split-json ./data/continuous_split_seed42.json   --batch-size 2 --seed 42 --num-inference-steps 10   --output ./eval_results/inference/diffusion_head_best/offline_seed42.json
```

现有扩散离线入口支持 v1/v2，标准闭环入口支持原 AeroVLA 和扩散模型。只准备任务清单、运行参数和结果表，不新增运行器或改变模型行为。新测试尚未启动。

## 来源与许可

基于 [AeroVLA](https://github.com/XuPeng23/AeroVLA)，参考 [NavDP](https://github.com/InternRobotics/NavDP) 与 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)。保留上游历史、署名和 [Apache License 2.0](LICENSE)。模型/数据遵循各自来源条款，报告区分已有训练与本次新增推理分析。
