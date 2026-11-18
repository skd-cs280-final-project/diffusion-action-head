# 三人闭环推理评测详细分工

课题：**《基于预训练扩散动作头的无人机视觉语言导航推理评测与分析》**。

所有人使用已有模型和现有命令，不训练、不改代码或框架，不做输入扰动、多候选选择或扩散步数实验。三人均负责自己的离线、闭环、统计和报告。

## 已冻结的场景分配

| 成员 | 完整负责场景（含已有 seen/unseen-object） | ModularPark 分片 | 正式任务 | 五模型 episode 次数 |
| --- | --- | --- | --- | --- |
| [A](member_a.md) | NYCEnvironmentMegapa、Carla_Town07、ModernCityMap | 337 | 1002 | 5010 |
| [B](member_b.md) | TropicalIsland、Carla_Town04、Carla_Town10HD | 310 | 1002 | 5010 |
| [C](member_c.md) | NewYorkCity、Carla_Town01、Carla_Town05、Carla_Town06、Carla_Town15 | 311 | 1001 | 5005 |

共 12 个场景、20 组场景/划分、3,005 个正式任务，五模型合计 15,025 次闭环。各人的任务清单已写入 [assignments](../../configs/course/closed_loop_plan.json)，不是等待后续再拆分。完整场景归一人负责，仅 ModularPark 拆成互不重叠的三份。

原任务 JSON 的每一行是帧记录，现有 loader 按 json 指向的轨迹去重；这里的任务数是去重后的轨迹数，不是行数。三个 ModularPark 文件每条轨迹保留一条原始记录，frame 不参与当前闭环起点选择。全体分片无重复、无遗漏，覆盖全部 3,005 个 category/trajectory 标识。

分配同时参考任务数和原始轨迹帧数，三人总参考帧数为 A 271,982、B 268,951、C 263,961，相差约 3%。这是预估工作量的依据，不等于实测耗时；C 场景较多、启动次数较多，但无需新增开发任务。各自负责全部流程，不集中给某一人评测或写报告。

## 所有人都跑的五个模型

| 编号 | 模型 | 原始权重目录 |
| --- | --- | --- |
| M0 | 原 AeroVLA | `checkpoints/aero_vla` |
| M1 | 扩散头 best | `checkpoints/best` |
| M2 | 联合 step 0 | `checkpoints/joint_lora_stage2_20261103/best` |
| M3 | 联合 step 1000 | `checkpoints/joint_lora_stage2_20261103/checkpoint-1000` |
| M4 | 联合 final | `checkpoints/joint_lora_stage2_20261103` |

联合 best 是 step 0 初始化对照，不能称联合更新后最优。联合中间状态来自同一次训练。全 26 个扩散状态另外做已有离线评估；A/B/C 的确切离线模型列表见个人文档，分别 8/9/9 个，A 另核对原参考模型。

## 参数与资源

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

| 成员 | 模型 GPU | 场景 GPU | 管理服务端口 | 首个场景 RPC | DDP 端口 |
| --- | --- | --- | --- | --- | --- |
| A | 0 | 0 | 30000 | 30001 | 20001 |
| B | 1 | 1 | 35000 | 35001 | 20002 |
| C | 2 | 2 | 40000 | 40001 | 20003 |

本机核对到四张 24GB RTX 4090，GPU 3 留作资源备用；空闲状态是核对时刻的状态。每人模型和场景共用一张卡，是否容得下须由预检验证，不能根据静态清单保证。显存不足时先串行调度，或将一人的场景临时放空闲 GPU 3 并记录资源变化；不改 batch、模型或去噪参数。三人不能共用同一场景管理服务。

## 完成顺序

1. 按个人文档准备环境，确认本地资源副本，查看已有入口的 dry-run 输出。
2. 先做个人指定场景的三任务预检，五个模型各跑一次，共每人 15 次；预检结果写 preflight，不计正式成绩。预检只是推理，没有训练更新。
3. 预检通过后先完成个人主场景（A NYC、B TropicalIsland、C NewYorkCity），再做其余完整场景，最后做 ModularPark 分片。每个场景/划分依次 M0–M4。
4. 完成分配的全量离线评估；与本人的闭环串行，避免抢同卡资源。
5. 核对每个正式运行的完成数，使用现有指标命令统计，整理表格、成功/失败案例与个人报告。组内合并结果。

正式共有 A 30、B 30、C 50 个“模型 × 场景/划分”运行包。启动次数不同不等于 episode 数量不同。每人另有 5 个预检包；预检总计 45 次，正式总计 15,025 次。

## 结果与交付

输出统一为 `eval_results/course_inference/<a|b|c>/formal/<model_id>/<category>/<scene>/`；控制台和指标日志放 `logs/course_inference/`，不能放进轨迹目录污染统计。个人文档提供全部命令，JSON 清单仅登记任务，不是新运行器。

每人提交：离线指标表、全部个人闭环运行的 full/easy/hard 表、每个场景至少一个成功和一个失败案例（若没有该类则如实注明）、实际命令/环境、完成及失败记录、个人报告。指标表同时列 expected/completed/missing；缺失任务不隐去。个人空结果表在 configs/course/results_template/a.json、b.json、c.json，总表在 summary.json；未运行指标均为 null。现有统计使用已保存结果作分母，必须先核对任务完整性再报正式成绩。

跨场景总成绩按完成的任务数加权；SR/OSR 用计数合并，SPL/NE 按各自有效样本数加权，不能把 20 个场景百分比等权平均。三份 ModularPark 用同样规则合并，保留其成员与分片字段，不另跑一遍整个场景。easy/hard 是同一批运行结果的分组统计，不是额外测试。某组无样本时可能输出 NaN，应记录“不适用（样本数 0）”，不能填成 0 分。

任务来自已有评估集合，不能称全新测试。现有全局 RNG 不恢复 episode 级状态，续跑会影响后续噪声序列；记录中断/续跑，核对不完整目录，不能把目录存在当作成功完成。当前准备未启动真实模型推理或仿真。
