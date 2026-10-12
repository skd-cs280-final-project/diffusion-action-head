# AeroVLA 文档导航

整理日期：2026-10-12。原 23 篇文档合并为 12 篇（含本页）。从下表按任务进入；各报告的统计截止时间保留在正文，实验快照不表示当前训练进程状态。

## 从这里开始

| 你想做什么 | 阅读入口 | 内容状态 |
| --- | --- | --- |
| 跑训练、保存权重、做离线或闭环评估 | [训练与评估指南](guides/training_and_evaluation.md) | 已实现功能的运行说明 |
| 理解 LMHead 替换、冻结方式、扩散结构与损失 | [扩散头架构与数据合同](design/diffusion_action_head.md) | 已实现；含历史数据就绪附录 |
| 看全量训练配置、监督检查和第三轮曲线 | [全量训练记录](experiments/full_training_20261010.md) | 固定快照至 step 70,568；含三轮验证 |
| 看周报和训练结果的详细分析 | [2026-10-11 周报](experiments/weekly_report_20261011.md) | 固定快照至 step 59,019；含两轮验证 |
| 比较扩散头与原 AeroVLA 的飞行效果 | [NYC 10 任务评估与对比](experiments/nyc10_evaluation_20261011.md) | 已完成闭环、小样本对比及可视化 |
| 查哪些任务早停、近障或有接触证据 | [终止原因分析](experiments/nyc10_evaluation_20261011.md#termination) | 同一份 NYC 报告的证据分析 |
| 解决 AirSim 取图通信延迟 | [环境排障](guides/troubleshooting.md) | 上游特定问题经验，按需参考 |
| 查每次实现改动与当时的验证 | [统一变更日志](development/changelog.md) | 合并原 10 份工作记录 |
| 查 GitHub 快照包含什么、复现还缺什么 | [GitHub 快照记录](development/github_snapshot_20261012.md) | 历史上传范围与外部依赖 |

## 下一步研究：尚未实施的方案

| 阅读顺序 | 文档 | 主要问题 |
| --- | --- | --- |
| 1 | [碰撞改进方案](research/collision_mitigation_20261012.md) | 终止口径、几何安全检查、候选动作与恢复数据 |
| 2 | [两阶段规划与动作专家](research/hierarchical_planner_20261012.md) | 可读中间计划、上下层接口与监督数据生产 |
| 3 | [历史轨迹与当前位姿调研](research/history_pose_20261012.md) | 机器人论文如何把历史、pose 和运动状态交给 LLM / action expert |

已有单步扩散头与训练/评估流程；上述三篇提出的新机制不能直接当作现有 checkpoint 的能力。

## 目录与维护约定

```text
docs/
├── README.md       文档首页
├── guides/         运行指南、环境排障（2 篇）
├── design/         已实现的架构与数据合同（1 篇）
├── experiments/    训练记录、周报、NYC 评估（3 篇）
├── research/       碰撞、两阶段、历史与位姿方案（3 篇）
├── development/    统一变更日志、GitHub 快照（2 篇）
├── reports/        图表、统计、生成脚本与固定日志快照
└── assets/         项目展示图片、动图
```

- 同一功能的运行命令集中维护在运行指南；小改动追加到统一变更日志的对应模块，避免再次生成零散 `*_worklog.md`。
- 同一实验的后续结果追加到对应报告，并标注日期和统计截止点；正式周报及其固定快照保持原统计范围。
- 新主题调研再新增独立文档，并在本页登记“已实现”或“建议/待验证”。命令默认从 AeroVLA 项目根目录执行。
- 图表和轻量证据保留在 `reports/` 原位置。链接到 `checkpoints/`、`eval_results/` 的文件是本地运行产物，GitHub 代码仓库未包含这些大型目录。
- 旧文件名请查[完整迁移表](development/changelog.md#docs-20261012)。仓库内链接已更新；此前聊天或外部收藏中的旧路径需按该表定位。
