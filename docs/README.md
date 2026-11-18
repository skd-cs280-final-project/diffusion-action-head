# 文档导航

当前只使用已训练模型和已有入口做推理测试，不再训练、改代码或开发新策略。协议格式保留 v2，当前修订为 existing_entrypoints_only。

| 文档 | 内容 |
| --- | --- |
| [课题简介](course_project.md) | 课题题目、研究问题与报告提纲 |
| [资源准备](team/resources.md) | 本机资源副本与迁移 |
| [项目首页](../README.md) | 范围与快速入口 |
| [三人分工](team/README.md) | 模型、闭环任务与交付均分 |
| [成员 A](team/member_a.md) | A 的执行任务 |
| [成员 B](team/member_b.md) | B 的执行任务 |
| [成员 C](team/member_c.md) | C 的执行任务 |
| [测试协议](experiments/protocol.md) | 离线、闭环与已有指标 |
| [动作头原理](design/diffusion_action_head.md) | 现有方法说明 |
| [现有入口](design/interfaces.md) | 不开发新接口的边界 |
| [运行指南](guides/training_and_evaluation.md) | 资源与命令 |
| [已有训练来源](guides/extensions.md) | 头训练、联合训练和历史扩展 |
| [排障](guides/troubleshooting.md) | AirSim RPC |
| [历史记录](archive/README.md) | 既有实验 |
| [图表证据](reports/README.md) | 固定快照 |
| [整理记录](maintenance.md) | 文档修订与验证 |

[模型清单](../configs/course/models.json)共 27 项：26 个扩散状态做离线；原 token 只做闭环。五个代表模型做共同闭环。当前修订未启动正式推理，登记不等于已加载测试；历史结果不可当作本次新结果。

三人场景、模型、参数、端口及全部命令已固定，任务量为 1002/1002/1001；共 15025 次正式模型-任务运行。详细登记见 [closed_loop_plan.json](../configs/course/closed_loop_plan.json)。
