# 文档导航

项目主线是 Diffusion Policy 和动作扩散头。阅读顺序为项目简介 → 设计与固定协议 → 三人分工 → 运行指南 → 历史证据。课程实验固定 DDPM、训练噪声步数 100、推理去噪步数 10；不安排扩散步数实验。

| 文档 | 用途 |
| --- | --- |
| [项目首页](../README.md) | 功能状态、目录与快速开始 |
| [动作头设计](design/diffusion_action_head.md) | 当前架构、候选结构、多步设计与限制 |
| [接口约定](design/interfaces.md) | 数据张量、预测、checkpoint、交付协议 |
| [固定实验协议](experiments/protocol.md) | A0–B4、训练预算、评估与统计 |
| [团队总览](team/README.md) | 文件归属、协作边界、里程碑 |
| [成员 A](team/member_a.md) | 单步头与条件注入 |
| [成员 B](team/member_b.md) | 多步序列与滚动执行 |
| [成员 C](team/member_c.md) | 评估、基线与结果分析 |
| [运行指南](guides/training_and_evaluation.md) | 安装、资源、训练、离线和闭环 |
| [可选扩展](guides/extensions.md) | 已有历史输入、联合 LoRA；不进入主实验 |
| [AirSim 专项排障](guides/troubleshooting.md) | RPC 取图延迟排障 |
| [历史归档](archive/README.md) | 旧实验、详细实现记录、证据边界 |
| [图表与证据](reports/README.md) | 固定报告快照及生成脚本 |
| [本次整理记录](maintenance.md) | 文件迁移、删除范围、验证方式 |

## 内容状态

- 已实现：单步 Transformer 扩散策略、训练/离线/闭环入口；历史输入和联合 LoRA 为可选扩展。
- 待实现：A1–A3、B1–B4，新的统一课程结果汇总与正式实验。
- 实验规格：`configs/course/` 只是共同配置与任务登记，不会被现有 CLI 自动读取。
- 历史材料：`archive/` 和 `reports/` 的数字保留原统计日期，不能混作课程新增结果。

## 维护规则

新方法只更新设计、对应成员页面和实验规格；完成实验后由 C 统一登记运行清单。不要新增临时进度周报或重复的运行指南。当前文档使用相对路径；归档中的绝对路径只表示原实验环境。报告原始大文件、训练日志和权重放到本地输出目录，不提交 Git。
