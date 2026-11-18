# 文档导航

当前协议为 **v2：全部已有正式模型的推理研究**。不进行新训练、不比较新网络或多步动作，不做扩散步数实验。旧训练型课程方案已由本版本替代，可从 Git 历史查阅。

| 文档 | 内容 |
| --- | --- |
| [项目首页](../README.md) | 当前范围、模型和快速入口 |
| [团队总览](team/README.md) | 均衡分工、公共工作分摊、交付 |
| [成员 A](team/member_a.md) | 生成机制与随机性 |
| [成员 B](team/member_b.md) | 条件依赖与鲁棒性 |
| [成员 C](team/member_c.md) | 动作选择与成本 |
| [固定协议](experiments/protocol.md) | 全模型覆盖、代表模型、实验矩阵 |
| [动作头设计](design/diffusion_action_head.md) | 当前结构、去噪、推理研究边界 |
| [接口约定](design/interfaces.md) | 变换、采样、选择、运行记录 |
| [推理指南](guides/training_and_evaluation.md) | 资源、路径、离线与闭环 |
| [历史扩展](guides/extensions.md) | 历史/联合训练来源，当前不执行 |
| [专项排障](guides/troubleshooting.md) | AirSim RPC |
| [历史归档](archive/README.md) | 既有训练与导航结果，不作为新实验结果 |
| [固定证据](reports/README.md) | 图表与结构化快照 |
| [整理记录](maintenance.md) | 文件组织与验证 |

机器可读清单在 [configs/course](../configs/course/README.md)。27 个登记状态全部进行共同离线评估，分配给 A/B/C 各 9 个；主要五个模型参与正式闭环和详细变体。权重不上传，登记只核对了元数据和文件存在性，不代表已经重新加载并完成评估。

所有新增推理实验状态都是 not_run。现有 184 项模型测试的历史通过记录，不替代新增推理适配器和正式实验验证。
