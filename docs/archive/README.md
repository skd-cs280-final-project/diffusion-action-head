# 历史归档与证据范围

这里保留课程重组之前的实现记录与实验。原日期、统计范围和大部分原始叙述保留；其中“当前”“运行中”等词只对应原记录时刻。归档不是本次课程实验的结果。

## 可引用结果

| 记录 | 内容与范围 |
| --- | --- |
| [冻结头训练](experiments/full_training_20261010.md) | 训练配置、监督统计和第三轮验证快照 |
| [历史周报](experiments/weekly_report_20261011.md) | 当时的阶段数据、标签与验证分析 |
| [NYC 10 条配对](experiments/nyc10_evaluation_20261011.md) | 小样本原 AeroVLA 与扩散比较，不能替代全量结论 |
| [原 best NYC 281 条](experiments/nyc_full_best_gpu3_20261012.md) | 164/281，SR 58.36%、OSR 62.28%、原协议 SPL 52.07% |
| [联合 LoRA 训练](experiments/joint_lora_20261103.md) | 23,080 次更新；512 帧验证 best 仍为 step 0 |
| [联合 final NYC 281 条](experiments/nyc_joint_final_seen281_20261104.md) | 166/281，SR 59.07%、OSR 61.92%、SPL 52.53%；含中断续跑 |
| [详细动作头与历史合同](design/diffusion_action_head.md) | 原实现、状态输入、因果索引与验证细节 |
| [历史状态调研](research/history_pose_20261012.md) | 数值状态分支依据及未实施方案 |
| [开发变更记录](development/changelog.md) | 原实现迭代和当时测试记录 |

图表、固定数据及生成脚本仍在 [reports](../reports/README.md)，以保持生成脚本的目录关系。归档中的 `checkpoints/`、`eval_results/`、数据与机器绝对路径指向本地运行产物，不随 Git 发布，链接不保证在新机器存在。

## 清理范围

重复的联合训练进度页、旧 GitHub 上传说明、当前不采用的碰撞安全层/两阶段规划提案，以及上游宣传图片已从主项目移除。详细联合训练结果和闭环最终报告仍保留。原长训练控制台日志不再嵌入文档目录，已保留结构化结果和曲线；训练曲线生成所需的固定 metrics 快照仍保留。

清理前所有本次涉及的代码与文档已有本地压缩备份，旧已提交文件也可从 Git 历史获取。迁移表和验证说明见[整理记录](../maintenance.md)。旧设计中的外部研究链接仅是历史参考，不改变本次课程范围。
