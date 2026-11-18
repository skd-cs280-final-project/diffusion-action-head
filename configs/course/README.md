# 课程实验规格

`protocol.json` 是固定控制变量；`a/experiments.json`、`b/experiments.json` 和 `eval/experiments.json` 分别登记 A/B/C 任务。

这些文件用于分工和复现记录，**不是现有训练脚本自动读取的配置**。A0 可按[运行指南](../../docs/guides/training_and_evaluation.md)调用现有 CLI。A1–A3、B1–B4 的模型/序列接口仍待对应成员实现，不能仅通过改 JSON 运行。

扩散模型统一固定 DDPM、100 个训练噪声步和 10 步推理去噪。没有扩散步数或采样器对比实验。`H` 是动作预测长度，`E` 是每次重新决策前执行的动作数。

`implementation_status` 描述代码能力，`run_status` 描述本次课程正式运行；历史模型已完成不等于课程 A0 已完成。B0 引用 A0，B4 复用 B1 权重，不新增训练。
