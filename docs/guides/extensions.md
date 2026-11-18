# 已有训练与扩展实现的来源

当前阶段只做已训练模型推理；这里解释权重来源和兼容边界，不安排新训练。联合训练模型已纳入[当前实验协议](../experiments/protocol.md)，数值历史扩展不作为本阶段变量。

## 仅动作头训练

`checkpoints/best` 来自冻结骨干、训练扩散动作头的阶段。保留 `src/train_diffusion_head.py` 和对应脚本用于解释与复现历史实现；当前三人无需重新执行。参数和当时验证范围见[历史训练记录](../archive/experiments/full_training_20261010.md)。

## 已完成的联合 LoRA

联合阶段从已训练扩散模型恢复，训练视觉与 Llama 的 LoRA、projector 和扩散头，基座主体参数仍冻结，不是全部 7B 参数全量微调。正式运行完成 1 epoch、23,080 次更新，保存中间检查点与 final。v2 必须加载对应 `joint_adapter.pt` 和动作头，仍依赖基座与原 adapter。

联合目录 `best/` 依据 noise MSE 选择，实际保留 step 0；它是初始化对照，不是联合更新后的最佳检查点。每 1,000 步的检查点来自同一次训练，不是不同随机种子训练。当前登记全部正式检查点做离线评估，并预先指定 step 1,000 和 final 等代表模型参加闭环。

结果与口径见[联合训练归档](../archive/experiments/joint_lora_20261103.md)和[281 条闭环](../archive/experiments/nyc_joint_final_seen281_20261104.md)。旧结果不等于当前新实验已完成，也不保证联合模型所有指标更好。

## UAV 数值历史

实现支持历史状态进入动作头，不进入 Llama；没有历史 RGB 或长期语义记忆。详细字段与坐标见[历史设计](../archive/design/diffusion_action_head.md#uav-history)。当前使用模型的 history_length=0，不能直接打开 K=4 当成无需训练的对照。已有状态审计和两步 smoke 只证明流程，不作为当前性能模型。

现有训练和 smoke 脚本保留，但 smoke 会更新参数，因此不属于本次“只推理”的验收入口。推理运行方式见[指南](training_and_evaluation.md)。
