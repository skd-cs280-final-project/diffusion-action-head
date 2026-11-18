# 模型与评估接口约定

本页是 A/B/C 的开发合同。当前生产实现仍是 `src/model_wrapper/` 下的单步模型；下面涉及多模型加载、动作块和统一记录的内容需要对应成员实现，不能直接假定现有 CLI 支持。

## 张量合同

| 字段 | 形状 / 类型 | 规则 |
| --- | --- | --- |
| condition | `[B,L,D]` 浮点 | 当前图文 token 特征，骨干冻结 |
| condition_mask | `[B,L]` bool | True 为有效 token，池化和 cross-attention 都使用 |
| actions | `[B,H,4]` FP32 | `[fwd,down,yaw,land]`，归一化至 [-1,1] |
| action_mask | `[B,H]` bool；必要时 `[B,H,4]` | True 参与监督，尾部 padding 不进入注意力有效上下文 |
| timesteps | `[B]` 整数 | 扩散噪声步 k，不是环境时刻 t |
| predicted_actions | `[B,H,4]` FP32 | 保持动作范围和维度顺序；不在头内执行仿真 |

现有 `DiffusionActionHead.compute_loss` / `sample` 已支持动作张量维度配置；但数据集、外层模型 schema、在线 wrapper 不等于已经支持多步课程路径。现有 loss mask 不自动等于双向 attention padding mask。

## 对外调用

现有单步 `predict_action` 和停止标志接口继续保留。课程扩展建议新增 `predict_action_chunk`，统一返回归一化动作块。执行层按固定 E 解码并执行，在 LAND / 环境终止后停止剩余动作。A1 回归模型忽略采样 RNG，但保留兼容的预测调用合同。

A 维护加载注册；B 的模型类型应显式登记 H、动作注意力、序列 schema；C 通过加载器调用预测，不在评估器内推断网络结构。检查点用独立格式版本保存，不能在旧 v1/v2 中改变字段含义。

## 数据与动作语义

fwd [0,5]、down [-5,5]、yaw [-1.1,1.1]；LAND = is_last_step OR is_penultimate，0/1 映射为 -1/+1，归一化第四维 >=0 触发停止。每个未来动作仍以它自身决策时刻的局部控制定义为准，不当作统一世界坐标增量。

主实验不新增数值状态、未来观测、未来指令、独立停止分支或新安全控制器。B 在报告中明确标签连续性、窗口与终止处理。C 保持原执行器的物理控制含义与成功判据。

## 检查点和交付

保存：模型类型/版本、头配置、H、动作 schema、条件方式、注意力方式、固定扩散配置（仅扩散模型）、tokenizer/processor、基座及 adapter 引用、权重、数据/划分哈希、种子、训练预算和代码提交。

现有权重不包含精确 resume 所需 optimizer/scheduler/RNG；不能把重新加载权重称为精确续训。迁移路径可以改资源位置，但不改数据 hash 或 schema。

交付使用[运行记录模板](../../evaluation/course/run_manifest.example.json)。未运行指标留 null，未来实现标记 planned；历史证据不得填入新的 A0/B1 等结果字段。
