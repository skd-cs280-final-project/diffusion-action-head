# 推理实验接口约定 v2

当前外层模型仍使用单步动作。新增接口只包装已有预测，不能训练、改权重、增加 H 或替换去噪网络。以下适配器尚待成员实现，不能直接假定现有 CLI 支持。

## 三个独立接口

| 接口 | 负责人 | 合同 |
| --- | --- | --- |
| prepare_variant | B | 原始当前观测/最终 prompt → 变体输入及变换元数据；identity 与原输入一致 |
| sample_candidates / trace | A | 固定条件 C 与局部 RNG → `[N,4]` 归一化动作、可选完整 10 步只读记录 |
| select_action | C | 候选及已冻结规则 → 一个完整四维动作、选择索引/投票和耗时 |

已有 condition 为 `[B,L,D]`，condition_mask `[B,L]` 中 True 为有效 token，动作 `[B,1,4]`。候选 N 与动作预测长度 H 独立；采样 N 个 H=1 的候选，不是预测 N 个未来动作。

原 predict_action/停止标志与旧 checkpoint 加载接口保留。no_grad/eval；对 v2 必须恢复 joint_adapter.pt 和完整 projector。现有 online setup 的 seed 存在硬编码，需要显式包装；不能把改变环境变量等同于已控制推理 RNG。

## 默认一致性

B identity、A trace关闭且 N=1、C 直接取唯一候选时应复现原路径（在相同设备/精度/RNG/输入下）。hook 不额外消费随机数。缓存只按模型哈希、输入内容、预处理配置命中，不能跨模型共享 VLM 特征。

同候选池比较选择规则，扰动 RNG 与采样 RNG 分开。原 token 模型无扩散候选，不人为复制它得到 N 个“候选”。解析失败必须记录；共同离线适配返回相同物理单位和停止规则。

## 随机数与分片

记录 run_seed、episode_uuid、观测/决策序号、采样 seed、变换 seed 和候选数。episode 级派生使用稳定哈希规则并写进协议，避免 Python hash 随进程变化。分片合并应和同协议不分片拥有相同 episode seed；续跑是否恢复 RNG 明确记录，不承诺跨硬件逐位一致。

## 运行交付

使用[运行模板](../../evaluation/course/run_manifest.example.json)，保存模型 ID、权重与资源 hash、数据/split/episode 清单 hash、协议版本、代码版本、成员/shard、输入变体、候选与选择规则、seed、完成数量、硬件、耗时和失败记录。指标缺失为 null，未运行为 planned/not_run。模板以扩散单候选为例；原 token 模型的 sampler/num_train_timesteps/num_inference_steps 等扩散专属字段改为 null，不套用虚假的扩散配置。

A/B/C 各自提交本线结果和图表，C 提供 schema/统计函数，各人使用相同定义。GT 仅用于评估计算，不能进入实际输入变换或候选选择。原始 checkpoint 保留，迁移只在副本改资源路径。
