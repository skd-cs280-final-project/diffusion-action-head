# 已训练扩散动作头与推理研究

[协议 v2](../experiments/protocol.md) · [接口](interfaces.md) · [三人分工](../team/README.md)

当前研究使用已训练模型，不再进行网络结构或训练预算对比。Diffusion Policy 是条件动作分布和执行策略；diffusion action head 是给定多模态条件与带噪动作，预测噪声的网络。

## 实际模型

所有登记扩散权重的 H=1、动作四维、history=0。共同头为 hidden256、4 层、8 heads 的 Transformer Decoder；当前图文 VLM token 投影后经 cross-attention 条件化；正弦噪声时间编码加入 memory，动作使用可学习位置编码。头训练 best 冻结过骨干，联合阶段则训练过视觉/Llama LoRA、projector 和头。推理时所有模型参数固定。

| 内容 | 当前约定 |
| --- | --- |
| 观测 | 前视/下视拼图、当前导航指令 |
| 动作 | fwd [0,5]、down [-5,5]、yaw [-1.1,1.1]、LAND 0/1，归一化到 [-1,1] |
| LAND | last OR penultimate；预测归一化第四维 >=0 停止 |
| 调度器 | DDPM，原 cosine 噪声日程，epsilon prediction |
| 噪声训练日程 / 采样 | 100 / 10，固定不做步数实验 |
| 执行 | 单次选择一个动作，H=1、E=1 |

## 扩散过程

环境时刻 t 和噪声步 k 不同。已有训练通过

\[
a_t^{(k)}=\sqrt{\bar\alpha_k}a_t+\sqrt{1-\bar\alpha_k}\epsilon,
\qquad \mathcal L=\|\epsilon_\theta(a_t^{(k)},k,C_t)-\epsilon\|^2
\]

学习条件动作去噪。当前阶段不计算梯度，加载模型后固定 10 次反向更新。A 记录完整去噪过程，比较不同初始噪声下的动作与 LAND，而不提前截断或更改步数。

中间 noisy sample 不代表可执行动作；可以另记录 scheduler 定义下的干净动作估计，但应保留裁剪等细节。动作多样性可能是有用模式，也可能是误差，必须结合固定任务证据；方差不直接解释成风险概率。

## 条件敏感性

B 保持模型结构与输入布局，改变一项可声明的视觉或语言条件。遮蔽、模糊和信息删除会引入分布偏移，研究的是当前模型依赖；不等于训练得到的单视角/其他条件网络消融。语义等价改写与方向/目标信息删除分开。

## 多候选推理设计

C 对同一条件生成 N=1/4/8 个候选，每个候选固定 10 次去噪。N 不是扩散步数，也不是未来动作长度。一次提取条件，复用在同一观测的候选生成中。

medoid 在归一化前三维距离上选择实际候选，保留该候选四维输出；LAND 投票版本先决定停止，再在继续飞行候选中选代表。固定平票规则，不使用标签、未来轨迹或执行成败排序，不预设多采样一定改善。

## 代码与边界

现有 [动作头](../../src/model_wrapper/diffusion_action_head.py)、[模型加载](../../src/model_wrapper/aerovla_diffusion.py)、[在线 wrapper](../../src/model_wrapper/aerovla_wrapper_ui.py) 保持生产路径。新增生成、条件与选择逻辑分别放在 `evaluation/course/generation/`、`conditions/`、`selection/`，按默认一致性检查接入。

不复用 H=1 权重伪造 H=4/8；不替换 MLP 或改变层数后声称结构实验成立；不把独立停止头/数值历史当作现有能力。原实现细节见[历史设计](../archive/design/diffusion_action_head.md)。

## 来源

[AeroVLA](https://github.com/XuPeng23/AeroVLA)、[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 和 [NavDP](https://github.com/InternRobotics/NavDP) 是方法与实现来源。本次新增贡献是受控推理策略、机制分析、鲁棒性和实证评估，而非声称重新训练了新架构。
