# Diffusion Policy 与动作扩散头设计

[文档导航](../README.md) · [实验协议](../experiments/protocol.md) · [接口约定](interfaces.md)

本项目研究固定视觉语言条件下的动作分布建模。Diffusion Policy 是条件策略与执行方式；diffusion action head 是负责条件去噪的网络模块。已有代码完成单步条件扩散，课程进一步比较网络、条件注入和动作序列策略。

## 问题与范围

三个研究问题：扩散相对直接回归是否有效；网络结构与条件注入怎样影响动作质量；多步预测与执行反馈怎样影响闭环导航和成本。固定 DDPM、训练噪声步数 100、推理去噪步数 10，不做扩散步数或 DDIM 消融。

视觉语言骨干、原 AeroVLA adapter 和输入信息在主对照中保持一致。状态历史、联合 LoRA、独立停止分类器、两阶段规划器、额外安全控制器不属于本次主实验。

## 当前模型

前视与下视分别缩放至 224×224，再上下拼接交给 processor。文本是当前导航提示加 `Action:`，不附动作答案。冻结骨干输出多模态条件 C，包含有效 token mask。

| 项目 | 当前实现 |
| --- | --- |
| 头 | Transformer Decoder，hidden 256，4 层，8 heads，dropout 0 |
| 条件 | 完整 VLM token 线性投影，通过 cross-attention 读取 |
| 动作 | `[B,1,4]`，fwd/down/yaw/land |
| 时间 | 正弦噪声时间编码，加入 memory |
| 动作位置 | 可学习位置 embedding，当前仅一槽位 |
| 动作 self-attention | 因果 mask，H=1 时无时序差异 |
| 调度 | DDPM，cosine beta 日程，epsilon prediction |
| 损失 | 有效动作坐标上的噪声 MSE |
| 执行 | H=1、E=1 |

前三维分别裁剪到 [0,5]、[-5,5]、[-1.1,1.1]，然后映射到 [-1,1]。LAND 使用 last 或 penultimate 标记，映射为 -1/+1；第四维 >=0 时触发停止。保留此定义以隔离动作头变量。

## 条件动作扩散

环境决策时刻为 t，预测序列长度为 H；噪声时间步为 k，两者不能混淆：

\[
A_t=[a_t,\ldots,a_{t+H-1}],\qquad C_t=f_{VLM}(I_t^{front},I_t^{down},instruction_t).
\]

\[
A_t^{(k)}=\sqrt{\bar\alpha_k}A_t+\sqrt{1-\bar\alpha_k}\epsilon,
\quad \epsilon\sim\mathcal N(0,I).
\]

\[
\mathcal L=\frac{\sum M\odot(\epsilon_\theta(A_t^{(k)},k,C_t)-\epsilon)^2}{\sum M}.
\]

M 为有效坐标 mask。训练随机采样噪声与时间步；推理从高斯动作开始，固定 10 次去噪后解码动作。冻结 VLM 一次提取条件，动作头迭代读取条件。真实多模态动作分布是否学到，需要导航证据；同条件采样有方差本身不是证明。

## A 的设计变量

- 确定性 Transformer：相同条件路径与相近容量，以可学习动作查询直接回归。
- MLP 扩散头：池化条件、带噪动作、时间步，经残差 MLP 预测噪声。
- 池化条件 Transformer：对有效条件 token 做池化，以已登记的调制方法注入，尽量保留主体结构。
- 当前 token cross-attention Transformer：共同参照。

池化与完整 token 的比较研究信息压缩的影响。MLP 与 Transformer 的比较还涉及容量和算力，必须报告实际参数量。训练 loss 不能跨不同学习目标直接排序；共同指标是动作、LAND、闭环及运行成本。

## B 的序列设计

当前数据集只接受 H=1。先审计轨迹中动作与决策时刻的连续性，确认没有重复状态被误当成独立动作。按轨迹划分后建窗口；所有 H 使用相同起始样本，尾部按有效性补齐并截断终止之后的标签。

- H=4/8、E=1：比较联合预测是否改善当前动作，不减少骨干调用次数。
- H=4、E=2：复用 H=4 权重，比较更长开环执行；遇到原停止/终止条件立即结束。
- H=4 的因果/双向 self-attention：整段带噪动作联合去噪，未来位置变量不等于未来观测。

双向注意力必须隔离 padding token；只遮 loss 不能防止污染有效动作。动作仍是各自未来决策时刻的局部控制量，不能直接相加生成世界坐标轨迹。数据审计不通过时，先解决监督问题，不通过改参数伪造多步能力。

## 代码导航与现有边界

| 文件 | 责任 |
| --- | --- |
| [diffusion_action_head.py](../../src/model_wrapper/diffusion_action_head.py) | 当前 Transformer 去噪、损失和采样 |
| [aerovla_diffusion.py](../../src/model_wrapper/aerovla_diffusion.py) | 骨干、冻结、条件编码、检查点 |
| [diffusion_dataset.py](../../src/diffusion_dataset.py) | 当前单步连续标签与 collator |
| [diffusion_split.py](../../src/diffusion_split.py) | 轨迹划分与 provenance |
| [aerovla_wrapper_ui.py](../../src/model_wrapper/aerovla_wrapper_ui.py) | 单步在线动作解码与扩展兼容 |
| [A 工作区](../../src/course_heads/README.md) | 待实现单步变体 |
| [B 工作区](../../src/course_sequence/README.md) | 待实现多步路径 |

头模块可配置 H 不表示外层数据、schema 和执行器已支持 H>1。旧 checkpoint 的四维动作和接口必须保持兼容。详尽的历史状态/联合模型实现依据见[归档设计](../archive/design/diffusion_action_head.md)。

## 参考

[AeroVLA](https://github.com/XuPeng23/AeroVLA) 提供视觉语言导航基础；[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 提供条件动作扩散和滚动执行思路；[NavDP](https://github.com/InternRobotics/NavDP) 为当前 Transformer 动作去噪实现提供参考。课程重点是适配与受控实验，不将这些通用方法声明为原创算法。
