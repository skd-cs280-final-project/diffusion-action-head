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

学习条件动作去噪。当前阶段固定模型参数，加载已有权重后使用原实现进行 10 步去噪和单动作预测，不记录内部去噪过程，不加入候选选择。

## 推理测试内容

对全部 26 个扩散状态使用已有离线入口，对预先指定的五个代表模型使用已有闭环入口。比较现有动作误差、LAND 与导航指标，分析头训练和联合训练模型的表现，以及同一联合训练轨迹上检查点的差异。

现有 [动作头](../../src/model_wrapper/diffusion_action_head.py)、[模型加载](../../src/model_wrapper/aerovla_diffusion.py)、[在线 wrapper](../../src/model_wrapper/aerovla_wrapper_ui.py) 均不修改。不添加去噪 hook、输入变换、采样选择、历史输入或新的动作预测长度。原实现细节见[历史设计](../archive/design/diffusion_action_head.md)。

## 来源

[AeroVLA](https://github.com/XuPeng23/AeroVLA)、[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 和 [NavDP](https://github.com/InternRobotics/NavDP) 是方法与实现来源。课程内容是已有方法说明、标准推理复现与结果分析，不声称提出新网络或推理算法。
