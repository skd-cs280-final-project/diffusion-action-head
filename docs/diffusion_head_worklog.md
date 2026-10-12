# 扩散动作头修改记录

## 2026-10-10：独立扩散模块与单元测试

- 参考 `NavDP/baselines/navdp/policy_network.py` 的 `input_embed`、Transformer decoder、`SinusoidalPosEmb`、可学习动作位置编码、因果注意力及 `DDPMScheduler`。
- 新增 `src/model_wrapper/diffusion_action_head.py`：以冻结 VLM 的隐藏特征作为交叉注意力条件，替代离散词表输出路径；输出 `[fwd, down, yaw, land]` 四维归一化动作，默认预测 1 步，可配置动作长度。
- 使用与 NavDP 一致的余弦 beta 调度、epsilon 预测、裁剪后的 DDPM 采样；NavDP 默认训练时间步数为 10，此实现默认训练时间步数为 100、推理步数为 10，两者均可配置并随配置保存。
- 将参数及配置分离为 `DiffusionActionHead` 和 `DiffusionActionHeadConfig`，支持配置字典/JSON 与 `state_dict` 保存恢复。条件投影、动作投影、decoder、位置编码及输出投影均属于可训练扩散头。
- 训练入口 `compute_loss` 对动作随机加噪，只在有效动作坐标计算 epsilon MSE；支持注入固定噪声/时间步供诊断，支持条件 padding mask、动作 padding mask、坐标级监督 mask。动作序列 padding 应位于末尾。
- 采样入口 `sample` 支持指定随机数生成器，输出 float32 归一化动作；动作物理单位转换及 land 阈值判断交由外部适配层。时序 mask/正弦频率注册为 buffer，兼容设备迁移。
- 条件 padding 在投影前清零，可容忍被 mask 的 NaN；无条件 token 时仍保留有效时间步 token。DDPM 运算与 loss 保持 float32，输入隐藏特征自动适配扩散头 dtype。
- 新增 `tests/test_diffusion_action_head.py`，覆盖前向加噪公式、epsilon 目标、有效元素损失分母、条件 mask、因果性、梯度/拟合、采样复现及范围、序列化、dtype 和非法输入。

验证结果：

```bash
HF_HOME=/tmp/aerovla-hf-cache /data/xiyang/.conda/envs/aero_vla/bin/python \
  -m unittest discover -s tests -p test_diffusion_action_head.py -v
```

- 10 项测试全部通过，运行时间 1.375 秒（CPU）。固定批次与固定噪声的 20 次优化中 loss 降至初值的一半以下。
- 额外手工检查整个扩散头转换为 bfloat16 后，CPU 上的 loss、反向传播及采样均得到有限值。
- 当前沙箱内 `torch.cuda.is_available()` 为 False；本次子模块验证仅覆盖 CPU，不将此结果表述为 GPU 或实际飞行效果验证。
- 使用项目现有 `/data/xiyang/.conda/envs/aero_vla` 环境：PyTorch 2.1.2+cu121，diffusers 0.30.3。

## 2026-10-10：整合审查后的调度一致性修正

- 发现默认 100 训练步 / 10 推理步配合 diffusers 的 `leading` 时间间隔，会从训练步 90 开始，而初始化样本是纯高斯噪声。改为 `trailing` 间隔，从训练步 99 开始，使初始化更接近训练终点的噪声分布。NavDP 的 10 / 10 等步数配置本来就覆盖终点，因此不存在同样的跳步差异。
- 阅读本地 diffusers 0.30.3 `scheduling_ddpm.py` 后确认：默认 `previous_timestep` 使用固定 `floor(T / N)`，对于 100 / 30 等不能整除的 trailing 间隔，与真实下一时间步不一致。采样时将生成的时间步列表再次作为显式 `timesteps` 注册，令反向更新系数准确使用下一时间步；最后一步回到 -1 对应的干净样本。
- `compute_loss` 增加公共接口的有效动作 `[-1, 1]` 范围校验，防止绕过数据集时用物理单位误训练扩散头。被 mask 的无效坐标不参与范围校验。
- 新增 100 / 10、100 / 30、100 / 1 的终端与相邻调度测试，并补充越界动作错误测试。
- 使用同一 unittest 命令复验：11 项测试全部通过，CPU 运行时间 0.514 秒。
