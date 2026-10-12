# AeroVLA 冻结骨干与扩散头适配变更记录

日期：2026-10-10。此记录覆盖 `src/model_wrapper/aerovla_diffusion.py` 与 `tests/test_aerovla_diffusion.py`；训练、数据和 DDPM 头由各自记录及总报告说明。

## 变更 1：以动作扩散头取代 LMHead

- 新增 `AeroVLADiffusionModel`。参考本仓库 `openvla-7b/modeling_prismatic.py` 的多模态前向，依次调用视觉骨干、projector、文本 embedding，将视觉 token 插入 BOS 后，再调用真实 Llama decoder。
- 加载基础权重及可选 AeroVLA LoRA 后执行 `merge_and_unload(safe_merge=True)`，既保留 LoRA 变更，也保留 `modules_to_save=["projector"]` 中已经训练的 projector。
- 移除 `backbone.language_model.lm_head`，直接使用 `get_decoder()` 返回的 decoder；没有语言 logits、token 生成或文字动作解析分支。全序列的最后一层 hidden states 和有效 token mask 作为扩散头条件。
- `backbone.requires_grad_(False)`、`backbone.eval()`、条件编码的 `torch.no_grad()` 同时启用；重写 `train()`，确保模型进入训练模式时骨干仍保持评估模式。优化器入口 `trainable_parameters()` 仅返回扩散头参数。默认骨干可为 BF16，头仍为 FP32。
- 校验右填充、非空文本、图像与文本批次一致性、head/LLM hidden size、一致的四维动作格式和模型上下文长度。图像 mask 与文本 mask 同步插入，拒绝左填充以避免把图像放到 PAD 后。

## 变更 2：独立头检查点

- `save_pretrained()` 只保存 `diffusion_head.pt` 和 `diffusion_config.json`，不复制冻结的 7B 骨干。
- 配置保存头参数、格式版本、基础模型与 LoRA 的本地绝对路径、词表长度、动作维度顺序/范围/降落语义以及训练元数据。
- `from_pretrained()` 自动重建骨干、合并 LoRA 后加载头；支持基础模型/适配器迁移路径覆盖。`load_head_checkpoint()` 使用 `weights_only=True` 与严格 `state_dict` 校验，拒绝配置、词表和动作语义不匹配。
- 临时文件替换正式权重/配置，JSON 序列化在落盘前校验。

## 变更 3：验证、入口兼容与边界修正

- 新增七项 CPU 测试。骨干采用随机初始化的微型真实 `LlamaForCausalLM`，视觉模块为小型替身，无需读取或下载 7B 权重。
- 训练验证检查骨干所有参数均冻结、梯度始终为空、一次优化后所有骨干权重逐位不变，且扩散头至少一个参数实际更新；视觉与 Llama dropout 在训练时保持关闭。
- 直接检查插入的视觉 embedding、BOS 位置、扩展 mask；改变 PAD token 的 id 不影响有效 hidden states 和固定随机种子的采样结果。
- 验证头检查点预测结果逐位一致，并拒绝动作顺序错误、头配置错误和缺失权重。
- 通过真实 PEFT 生成并保存微型 LoRA adapter，再从模型加载入口恢复，验证 LoRA 增量与 adapter 保存的 projector 确实合入冻结骨干。
- 将头导入改为包内相对导入，使 `python src/train_diffusion_head.py` 与包方式调用兼容；对不支持的字典图像输入返回明确错误。额外拒绝空批次、空动作 schema 和布尔词表大小。

执行命令（工作目录 `AeroVLA`）：

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 /data/xiyang/.conda/envs/aero_vla/bin/python -m pytest tests/test_aerovla_diffusion.py -q
env -u PYTHONPATH /data/xiyang/.conda/envs/aero_vla/bin/python src/train_diffusion_head.py --help
```

结果：七项测试全部通过；训练脚本 `--help` 正常退出。测试有 PyTorch TypedStorage 与 pkg_resources 的弃用警告；直接执行脚本还会提示默认 Hugging Face 缓存目录不可写。后者由训练启动脚本把 `HF_HOME` / `HF_MODULES_CACHE` 指到项目内可写目录处理。

本记录中的七项测试验证模型连接、冻结策略、PEFT 保留和序列化正确性；不代表已经使用真实飞行数据完成收敛训练，也不代表评估过飞行成功率。
