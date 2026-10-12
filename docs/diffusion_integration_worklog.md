# 扩散头训练及推理集成改动记录

日期：2026-10-10。仅修改 AeroVLA，NavDP 作为只读参考。原有未跟踪的权重、环境、数据软链接和调研文档保留。

| 批次 | 文件 | 改动及原因 | 验证状态 |
| --- | --- | --- | --- |
| 1 | `requirements.txt` | 增加本机已安装的 `diffusers==0.30.3`，用于与 NavDP 一致的 DDPM 调度器。 | 已确认本地版本。 |
| 2 | `src/train_diffusion_head.py`、`scripts/train_diffusion_head.sh` | 新增单进程训练入口；冻结参数检查、仅头优化器、按样本计数的梯度累积、有限值检查、余弦学习率、JSONL 日志及仅头检查点。缺少训练 JSON 时提前报错；基座默认加载并冻结现有 AeroVLA LoRA。 | 等待模型和数据模块完成后集成验证。 |
| 3 | `src/model_wrapper/aerovla_wrapper_ui.py`、`scripts/eval_aerovla.sh` | 评测自动识别扩散检查点，去噪后反归一化连续动作，LAND 以归一化第四维 >=0 判断；允许环境变量指定检查点和 GPU。原 LoRA 检查点仍使用原路径。 | 等待检查点往返及动作转换验证。 |
| 4 | `tests/test_train_diffusion_head.py` | 新增不等微批大小的梯度累积等价性、余批更新、max_steps、冻结参数、非有限 loss 及现有评测接口动作转换测试。 | 待集成运行。 |
| 5 | `README.md`、`docs/diffusion_action_head_report.md` | 增加新训练入口链接及中文架构、动作规范、运行命令、验证边界和日志索引。 | 全套 pytest 36 项通过；CLI、compileall、bash 语法检查、diff 空白检查通过。GPU 集成待完成。 |
| 6 | `scripts/eval_aerovla.sh`、`README.md`、总报告 | 评测支持 `PYTHON` 选择与可写 HF 缓存；补充最终 instruction 约束，修正文档命令环境并记录真实 GPU 两步训练结果。 | GPU 1 训练 exit 0；36 项测试通过。保存的 tokenizer/processor 均离线重载成功，完整重载推理继续验证。 |
| 7 | 总报告及本记录 | 补入完整 7B 检查点重载与默认 10 步推理结果、精度差异边界、实际 GPU 验证命令及 AirSim 前置条件。 | CPU float32 完整重载推理 exit 0；输出 `[1,1,4]` 有限且归一化范围正确，LMHead 不存在、backbone 仍冻结。最终 shell 语法和 diff 空白检查通过。 |

## 最终验证结论

- 批次 2–4 的待验证项目全部完成：完整 pytest 36/36，覆盖模型 7、头 11、数据 13、训练与推理接口 5。
- 本地 7B + 原 AeroVLA LoRA 在 GPU 1 上以真实图片、人工标签完成两步训练，loss 分别为 1.2759649753570557 和 0.8945425152778625，运行退出码 0。
- 保存了 step 1、step 2 及最终头；头文件 21,088,629 bytes。所有保存的处理器重载成功，随后完整模型 CPU 重载与采样成功。
- 验证中冻结 7,409,644,992 个参数，仅训练 5,265,668 个头参数。真实小型 Llama 测试逐位验证优化前后全部骨干权重不变。
- 缺少正式 `data/aerovla_train_dataset.json`；未执行正式收敛训练或 AirSim 指标评测。人工标签及其临时检查点不代表正式训练产物。

注意：`unittest discover` 仅收集 29 项 unittest 用例；完整验证必须使用 pytest，它会同时收集模型模块的 7 项 pytest 用例。

## 新增真实数据后的就绪核验

- 用户新增 `data/aerovla_train_dataset.json` 与 `data/continuous_split_seed42.json` 后，核对全部 410,450 条标注、轨迹划分及 820,900 个图片路径；schema 通过、图片缺失 0，32 条真实样本处理通过。
- 识别出当前训练入口尚未消费 split、缺少离线动作验证入口；未直接启动会混入验证集的全量训练。
- 新增 `docs/diffusion_data_readiness_report.md`，在总报告顶部增加最新状态提示。本批只改文档，不改训练或评测行为。

后续每批修改及其验证结果继续追加到本文件；最终使用说明和验证汇总位于 `diffusion_action_head_report.md`。
