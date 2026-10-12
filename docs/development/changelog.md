# AeroVLA 开发与文档变更日志

[返回文档首页](../README.md)

> **追溯记录。** 原 10 份工作日志按模块合并。各节的日期、测试数量、待办与机器状态均对应当时；36、66、87 等测试数不能相加。人工标签检查、真实数据两步检查和正式训练是不同实验。当前运行方式见[指南](../guides/training_and_evaluation.md)。

## 按主题查阅

[2026-10-12 文档整理与旧路径迁移](#docs-20261012)

- [扩散去噪网络与调度器](#head)
- [骨干冻结、模型连接与检查点](#model)
- [连续动作与图像文本数据管线](#dataset)
- [首次训练与推理集成](#integration)
- [轨迹划分与数据隔离](#split)
- [训练、验证与真实数据流程检查](#train-eval)
- [离线动作评估](#eval)
- [闭环启动与路径保存修复](#closed-loop)
- [监督标签审计](#supervision)
- [训练监控与产物导出](#monitor)

<a id="docs-20261012"></a>

## 2026-10-12：文档合并与分类整理

- 原有 23 篇 Markdown 整理为 12 篇（含新首页），减少 11 个文件；`docs/` 根目录仅保留 `README.md`。
- 10 份开发记录合成本文；数据就绪报告并入架构附录；训练曲线更新并入全量训练记录；NYC 运行和基线对比并为一篇，分别保留运行与对比章节。
- 新增按任务查阅的首页和每篇返回入口，统一文档、代码、图表与运行产物的相对链接。旧文件与其合并位置一一对应见下表，不额外保留跳转文件。
- 明确固定实验快照与待实现研究方案；修正运行指南中的过时状态提示。历史正文保留原指标、配置和验证记录；文中路径引用统一为整理后的地址。
- 本次只整理 Markdown，未修改运行代码、模型权重、原始实验数据及 `docs/reports/` 中图表、脚本、快照。

### 原路径与新位置

路径相对 `docs/`。保留此表便于按旧文件名检索。

| 原文件 | 新位置 |
| --- | --- |
| `diffusion_training_evaluation_guide.md` | [guides/training_and_evaluation.md](../guides/training_and_evaluation.md) |
| `assets/troubleshooting.md` | [guides/troubleshooting.md](../guides/troubleshooting.md) |
| `diffusion_action_head_report.md` | [design/diffusion_action_head.md](../design/diffusion_action_head.md) |
| `diffusion_data_readiness_report.md` | [design/diffusion_action_head.md#data-readiness](../design/diffusion_action_head.md#data-readiness) |
| `diffusion_full_training_20261010.md` | [experiments/full_training_20261010.md#launch](../experiments/full_training_20261010.md#launch) |
| `diffusion_training_trend_updates.md` | [experiments/full_training_20261010.md#epoch3](../experiments/full_training_20261010.md#epoch3) |
| `weekly_report_20261011.md` | [experiments/weekly_report_20261011.md](../experiments/weekly_report_20261011.md) |
| `diffusion_nyc10_gpu2_20261011.md` | [experiments/nyc10_evaluation_20261011.md#run](../experiments/nyc10_evaluation_20261011.md#run) |
| `diffusion_nyc10_baseline_comparison_20261011.md` | [experiments/nyc10_evaluation_20261011.md#comparison](../experiments/nyc10_evaluation_20261011.md#comparison) |
| `diffusion_collision_mitigation_plan_20261012.md` | [research/collision_mitigation_20261012.md](../research/collision_mitigation_20261012.md) |
| `hierarchical_planner_action_expert_proposal_20261012.md` | [research/hierarchical_planner_20261012.md](../research/hierarchical_planner_20261012.md) |
| `history_pose_literature_review_20261012.md` | [research/history_pose_20261012.md](../research/history_pose_20261012.md) |
| `github_snapshot_20261012.md` | [development/github_snapshot_20261012.md](github_snapshot_20261012.md) |
| `diffusion_head_worklog.md` | [development/changelog.md#head](changelog.md#head) |
| `diffusion_model_worklog.md` | [development/changelog.md#model](changelog.md#model) |
| `diffusion_dataset_worklog.md` | [development/changelog.md#dataset](changelog.md#dataset) |
| `diffusion_integration_worklog.md` | [development/changelog.md#integration](changelog.md#integration) |
| `diffusion_split_worklog.md` | [development/changelog.md#split](changelog.md#split) |
| `diffusion_train_eval_worklog.md` | [development/changelog.md#train-eval](changelog.md#train-eval) |
| `diffusion_eval_worklog.md` | [development/changelog.md#eval](changelog.md#eval) |
| `diffusion_closed_loop_worklog.md` | [development/changelog.md#closed-loop](changelog.md#closed-loop) |
| `diffusion_supervision_worklog.md` | [development/changelog.md#supervision](changelog.md#supervision) |
| `diffusion_monitor_worklog.md` | [development/changelog.md#monitor](changelog.md#monitor) |

### 本次核验

- 原 23 篇文档均在迁移表中一一对应；整理后共 12 篇，顶层只有导航首页，首页覆盖其余全部 11 篇。
- 已核验 223 处本地 Markdown 链接、68 处 Markdown 章节锚点及 README 的 HTML 图片引用，无缺失目标；这不包含外部论文网站的联网可用性检查。
- 对照整理前原文：37 个代码块保留，443 行原有表格内容保留（链接及文档路径按新位置归一化比较）；合并时检查全部 23 篇正文已纳入新文件。随后仅调整运行指南的过时状态提示及终止原因的章节引用。
- 所有新文档的代码围栏配对、单一一级标题和行尾空白检查通过；`git diff --check` 通过。
- `src/`、`scripts/`、`tests/`、`docs/reports/` 和原展示资源无差异；本次未运行训练/推理测试，也未重算已有实验指标。
- 独立复审确认迁移清单、合并章节、历史结果与待实现方案的标识完整。

### 后续 GitHub 更新（2026-10-12）

用户随后要求上传本次更新，发布目标为个人仓库 [xiyang-666/AeroVLA](https://github.com/xiyang-666/AeroVLA) 的 `main` 分支，使用 `github` 远程。

本次文档提交包含目录合并、导航与链接修复，以及此前新增的历史轨迹/位姿文献调研。与首次代码快照分开提交；此次未重新运行训练或仿真，也未纳入模型权重、训练数据或大型运行产物。提交前复核文档链接、代码块与表格保留情况，并检查暂存范围和空白格式。

<a id="head"></a>

## 扩散去噪网络与调度器

### 2026-10-10：独立扩散模块与单元测试

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

### 2026-10-10：整合审查后的调度一致性修正

- 发现默认 100 训练步 / 10 推理步配合 diffusers 的 `leading` 时间间隔，会从训练步 90 开始，而初始化样本是纯高斯噪声。改为 `trailing` 间隔，从训练步 99 开始，使初始化更接近训练终点的噪声分布。NavDP 的 10 / 10 等步数配置本来就覆盖终点，因此不存在同样的跳步差异。
- 阅读本地 diffusers 0.30.3 `scheduling_ddpm.py` 后确认：默认 `previous_timestep` 使用固定 `floor(T / N)`，对于 100 / 30 等不能整除的 trailing 间隔，与真实下一时间步不一致。采样时将生成的时间步列表再次作为显式 `timesteps` 注册，令反向更新系数准确使用下一时间步；最后一步回到 -1 对应的干净样本。
- `compute_loss` 增加公共接口的有效动作 `[-1, 1]` 范围校验，防止绕过数据集时用物理单位误训练扩散头。被 mask 的无效坐标不参与范围校验。
- 新增 100 / 10、100 / 30、100 / 1 的终端与相邻调度测试，并补充越界动作错误测试。
- 使用同一 unittest 命令复验：11 项测试全部通过，CPU 运行时间 0.514 秒。

<a id="model"></a>

## 骨干冻结、模型连接与检查点

日期：2026-10-10。此记录覆盖 `src/model_wrapper/aerovla_diffusion.py` 与 `tests/test_aerovla_diffusion.py`；训练、数据和 DDPM 头由各自记录及总报告说明。

### 变更 1：以动作扩散头取代 LMHead

- 新增 `AeroVLADiffusionModel`。参考本仓库 `openvla-7b/modeling_prismatic.py` 的多模态前向，依次调用视觉骨干、projector、文本 embedding，将视觉 token 插入 BOS 后，再调用真实 Llama decoder。
- 加载基础权重及可选 AeroVLA LoRA 后执行 `merge_and_unload(safe_merge=True)`，既保留 LoRA 变更，也保留 `modules_to_save=["projector"]` 中已经训练的 projector。
- 移除 `backbone.language_model.lm_head`，直接使用 `get_decoder()` 返回的 decoder；没有语言 logits、token 生成或文字动作解析分支。全序列的最后一层 hidden states 和有效 token mask 作为扩散头条件。
- `backbone.requires_grad_(False)`、`backbone.eval()`、条件编码的 `torch.no_grad()` 同时启用；重写 `train()`，确保模型进入训练模式时骨干仍保持评估模式。优化器入口 `trainable_parameters()` 仅返回扩散头参数。默认骨干可为 BF16，头仍为 FP32。
- 校验右填充、非空文本、图像与文本批次一致性、head/LLM hidden size、一致的四维动作格式和模型上下文长度。图像 mask 与文本 mask 同步插入，拒绝左填充以避免把图像放到 PAD 后。

### 变更 2：独立头检查点

- `save_pretrained()` 只保存 `diffusion_head.pt` 和 `diffusion_config.json`，不复制冻结的 7B 骨干。
- 配置保存头参数、格式版本、基础模型与 LoRA 的本地绝对路径、词表长度、动作维度顺序/范围/降落语义以及训练元数据。
- `from_pretrained()` 自动重建骨干、合并 LoRA 后加载头；支持基础模型/适配器迁移路径覆盖。`load_head_checkpoint()` 使用 `weights_only=True` 与严格 `state_dict` 校验，拒绝配置、词表和动作语义不匹配。
- 临时文件替换正式权重/配置，JSON 序列化在落盘前校验。

### 变更 3：验证、入口兼容与边界修正

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

<a id="dataset"></a>

## 连续动作与图像文本数据管线

### 2026-10-10：建立连续动作数据接口

- 新增 `src/diffusion_dataset.py`，复用 `AeroVLADataset._get_images` 的双视角拼图，保持前视图在上、下视图在下，每幅图先缩放到 224×224。
- 标签顺序为 `[fwd, down, yaw, land]`。前三维沿用原范围 `[0,5]`、`[-5,5]`、`[-1.1,1.1]`，裁剪后线性映射至 `[-1,1]`，取消原 99 桶量化。
- `land = is_last_step OR is_penultimate`，继续飞行为 -1，降落为 +1，保留原 `LAND` token 的语义。公开归一化、反归一化函数及范围常量，供推理与检查点元数据共用。
- 模型输入仅包含 `instruction.strip() + "\nAction: "`；不加入动作数值、`LAND`、目标 EOS 或语言建模标签。检查 tokenizer 没有自动添加 EOS。
- 新增 `DiffusionDataCollator`，输出 `pixel_values`、`input_ids`、`attention_mask`、`actions`。动作形状为 `[B,1,4]`；图片兼容当前的通道拼接 tensor 和字典格式视觉输入。
- 对超长 prompt、非有限动作、非法 LAND 标记、错误维度明确报错。禁止静默截断，以免丢失导航指令或 `Action:` 后缀。
- 当前训练 JSON 只明确描述单帧动作，默认且仅支持 `action_horizon=1`；未将相邻 JSON 记录擅自解释为连续轨迹。
- 未改动旧数据集或旧 LM/LoRA 训练入口。

### 本地数据条件

- `dataset_raw` 链接至 `/data/xiyang/dataset/OpenUAV/TravelUAV`，有真实 frontcamera/downcamera 图像。
- `data/aerovla_train_dataset.json` 缺失；在 `/data/xiyang/dataset` 和 `/data/xiyang/project_orig` 中未发现同名训练标注。现有 `data/uav_dataset` 是评估拆分，不能替代监督训练标签。
- 本地 tokenizer 配置为 `add_bos_token=true`、`add_eos_token=false`、右侧 padding；适合仅 prompt 的扩散训练输入。

### 验证记录

#### 第二次修改：补充数据合同测试

- 新增 `tests/test_diffusion_dataset.py`，共 13 项测试。
- 覆盖范围：归一化端点/中点/反变换与输入不变性、范围裁剪、非法维度和 NaN/Inf、双视角拼图位置、完整 LAND 规则、修改动作标签不改变模型输入、单步 horizon 约束、标注格式错误、缺图报错、左右 padding mask、动作维度与 dtype、字典图像输入、超长 prompt 报错、自动 EOS/空 batch/非法目标拒绝。
- 执行命令：`/data/xiyang/.conda/envs/aero_vla/bin/python -m unittest discover -s tests -p 'test_diffusion_dataset.py' -v`。
- 结果：13/13 通过，测试主体耗时约 0.12 秒。

#### 本地真实处理器与图片验证

- 使用本地 `openvla-7b` 的 `AutoTokenizer`、`AutoProcessor`，开启 `local_files_only=True` 和离线环境变量，未下载任何模型或数据。
- 输入 TravelUAV 真实样本图片 `Japanese_Street/28725a09-4eea-4708-89bc-c7402931d4ba/{frontcamera,downcamera}/000000.png`。
- 用临时 JSON 提供人工构造标签 `[2.5,0,0,0]`，仅用于接口 smoke 验证，**不是真实监督训练样本，也没有据此报告真实训练效果**。
- 输出形状：`pixel_values=[1,6,224,224]`、`input_ids=[1,11]`、`attention_mask=[1,11]`、`actions=[1,1,4]`；归一化标签为 `[0,0,0,-1]`。
- 解码输入为 `<s> Fly forward toward the building.\nAction: `；确认无 EOS 和目标 LAND，符合冻结 VLM 提取条件特征的要求。

<a id="integration"></a>

## 首次训练与推理集成

日期：2026-10-10。仅修改 AeroVLA，NavDP 作为只读参考。原有未跟踪的权重、环境、数据软链接和调研文档保留。

| 批次 | 文件 | 改动及原因 | 验证状态 |
| --- | --- | --- | --- |
| 1 | `requirements.txt` | 增加本机已安装的 `diffusers==0.30.3`，用于与 NavDP 一致的 DDPM 调度器。 | 已确认本地版本。 |
| 2 | `src/train_diffusion_head.py`、`scripts/train_diffusion_head.sh` | 新增单进程训练入口；冻结参数检查、仅头优化器、按样本计数的梯度累积、有限值检查、余弦学习率、JSONL 日志及仅头检查点。缺少训练 JSON 时提前报错；基座默认加载并冻结现有 AeroVLA LoRA。 | 等待模型和数据模块完成后集成验证。 |
| 3 | `src/model_wrapper/aerovla_wrapper_ui.py`、`scripts/eval_aerovla.sh` | 评测自动识别扩散检查点，去噪后反归一化连续动作，LAND 以归一化第四维 >=0 判断；允许环境变量指定检查点和 GPU。原 LoRA 检查点仍使用原路径。 | 等待检查点往返及动作转换验证。 |
| 4 | `tests/test_train_diffusion_head.py` | 新增不等微批大小的梯度累积等价性、余批更新、max_steps、冻结参数、非有限 loss 及现有评测接口动作转换测试。 | 待集成运行。 |
| 5 | `README.md`、`docs/design/diffusion_action_head.md` | 增加新训练入口链接及中文架构、动作规范、运行命令、验证边界和日志索引。 | 全套 pytest 36 项通过；CLI、compileall、bash 语法检查、diff 空白检查通过。GPU 集成待完成。 |
| 6 | `scripts/eval_aerovla.sh`、`README.md`、总报告 | 评测支持 `PYTHON` 选择与可写 HF 缓存；补充最终 instruction 约束，修正文档命令环境并记录真实 GPU 两步训练结果。 | GPU 1 训练 exit 0；36 项测试通过。保存的 tokenizer/processor 均离线重载成功，完整重载推理继续验证。 |
| 7 | 总报告及本记录 | 补入完整 7B 检查点重载与默认 10 步推理结果、精度差异边界、实际 GPU 验证命令及 AirSim 前置条件。 | CPU float32 完整重载推理 exit 0；输出 `[1,1,4]` 有限且归一化范围正确，LMHead 不存在、backbone 仍冻结。最终 shell 语法和 diff 空白检查通过。 |

### 最终验证结论

- 批次 2–4 的待验证项目全部完成：完整 pytest 36/36，覆盖模型 7、头 11、数据 13、训练与推理接口 5。
- 本地 7B + 原 AeroVLA LoRA 在 GPU 1 上以真实图片、人工标签完成两步训练，loss 分别为 1.2759649753570557 和 0.8945425152778625，运行退出码 0。
- 保存了 step 1、step 2 及最终头；头文件 21,088,629 bytes。所有保存的处理器重载成功，随后完整模型 CPU 重载与采样成功。
- 验证中冻结 7,409,644,992 个参数，仅训练 5,265,668 个头参数。真实小型 Llama 测试逐位验证优化前后全部骨干权重不变。
- 缺少正式 `data/aerovla_train_dataset.json`；未执行正式收敛训练或 AirSim 指标评测。人工标签及其临时检查点不代表正式训练产物。

注意：`unittest discover` 仅收集 29 项 unittest 用例；完整验证必须使用 pytest，它会同时收集模型模块的 7 项 pytest 用例。

### 新增真实数据后的就绪核验

- 用户新增 `data/aerovla_train_dataset.json` 与 `data/continuous_split_seed42.json` 后，核对全部 410,450 条标注、轨迹划分及 820,900 个图片路径；schema 通过、图片缺失 0，32 条真实样本处理通过。
- 识别出当前训练入口尚未消费 split、缺少离线动作验证入口；未直接启动会混入验证集的全量训练。
- 新增 `docs/design/diffusion_action_head.md#data-readiness`，在总报告顶部增加最新状态提示。本批只改文档，不改训练或评测行为。

后续每批修改及其验证结果继续追加到本文件；最终使用说明和验证汇总位于 `docs/design/diffusion_action_head.md`。

<a id="split"></a>

## 轨迹划分与数据隔离

日期：2026-10-10。

### 修改 1：接入严格轨迹划分

- 新增 `src/diffusion_split.py`，按 `version=1`、`split_by=trajectory` 读取用户提供的划分。
- 要求 train、val 均为非空轨迹列表；拒绝重复轨迹、轨迹交集、缺失或额外轨迹、重复帧，以及可能造成同轨迹别名的非规范相对路径。
- 验证全集轨迹 SHA256：排序轨迹 ID，以换行连接，不加末尾换行，再按 UTF-8 计算 SHA256。
- 修改 `src/diffusion_dataset.py`，新增 `split_json`、`split` 参数，在任何抽样前校验完整标注，并按原 JSON 顺序选择 train/val。旧的不带划分调用保持兼容；仅给 split 而不给文件会报错。
- `dataset.split_info` 保存两个分区的帧数与轨迹数、全集及各分区轨迹摘要、所选原始索引摘要、划分文件摘要、数据 JSON 摘要及绝对路径，可写入训练检查点和评估结果核对来源。数据 JSON 摘要为文件字节 SHA256，标签变化也会使其变化。

### 修改 2：可复现的小规模抽样

- 新增 `deterministic_subset(dataset, max_samples, seed)`，使用独立 Python RNG 随机抽帧并按原顺序排列，避免只取数据开头导致地图/轨迹偏置。
- 仅 `max_samples=None` 表示不限制；0、负数及非整数会报错。超出数据规模时返回完整数据集。
- 抽样返回保留 `split_info` 的 PyTorch `Subset`，记录随机种子、抽样数量、抽样前帧数及选中索引 SHA256，不修改原数据集元信息。
- 抽帧仅供诊断或限制运行规模，先执行完整划分校验；不会改变训练/验证的轨迹边界。

### 修改 3：验证与测试

- 新增 `tests/test_diffusion_split.py`，覆盖训练/验证隔离、完整覆盖、元信息摘要、重新划分检测、非法 schema、交叉/重复轨迹、未选中分区中的重复帧、路径别名、参数配对以及抽样可复现性与 RNG 隔离。
- 执行 `/data/xiyang/.conda/envs/aero_vla/bin/python -m unittest discover -s tests -p 'test_diffusion_split.py' -v`：新增 10 项全部通过。
- 执行同环境下 `test_diffusion_dataset.py`：原有 13 项全部通过，图像拼接、动作归一化及 prompt-only 输入行为保持通过。
- 实际加载全部 410,450 条真实标注并分别构造 train/val 数据集：全部严格校验通过；train 为 369,274 帧 / 6,850 条轨迹，val 为 41,176 帧 / 761 条轨迹。
- 固定 seed=42 分别抽取 8 帧，均仅来自对应分区；完整分区校验发生在抽样前。这里验证的是元信息与划分，未在此步骤解码图片或执行模型。

真实文件摘要：

| 对象 | SHA256 |
| --- | --- |
| `continuous_split_seed42.json` 文件字节 | `50c4c84f90058bca2cc14ab19bbbccd41c0605e05bd924f530e743ad99236c4b` |
| `aerovla_train_dataset.json` 文件字节 | `1c1e32787cedb096fe5df2511e5bf3cc81901b8a30d261df73ebcb800c868a67` |
| 全集轨迹 ID | `831856ffb9a8d626f242c9d64a8177a0c16302912b44fbbdef5fd6c91975ce75` |

<a id="train-eval"></a>

## 训练、验证与真实数据流程检查

本记录承接 `docs/design/diffusion_action_head.md` 与 `docs/design/diffusion_action_head.md#data-readiness`，日期 2026-10-10。原数据、基座和 adapter 权重保留。

| 批次 | 文件 | 改动 | 验证 |
| --- | --- | --- | --- |
| 1 | `src/train_diffusion_head.py` | 默认要求轨迹 split，训练仅使用 train、验证仅使用 val；固定种子随机子集；新增不加载 7B 的 `--check-data`；epoch/最终步/可选周期验证，按验证 noise_mse 保存 best；保存划分与验证元数据。 | 模块并行实现完成后统一测试。 |
| 2 | `tests/test_train_diffusion_head.py` | 测试验证回调仅在 optimizer 更新后触发；重合的 epoch/周期/最终步不重复；验证后恢复头训练状态且骨干保持 eval。 | 等待统一测试。 |
| 3 | `scripts/train_diffusion_head.sh`、`scripts/smoke_diffusion_pipeline.sh` | 配置可覆盖的 CPU 线程数；新增真实标注串行流程检查：train4/val8、两步更新、独立评估重载、同种子指标一致性与数据隔离断言。 | 待运行，检查点明确仅为流程验证。 |
| 4 | `README.md`、两份前序报告、新运行指南 | 加入当前训练/离线评估/闭环配置命令，区分历史验证与现状，解释指标含义和检查点边界。 | 真实数据 `--check-data` 通过；首轮 58 项测试通过；GPU 短程流程运行中。 |
| 5 | `src/train_diffusion_head.py` | 独立审查后将 `--check-data` 的形状记录改为递归，兼容数据处理器的字典型图像输出。 | 当前本地 OpenVLA tensor 图片路径已通过，后续执行完整回归。 |
| 6 | smoke 脚本和运行指南 | 流程核验从实际检查点读取并断言 train/val 帧数及 optimizer 步数，避免仅按常量输出报告；更新真实两步训练、独立重载指标与完整测试结果。 | 真实 7B GPU 流程 exit 0；重载指标一致；实际检查点计数及 split 副本校验通过；全套 66 项通过。 |

### 最终验证记录

- `scripts/train_diffusion_head.sh --check-data --max-samples 4 --max-val-samples 8 --num-workers 0`：成功验证 train/val 完整划分和实际预处理，模型未加载。
- `PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python CUDA_VISIBLE_DEVICES=1 bash scripts/smoke_diffusion_pipeline.sh`：沙箱内明确检测 CUDA 不可用后，经命令权限批准在物理 GPU 1 完成。未占用 GPU 0 已有任务，未操作模拟器。
- GPU 真实训练更新两次，loss=1.2125136852264404、1.2360974550247192；验证 noise_mse=1.50709730386734、归一化前三维 MSE=0.7305838082736672。独立进程重载最终检查点所得指标相同。
- 仅训练 5,265,668 个扩散头参数，冻结 7,409,644,992 个上游参数。实际使用 4 条 train / 8 条 val 真实标注，未跨越轨迹划分。
- 已核对最终/best/step1/step2 检查点、元数据计数、SHA256、复制后的 `data_split.json` 与输入文件字节一致。
- `python -m pytest tests -q` 最终 **66 passed**（6.27 秒）；Python 编译、四个 shell 语法、diff 空白检查通过。
- 用户的原始数据、基座权重、AeroVLA adapter、已有模拟器任务均未修改。未自行提交 git commit。尚未进行完整收敛训练或真实闭环指标评测。

数据划分、离线评估、闭环启动脚本的每次改动，分别记于 `docs/development/changelog.md#split`、`docs/development/changelog.md#eval`、`docs/development/changelog.md#closed-loop`。

<a id="eval"></a>

## 离线动作评估

日期：2026-10-10。

### 记录 1：离线评估实现

- 新增 `src/eval_diffusion_head.py`，公开 `evaluate_diffusion`，供训练循环复用；独立 CLI 从检查点恢复 tokenizer、processor、冻结骨干及动作扩散头。
- 每批只编码一次图像和导航指令，共用条件计算 epsilon MSE 与 DDPM 动作采样；目标动作不进入条件编码器。
- 噪声、时间步、采样分别使用独立固定种子生成器；保存并恢复 torch 全局 RNG、DataLoader/sampler 的显式 generator 和每个模块的原始 train/eval 模式；评估全程无梯度。
- 按有效坐标累计噪声误差和前三维归一化 MSE；fwd/down/yaw 分别报告 MAE、RMSE 和坐标数，采用训练时相同范围裁剪，不混合距离与角度单位。LAND 报告混淆矩阵、precision/recall/F1/accuracy、正负样本数，无分母时 precision/recall/F1 为 0。
- CLI 固定使用 val 分割，默认完整验证集；可固定种子抽样用于接口验证。检查训练元数据的数据/划分哈希；旧检查点缺少分割证据时明确标为 `unknown`，不声称独立留出验证。
- 新增 `scripts/eval_diffusion_head.sh`，提供与训练入口一致的本地缓存设置。

相同模型、样本顺序、batch size、设备和精度下重复评估可复现。更换 batch size 或设备时 DDPM 的随机数布局及浮点运算可能变化，不承诺逐位一致。指标只表示离线动作预测效果，不能替代 AirSim 闭环导航指标。

### 记录 2：回归检验与输出兼容

- 增加 `--output-json`（`--output` 的别名），便于自动化训练→重载→评估脚本调用；报告将数值集中保存在 `metrics`，不混入耗时字段。
- `tests/test_eval_diffusion_head.py` 使用手算三帧数据核验全维噪声均值、前三维 MSE、逐物理维误差、LAND 混淆矩阵和最后单帧批次。
- 补充缺失坐标 mask、范围裁剪、零正样本 LAND、评估异常恢复、真实微型 DDPM 确定性、参数/梯度不变和训练 RNG/Loader generator 不变测试。
- 补充旧检查点未知来源、错误 train/val 身份、数据/划分 SHA 不匹配拒绝测试；CLI 测试验证从检查点恢复处理器、输出 JSON，并在划分冲突时提前拒绝加载大模型。
- 首轮 7 项测试全部通过（2.65 秒）；shell `bash -n` 和 CLI `--help` 通过。最终增加 CLI 集成回归后 **8 项全部通过**（3.35 秒）。验证命令：`OMP_NUM_THREADS=2 HF_HOME=/data/xiyang/project_orig/AeroVLA/.cache/huggingface /data/xiyang/.conda/envs/aero_vla/bin/python -m pytest tests/test_eval_diffusion_head.py -q`。

### 使用命令

```bash
cd /data/xiyang/project_orig/AeroVLA
PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python \
CUDA_VISIBLE_DEVICES=1 bash scripts/eval_diffusion_head.sh \
  --checkpoint ./checkpoints/aero_vla_diffusion \
  --data-json ./data/aerovla_train_dataset.json \
  --split-json ./data/continuous_split_seed42.json \
  --batch-size 2 --num-workers 4 \
  --output ./checkpoints/aero_vla_diffusion/eval_val.json
```

省略 `--max-samples` 会完整评估 val；短程接口验证可加 `--max-samples 8 --num-workers 0`。实际运行模型需要可访问的 GPU，或显式 `--device cpu --backbone-dtype float32`。完整 7B、正式标签和保存检查点的联合验证由主训练集成记录汇总。

<a id="closed-loop"></a>

## 闭环启动与路径保存修复

日期：2026-10-10。

### 修改 1：拆分模型 GPU 和模拟器连接配置

- `scripts/eval_aerovla.sh` 从脚本位置定位并切换 AeroVLA 项目根目录，可从其他目录启动，所有路径参数采用 Bash 数组传递，支持含空格路径。显式传递 `--project_prefix`，使日志根目录与项目一致。
- 保留默认模型 `checkpoints/aero_vla`、默认任务 `seen_valset/NYCEnvironmentMegapa`、batch 1、最多 200 步。
- `GPU_ID` 仅控制模型进程的 `CUDA_VISIBLE_DEVICES`；`SIMULATOR_GPU_ID` 对应传给服务的物理 GPU 编号 `--gpu_id`，默认继承 `GPU_ID`。根据 `src/vlnce_src/env_uav.py`，该编号会写入发送给模拟器服务的 `machines_info[*]['gpus']`。
- `SIMULATOR_PORT` 独立指定模拟器服务端口；默认仍为 `30000 + SIMULATOR_GPU_ID * 5000`。更改模型 GPU 时如需复用某个已准备好的服务，必须显式设置模拟器 GPU/端口。
- 原固定 `DDP_MASTER_PORT=80005` 超出 TCP 端口范围，改为 `src/common/param.py` 中的合法默认 `20001`，允许环境变量覆盖。两个端口都在启动 Python 前校验为 1–65535。
- `MODEL_DIR`、`MODEL_BASE`、`TASK_ID`、`PYTHON`、`TEST_JSON`、`SAVE_DIR`、`DATASET_PATH`、`BATCH_SIZE`、`MAX_WAYPOINTS`、`MAP_SPAWN_AREA_JSON_PATH`、`OBJECT_NAME_JSON_PATH` 均支持环境变量覆盖。
- 新增 `--dry-run`，仅打印配置与完整 shell 转义命令，不加载模型、不访问模拟器。`--` 后的其他参数转发到原评估 Python 入口。

示例（先检查参数，不会启动导航）：

```bash
PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python \
MODEL_DIR=./checkpoints/aero_vla_diffusion \
TASK_ID=seen_valset/NYCEnvironmentMegapa \
GPU_ID=1 SIMULATOR_GPU_ID=0 SIMULATOR_PORT=30000 \
bash /data/xiyang/project_orig/AeroVLA/scripts/eval_aerovla.sh --dry-run
```

该示例仅说明 GPU/端口如何分开配置，不表示端口 30000 的现有模拟器可被当前实验占用。真正闭环评估还需要准备或确认可用服务、地图资源、episode JSON 及训练好的检查点；移除 `--dry-run` 后原入口才会连接服务并执行导航。`continuous_split_seed42.json` 属于逐帧离线验证划分，不是该入口使用的 episode JSON。

### 修改 2：启动边界回归测试

新增 `tests/test_eval_launcher.py`，用 stub Python 接收参数，不导入模型或模拟器。检查项目路径和 Python 路径包含空格、从项目外启动、默认值、模型与模拟器 GPU 解耦、旧 GPU→端口默认规则、各环境参数、参数转发、dry-run 不执行 Python、非法端口及任务格式拒绝。

### 验证结果

- `bash -n scripts/eval_aerovla.sh` 通过。
- `/data/xiyang/.conda/envs/aero_vla/bin/python -m pytest tests/test_eval_launcher.py -q`：**10 项通过**，耗时 0.59 秒。
- 从 `/tmp` 执行上述 `--dry-run` 示例通过，输出模型 GPU 1、模拟器 GPU 0/端口 30000、DDP 端口 20001，项目资源路径均定位到 AeroVLA，模型相对路径将在项目根目录解析。

本次闭环范围只验证启动命令；此次训练/离线评估的运行结果由总报告记录，未声称已完成真实 AirSim 导航或获得 SR/SPL/NE 闭环指标，也未连接、重启现有模拟器。

### 修改 3：保留轨迹目录的绝对路径并修正加载数量日志

日期：2026-10-11。适用任务：GPU 2 使用 `checkpoint-70000` 评估 NYCEnvironmentMegapa 的随机 10 条独立 episode。

- 问题：`src/vlnce_src/env_uav.py` 的 `load_my_datasets()` 先用 `merged_json.strip('/').split('/')` 提取地图和 UUID，再将各段拼回轨迹目录。输入绝对 `dataset_path` 时，重建目录丢失开头的 `/`，导致 `trajectory_dir` 错误地变成相对路径；episode 保存阶段复制 `object_description.json` 时可能出现 `FileNotFoundError`，保存的 `ori_info.json` 也会记录错误路径。
- 修复：仅将轨迹目录的构造改为 `str(Path(merged_json).parent)`，保留绝对/相对路径的语义。地图、UUID、轨迹、指令、目标及对象字段保持原逻辑。
- 日志修复：`Loaded dataset` 由输出评估 JSON 路径的字符串长度改为输出实际加载的 episode 数量 `len(load_data)`。
- 验证：使用 `aero_vla` 环境执行 CPU 检查，从 AST 提取真实 `load_my_datasets()` 和 `find_closest_area()`，仅以简单值对象替代 AirSim 的 Pose/Vector/Quaternion 构造；未导入或启动模型、模拟器和 GPU。读取本次 `eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/selected_episodes.json`，分别以绝对根目录、`dataset_raw`、`./dataset_raw` 加载，每种均得到 10 条 episode。30 个路径案例的 merged/mark/object JSON 均可读，并实际执行 `shutil.copy2` 复制对象描述到临时目录成功。
- 对照检查：修复前后除 `trajectory_dir` 外所有加载字段相同；随机数状态保持不变；三种根目录解析得到相同的真实轨迹路径；加载数量日志使用实际列表长度。代码语法检查通过。
- 此次变更不修改动作、控制逻辑、模型输入、停止阈值或 `trace_eval.py`。仿真重启及闭环执行结果由本次独立评估报告记录。

<a id="supervision"></a>

## 监督标签审计

日期：2026-10-10。对应运行：`checkpoints/aero_vla_diffusion_full_20261010_200045/`。

用户要求整理训练过程中的监督数据，方便检查。本次提供全量静态标签审计和可追溯的固定抽查样本；训练运行状态与模型监控由全量训练报告单独记录。

### 逐次变更

1. 新增 `scripts/audit_diffusion_supervision.py`。读取原始数据和轨迹 split，复用训练的 split 校验及 float32 `normalize_actions`。验证全部监督标签、指令和 LAND 布尔标志；缺失、非有限、float32 溢出等情况明确写入 `audit_errors.json` 并失败，不跳过坏记录。
2. 增加来源核验：数据 JSON、split JSON、全轨迹集合、各 split 轨迹集合、原始清单选中索引 SHA256，以及帧数/轨迹数，必须与当前 `run_config.json` 的 train/val 记录匹配。保存脚本、归一化实现、动作边界定义和 split 校验代码的 SHA256，便于追溯。
3. 增加全量 train/val 的 raw、clipped、normalized 四轴范围/均值/总体标准差和逐轴裁剪计数/比例；LAND 正负、last、penultimate、两标志同时为真的数量分别统计，避免重复计算 LAND 阳性。输出完整地图和逐轨迹分布。
4. 增加每 split/map 固定代表样本。选择原始顺序下的 LAND 正负、last、penultimate、两者同时、任意裁剪和各轴上下越界的首例，合并重复样本。保留原始全清单 0-based `sample_id`、指令、原始 label、4D 物理/裁剪/归一化动作、LAND 规则与 front/down 路径；输出 JSONL、便于表格检查的 CSV、中文 Markdown 摘要。不复制图像、不分词、不加载模型、不使用 GPU。
5. 新增 `tests/test_diffusion_supervision.py`，覆盖训练归一化逐元素一致、端点与裁剪、总体统计、LAND 规则、来源哈希失配、非有限/缺失标签、全局样本索引以及 Python/NumPy/Torch 随机数状态不变。
6. 自查时修正 CSV 中 `count` 潜在命名冲突：统计样本数继续用 `count`，原始标签裁剪字段使用 `raw_clipping_*` 前缀；增加 CSV 回归断言。摘要中的总帧数按实际统计生成，避免硬编码。

### 数值与使用约定

- 原始标签统计使用 float64 保留 JSON 数值；训练动作先转 float32，再由生产归一化函数生成目标。因此 CSV 中 float32 的 `clipped_4d` 可能显示 `1.100000023841858`，这是 float32 表示误差，不是更改动作上界。
- 裁剪比例按原始数值是否超出声明物理上下界计算；等于边界不算裁剪。LAND 原始值由 `is_last_step or is_penultimate` 生成。
- 固定抽查样本不是训练实际 batch，也不含预测值；训练原始标签完整保留在 `data/aerovla_train_dataset.json`。
- 审计仅为标签与来源检查，不表示图片像素解码、模型预测或导航性能已通过评测。

### 执行与验证

在 AeroVLA 根目录运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/xiyang/.conda/envs/aero_vla/bin/python scripts/audit_diffusion_supervision.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/xiyang/.conda/envs/aero_vla/bin/python -m unittest discover \
  -s tests -p test_diffusion_supervision.py -v
```

默认输出：`checkpoints/aero_vla_diffusion_full_20261010_200045/supervision/`。

- 定向测试：8 项通过。
- 语法检查：新增脚本和测试均通过 `py_compile`。
- 真实数据审计：2026-10-10 20:09:40（北京时间）完成；全部 410,450 帧通过，来源与当前 run_config 完全匹配。
- 产物独立复核：139 个固定样本的归一化动作与生产函数逐元素相同；278 个图像路径均存在（未解码）；7,611 条轨迹分布的帧数/LAND 阳性汇总与 train/val 总计一致；保存的脚本与 run_config SHA256 再次匹配。

### 真实数据审计结果

| 项目 | train | val |
| --- | ---: | ---: |
| 帧数 | 369,274 | 41,176 |
| 轨迹数 | 6,850 | 761 |
| LAND 阳性 / 阴性 | 8,203 / 361,071 | 910 / 40,266 |
| LAND 阳性比例 | 2.2214% | 2.2100% |
| last / penultimate / 同时为真 | 6,850 / 1,353 / 0 | 761 / 149 / 0 |
| 任一轴需裁剪帧数 | 9,605 | 1,051 |
| fwd 裁剪数 / 比例 | 4,669 / 1.2644% | 532 / 1.2920% |
| down 裁剪数 / 比例 | 807 / 0.2185% | 72 / 0.1749% |
| yaw 裁剪数 / 比例 | 8,308 / 2.2498% | 896 / 2.1760% |

统计忠实保留原始 LAND 标记；并非每条轨迹均有 `is_penultimate=True` 的帧。未重写标签或增加倒数第二帧标记。部分原始动作超出训练动作范围，审计明确记录裁剪范围和比例；训练继续使用前序已验证的裁剪规则。

审计目录约 1.2 MB。固定抽查共 139 帧，覆盖 11 个地图各自的 train/val。产物入口：

- [中文审计摘要](../../checkpoints/aero_vla_diffusion_full_20261010_200045/supervision/README.md)
- [固定样本 CSV](../../checkpoints/aero_vla_diffusion_full_20261010_200045/supervision/samples.csv)
- [固定样本 JSONL](../../checkpoints/aero_vla_diffusion_full_20261010_200045/supervision/samples.jsonl)
- [完整统计 JSON](../../checkpoints/aero_vla_diffusion_full_20261010_200045/supervision/summary.json)
- [逐轨迹分布](../../checkpoints/aero_vla_diffusion_full_20261010_200045/supervision/trajectory_distribution.csv)

数据 SHA256：`1c1e32787cedb096fe5df2511e5bf3cc81901b8a30d261df73ebcb800c868a67`。

Split SHA256：`50c4c84f90058bca2cc14ab19bbbccd41c0605e05bd924f530e743ad99236c4b`。

<a id="monitor"></a>

## 训练监控与产物导出

日期：2026-10-10（Asia/Shanghai）。本记录对应全量训练
`aero_vla_diffusion_full_20261010_200045` 的独立 CPU 监控工具。

### 修改 1：只读监控与检查产物

新增 `src/monitor_diffusion_training.py`。不修改正在运行的训练代码、不加载模型、不读取完整训练数据、不使用 CUDA；只增量读取训练目录的 `metrics.jsonl` / `validation_metrics.jsonl`，以及小文件 `run_config.json` 和日志目录的 `status.json`。

在 `<run-dir>/monitoring/` 原子更新以下文件：

| 文件 | 用途 |
| --- | --- |
| `train_metrics.csv` | 完整训练记录及全部现有字段，包含 step、epoch、loss、lr、grad_norm、samples |
| `validation_metrics.csv` | 完整验证记录，嵌套 physical / land 指标使用点号展开 |
| `latest_summary.json` | 最新记录、最近 100 条 loss 均值、进度、观测吞吐、训练 ETA、告警与检查点摘要 |
| `status.md` | 便于阅读的训练状态、最新验证和告警表 |
| `training_curves.png` | 六组英文标注曲线：训练 loss 与 100 次更新滚动均值、学习率、裁剪前梯度范数、验证 noise/action MSE、动作 MAE、LAND 指标 |
| `warnings.csv` | 非有限标量值、无效 JSONL 行、绘图不可用原因 |
| `checkpoints.csv` | root、best、checkpoint-* 权重存在性、大小、mtime 与 config 中的 global_step |

JSONL 原文件保留不改；末尾未写完的行（包括拆开的 UTF-8 字符）暂存到下一次轮询，避免误报。无效完整行记录告警后跳过。JSON 摘要中非有限数值写为 `null`，原 CSV 保留原始数值字符串，并单独记录告警。

CSV 保留全量记录；图片最多采样约 5,000 个训练点，滚动均值先基于完整记录计算。监控在尚无 config / metrics 时仍生成空表并等待；收到 `completed` 或 `failed` 时进行最后一次导出后退出。收到终态后再次轮询 JSONL，避免首次轮询与状态读取之间完成训练导致漏掉最终 step / validation 的竞态。

### 修改 2：计时口径和检查点口径

原始 metrics 未记录逐步 wall time，因此没有补造历史时间戳。吞吐只用监控进程首次观测的 step 与后续 step 差值、`time.monotonic()` 观测窗口计算；同时显示训练日志 `mtime` 与距上次更新秒数。跨 epoch 或出现新验证记录时重置吞吐窗口；step 静止时保留最近一次有效吞吐。

ETA 仅估计剩余训练更新耗时，**不包含后续全量验证耗时**；保存、数据加载与轮询误差可能包含在观测速率中。验证期间训练日志不增长属于正常现象，日志年龄不用于武断判断训练挂起。loss 是一个梯度累积窗口内按样本数加权的 epsilon 噪声 MSE；`grad_norm` 是裁剪前范数，超过配置的裁剪阈值不单独判为异常。

检查点列表只扫描 root、best、checkpoint-* 的文件元信息和小配置，不读取 `.pt` 内容。`files_present` **只表示权重和配置文件路径存在**，不表示权重内容通过加载验证，也不保证当前正在写入的保存已完成。

### 修改 3：绘图依赖降级和缓存

已确认现有 `aero_vla` Python 环境带 matplotlib 3.10.9，无需安装依赖。默认使用 Agg 后端，并将尚未配置的 Matplotlib / XDG 缓存指向 monitoring 下可写目录，避免受限环境首次字体缓存无法写入。缺少 matplotlib 或绘图失败会进入 `warnings.csv`，CSV、JSON、Markdown 仍继续生成。可用 `--no-plots` 显式关闭绘图。

### 验证

新增 `tests/test_training_monitor.py`，13 项测试通过，覆盖：

- 不完整 JSONL / UTF-8 尾行补齐后仅消费一次，源文件保持不变。
- 无记录 / 无配置时输出空表，后续配置到达时正常更新。
- CSV 保留新增字段，验证动作 / LAND 指标正确展平。
- NaN / Infinity 与异常完整行生成告警表，摘要为标准 JSON。
- 吞吐只来自监控观测窗口，验证或 epoch 改变后重置，验证阶段正确标记。
- JSONL 截短后重新读取。
- completed / failed 输出最后快照后不再 sleep；terminal 读取竞态不漏最终记录。
- 无 matplotlib 时仍导出监督文件。
- 检查点只读元信息，未对伪权重内容作加载或校验结论。
- 训练总步数正确处理尾批次、梯度累积和 max_steps；滚动均值忽略非有限值。

执行命令：

```bash
/data/xiyang/.conda/envs/aero_vla/bin/python -m unittest discover -s tests -p test_training_monitor.py -v
```

另以临时目录中的 5 条人工训练记录和 1 条验证记录进行真实 matplotlib 绘图，输出 PNG（163,963 字节）且无监督告警；该检查只用于工具验证，不属于真实训练结果。最初使用 `python -m unittest tests.test_training_monitor` 因本项目 tests 不是 Python package 而找不到模块，改为上述 discover 命令后 13 项测试全部通过。

### 使用方式

在 AeroVLA 项目目录执行一次快照：

```bash
/data/xiyang/.conda/envs/aero_vla/bin/python -u src/monitor_diffusion_training.py \
  --run-dir checkpoints/aero_vla_diffusion_full_20261010_200045 \
  --log-dir logs/aero_vla_diffusion_full_20261010_200045 \
  --interval 60 --plot-interval 60 --once
```

去掉 `--once` 后持续每 60 秒更新，直到 launcher 状态变为 completed / failed；failed 退出码为 1，completed / 单次快照正常退出码为 0。后台启动和真实训练观测由全量训练主流程统一管理，本工具开发验证未自行启动第二个真实监控进程。
