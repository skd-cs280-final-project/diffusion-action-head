# 扩散动作头数据管线修改记录

## 2026-10-10：建立连续动作数据接口

- 新增 `src/diffusion_dataset.py`，复用 `AeroVLADataset._get_images` 的双视角拼图，保持前视图在上、下视图在下，每幅图先缩放到 224×224。
- 标签顺序为 `[fwd, down, yaw, land]`。前三维沿用原范围 `[0,5]`、`[-5,5]`、`[-1.1,1.1]`，裁剪后线性映射至 `[-1,1]`，取消原 99 桶量化。
- `land = is_last_step OR is_penultimate`，继续飞行为 -1，降落为 +1，保留原 `LAND` token 的语义。公开归一化、反归一化函数及范围常量，供推理与检查点元数据共用。
- 模型输入仅包含 `instruction.strip() + "\nAction: "`；不加入动作数值、`LAND`、目标 EOS 或语言建模标签。检查 tokenizer 没有自动添加 EOS。
- 新增 `DiffusionDataCollator`，输出 `pixel_values`、`input_ids`、`attention_mask`、`actions`。动作形状为 `[B,1,4]`；图片兼容当前的通道拼接 tensor 和字典格式视觉输入。
- 对超长 prompt、非有限动作、非法 LAND 标记、错误维度明确报错。禁止静默截断，以免丢失导航指令或 `Action:` 后缀。
- 当前训练 JSON 只明确描述单帧动作，默认且仅支持 `action_horizon=1`；未将相邻 JSON 记录擅自解释为连续轨迹。
- 未改动旧数据集或旧 LM/LoRA 训练入口。

## 本地数据条件

- `dataset_raw` 链接至 `/data/xiyang/dataset/OpenUAV/TravelUAV`，有真实 frontcamera/downcamera 图像。
- `data/aerovla_train_dataset.json` 缺失；在 `/data/xiyang/dataset` 和 `/data/xiyang/project_orig` 中未发现同名训练标注。现有 `data/uav_dataset` 是评估拆分，不能替代监督训练标签。
- 本地 tokenizer 配置为 `add_bos_token=true`、`add_eos_token=false`、右侧 padding；适合仅 prompt 的扩散训练输入。

## 验证记录

### 第二次修改：补充数据合同测试

- 新增 `tests/test_diffusion_dataset.py`，共 13 项测试。
- 覆盖范围：归一化端点/中点/反变换与输入不变性、范围裁剪、非法维度和 NaN/Inf、双视角拼图位置、完整 LAND 规则、修改动作标签不改变模型输入、单步 horizon 约束、标注格式错误、缺图报错、左右 padding mask、动作维度与 dtype、字典图像输入、超长 prompt 报错、自动 EOS/空 batch/非法目标拒绝。
- 执行命令：`/data/xiyang/.conda/envs/aero_vla/bin/python -m unittest discover -s tests -p 'test_diffusion_dataset.py' -v`。
- 结果：13/13 通过，测试主体耗时约 0.12 秒。

### 本地真实处理器与图片验证

- 使用本地 `openvla-7b` 的 `AutoTokenizer`、`AutoProcessor`，开启 `local_files_only=True` 和离线环境变量，未下载任何模型或数据。
- 输入 TravelUAV 真实样本图片 `Japanese_Street/28725a09-4eea-4708-89bc-c7402931d4ba/{frontcamera,downcamera}/000000.png`。
- 用临时 JSON 提供人工构造标签 `[2.5,0,0,0]`，仅用于接口 smoke 验证，**不是真实监督训练样本，也没有据此报告真实训练效果**。
- 输出形状：`pixel_values=[1,6,224,224]`、`input_ids=[1,11]`、`attention_mask=[1,11]`、`actions=[1,1,4]`；归一化标签为 `[0,0,0,-1]`。
- 解码输入为 `<s> Fly forward toward the building.\nAction: `；确认无 EOS 和目标 LAND，符合冻结 VLM 提取条件特征的要求。
