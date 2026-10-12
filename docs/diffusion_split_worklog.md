# 扩散头训练与验证：轨迹划分修改记录

日期：2026-10-10。

## 修改 1：接入严格轨迹划分

- 新增 `src/diffusion_split.py`，按 `version=1`、`split_by=trajectory` 读取用户提供的划分。
- 要求 train、val 均为非空轨迹列表；拒绝重复轨迹、轨迹交集、缺失或额外轨迹、重复帧，以及可能造成同轨迹别名的非规范相对路径。
- 验证全集轨迹 SHA256：排序轨迹 ID，以换行连接，不加末尾换行，再按 UTF-8 计算 SHA256。
- 修改 `src/diffusion_dataset.py`，新增 `split_json`、`split` 参数，在任何抽样前校验完整标注，并按原 JSON 顺序选择 train/val。旧的不带划分调用保持兼容；仅给 split 而不给文件会报错。
- `dataset.split_info` 保存两个分区的帧数与轨迹数、全集及各分区轨迹摘要、所选原始索引摘要、划分文件摘要、数据 JSON 摘要及绝对路径，可写入训练检查点和评估结果核对来源。数据 JSON 摘要为文件字节 SHA256，标签变化也会使其变化。

## 修改 2：可复现的小规模抽样

- 新增 `deterministic_subset(dataset, max_samples, seed)`，使用独立 Python RNG 随机抽帧并按原顺序排列，避免只取数据开头导致地图/轨迹偏置。
- 仅 `max_samples=None` 表示不限制；0、负数及非整数会报错。超出数据规模时返回完整数据集。
- 抽样返回保留 `split_info` 的 PyTorch `Subset`，记录随机种子、抽样数量、抽样前帧数及选中索引 SHA256，不修改原数据集元信息。
- 抽帧仅供诊断或限制运行规模，先执行完整划分校验；不会改变训练/验证的轨迹边界。

## 修改 3：验证与测试

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
