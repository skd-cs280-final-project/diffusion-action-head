# 扩散头监督数据整理记录

日期：2026-10-10。对应运行：`checkpoints/aero_vla_diffusion_full_20261010_200045/`。

用户要求整理训练过程中的监督数据，方便检查。本次提供全量静态标签审计和可追溯的固定抽查样本；训练运行状态与模型监控由全量训练报告单独记录。

## 逐次变更

1. 新增 `scripts/audit_diffusion_supervision.py`。读取原始数据和轨迹 split，复用训练的 split 校验及 float32 `normalize_actions`。验证全部监督标签、指令和 LAND 布尔标志；缺失、非有限、float32 溢出等情况明确写入 `audit_errors.json` 并失败，不跳过坏记录。
2. 增加来源核验：数据 JSON、split JSON、全轨迹集合、各 split 轨迹集合、原始清单选中索引 SHA256，以及帧数/轨迹数，必须与当前 `run_config.json` 的 train/val 记录匹配。保存脚本、归一化实现、动作边界定义和 split 校验代码的 SHA256，便于追溯。
3. 增加全量 train/val 的 raw、clipped、normalized 四轴范围/均值/总体标准差和逐轴裁剪计数/比例；LAND 正负、last、penultimate、两标志同时为真的数量分别统计，避免重复计算 LAND 阳性。输出完整地图和逐轨迹分布。
4. 增加每 split/map 固定代表样本。选择原始顺序下的 LAND 正负、last、penultimate、两者同时、任意裁剪和各轴上下越界的首例，合并重复样本。保留原始全清单 0-based `sample_id`、指令、原始 label、4D 物理/裁剪/归一化动作、LAND 规则与 front/down 路径；输出 JSONL、便于表格检查的 CSV、中文 Markdown 摘要。不复制图像、不分词、不加载模型、不使用 GPU。
5. 新增 `tests/test_diffusion_supervision.py`，覆盖训练归一化逐元素一致、端点与裁剪、总体统计、LAND 规则、来源哈希失配、非有限/缺失标签、全局样本索引以及 Python/NumPy/Torch 随机数状态不变。
6. 自查时修正 CSV 中 `count` 潜在命名冲突：统计样本数继续用 `count`，原始标签裁剪字段使用 `raw_clipping_*` 前缀；增加 CSV 回归断言。摘要中的总帧数按实际统计生成，避免硬编码。

## 数值与使用约定

- 原始标签统计使用 float64 保留 JSON 数值；训练动作先转 float32，再由生产归一化函数生成目标。因此 CSV 中 float32 的 `clipped_4d` 可能显示 `1.100000023841858`，这是 float32 表示误差，不是更改动作上界。
- 裁剪比例按原始数值是否超出声明物理上下界计算；等于边界不算裁剪。LAND 原始值由 `is_last_step or is_penultimate` 生成。
- 固定抽查样本不是训练实际 batch，也不含预测值；训练原始标签完整保留在 `data/aerovla_train_dataset.json`。
- 审计仅为标签与来源检查，不表示图片像素解码、模型预测或导航性能已通过评测。

## 执行与验证

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

## 真实数据审计结果

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

- [中文审计摘要](../checkpoints/aero_vla_diffusion_full_20261010_200045/supervision/README.md)
- [固定样本 CSV](../checkpoints/aero_vla_diffusion_full_20261010_200045/supervision/samples.csv)
- [固定样本 JSONL](../checkpoints/aero_vla_diffusion_full_20261010_200045/supervision/samples.jsonl)
- [完整统计 JSON](../checkpoints/aero_vla_diffusion_full_20261010_200045/supervision/summary.json)
- [逐轨迹分布](../checkpoints/aero_vla_diffusion_full_20261010_200045/supervision/trajectory_distribution.csv)

数据 SHA256：`1c1e32787cedb096fe5df2511e5bf3cc81901b8a30d261df73ebcb800c868a67`。

Split SHA256：`50c4c84f90058bca2cc14ab19bbbccd41c0605e05bd924f530e743ad99236c4b`。
