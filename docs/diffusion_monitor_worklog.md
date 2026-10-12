# 扩散头训练监督工具修改记录

日期：2026-10-10（Asia/Shanghai）。本记录对应全量训练
`aero_vla_diffusion_full_20261010_200045` 的独立 CPU 监控工具。

## 修改 1：只读监控与检查产物

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

## 修改 2：计时口径和检查点口径

原始 metrics 未记录逐步 wall time，因此没有补造历史时间戳。吞吐只用监控进程首次观测的 step 与后续 step 差值、`time.monotonic()` 观测窗口计算；同时显示训练日志 `mtime` 与距上次更新秒数。跨 epoch 或出现新验证记录时重置吞吐窗口；step 静止时保留最近一次有效吞吐。

ETA 仅估计剩余训练更新耗时，**不包含后续全量验证耗时**；保存、数据加载与轮询误差可能包含在观测速率中。验证期间训练日志不增长属于正常现象，日志年龄不用于武断判断训练挂起。loss 是一个梯度累积窗口内按样本数加权的 epsilon 噪声 MSE；`grad_norm` 是裁剪前范数，超过配置的裁剪阈值不单独判为异常。

检查点列表只扫描 root、best、checkpoint-* 的文件元信息和小配置，不读取 `.pt` 内容。`files_present` **只表示权重和配置文件路径存在**，不表示权重内容通过加载验证，也不保证当前正在写入的保存已完成。

## 修改 3：绘图依赖降级和缓存

已确认现有 `aero_vla` Python 环境带 matplotlib 3.10.9，无需安装依赖。默认使用 Agg 后端，并将尚未配置的 Matplotlib / XDG 缓存指向 monitoring 下可写目录，避免受限环境首次字体缓存无法写入。缺少 matplotlib 或绘图失败会进入 `warnings.csv`，CSV、JSON、Markdown 仍继续生成。可用 `--no-plots` 显式关闭绘图。

## 验证

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

## 使用方式

在 AeroVLA 项目目录执行一次快照：

```bash
/data/xiyang/.conda/envs/aero_vla/bin/python -u src/monitor_diffusion_training.py \
  --run-dir checkpoints/aero_vla_diffusion_full_20261010_200045 \
  --log-dir logs/aero_vla_diffusion_full_20261010_200045 \
  --interval 60 --plot-interval 60 --once
```

去掉 `--once` 后持续每 60 秒更新，直到 launcher 状态变为 completed / failed；failed 退出码为 1，completed / 单次快照正常退出码为 0。后台启动和真实训练观测由全量训练主流程统一管理，本工具开发验证未自行启动第二个真实监控进程。
