# 训练与评估接入工作记录

本记录承接 `diffusion_action_head_report.md` 与 `diffusion_data_readiness_report.md`，日期 2026-10-10。原数据、基座和 adapter 权重保留。

| 批次 | 文件 | 改动 | 验证 |
| --- | --- | --- | --- |
| 1 | `src/train_diffusion_head.py` | 默认要求轨迹 split，训练仅使用 train、验证仅使用 val；固定种子随机子集；新增不加载 7B 的 `--check-data`；epoch/最终步/可选周期验证，按验证 noise_mse 保存 best；保存划分与验证元数据。 | 模块并行实现完成后统一测试。 |
| 2 | `tests/test_train_diffusion_head.py` | 测试验证回调仅在 optimizer 更新后触发；重合的 epoch/周期/最终步不重复；验证后恢复头训练状态且骨干保持 eval。 | 等待统一测试。 |
| 3 | `scripts/train_diffusion_head.sh`、`scripts/smoke_diffusion_pipeline.sh` | 配置可覆盖的 CPU 线程数；新增真实标注串行流程检查：train4/val8、两步更新、独立评估重载、同种子指标一致性与数据隔离断言。 | 待运行，检查点明确仅为流程验证。 |
| 4 | `README.md`、两份前序报告、新运行指南 | 加入当前训练/离线评估/闭环配置命令，区分历史验证与现状，解释指标含义和检查点边界。 | 真实数据 `--check-data` 通过；首轮 58 项测试通过；GPU 短程流程运行中。 |
| 5 | `src/train_diffusion_head.py` | 独立审查后将 `--check-data` 的形状记录改为递归，兼容数据处理器的字典型图像输出。 | 当前本地 OpenVLA tensor 图片路径已通过，后续执行完整回归。 |
| 6 | smoke 脚本和运行指南 | 流程核验从实际检查点读取并断言 train/val 帧数及 optimizer 步数，避免仅按常量输出报告；更新真实两步训练、独立重载指标与完整测试结果。 | 真实 7B GPU 流程 exit 0；重载指标一致；实际检查点计数及 split 副本校验通过；全套 66 项通过。 |

## 最终验证记录

- `scripts/train_diffusion_head.sh --check-data --max-samples 4 --max-val-samples 8 --num-workers 0`：成功验证 train/val 完整划分和实际预处理，模型未加载。
- `PYTHON=/data/xiyang/.conda/envs/aero_vla/bin/python CUDA_VISIBLE_DEVICES=1 bash scripts/smoke_diffusion_pipeline.sh`：沙箱内明确检测 CUDA 不可用后，经命令权限批准在物理 GPU 1 完成。未占用 GPU 0 已有任务，未操作模拟器。
- GPU 真实训练更新两次，loss=1.2125136852264404、1.2360974550247192；验证 noise_mse=1.50709730386734、归一化前三维 MSE=0.7305838082736672。独立进程重载最终检查点所得指标相同。
- 仅训练 5,265,668 个扩散头参数，冻结 7,409,644,992 个上游参数。实际使用 4 条 train / 8 条 val 真实标注，未跨越轨迹划分。
- 已核对最终/best/step1/step2 检查点、元数据计数、SHA256、复制后的 `data_split.json` 与输入文件字节一致。
- `python -m pytest tests -q` 最终 **66 passed**（6.27 秒）；Python 编译、四个 shell 语法、diff 空白检查通过。
- 用户的原始数据、基座权重、AeroVLA adapter、已有模拟器任务均未修改。未自行提交 git commit。尚未进行完整收敛训练或真实闭环指标评测。

数据划分、离线评估、闭环启动脚本的每次改动，分别记于 `diffusion_split_worklog.md`、`diffusion_eval_worklog.md`、`diffusion_closed_loop_worklog.md`。
