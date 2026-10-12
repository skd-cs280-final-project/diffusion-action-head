# 扩散头离线评估逐次改动记录

日期：2026-10-10。

## 记录 1：离线评估实现

- 新增 `src/eval_diffusion_head.py`，公开 `evaluate_diffusion`，供训练循环复用；独立 CLI 从检查点恢复 tokenizer、processor、冻结骨干及动作扩散头。
- 每批只编码一次图像和导航指令，共用条件计算 epsilon MSE 与 DDPM 动作采样；目标动作不进入条件编码器。
- 噪声、时间步、采样分别使用独立固定种子生成器；保存并恢复 torch 全局 RNG、DataLoader/sampler 的显式 generator 和每个模块的原始 train/eval 模式；评估全程无梯度。
- 按有效坐标累计噪声误差和前三维归一化 MSE；fwd/down/yaw 分别报告 MAE、RMSE 和坐标数，采用训练时相同范围裁剪，不混合距离与角度单位。LAND 报告混淆矩阵、precision/recall/F1/accuracy、正负样本数，无分母时 precision/recall/F1 为 0。
- CLI 固定使用 val 分割，默认完整验证集；可固定种子抽样用于接口验证。检查训练元数据的数据/划分哈希；旧检查点缺少分割证据时明确标为 `unknown`，不声称独立留出验证。
- 新增 `scripts/eval_diffusion_head.sh`，提供与训练入口一致的本地缓存设置。

相同模型、样本顺序、batch size、设备和精度下重复评估可复现。更换 batch size 或设备时 DDPM 的随机数布局及浮点运算可能变化，不承诺逐位一致。指标只表示离线动作预测效果，不能替代 AirSim 闭环导航指标。

## 记录 2：回归检验与输出兼容

- 增加 `--output-json`（`--output` 的别名），便于自动化训练→重载→评估脚本调用；报告将数值集中保存在 `metrics`，不混入耗时字段。
- `tests/test_eval_diffusion_head.py` 使用手算三帧数据核验全维噪声均值、前三维 MSE、逐物理维误差、LAND 混淆矩阵和最后单帧批次。
- 补充缺失坐标 mask、范围裁剪、零正样本 LAND、评估异常恢复、真实微型 DDPM 确定性、参数/梯度不变和训练 RNG/Loader generator 不变测试。
- 补充旧检查点未知来源、错误 train/val 身份、数据/划分 SHA 不匹配拒绝测试；CLI 测试验证从检查点恢复处理器、输出 JSON，并在划分冲突时提前拒绝加载大模型。
- 首轮 7 项测试全部通过（2.65 秒）；shell `bash -n` 和 CLI `--help` 通过。最终增加 CLI 集成回归后 **8 项全部通过**（3.35 秒）。验证命令：`OMP_NUM_THREADS=2 HF_HOME=/data/xiyang/project_orig/AeroVLA/.cache/huggingface /data/xiyang/.conda/envs/aero_vla/bin/python -m pytest tests/test_eval_diffusion_head.py -q`。

## 使用命令

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
