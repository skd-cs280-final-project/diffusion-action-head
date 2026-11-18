# 可选历史输入与联合 LoRA

这些实现来自课程重组之前，保留用于已有检查点兼容与后续研究。课程主实验使用 history_length=0、冻结骨干，不包含这些变量。

## UAV 数值历史

`--history-length 4` 表示当前及最多三个过去的观测；1 为当前状态，0 为关闭。状态进入动作头，不进入 Llama；没有历史 RGB 或长期语义记忆。详细字段、坐标和缺失值规则见[历史设计](../archive/design/diffusion_action_head.md#uav-history)。

```bash
PYTHON=python bash scripts/train_diffusion_head.sh   --history-length 4 --check-data --max-samples 4 --max-val-samples 8 --num-workers 0
PYTHON=python CUDA_VISIBLE_DEVICES=0 HISTORY_LENGTH=4 SMOKE_OUTPUT_DIR=./checkpoints/history_smoke_new bash scripts/smoke_diffusion_pipeline.sh
```

需要原始逐帧 log；评估从 checkpoint 自动恢复历史长度。K=0 权重不能直接当作 K=4；历史训练入口没有自动扩展旧权重的 warm-start。状态日志缺速度/时间时通过有效位表达，K=1 的相对位置恒为零。

静态审计读取保留的源清单：

```bash
python scripts/audit_uav_history.py --output-dir ./eval_results/history_audit_new
```

历史 smoke 与特征审计只证明流程，不证明新分支闭环收益。

## 联合 LoRA

从已训练扩散 checkpoint 恢复，新增视觉/Llama LoRA，同时训练 projector 与动作头。继承原头的 H 和历史配置，不属于精确 optimizer/RNG 续训。

```bash
PYTHON=python CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/train_diffusion_joint.sh   --checkpoint ./checkpoints/course_A0_seed42   --model-path ./openvla-7b --adapter-path ./checkpoints/aero_vla   --output-dir ./checkpoints/joint_extension_new   --epochs 1 --batch-size 2 --gradient-accumulation-steps 4 --eval-batch-size 2   --lora-rank 16 --lora-alpha 32   --lora-lr 2e-5 --projector-lr 1e-5 --head-lr 1e-5   --max-val-samples 512 --eval-steps 1000 --save-steps 1000
```

联合入口中 `--max-val-samples 0` 表示完整 val，单步训练/离线入口则要求正数或省略。step 0 先验证，best 可能停在初始化。v2 保存 joint_adapter.pt 和动作头，仍依赖基座及原 adapter。

`scripts/smoke_diffusion_joint.sh` 固定使用 `checkpoints/best`，解释器需显式设置 `PYTHON=python`；使用其他初始化权重时改用上面的正式入口并加小样本/max-steps 参数。结果参见[联合训练归档](../archive/experiments/joint_lora_20261103.md)和[281 条闭环](../archive/experiments/nyc_joint_final_seen281_20261104.md)。
