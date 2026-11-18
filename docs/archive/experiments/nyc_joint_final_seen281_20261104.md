# Joint final step 23080 · NYC seen 完整验证（281 条）· model GPU 0 / simulator GPU 1

> 历史归档：保留原统计日期和实验范围；不代表本课程实验已完成。入口见 [归档说明](../README.md)。

此页由 summarize_results.py 自动生成，每次汇总覆盖更新。

更新时间：2026-11-05T17:37:28.250203+00:00（UTC）。当前状态：`completed`。
范围：`Complete official NYC seen 281 tasks, joint final step 23080; 81 audited completed tasks carried forward, 200 pending tasks resumed.`。已完成 **281/281**；pending：**0**。
结果性质：**全部完成且独立核验通过**。
Checkpoint：`/data/xiyang/project_orig/AeroVLA_stat/checkpoints/joint_lora_stage2_20261103`；step：`23080`；history_disabled：`True`；物理 GPU：`0`。
本次续跑已用时：8.247 小时；预计剩余：0.000 小时。
粗估：本次续跑墙钟耗时 / 本次新增完成数 × pending；已保留的 81 条不进入续跑速度分母，任务时长不均，不保证完工时间。

以下 SR、SPL、目标距离只针对 completed；未完成任务不计作失败，阶段结果不能代表请求全量。

| 已完成任务 overall 指标 | 数值 |
|---|---:|
| SR | 166/281 = 59.07% |
| OSR | 61.92% |
| SPL（原协议） | 52.53% |
| 目标距离均值（m） | 66.538 |
| 参考终点距离均值（原 NE，m） | 66.706 |
| 推理平均 / P95（s） | 0.165 / 0.174 |
| 动作数 / STOP | 23543 / 168 |

| 分组 | completed / requested | SR | OSR | SPL | 目标距离均值 m |
|---|---:|---:|---:|---:|---:|
| `official_seen` | 281 / 281 | 59.07% | 61.92% | 52.53% | 66.538 |

## 与原 best 的完整 seen 对比

原 best（step 115400）的 281 条结果已独立核验。联合 final（step 23080）仅在全部完成且核验通过后填入下表；阶段指标见上方。

| 指标 | 原 best：281 条 | 联合 final：281 条 | 变化 |
|---|---:|---:|---:|
| SR | 58.36% | 59.07% | +0.71 个百分点 |
| OSR | 62.28% | 61.92% | -0.36 个百分点 |
| SPL（原协议） | 52.07% | 52.53% | +0.46 个百分点 |
| 最终目标距离均值（m） | 63.100 | 66.538 | +3.439 m |
| NE（参考终点距离，m） | 63.090 | 66.706 | +3.615 m |

两次任务清单完全相同，沿用 seed=100、每任务最多 200 动作、DDPM 10 步、目标方向提示和原成功/终止协议；不保证逐动作相同随机噪声。两份权重 history_length=0，本次评估联合微调效果，未启用状态历史新分支。

原 baseline 用 GPU 3 同时运行模型和仿真；本次模型 GPU 0、仿真 GPU 1，耗时差异不能仅归因于模型。旧 baseline 曾耗时约 9 小时 43 分钟；任务按参考轨迹长度排序，早期进度外推可能低估后续耗时。

## 中断与续跑说明

首次进程在 2026-11-04 21:53（北京时间）完成 81/281 后，因 AirSim 零尺寸 RGB 图像中断。已修复尺寸/数据长度检查，使异常进入原有最多 11 次取图重试；模型权重、控制和指标协议不变。

81 条已完成轨迹独立核验通过，并在此目录保留；中断任务的 45 条未完成动作不参与统计，剩余 200 条从各自 frame 0 开始。续跑重新设 seed=100，不是恢复原进程的随机数状态。总分母仍为全部 281 条，每个 UUID 仅统计一次。

中断前固定同 81 条对照（不是全量结果）：

| 指标 | 原 best，同 81 条 | 联合 final，同 81 条 | 变化 |
|---|---:|---:|---:|
| SR | 64.20%（52/81） | 67.90%（55/81） | +3.70 个百分点 |
| OSR | 70.37%（57/81） | 69.14%（56/81） | −1.23 个百分点 |
| SPL（原协议） | 52.27% | 55.63% | +3.36 个百分点 |
| NE（m） | 35.34 | 33.15 | −2.19 m |

这 81 条优先执行较短参考轨迹，不能据此下全量改善结论。配对共同成功 41 条、旧失败新成功 14 条、旧成功新失败 11 条、共同失败 15 条；精确对照和部分核验证据见运行目录中的 matched81_comparison_20261105.json、partial_audit_20261105.json。

## 运行与复核入口

运行目录：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105`。

- [selected_episodes.json](../../../eval_results/joint_final_nyc_seen281_resume_20261105/selected_episodes.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/selected_episodes.json`
- [selection_manifest.json](../../../eval_results/joint_final_nyc_seen281_resume_20261105/selection_manifest.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/selection_manifest.json`
- [launch_manifest.json](../../../eval_results/joint_final_nyc_seen281_resume_20261105/launch_manifest.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/launch_manifest.json`
- [status.json](../../../eval_results/joint_final_nyc_seen281_resume_20261105/status.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/status.json`
- [baseline_reference.json](../../../eval_results/joint_final_nyc_seen281_resume_20261105/baseline_reference.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/baseline_reference.json`
- [resume_manifest.json](../../../eval_results/joint_final_nyc_seen281_resume_20261105/resume_manifest.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/resume_manifest.json`
- [source_snapshot.json](../../../eval_results/joint_final_nyc_seen281_resume_20261105/source_snapshot.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/source_snapshot.json`
- [partial_audit_20261105.json](../../../eval_results/joint_final_nyc_seen281_resume_20261105/partial_audit_20261105.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/partial_audit_20261105.json`
- [matched81_comparison_20261105.json](../../../eval_results/joint_final_nyc_seen281_resume_20261105/matched81_comparison_20261105.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/matched81_comparison_20261105.json`
- [resume_support.py](../../../eval_results/joint_final_nyc_seen281_resume_20261105/resume_support.py)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/resume_support.py`
- [prepare_resume.py](../../../eval_results/joint_final_nyc_seen281_resume_20261105/prepare_resume.py)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/prepare_resume.py`
- [summary.json](../../../eval_results/joint_final_nyc_seen281_resume_20261105/summary.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/summary.json`
- [results.md](../../../eval_results/joint_final_nyc_seen281_resume_20261105/results.md)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/results.md`
- [episode_results.csv](../../../eval_results/joint_final_nyc_seen281_resume_20261105/episode_results.csv)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/episode_results.csv`
- [episode_metrics.jsonl](../../../eval_results/joint_final_nyc_seen281_resume_20261105/episode_metrics.jsonl)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/episode_metrics.jsonl`
- [action_trace.jsonl](../../../eval_results/joint_final_nyc_seen281_resume_20261105/action_trace.jsonl)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/action_trace.jsonl`
- [verification.json](../../../eval_results/joint_final_nyc_seen281_resume_20261105/verification.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/verification.json`
- [episodes](../../../eval_results/joint_final_nyc_seen281_resume_20261105/episodes)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/episodes`
- [eval.log](../../../eval_results/joint_final_nyc_seen281_resume_20261105/eval.log)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/eval.log`
- [simulator.log](../../../eval_results/joint_final_nyc_seen281_resume_20261105/simulator.log)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/simulator.log`
- [supervisor.log](../../../eval_results/joint_final_nyc_seen281_resume_20261105/supervisor.log)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/supervisor.log`
- [launch.sh](../../../eval_results/joint_final_nyc_seen281_resume_20261105/launch.sh)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/launch.sh`
- [launch_job.py](../../../eval_results/joint_final_nyc_seen281_resume_20261105/launch_job.py)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/launch_job.py`
- [trace_eval.py](../../../eval_results/joint_final_nyc_seen281_resume_20261105/trace_eval.py)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/trace_eval.py`
- [summarize_results.py](../../../eval_results/joint_final_nyc_seen281_resume_20261105/summarize_results.py)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/summarize_results.py`
- [verify_results.py](../../../eval_results/joint_final_nyc_seen281_resume_20261105/verify_results.py)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/verify_results.py`

```bash
/data/xiyang/.conda/envs/aero_vla/bin/python /data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/summarize_results.py
/data/xiyang/.conda/envs/aero_vla/bin/python /data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/verify_results.py
```

verify 仅在全部 selected 任务已完成且状态为 verifying/completed 时核验；partial 不会得到 passed=true。

实际 server_command：

```bash
/data/xiyang/.conda/envs/aero_vla/bin/python -u /data/xiyang/project_orig/AeroVLA_stat/airsim_plugin/AirVLNSimulatorServerTool.py --gpus 1 --port 45010 --root_path /data/xiyang/project_orig/AeroVLA_stat/envs
```

实际 eval_command：

```bash
/data/xiyang/.conda/envs/aero_vla/bin/python -u /data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/trace_eval.py --project_prefix /data/xiyang/project_orig/AeroVLA_stat --run_type eval --name Joint_Final_NYC_Seen281_Resume --gpu_id 1 --simulator_tool_port 45010 --DDP_MASTER_PORT 20005 --batchSize 1 --maxWaypoints 200 --dataset_path /data/xiyang/project_orig/AeroVLA_stat/dataset_raw --eval_save_path /data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/episodes --model_path /data/xiyang/project_orig/AeroVLA_stat/checkpoints/joint_lora_stage2_20261103 --model_base /data/xiyang/project_orig/AeroVLA_stat/openvla-7b --eval_json_path /data/xiyang/project_orig/AeroVLA_stat/eval_results/joint_final_nyc_seen281_resume_20261105/selected_episodes.json --map_spawn_area_json_path /data/xiyang/project_orig/AeroVLA_stat/data/meta/map_spawnarea_info.json --object_name_json_path /data/xiyang/project_orig/AeroVLA_stat/data/meta/object_description.json
```

## 指标边界

- **goal_hint**：Original model prompt includes a semantic direction derived from GT target position.
- **success**：Original evaluator stop signal within 20 meters, subject to done/early_end logic.
- **oracle_success**：success OR oracle_success, matching legacy metric.py directory protocol.
- **distance**：final_target_distance is to spawned target; reference_endpoint_distance is to reference last point (legacy NE). Neither is normalized.
- **legacy_spl**：success * max(0, (reference_length - 20) / max(reference_length - 20, pred_length)); null if successful with zero denominator.
- **collision**：Combined collision/depth/stuck/distance-increase flag, not a pure physical collision rate.
- **latency**：Wall seconds of original model.run(), whole batch (batch size 1), excludes simulator action and observation time.
- **denominator**：Only finished selected episodes. Pending episodes are not counted as failures; zero finished means null aggregate.
