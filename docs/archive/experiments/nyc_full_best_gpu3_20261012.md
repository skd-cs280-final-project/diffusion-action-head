# Best checkpoint · NYC seen 完整验证（281 条）· GPU 3

> 历史归档：保留原统计日期和实验范围；不代表本课程实验已完成。入口见 [归档说明](../README.md)。

此页由 summarize_results.py 自动生成，每次汇总覆盖更新。

更新时间：2026-10-12T18:14:12.634129+00:00（UTC）。当前状态：`completed`。
范围：`Complete official seen validation split for NYCEnvironmentMegapa: all 281 unique episodes; no sampling. Latest user correction explicitly excludes the remaining raw trajectories.`。已完成 **281/281**；pending：**0**。
结果性质：**全部完成且独立核验通过**。
Checkpoint：`/data/xiyang/project_orig/AeroVLA_stat/checkpoints/best`；step：`115400`；history_disabled：`True`；物理 GPU：`3`。
Launch 后已用时：9.714 小时；预计剩余：0.000 小时。
粗估：从 launch 开始的墙钟耗时 / completed × pending；包含启动开销，任务时长不均，不保证完工时间。

以下 SR、SPL、目标距离只针对 completed；未完成任务不计作失败，阶段结果不能代表请求全量。

| 已完成任务 overall 指标 | 数值 |
|---|---:|
| SR | 164/281 = 58.36% |
| OSR | 62.28% |
| SPL（原协议） | 52.07% |
| 目标距离均值（m） | 63.100 |
| 参考终点距离均值（原 NE，m） | 63.090 |
| 推理平均 / P95（s） | 0.139 / 0.147 |
| 动作数 / STOP | 23872 / 167 |

| 分组 | completed / requested | SR | OSR | SPL | 目标距离均值 m |
|---|---:|---:|---:|---:|---:|
| `official_seen` | 281 / 281 | 58.36% | 62.28% | 52.07% | 63.100 |

## 运行与复核入口

运行目录：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012`。

- [selected_episodes.json](../../../eval_results/best_nyc_full_gpu3_20261012/selected_episodes.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/selected_episodes.json`
- [selection_manifest.json](../../../eval_results/best_nyc_full_gpu3_20261012/selection_manifest.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/selection_manifest.json`
- [launch_manifest.json](../../../eval_results/best_nyc_full_gpu3_20261012/launch_manifest.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/launch_manifest.json`
- [status.json](../../../eval_results/best_nyc_full_gpu3_20261012/status.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/status.json`
- [summary.json](../../../eval_results/best_nyc_full_gpu3_20261012/summary.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/summary.json`
- [results.md](../../../eval_results/best_nyc_full_gpu3_20261012/results.md)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/results.md`
- [episode_results.csv](../../../eval_results/best_nyc_full_gpu3_20261012/episode_results.csv)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/episode_results.csv`
- [episode_metrics.jsonl](../../../eval_results/best_nyc_full_gpu3_20261012/episode_metrics.jsonl)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/episode_metrics.jsonl`
- [action_trace.jsonl](../../../eval_results/best_nyc_full_gpu3_20261012/action_trace.jsonl)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/action_trace.jsonl`
- [verification.json](../../../eval_results/best_nyc_full_gpu3_20261012/verification.json)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/verification.json`
- [episodes](../../../eval_results/best_nyc_full_gpu3_20261012/episodes)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/episodes`
- [eval.log](../../../eval_results/best_nyc_full_gpu3_20261012/eval.log)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/eval.log`
- [simulator.log](../../../eval_results/best_nyc_full_gpu3_20261012/simulator.log)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/simulator.log`
- [supervisor.log](../../../eval_results/best_nyc_full_gpu3_20261012/supervisor.log)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/supervisor.log`
- [launch.sh](../../../eval_results/best_nyc_full_gpu3_20261012/launch.sh)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/launch.sh`
- [launch_job.py](../../../eval_results/best_nyc_full_gpu3_20261012/launch_job.py)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/launch_job.py`
- [trace_eval.py](../../../eval_results/best_nyc_full_gpu3_20261012/trace_eval.py)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/trace_eval.py`
- [summarize_results.py](../../../eval_results/best_nyc_full_gpu3_20261012/summarize_results.py)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/summarize_results.py`
- [verify_results.py](../../../eval_results/best_nyc_full_gpu3_20261012/verify_results.py)：`/data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/verify_results.py`

```bash
/data/xiyang/.conda/envs/aero_vla/bin/python /data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/summarize_results.py
/data/xiyang/.conda/envs/aero_vla/bin/python /data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/verify_results.py
```

verify 仅在全部 selected 任务已完成且状态为 verifying/completed 时核验；partial 不会得到 passed=true。

实际 server_command：

```bash
/data/xiyang/.conda/envs/aero_vla/bin/python -u /data/xiyang/project_orig/AeroVLA_stat/airsim_plugin/AirVLNSimulatorServerTool.py --gpus 3 --port 45000 --root_path /data/xiyang/project_orig/AeroVLA_stat/envs
```

实际 eval_command：

```bash
/data/xiyang/.conda/envs/aero_vla/bin/python -u /data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/trace_eval.py --project_prefix /data/xiyang/project_orig/AeroVLA_stat --run_type eval --name Best_NYC_Full_GPU3 --gpu_id 3 --simulator_tool_port 45000 --DDP_MASTER_PORT 20004 --batchSize 1 --maxWaypoints 200 --dataset_path /data/xiyang/project_orig/AeroVLA_stat/dataset_raw --eval_save_path /data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/episodes --model_path /data/xiyang/project_orig/AeroVLA_stat/checkpoints/best --model_base /data/xiyang/project_orig/AeroVLA_stat/openvla-7b --eval_json_path /data/xiyang/project_orig/AeroVLA_stat/eval_results/best_nyc_full_gpu3_20261012/selected_episodes.json --map_spawn_area_json_path /data/xiyang/project_orig/AeroVLA_stat/data/meta/map_spawnarea_info.json --object_name_json_path /data/xiyang/project_orig/AeroVLA_stat/data/meta/object_description.json
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
