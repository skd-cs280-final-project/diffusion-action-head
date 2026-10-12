# 闭环评估启动脚本修改记录

日期：2026-10-10。

## 修改 1：拆分模型 GPU 和模拟器连接配置

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

## 修改 2：启动边界回归测试

新增 `tests/test_eval_launcher.py`，用 stub Python 接收参数，不导入模型或模拟器。检查项目路径和 Python 路径包含空格、从项目外启动、默认值、模型与模拟器 GPU 解耦、旧 GPU→端口默认规则、各环境参数、参数转发、dry-run 不执行 Python、非法端口及任务格式拒绝。

## 验证结果

- `bash -n scripts/eval_aerovla.sh` 通过。
- `/data/xiyang/.conda/envs/aero_vla/bin/python -m pytest tests/test_eval_launcher.py -q`：**10 项通过**，耗时 0.59 秒。
- 从 `/tmp` 执行上述 `--dry-run` 示例通过，输出模型 GPU 1、模拟器 GPU 0/端口 30000、DDP 端口 20001，项目资源路径均定位到 AeroVLA，模型相对路径将在项目根目录解析。

本次闭环范围只验证启动命令；此次训练/离线评估的运行结果由总报告记录，未声称已完成真实 AirSim 导航或获得 SR/SPL/NE 闭环指标，也未连接、重启现有模拟器。

## 修改 3：保留轨迹目录的绝对路径并修正加载数量日志

日期：2026-10-11。适用任务：GPU 2 使用 `checkpoint-70000` 评估 NYCEnvironmentMegapa 的随机 10 条独立 episode。

- 问题：`src/vlnce_src/env_uav.py` 的 `load_my_datasets()` 先用 `merged_json.strip('/').split('/')` 提取地图和 UUID，再将各段拼回轨迹目录。输入绝对 `dataset_path` 时，重建目录丢失开头的 `/`，导致 `trajectory_dir` 错误地变成相对路径；episode 保存阶段复制 `object_description.json` 时可能出现 `FileNotFoundError`，保存的 `ori_info.json` 也会记录错误路径。
- 修复：仅将轨迹目录的构造改为 `str(Path(merged_json).parent)`，保留绝对/相对路径的语义。地图、UUID、轨迹、指令、目标及对象字段保持原逻辑。
- 日志修复：`Loaded dataset` 由输出评估 JSON 路径的字符串长度改为输出实际加载的 episode 数量 `len(load_data)`。
- 验证：使用 `aero_vla` 环境执行 CPU 检查，从 AST 提取真实 `load_my_datasets()` 和 `find_closest_area()`，仅以简单值对象替代 AirSim 的 Pose/Vector/Quaternion 构造；未导入或启动模型、模拟器和 GPU。读取本次 `eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/selected_episodes.json`，分别以绝对根目录、`dataset_raw`、`./dataset_raw` 加载，每种均得到 10 条 episode。30 个路径案例的 merged/mark/object JSON 均可读，并实际执行 `shutil.copy2` 复制对象描述到临时目录成功。
- 对照检查：修复前后除 `trajectory_dir` 外所有加载字段相同；随机数状态保持不变；三种根目录解析得到相同的真实轨迹路径；加载数量日志使用实际列表长度。代码语法检查通过。
- 此次变更不修改动作、控制逻辑、模型输入、停止阈值或 `trace_eval.py`。仿真重启及闭环执行结果由本次独立评估报告记录。
