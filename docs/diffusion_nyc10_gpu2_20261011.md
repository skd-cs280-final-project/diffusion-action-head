# checkpoint-70000：GPU2 上 NYC 随机 10 个任务闭环评估

## 最终结果

**已在 `aero_vla` 环境、物理 GPU2 上完成全部 10 条任务，成功 7 条，SR=70%。** 正式运行于北京时间 2026-10-11 22:55:51 启动，23:13:53 完成，包含模型和场景加载共约 **18 分 02 秒**。评估器退出码为 0，任务集合与抽样清单完全一致。

| 指标 | 结果 |
|---|---:|
| 完整任务数 | 10/10 |
| SR / OSR | 70.00% / 70.00% |
| 原协议 SPL | 62.03% |
| 平均目标距离（三维） | 74.855 m |
| 平均参考终点距离（原 NE） | 74.879 m |
| 7 条成功任务的平均目标距离 | 2.576 m |
| 成功任务目标距离范围 | 1.312–3.943 m |
| 深度近障终止 | 3/10 |
| 提前 STOP 标志 | 0/10 |
| 高层动作 / STOP 信号 | 693 / 7 |
| 单次模型推理平均 / P95 | 0.136 / 0.152 秒 |

结果表明 checkpoint-70000 已能在这组任务中完成图像输入、动作扩散采样、仿真执行、反馈观测和停止判断的闭环。3 条失败均较早因深度近障规则终止，拉高了全体平均目标距离；不能用成功子集的 2.576 m 代替全体指标，也不能用本次同一地图的 10 条小样本结果代表完整基准。各条任务的指标见[最终逐任务报告](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/results.md)。

![10 条任务轨迹对照](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/trajectory_overview.png)

## 请求与配置

- 模型：`checkpoints/aero_vla_diffusion_full_20261010_200045/checkpoint-70000`。
- 数据地图：`dataset_raw/NYCEnvironmentMegapa`；这里的 10 个场景是同一地图的 10 条独立 episode，各自有起点、目标与指令。
- 环境：`/data/xiyang/.conda/envs/aero_vla/bin/python`。
- 模型与 Unreal 场景：物理 GPU2；模型环境变量 `CUDA_VISIBLE_DEVICES=2`，场景由 `-GraphicsAdapter=2` 指定。
- 新建仿真服务端口 `40000`，首个场景端口 `40001`，评估 DDP 检查端口 `20003`。
- 单任务 batch=1，最多 200 次高层动作；预测 1 步、执行 1 步，每次动作采样去噪 10 次。
- 抽样 seed=42；推理沿用原入口 `setup()` 的随机 seed=100，二者含义不同。

## 抽样与数据核验

从官方 `data/uav_dataset/seen_valset_splits/NYCEnvironmentMegapa.json` 的 281 条独立任务中，先按轨迹路径排序去重，再使用 `random.Random(42).sample(..., 10)` 抽样。没有对逐帧记录直接抽样，以免同一任务重复出现。

10 条任务与本轮扩散头训练轨迹、离线验证轨迹均无交集；基座及既有 adapter 的历史训练暴露未知，不能据此宣称完整模型从未见过这些任务。

[抽样 JSON](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/selected_episodes.json)和[完整抽样清单](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/selection_manifest.json)保存了抽样顺序、环境实际排序、指令、资源检查、训练交集检查和输入/权重 SHA256。checkpoint 权重哈希为 `4d5259f3d89c8613d7ca0d28577bb14821ab0e90f57c5a3a6235133944ac7fdf`。

## 本次实施记录

1. 核对 checkpoint、地图程序、仿真客户端和 `aero_vla` 依赖；GPU2 初次检查空闲，GPU1 正在训练。沙箱内 CUDA 不可见，经沙箱外检查确认 GPU2 可用。
2. 新增独立抽样 JSON 与 provenance 清单，检查 10 条任务的 merged/mark/object JSON、初始姿态、地图 spawn 和描述资源。
3. 新增本次任务目录内的 `launch_job.py` 和 `launch.sh`，在持久会话中管理专用仿真服务、评估子进程、日志和完成状态；只清理本次启动的进程与服务。
4. 新增 `trace_eval.py`，包装原评估入口以记录逐步模型动作、停止标志、推理耗时和每个任务的结束指标；保持原输入处理、控制和终止逻辑。
5. 首次运行在第一个任务保存时触发 `FileNotFoundError`：`env_uav.py` 对绝对路径分段后丢失了开头的 `/`。改为 `str(Path(merged_json).parent)`，并修正数据加载数量日志。CPU 核验三种路径形式、共 30 个任务加载和对象描述复制通过；首次尝试日志与输出保存在 `attempts/initial_absolute_path_issue/`。详情见[闭环修改记录](diffusion_closed_loop_worklog.md)。
6. 新增 `summarize_results.py`，生成 JSON、逐任务 CSV 和 Markdown。用 CPU 临时数据核验 0/10、部分完成、10/10、重复任务、未写完日志尾行及指标口径。
7. 新增 `plot_trajectories.py`，从已保存的状态日志绘制参考轨迹与实际轨迹的 XY 对照图，输出 PNG/PDF；目标距离仍按三维坐标计算，未完成任务不填入最终结果。
8. 新增 `verify_results.py`，完成后从独立保存的逐帧位置重算指标，检查 10 个任务身份、逐步动作序号与范围、传感器日志/相机帧完整性、对象描述复制和 checkpoint SHA256，并将核验结果写入 `verification.json`。
9. 交叉复核时修正核验脚本的相机帧索引断言：原控制器每步记录 5 条状态，仅最后一条带图像；因此日志数应为 `1 + 5 × 动作数`，各相机图像数为 `1 + 动作数`，图像索引为 `000000/000005/000010/...`。没有改动评估器的保存频率。

模型与训练代码、checkpoint 权重不作修改。原评估器每个任务保存传感器日志和图像；本次另保存 `action_trace.jsonl`、`episode_metrics.jsonl`，便于检查。

## 评估口径

沿用项目原有闭环评估协议：输入提示含根据目标位置计算的相对方向；成功要求模型发出停止信号且满足 20 的目标距离阈值及原终止条件。`collision` 标志也可能来自停滞或距离持续增加等启发式判断，不能一概解释为真实物理碰撞。

报告中的目标距离与到参考轨迹终点的距离分别保存。legacy SPL 沿用原 `utils/metric.py` 的参考路径长度减 20 口径，不将其冒称为基于真实最短路径的标准 SPL。

## 已核实的失败原因

| 任务前缀 | 高层动作数 | 触发规则 | 触发相机中深度灰度值 ≤1 的像素占比 | 结束时目标距离 |
|---|---:|---|---:|---:|
| `4536c7ba` | 8 | 右视深度近障阈值 | 27.203% | 114.133 m |
| `0c414bd0` | 6 | 前视深度近障阈值 | 43.234% | 260.431 m |
| `c4261e45` | 18 | 前视深度近障阈值，末状态另有 AirSim 接触标志 | 94.612% | 355.950 m |

前两条记录在 `eval.log` 中均先出现具体相机的 `collision type: close`，随后终止。`assist.py` 对任一深度图中灰度值 ≤1 的像素占比超过 10% 时设置碰撞和结束标志；这里的 1 是保存/处理后的深度图数值，不能直接表述为 1 m。前两条任务均未发出 STOP，也未触发提前停止标志，执行次数不足以达到十次持续远离或超过十五次停滞的终止条件。

第一条初始传感器日志曾出现 `has_collided=True`、碰撞对象名为空，后续及终止帧均为 False；第二条所有保存的传感器日志均为 False。因此，报告使用“深度近障判据触发”，不将其等同为两次实际撞机。第一条由右视触发，而模型只接收前视和下视，这是观测范围的限制；现有记录尚不能确定近障、深度渲染和策略行为各自的因果贡献。

第三条同样由前视深度近障规则直接终止（`eval.log:1182–1185`）；其最后一次动作后 `has_collided=True`，此前一条决策帧为 False，说明还出现了 AirSim 接触标志，但对象名为空，无法识别碰撞物体。18 次动作均没有 STOP，其中 15 次 `down<0`，11 次 `down≤−4`；NED 坐标 z 从 −7.836 到 −64.631，净上升约 56.79 m，XY 净位移仅约 13.42 m。可确认存在持续爬升、水平推进有限的行为，不能仅由此确定模型或控制器的根因。

## 核验与保存产物

`verify_results.py` 已通过：10 个唯一任务、693 条逐步动作、3,475 条传感器状态日志、7,030 张相机 PNG，所有相机文件索引与原保存频率一致；由保存坐标独立重算各项距离、路径长度和 SPL，与汇总匹配。checkpoint 清单中所有文件的 SHA256 均未变化。核验文件见[verification.json](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/verification.json)。

另使用项目原 `utils/metric.py` 读取同一结果目录复算，得到 SR=70%、OSR=70%、NE=74.88、SPL=62.03%，与新汇总一致；原始输出保存在[original_metric.log](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/original_metric.log)。核心路径修复的 `git diff --check` 通过。

运行目录：[diffusion_ckpt70000_nyc10_seed42_gpu2_20261011](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/)。

- [最终报告](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/results.md)、[逐任务 CSV](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/episode_results.csv)、[汇总 JSON](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/summary.json)。
- [逐动作记录](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/action_trace.jsonl)：fwd/down/yaw、STOP、动作前位置、模型推理耗时；[逐任务原始指标](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/episode_metrics.jsonl)。
- [图像和状态日志](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/episodes/)：每条任务各自保存五路 RGB、五路深度图、逐帧状态、对象描述及原轨迹索引；这些是闭环评估输出，专家参考轨迹仍保留在原数据目录。
- [轨迹图 PNG](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/trajectory_overview.png)、[PDF](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/trajectory_overview.pdf)：XY 俯视图，标注的目标距离按三维计算。
- [运行状态](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/status.json)、[启动命令与配置](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/launch_manifest.json)、[评估日志](../eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/eval.log)。

修复后运行使用持久会话 `aerovla_nyc10_ckpt70000_gpu2_20261011_retry`。结束时 `status.json` 为 `completed`，`scene_cleanup_success=true`；端口 40000、40001、20003 已释放，GPU2 显存回到约 21 MiB。GPU1 的训练仍在运行。

如需重新生成汇总与图像，可在 AeroVLA 根目录执行下列 CPU 命令；它们不会重新运行模型或仿真：

```bash
/data/xiyang/.conda/envs/aero_vla/bin/python eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/plot_trajectories.py
/data/xiyang/.conda/envs/aero_vla/bin/python eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/summarize_results.py
/data/xiyang/.conda/envs/aero_vla/bin/python eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011/verify_results.py
```
