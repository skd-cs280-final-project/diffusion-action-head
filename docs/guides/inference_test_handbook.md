# 推理测试执行与记录手册

课题：**《基于预训练扩散动作头的无人机视觉语言导航推理评测与分析》**。

本页说明每次测试如何执行、何时算完成、记录哪些指标以及如何交付；具体场景与命令从个人任务页取用。只调用现有入口，不训练、不改代码或框架，不做扩散步数、输入扰动、多候选实验。

[分工总表](../team/README.md) · [A 命令](../team/member_a_commands.md) · [B 命令](../team/member_b_commands.md) · [C 命令](../team/member_c_commands.md) · [可视化手册](inference_visualization.md)

## 1. 先辨认三种数量

| 名称 | 含义 | 本项目数量 |
| --- | --- | --- |
| 任务 episode | 一个 category/scene/trajectory，不是一条帧记录 | 正式 3,005 个，A/B/C 分别 1002/1002/1001 |
| 闭环运行包 run | 一个模型在一个场景/划分或成员分片上的完整运行 | 正式 A/B/C 分别 30/30/50 包，共 110 包 |
| 模型-任务执行 | 一个模型执行一条任务 | 五模型正式 15,025 次，预检另有 45 次 |

每人先跑三个预检任务 × 五模型，共 15 次，再运行自己的正式任务。原任务 JSON 有多条帧记录，现有 loader 按 json 指向的轨迹去重；不能把 JSON 行数当 episode 数。ModularPark 必须使用自己的冻结分片。预检写 preflight，正式写 formal，不能合并统计。

每个正式任务都比较 M0 原 AeroVLA、M1 扩散头 best、M2 联合初始化 step0、M3 联合 step1000、M4 联合 final。全部 26 个扩散状态另做全量离线；原 token 模型没有本阶段离线指标。

## 2. 开始前：逐项确认

1. 打开自己的 assignments JSON，确认 owner、run_id、model_id、task_id、test_json、expected_episodes、save_dir 和日志路径；不自行换模型或缩小正式清单。
2. 核对[资源说明](../team/resources.md)。本机已准备 resources/models 路径副本，原权重不变；原 checkpoint 里的旧绝对路径不能直接照搬。
3. 在自己的终端设置个人命令页的 PYTHON、GPU、端口、线程和路径。闭环 batch=1、maxWaypoints=200；离线 batch=2、seed42、完整 val；扩散固定 DDPM/100/10、H=1、E=1、history=0。
4. 确认输出目录是本次运行目录。若已存在，先按第 6 节区分续跑与独立新运行；不能直接覆盖日志。
5. 记录代码版本、实际命令、输入清单 hash 和模型版本。模型文件 hash 可对原文件计算一次供三人共享，不必每个场景重算；尚未计算填 null，不能拿目录名当 hash。
6. 确认 GPU 与端口当前可用，再启动自己独立的场景管理服务。每张卡同一时刻只跑本人一个离线或闭环任务，不同时启动两套模型。

以下仅记录环境，不加载模型或启动仿真。从仓库根目录运行，记录目录放在日志区，不放轨迹目录：

```bash
mkdir -p logs/course_inference/a/records
# B/C 将 a 改为自己的成员目录
{
  date -Iseconds
  git rev-parse HEAD
  git status --short
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
  "$PYTHON" -c 'import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda)'
} > logs/course_inference/a/records/environment.txt 2>&1
```

工作区若有未提交修改，记录状态并保存对应差异说明；commit SHA 不能独自代表未提交的工作树。环境记录不要加入访问令牌、完整环境变量转储等无关内容。

## 3. 预检：确认可以跑，不用来选模型

按个人命令页顺序：先 dry-run，再三任务清单上的 M0–M4。dry-run 只打印命令，即使退出码为 0 也不表示权重加载或渲染成功。

每个模型预检都检查：

- 控制台实际加载的模型路径与 run_id 对应；联合 v2 使用配套 joint_adapter.pt，不只加载动作头。
- 场景启动、前视/下视图像获取、控制与结束保存正常；没有持续空图、资源路径异常或 OOM。
- 请求三条任务，各有完整可读的结果；成功和导航失败都算已执行，程序异常中断不算完成。
- 对应 metrics 日志可读，Total Count 与核验的任务数一致；空 easy/hard 组允许“不适用”，但不能把读取异常当作空组。
- 记录开始/结束时刻、退出状态和异常。三任务结果不用于挑选最优 checkpoint，也不替代正式测试。

任一模型预检未通过，先处理资源或运行环境问题并记录，不能只跳过它继续报告四模型正式对比。需要改推理代码的问题超出本阶段范围，应明确反馈，不能静默改算法。

## 4. 每个正式运行包的操作顺序

1. 从个人命令页复制对应场景/划分和模型的命令。先做主场景，再其他完整场景，最后 ModularPark 分片；每行任务依次 M0–M4。
2. 在[运行记录模板](../../evaluation/course/run_manifest.example.json)的副本中填 run_id、模型、输入清单、路径、实际命令和开始时刻，status 置 running。用独立记录文件，不修改仓库里的空模板。
3. 执行已有 `scripts/eval_aerovla.sh`。另一终端查看本人控制台日志和场景日志；程序仍在运行时不生成最终成绩。
4. 运行结束后记录退出状态、结束时刻和是否中断。闭环模型推理只读；不运行含参数更新的 smoke/train 脚本。
5. 先按第 5 节查重、查漏、检查结果结构，再执行命令页中的 `utils/metric.py`。
6. 将 full/easy/hard 的 Total Count、SR、OSR、NE、SPL 填入个人结果表，保留原 metrics 日志；完成数不足标 partial，不能标 complete。
7. 所有五模型都完成后，才对该场景做模型间对比和选取案例。未齐全时保留缺失项，不补 0 冒充结果。

结果目录只放现有程序保存的轨迹。环境、计时、汇总表、图片和视频放在 `logs/course_inference/<owner>/records/` 或 `eval_results/course_inference/<owner>/analysis/`，不要直接放进 formal/.../<scene>/；现有指标脚本会遍历其中条目。

## 5. 完整性核验：先检查任务，再计算成绩

对每个 run 使用自己的 TEST_JSON 去重后得到 expected 任务集合，以 category + scene + UUID 作为身份。同一 UUID 在不同划分中分别登记，不只用裸 UUID 汇总全项目。

现有结果目录通常是 `success_<uuid>`、`oracle_<uuid>` 或 `<uuid>`。去掉开头的 success_/oracle_ 后对照任务表。不要用模糊字符串包含匹配去混淆两个任务。

| 检查项 | 合格条件 | 不合格时记录 |
| --- | --- | --- |
| 任务覆盖 | 每个预期任务恰好有一份被接受的结果 | missing、duplicate、unexpected 分别列出 |
| ori_info.json | 可解析，引用正确的原轨迹，merged_data.json 可读 | invalid_reference，不计算距离指标 |
| log/*.json | 非空、编号连续且可解析，位置坐标有限；检查首尾与异常文件 | partial/corrupt，不当成导航失败 |
| 结束与保存 | 对应任务的结束/保存记录和完整目录相符，运行无未处理异常 | 不能仅凭目录存在认定完成 |
| 图像证据 | 案例需要的 frontcamera/downcamera 图像可读 | 记录缺失，不伪造画面；是否影响指标另查日志 |
| 总数 | expected = accepted_complete + missing_or_invalid，重复不得重复计数 | 报告完成率与异常清单 |

图像采集和保存频率可能不同于状态记录；不要要求图片数量等于 log 数量。一次控制可保存多个状态点，log 文件数也不是“模型调用次数”或“控制决策步数”。

completed 包括正常结束的成功与失败任务；缺失、加载失败、OOM、RPC 中断和残缺保存单独记为运行异常。最终成绩要求完成整个冻结清单。若报告阶段性结果，明确完成比例，并在模型比较时注明任务集合是否一致；共同完成子集只能作为标明范围的临时对照。

## 6. 中断、续跑和异常处理

现有 loader 会按输出目录中已有任务名跳过任务，目录存在不等于该任务保存完整。中断后按以下顺序处理：

1. 保留原控制台日志、出错时间、最后任务和原结果，不覆盖证据。
2. 完整成功、完整导航失败都保留，不因失败重跑挑更好的结果。
3. 将损坏或未完整保存的条目记录到异常表，移到指标目录之外的 quarantine 位置后再恢复缺失任务；操作前核对具体目录，不能删除整个场景结果。
4. 同一清单继续运行时保留已接受任务，新的控制台日志使用 attempt02 等独立名称。也可在独立恢复目录仅运行明确的缺失清单，保留恢复清单与来源，不改变任务内容。
5. 合并时每个 category/scene/UUID 只接受一份有效结果，记录采用哪个 attempt 和理由，绝不能按结果优劣选择。
6. 续跑不恢复 episode 级 RNG。即使默认 seed 相同，后续随机序列也可能不同；resumed=true、rng_restored=false，并在报告中说明。

终止日志只用于解释已有实现的事件：Global Stuck、距离持续增大、深度近障或返回状态不足都可能进入 collision 标志。故应记录 `legacy_collision_or_blockage`，不能统一称“物理碰撞”。Early End、Timeout 需要对应任务日志证据，不能只看没有 success_ 前缀就猜原因。证据不足写 unknown。

## 7. 闭环必须记录的指标与口径

现有指标来源为 [utils/metric.py](../../utils/metric.py)，结束标志来自 [closeloop_util.py](../../src/vlnce_src/closeloop_util.py) 和 [env_uav.py](../../src/vlnce_src/env_uav.py)。沿用原实现，不重定义评分规则。

| 字段 | 定义与读取方式 | 单位/方向 |
| --- | --- | --- |
| expected / completed / missing | 冻结清单、核验的完整任务与未完成任务 | 个；completed 含导航失败 |
| success_count | 已接受且带 success_ 的任务数 | 个 |
| oracle_success_count | success_ 与 oracle_ 两类总和，不只是 oracle_ 数量 | 个；应 >= success_count |
| SR | success_count / 有效完整任务数 | %，越高越好 |
| OSR | oracle_success_count / 有效完整任务数 | %，越高越好 |
| SPL | 现有成功与路径效率指标，失败贡献 0 | %，越高越好 |
| NE | 预测最后位置到原参考轨迹最后位置的欧氏距离，所有有效任务均计入 | m，越低越好；不是归一化误差或目标距离 |
| metric_count / spl_count / ne_count | 脚本统计的分母及各指标可用样本数 | 个；异常时不能默认相等 |
| full / easy / hard | 同一次运行的结果分组；原参考轨迹长度 <=250m 为 easy，>250m 为 hard | 非额外实验 |
| status / attempts / error_note | complete、partial、failed；尝试记录和异常原因 | 无结果留 null |

SR 包含停止和既有终止逻辑：原代码在目标距离 <=20 且满足停止/early_end 规则时判成功。OSR 使用现有环境在动作后状态检查到目标邻域的标记（阈值 20），不要求最终停下；它不是重新扫描所有连续飞行点得到的无限频率 oracle。直接保留程序结果，不用 NE 阈值重新判断成功。

SPL 的实际口径：原始参考轨迹三维折线长度记 L，代码使用 l=L-20；实际轨迹折线长度记 p，成功任务贡献 max(l/max(l,p),0)，失败为 0，最后均值 ×100。遇到非有限值保留异常，不自行改公式。easy/hard 划分使用未减 20 的 L。

full 是所有任务，easy/hard 是分区；汇总不能把 full+easy+hard 相加重复计算。原脚本 NE 日志叫 Average Normalized Error，但报告统一标“参考终点距离 NE (m)”。若要真实目标距离，需核对环境最终选定目标，不能直接使用 mark.json 原位置或参考终点替代；本阶段不新增该指标要求。

## 8. 离线执行与必记指标

按个人命令页运行分配的 8/9/9 个扩散状态。全部使用相同完整 val、batch=2、seed42、10 步推理，不能加 --max-samples 作为正式测试。当前固定数据的全 val 是 41,176 帧/761 条轨迹；最终以 JSON 的 split 与实际 num_samples 核对。没有统一 token 离线入口，因此 M0 各项填“不适用”，不填 0。

现有 JSON 的指标位于顶层 metrics 下。程序成功时写正式 JSON 并输出 Evaluation report；只有 .tmp 文件或尾部截断日志不能视为成功完成。

| JSON 字段（均在 metrics 下） | 记录内容 | 注意事项 |
| --- | --- | --- |
| num_samples / num_valid_samples | 评估帧数、含有效动作坐标的帧数 | 不同模型应使用相同样本集合 |
| num_valid_coordinates / num_valid_continuous_coordinates | 全四维与前三维的有效坐标数 | 加权分母；有 mask 时不机械当成帧数乘维度 |
| noise_mse | 噪声预测 MSE | 无量纲、越低越好，仅扩散内比较，不是训练 loss 曲线 |
| normalized_continuous_mse | 裁剪/归一化后的 fwd/down/yaw 动作 MSE，不含 LAND | 无量纲、越低越好 |
| physical.fwd.mae / rmse / count / unit | 前向动作误差 | 原距离单位；本数据通常为米，保留原 unit 字段 |
| physical.down.mae / rmse / count / unit | 垂直动作误差 | 原距离单位，不与 yaw 直接求和 |
| physical.yaw.mae / rmse / count / unit | 偏航动作误差 | radians；转成度展示时注明乘 180/pi，保留原值 |
| land.tp / tn / fp / fn | LAND 混淆统计 | 正类为需要停止，真值 last OR penultimate |
| land.precision / recall / f1 / accuracy | LAND 分类指标 | 原 JSON 是 0–1，制图转 % 时乘 100，不能再重复转换 |
| land.support_positive / support_negative / predicted_positive | 正/负例与预测正例数 | 与混淆矩阵一起记录 |
| land.threshold_normalized / zero_division | 阈值 0、零分母返回规则 | 归一化第四维 >=0 判 LAND |
| seed / num_inference_steps / sample_actions | 运行参数核验 | 42 / 10 / true |

同时保留顶层 split、split_provenance、runtime、target_semantics、checkpoint 和数据路径。数据 provenance 不一致时先查来源，不跳过校验。动作标签按原范围裁剪，因此物理误差不是未裁剪的原始控制误差。

当没有正例或没有正预测时，程序按 zero_division=0 产生数值 0。保留原 JSON 数值，展示时同时标注支持数和“零分母按 0 处理”；无正例不能据此评价停止能力。不要人为补造正例、删样本或将 undefined 解释成可靠的差表现。

现有离线 JSON 不保存逐帧预测和置信度，所以本阶段不要求误差直方图、预测-真值散点、校准曲线、逐样本动作分布图或去噪内部轨迹。不能由 MAE/RMSE 反推这些数据。

## 9. 需要保存的记录文件

| 层次 | 记录方式 | 内容 |
| --- | --- | --- |
| 每个 run | [run_manifest 模板](../../evaluation/course/run_manifest.example.json)的独立副本 | 模型/数据/命令、时间、资源、状态、完整性、原始结果与日志路径 |
| 逐任务 | [episode_record 示例](../../evaluation/course/episode_record.example.json) | UUID、采纳 attempt、完整性、程序结局、证据路径；优先保证全量覆盖索引，详细人工说明用于案例/异常 |
| 离线总表 | [offline 空表](../../configs/course/results_template/offline.json) | 26 个扩散状态的原始指标和来源路径 |
| 闭环成员表 | configs/course/results_template/a.json、b.json、c.json | 每 run 的 full/easy/hard，预填 expected，未运行指标为 null |
| 总表 | configs/course/results_template/summary.json | 五模型在 all/seen/unseen-object/unseen-map 下汇总 |
| 案例与图片 | [case_record 示例](../../evaluation/course/case_record.example.json)、[可视化要求](inference_visualization.md) | 任务、选择规则、五模型结果、轨迹/关键帧来源与解释 |

将模板副本填在本地 analysis/records 下；仓库空模板保持空。逐任务索引可从已有目录与 ori_info 汇总，JSON/CSV/表格软件均可，不要求手工写 15,025 份长篇 JSON。episode_record 定义的是应保留的字段，不是要求新增程序日志。

运行时间记录开始、结束、wall_seconds 与运行条件即可。若只观察 nvidia-smi 的某几个采样点，应叫“观测到的显存占用”，不能叫精确峰值；它还可能含同卡渲染。没有额外测量就把 gpu_hours、simulator_hours、单动作延迟/P95 留空，不用总时间除日志帧数伪造模型延迟。失败提前终止导致用时更短，也不代表策略更高效。

## 10. 合并与复核

1. 合并逐任务索引，先查 category/scene/UUID 是否重复和遗漏，再对同一模型汇总。
2. SR=总 success_count / 总 metric_count；OSR 同理。不能等权平均场景百分比。
3. SPL/NE 按对应有效样本数加权。原 metrics 日志只显示两位小数，用这些数加权得到的是近似汇总，必须标“由已舍入分组均值加权”；不能宣称逐任务精确重算。
4. 若需要与原脚本一致的整体精度，可在独立分析目录创建只含已核验轨迹的符号链接视图后调用同一 utils/metric.py。链接名须保留 success_/oracle_ 前缀，并加 category/scene 避免重名；ori_info 保持原内容。视图只能含已核验轨迹，不能包含原场景目录或控制台日志。该操作只整理结果文件，不改指标代码。
5. 三份 ModularPark 只合并一次；full 与 easy/hard 分开汇总。每个模型正式 expected=3005，其中 seen=1418、unseen-object=629、unseen-map=958。
6. 离线每个 checkpoint 是一行，不能把不同 checkpoint 的 MSE 平均成“一个模型成绩”。同一次联合训练的检查点也不是独立重复，不能画成训练随机种子的误差条。
7. 每人交叉检查另一人的一个完整运行包：输入清单、5 模型结果身份、原始日志、表格与图是否一致，记录复核人和日期。

## 11. 最终验收清单

- 自己的所有正式运行完成或明确列出阻塞，预检与正式完全分开。
- 每个模型的请求/完成/缺失数量可追溯，导航失败未被删去，程序异常未被冒充正常失败。
- 已填离线表、闭环表、异常/续跑记录；所有成绩有原始 JSON 或 metrics 日志路径。
- 按可视化手册完成规定图表、同任务案例和图注，原数据/作图表格与导出图一起保留。
- 提交个人场景分析：模型差异、easy/hard 与场景划分影响、成功/失败案例、数据与方法局限。
- 未运行留 null/not_run；没有真实结果前不生成填有虚构数值的成绩图。报告中的计划次数与完成次数分开写。
