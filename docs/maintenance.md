# 项目重组记录

日期：2026-11-18。项目改为 Diffusion Policy 与动作扩散头课程研究，固定 DDPM/100/10；不包含扩散步数实验。

## 初次重组记录（后续已由 v2 推理计划替代）

- 保留全部现有训练、模型、数据、仿真、测试代码和历史扩展；未启动新训练或仿真。两份新增测试文件仅清除末尾多余空行，AST 未变。
- 当前设计、分工、接口和实验规格独立于历史结果，A1–A3/B1–B4 明确待实现。
- 原根目录可视化脚本移至 tools/visualization，第三方 RPC 包移至 third_party，执行逻辑与包内容不变。
- 原实验记录、详细设计和变更历史移至 docs/archive；图表及必要指标快照保留在 docs/reports。
- 删除重复进度页、旧发布说明、未采用的安全/分层规划提案、上游宣传素材和冗长控制台日志；绘图缓存清理。
- 数据、模型、仿真环境、结果目录和本地缓存仍不上传。

## 文件迁移

| 原位置 | 新位置 |
| --- | --- |
| `docs/development/changelog.md` | `docs/archive/development/changelog.md` |
| `docs/experiments/nyc10_evaluation_20261011.md` | `docs/archive/experiments/nyc10_evaluation_20261011.md` |
| `docs/experiments/nyc_joint_final_seen281_20261104.md` | `docs/archive/experiments/nyc_joint_final_seen281_20261104.md` |
| `docs/experiments/weekly_report_20261011.md` | `docs/archive/experiments/weekly_report_20261011.md` |
| `docs/experiments/nyc_full_best_gpu3_20261012.md` | `docs/archive/experiments/nyc_full_best_gpu3_20261012.md` |
| `docs/experiments/full_training_20261010.md` | `docs/archive/experiments/full_training_20261010.md` |
| `docs/experiments/joint_lora_20261103.md` | `docs/archive/experiments/joint_lora_20261103.md` |
| `docs/research/history_pose_20261012.md` | `docs/archive/research/history_pose_20261012.md` |
| `docs/design/diffusion_action_head.md` | `docs/archive/design/diffusion_action_head.md` |
| `get_video.py` | `tools/visualization/get_video.py` |
| `get_2pcd_from_depth.py` | `tools/visualization/get_2pcd_from_depth.py` |
| `visual_map_traj.py` | `tools/visualization/visual_map_traj.py` |
| `msgpack-rpc-python-fix-msgpack-dep.zip` | `third_party/msgpack-rpc-python-fix-msgpack-dep.zip` |

## 移除项

- `docs/development/github_snapshot_20261012.md`
- `docs/experiments/joint_lora_progress_20261103.md`
- `docs/research/collision_mitigation_20261012.md`
- `docs/research/hierarchical_planner_20261012.md`
- `docs/assets/pipeline.png`
- `docs/assets/star-history.png`
- `docs/assets/teaser_figure.png`
- `docs/assets/AerialVLA_Demo_2.gif`
- `docs/assets/.gitkeep`
- `docs/reports/joint_lora_20261103/training.log`

`index.html` 在本次整理前已被删除，本次保持删除状态。已提交的旧版本可由 Git 历史查找；清理前完整工作区代码和文档另有本地压缩备份（不上传）。

## 验证

运行 `python scripts/check_course_repository.py` 检查实验协议、主文档链接、Shell 示例和 Python 语法。历史归档的本地检查点/原始结果链接单独统计，不把不存在的历史产物视为当前交付。回归测试命令见运行指南。实际本次结果记录在 [验证摘要](validation.json)。

发布保留上游 Git 历史和 LICENSE；使用独立发布 checkout 形成提交，避免覆盖原工作区已有 origin/github 远程。仅推送用户指定的 diffusion-head 仓库，不改写旧仓库。

## 同日修订：三人均衡的预训练推理计划 v2

使用全部已保存正式模型和初始化对照，共登记 27 项；每人 9 项共同离线评估，共同闭环按任务三等分并对每份运行五个代表模型。A/B/C 分别负责生成随机性、条件鲁棒性、候选选择与成本，独立完成自己变体的闭环、图表与写作。

移除 v1 的重新训练架构/序列实验规格和对应空占位目录；新增 `models.json`、shared/c 规格、`evaluation/course/generation|conditions|selection|shared` 与 `scripts/course_infer/a|b|c` 的任务说明。当前入口扩展仍待实现，未启动任何新训练或仿真，也未修改已有模型实现或权重。固定 DDPM/100/10 与 H=1、E=1。

## 后续收缩：只运行已有推理入口

根据用户要求撤销生成过程插桩、条件扰动、多候选选择、适配器与新运行器开发。三人改为同类型模型测试、闭环分片和报告。26 个扩散状态做已有离线评估，原 token 仅做已有闭环；任务编号保留为执行清单。只更新 Markdown 与任务/记录 JSON，代码、框架目录、权重、数据和既有脚本保持不变。此前占位目录仅标明停用，不删改结构。当前未启动正式测试。

## 详细场景与命令分工

课题定名《基于预训练扩散动作头的无人机视觉语言导航推理评测与分析》。新增三人明确场景、参数、模型与可复制命令、冻结任务 JSON 和空结果表。20 组原始场景/划分共 3005 条任务，完整覆盖且无重复，五模型正式 15025 次。仅本地准备资源路径副本与权重链接；原配置、权重和已有代码保持不变。没有启动模型加载、场景或正式实验。
