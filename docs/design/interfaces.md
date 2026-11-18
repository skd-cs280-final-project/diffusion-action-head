# 现有入口与使用边界

本阶段不新增接口、不修改模型或框架，只调用现成入口。

| 用途 | 入口 | 使用方式 |
| --- | --- | --- |
| 扩散离线 | scripts/eval_diffusion_head.sh | 指定 checkpoint、已有数据/split、batch、seed 和输出 |
| 标准闭环 | scripts/eval_aerovla.sh | 通过 MODEL_DIR、TEST_JSON、SAVE_DIR 和 GPU/端口变量运行 |
| 导航指标 | utils/metric.py | 读取现有轨迹并统计 |

原 AeroVLA 只参加闭环；不新增 token 离线适配。联合 v2 一起加载 joint_adapter.pt、动作头和原资源。部署时允许使用已有路径参数，或只修正独立副本中的资源路径，保持原权重与模型设置。

不开发 prepare_variant、sample_candidates、select_action、去噪 hook、episode RNG 或统一运行器。此前占位目录留存说明，不作为开发任务。

[运行记录模板](../../evaluation/course/run_manifest.example.json)仅供手工填表，不改变程序输出。未测量指标留 null。原 token 的扩散专属字段为 null。记录现有随机数规则，不宣称 episode 级噪声配对或跨硬件逐位一致。
