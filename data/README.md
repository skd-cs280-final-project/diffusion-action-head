# 数据资源

Git 保留 `continuous_split_seed42.json`、`uav_dataset/` 闭环任务与 `meta/` 场景元数据。帧级训练清单 `aerovla_train_dataset.json` 和原始 TravelUAV 图像/日志需要另行准备，不随代码发布。

训练清单、轨迹 split、闭环 episode 是三种不同格式，不能互换。当前帧级清单有 train 369,274 帧/6,850 轨迹，val 41,176 帧/761 轨迹；程序校验覆盖、重复与来源。迁移到其他数据集时需要对应的新 split，并记录协议变更。

默认原始数据布局为 `dataset_raw/<map>/<trajectory>/frontcamera|downcamera/<frame>.png`。历史状态需要同轨迹 `log/`，闭环指标会引用原始 `merged_data.json`。本阶段使用相同的固定观测进行推理比较；B 负责开发/最终任务清单与分片，以及视觉和语言扰动。不得根据结果更换任务或将扰动作用到真值标签；不构造多步训练监督。

准备步骤见[运行指南](../docs/guides/training_and_evaluation.md)。
