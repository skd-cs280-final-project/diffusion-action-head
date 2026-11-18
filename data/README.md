# 数据资源

Git 保留 `continuous_split_seed42.json`、`uav_dataset/` 闭环任务与 `meta/` 场景元数据。帧级训练清单 `aerovla_train_dataset.json` 和原始 TravelUAV 图像/日志需要另行准备，不随代码发布。

训练清单、轨迹 split、闭环 episode 是三种不同格式，不能互换。当前帧级清单有 train 369,274 帧/6,850 轨迹，val 41,176 帧/761 轨迹；程序校验覆盖、重复与来源。迁移到其他数据集时需要对应的新 split，并记录协议变更。

默认原始数据布局为 `dataset_raw/<map>/<trajectory>/frontcamera|downcamera/<frame>.png`。历史状态需要同轨迹 `log/`，闭环指标会引用原始 `merged_data.json`。本阶段沿用已有标注和任务，只准备三人的不重叠闭环分片清单，经现有 TEST_JSON 参数运行；不生成训练监督、不添加输入扰动。每份清单内所有代表模型使用相同任务和顺序。

准备步骤见[运行指南](../docs/guides/training_and_evaluation.md)。
