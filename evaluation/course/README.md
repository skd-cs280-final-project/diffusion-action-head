# 成员 C 课程评估工作区

这里归属模型无关评估、任务清单、配对统计、成本测量和作图。当前目录提供运行记录模板，不宣称已实现新的多模型评估器。

现有离线入口为 [eval_diffusion_head.py](../../src/eval_diffusion_head.py)，专用于扩散；现有闭环入口为 [eval_aerovla.sh](../../scripts/eval_aerovla.sh)。C 在[接口约定](../../docs/design/interfaces.md)确定后接入新增模型。

复制 `run_manifest.example.json` 到本地结果目录并填写真实值；示例中的 null / planned 不是结果。完整权重、图像和日志不提交此目录。协议见[实验说明](../../docs/experiments/protocol.md)。
