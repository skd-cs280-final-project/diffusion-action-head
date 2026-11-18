# 运行资源与路径准备

[分工总表](README.md) · [运行指南](../guides/training_and_evaluation.md)

模型权重、场景与闭环任务元数据可按成员分工从 [GitHub Release 下载](../guides/resource_download.md)。附件独立于 Git 历史，克隆仓库不会自动下载大文件。

## 当前本机已完成

已核对 20 组原任务清单对应的 3,005 条轨迹，其 merged_data.json、mark.json、object_description.json 均存在；12 个场景的启动脚本和对应二进制存在。ModernCityMap 的真实二进制目录名是 ModernCityDowntown，使用现有启动脚本即可，不改场景映射。

本机有四张 24GB RTX 4090；三人分配 GPU 0/1/2，GPU 3 备用。这些是静态资源检查，尚未启动场景、加载全部模型或确认显存足够。

旧 diffusion_config.json 中有不存在的绝对路径，因此已经为 27 项模型准备 `eval_results/course_inference/resources/models/<model_id>/`：文件链接到原始文件，只有 diffusion_config.json 是独立副本，并且仅修正 base_model_path 和 adapter_path。未修改原配置、模型参数、代码或框架。详细本机记录在 `eval_results/course_inference/resources/preparation.json`，不随 Git 上传。

三人的完整命令均使用这个已准备目录。MODEL_BASE 不能覆盖 diffusion_config.json 内的原 adapter 路径，所以不能只设置 MODEL_BASE 而跳过资源准备。

## 换机器时

闭环评测先准备完整基座、五模型所需权重、个人场景和个人轨迹元数据；无需下载 TravelUAV 原始摄像头图像。离线评测还需自己负责的检查点、原帧级标注 JSON、数据划分和验证集前视/下视图片。Release 默认下载闭环资源，`--scope all` 增加个人离线检查点及标注，但不含验证集图片。解释器路径按本机修改，模型和数据布局保持仓库指南要求。以下是一次性资源文件准备命令，不创建新的推理脚本，也不改变模型结构；当前机器已完成，重复执行时核对已有副本而不覆盖。

```bash
python - <<'PYRESOURCE'
import json
from pathlib import Path
root = Path.cwd()
base = (root / 'openvla-7b').resolve()
adapter = (root / 'checkpoints/aero_vla').resolve()
assert base.is_dir() and adapter.is_dir()
registry = json.loads((root / 'configs/course/models.json').read_text())
for model in registry['models']:
    if model['id'] in registry['core_model_ids']:
        source = root / model['path']
        assert all((source / name).is_file() for name in model['required_files']), model['id']
for model in registry['models']:
    source = root / model['path']
    if not all((source / name).is_file() for name in model['required_files']):
        print('Not downloaded; skipping optional checkpoint:', model['id'])
        continue
    target = root / 'eval_results/course_inference/resources/models' / model['id']
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.name != 'diffusion_config.json':
            output = target / item.name
            if not output.exists() and not output.is_symlink():
                output.symlink_to(item.resolve())
            else:
                assert output.resolve() == item.resolve(), output
    if model['family'] == 'diffusion':
        config = json.loads((source / 'diffusion_config.json').read_text())
        config['base_model_path'] = str(base)
        config['adapter_path'] = str(adapter)
        output = target / 'diffusion_config.json'
        assert not output.is_symlink(), output
        if output.exists():
            assert json.loads(output.read_text()) == config, output
        else:
            output.write_text(json.dumps(config, indent=2) + '\n')
print('Resource views ready; model inference has not been run.')
PYRESOURCE
```

该目录依赖原文件，不能仅打包符号链接发送给另一人而缺少模型本体。Release 提供实际文件，原权重不进入 Git 历史。

## 预检与故障记录

每人先完成个人命令页的三任务 × 五模型预检；遇到文件、依赖、端口或显存问题时记录原因，修正部署资源或调整运行时间。不能因结果差而替换模型/任务。需要推理代码变更的问题不在本次范围，不静默打补丁。

场景管理服务会管理自己的 RPC 端口；A/B/C 使用独立管理端口及不重叠的 RPC 范围。不得把两人的运行连接到同一服务。单卡显存不够时优先错开任务，或安排空闲 GPU 3 承担某人的渲染并同步修改该人的 --gpus 和 SIMULATOR_GPU_ID；模型与场景硬件变化如实登记，不改变科学参数。

已有 loader 会跳过输出目录中已存在的任务目录；程序中断时有目录不代表轨迹完整。保留中断日志，查缺失记录，使用独立恢复目录/清单记录补跑，不将残缺轨迹算成功完成。不要在轨迹目录放汇总文本、图片或日志。
