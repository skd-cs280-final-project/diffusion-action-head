# 模型权重与仿真环境下载

[资源准备](../team/resources.md) · [三人分工](../team/README.md) · [推理测试手册](inference_test_handbook.md)

下载入口：[GitHub Release：inference-assets-v1](https://github.com/skd-cs280-final-project/diffusion-action-head/releases/tag/inference-assets-v1)。这些附件独立于 Git 历史，`git clone` 和 GitHub 的 Source code 压缩包都不包含权重或场景。

## 下载内容

| 资源 | 包含内容 | 用途 |
|---|---|---|
| 基座 `model-openvla-7b` | 本项目使用的完整 OpenVLA 7B、tokenizer、配置和自定义模型文件 | 所有模型共同依赖 |
| 五个核心模型包 | 原 AeroVLA adapter、diffusion head best、joint init、joint step 1000、joint final | 五模型闭环对比 |
| 其余 22 个检查点 | joint step 2000–23000，每 1000 步一个 | 既定离线评测；只下载自己负责的版本 |
| 场景资源 | 12 个任务场景、启动脚本和运行依赖 | 三人分别下载自己负责的场景 |
| `tasks-a/b/c` | 所负责轨迹的 merged_data.json、mark.json、object_description.json | 指令、初始状态、目标、参考轨迹与指标计算 |
| `offline-annotations` | 原帧级标注 JSON | 离线评测；验证集图片仍需另行准备 |

权重保持原参数和配置内容，**解压后仍需修正本机模型路径**。`joint_init` 是第 0 步对照，目录名 `best` 不代表联合训练最优模型。所有 27 项登记模型均提供；其中原 AeroVLA 仅参加闭环评测，另 26 项可参加现有离线评测。

场景包来自当前项目实际使用的 `envs/`，对应 Travel_env 的同名目录。仅排除不参与本次任务的场景、`Saved/` 运行状态与日志、`.debug` 和 `.sym` 调试符号，保留可执行程序、`.so`、`.pak`、启动脚本及运行配置。共享 `Engine` 单独打包。ModernCityMap 的资源目录实际为 ModernCityDowntown，沿用原启动脚本。

每个附件最多 1 GiB，是一个 `.tar.zst` 压缩包的连续分卷，**单个 part 不能独立解压**。`assets-manifest.json` 记录包归属、文件清单、分卷顺序、字节数与 SHA-256；`SHA256SUMS` 可供手动校验。共 45 个资源包、93 个分卷；原文件约 113.30 GiB，压缩后约 65.12 GiB。发布前已核对全部压缩包的完整性、文件清单、大小和权限，并将全部分卷与 GitHub 返回的 SHA-256 对比。

| 默认闭环下载 | 压缩分卷 | 解压文件 | 两者合计，尚未计额外余量 |
|---|---:|---:|---:|
| A | 25.62 GiB | 35.43 GiB | 61.05 GiB |
| B | 24.70 GiB | 41.19 GiB | 65.89 GiB |
| C | 33.94 GiB | 62.19 GiB | 96.13 GiB |

## 按个人分工下载

在 Linux 的项目根目录执行。需要 Python 3、curl、tar、zstd；压缩包还原的是 Linux 仿真程序。Debian/Ubuntu 可用 `sudo apt-get install curl zstd` 安装下载与解压工具。GPU 驱动、Python 环境及项目依赖仍按运行指南准备。

先查看 A 成员的闭环下载清单和空间需求：

```bash
python scripts/download_release_assets.py --member A
```

确认使用新克隆的项目目录后下载、校验并解压：

```bash
python scripts/download_release_assets.py --member A --download --extract
```

B、C 成员分别把 `A` 替换为 `B`、`C`。默认只准备闭环评测所需资源：共享基座、五模型、个人场景、个人任务元数据。

| 成员 | closeloop 场景 | Town 场景 |
|---|---|---|
| A | NYCEnvironmentMegapa、ModernCityMap、ModularPark | Town07 |
| B | TropicalIsland、ModularPark | Town04、Town10HD |
| C | NewYorkCity、ModularPark | Town01、Town05、Town06、Town15 |

下载程序会为每个人加入 closeloop 共享 Engine。需要原计划中的离线检查点和标注时：

```bash
python scripts/download_release_assets.py --member A --scope all --download --extract
```

`--scope all` **不包含验证集图像**；运行离线指标前，仍需准备原验证集前视/下视图片，并保持原标注和数据划分不变。仅进行闭环评测则不需要这些图片。

集中准备全组资源：

```bash
python scripts/download_release_assets.py --member all --scope all --download --extract
```

## 存储、重试与已有文件

- 默认下载缓存：`eval_results/release_downloads/`。可用 `--cache /另一个磁盘/下载缓存` 修改。
- 默认解压位置：当前目录。可用 `--destination /新克隆的项目目录` 修改。
- 需要同时容纳压缩分卷和解压资源，脚本预览会分别显示两者大小；额外留出文件系统余量。
- 网络中断后重新执行原命令；`.partial` 文件续传，已完成分卷先校验 SHA-256，再决定是否跳过。
- 校验失败会停止，不使用损坏资源。按报错路径移走损坏文件后重试。
- 解压时拒绝覆盖已有资源。请使用新克隆的目录，避免覆盖现有实验权重或数据。正常完成的解压有本地记录，重复执行会跳过；解压过程本身中断时，请在干净目录恢复。
- 下载器只负责资源部署，不启动仿真、推理或训练，不改变模型结构和评测参数。

## 解压后还需要做什么

目录布局如下：

```text
项目根目录/
├── openvla-7b/
├── checkpoints/
│   ├── aero_vla/
│   ├── best/
│   └── joint_lora_stage2_20261103/
├── envs/
│   ├── closeloop_envs/
│   └── carla_town_envs/
└── dataset_raw/
    └── 场景/轨迹 UUID/{merged_data.json,mark.json,object_description.json}
```

1. 按[资源准备](../team/resources.md)运行模型路径准备代码；下载程序不会改写原 diffusion_config.json。
2. 按个人命令页设置 Python 路径、GPU 和端口，保持 `MODEL_BASE="$PWD/openvla-7b"`、`DATASET_PATH="$PWD/dataset_raw"`，场景服务使用 `--root_path "$PWD/envs"`。
3. 完成三任务 × 五模型预检后，再开展正式闭环评测。

发布文件的完整性检查不等于 GPU 加载或场景渲染验证；首次部署仍需执行预检。资源来自本项目已有检查点、OpenVLA 和 TravelUAV/Unreal/AirSim 环境，沿用原资源各自的许可与说明，代码仓库许可不替代第三方资源许可。
