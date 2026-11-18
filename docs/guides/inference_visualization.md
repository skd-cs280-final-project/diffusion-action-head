# 推理结果可视化与案例制作手册

[执行与指标手册](inference_test_handbook.md) · [三人分工](../team/README.md) · [图表交付清单](../../configs/course/results_template/visualizations.json)

所有图只来自已经核验的现有输出。先填结果表再画图，可使用 Excel、LibreOffice、Origin 等工具；不需要改推理代码。没有测量结果时保留空清单，不画假数据示意成实验结果。

## 1. 全队共用规范

固定模型顺序 M0 原 AeroVLA、M1 扩散头 best、M2 联合 step0、M3 联合 step1000、M4 联合 final。建议颜色依次为灰 #7F7F7F、蓝 #0072B2、橙 #E69F00、绿 #009E73、紫 #CC79A7，同时用文字或线型区分，不能只依赖颜色。

SR/OSR/SPL 和 LAND P/R/F1 展示为百分数，NE 为 m，yaw 为 rad 或明确转换后的 degree；图注必须注明方向、模型、划分、样本数和数据文件。柱状图从 0 起，NE 用独立面板，不与百分数共用纵轴。不同场景热力图使用同一色标。

只有一次既定推理，没有独立训练/推理重复的结果就不画标准差误差条。缺失结果显示 NA 或斜线，不能画成 0；数据不全的图明确写 partial 和各模型 n。每张图同时保存 PNG（建议 300dpi）与 PDF/SVG、作图用的 CSV/表格、图注和来源索引。路径放 analysis/figures，不放正式轨迹目录。

## 2. 必做图及分工

| 图号 | 内容 | 数据与绘制方式 | 负责人 |
| --- | --- | --- | --- |
| V01 | 五模型总体与三类划分对比 | all、seen、unseen-object、unseen-map；SR/OSR/SPL/NE 四个独立面板，模型分组柱图 | A 汇总；三人提供已核验数据 |
| V02 | 场景表现矩阵 | 20 组 category/scene × 5 模型；SR 和 SPL 各一张热力图，单元格附 n；ModularPark 合并三片为一行 | B 汇总 |
| V03 | 难度对比 | 每模型 easy/hard 的 SR、SPL 对比，分组柱图；同时列各组 n，空组 NA | C 汇总 |
| V04 | 联合检查点离线变化 | 横轴联合训练 step=0、1000…23000、23080；纵轴 noise MSE、归一化动作 MSE、LAND F1 分面板 | A 汇总三人的离线表 |
| V05 | LAND 混淆矩阵 | M1/M2/M3/M4 各 2×2，显示 TP/TN/FP/FN 原始计数，并附 P/R/F1 与正例数 | B 汇总；M0 不适用 |
| V06 | 同任务轨迹案例 | 每人至少两个 UUID，各包含五模型和参考轨迹，XY 平面等比例显示 | 各人独立完成自己的案例 |
| V07 | 案例关键帧 | 每人至少一张与 V06 对应的前视/下视关键帧板，标模型、UUID、记录帧号与结局 | 各人独立完成 |

这只是结果整理分工，不增加新的推理策略。A/B/C 各自仍完成自己的原始统计和场景案例；图表汇总人不替其他成员跑测试或审核所有日志。

每个负责场景的至少一成功/一失败案例写入案例索引；V06/V07 从中选择代表任务制作完整图，不要求给每条导航任务制作视频。

## 3. 各图的具体操作和图注

### V01：总体导航指标

从 summary 空表的真实填充值取数据，每个 category 建一张表：行 M0–M4，列 n、SR、OSR、SPL、NE。分别插入四张分组柱图，百分数轴 0–100，NE 轴单独设置。总表未完成时只展示已完成场景，不能把缺失项补为 0。

全量时每模型 all=3005、seen=1418、unseen-object=629、unseen-map=958。图注示例格式：“固定 DDPM/10 步，五模型标准策略；统计范围……；每模型 n=……；SPL/NE 沿用仓库定义；汇总为逐任务重算/已舍入分组均值加权（选实际方式）。”图中不能写尚未得到的数值。

### V02：场景热力图

行按 category 再按 scene 排序，20 行的清单来自 closed_loop_plan.json 的 source_inventory。列固定 M0–M4。SR、SPL 分图，色标均 0–100%；每个单元格显示成绩和有效 n。三份 ModularPark 先合并，不画成三种不同场景。可另画 M4−M1 的百分点差值图，但必须同时给原始成绩与 n，不能把百分点差称相对百分比提升。

### V03：easy/hard

使用同一次评测的 full/easy/hard 统计，easy 原参考长度 <=250m，hard >250m。不要再跑一套“难度实验”。先检查 full 数量等于 easy+hard，再汇总每模型的分区 SR/SPL；不同场景难度比例不同，不直接用未经配平的跨场景数字断言某地图更难。可以保留逐场景小面板帮助解释。

### V04：离线检查点变化

联合 step0、23 个周期检查点和 final 合计 25 个点；各点来自同一份完整 val 的重新离线评测。M1 头 best 是额外参考，可画水平线，不能把其 step115400 混到联合训练横轴。折线表示同一训练轨迹已保存模型的推理指标变化，不是本次新训练曲线，不配训练种子误差条。noise MSE 与动作误差分别标注，不能把最低 noise MSE 当作最佳导航成功率。

### V05：LAND 混淆矩阵

矩阵行是真值 `[继续, LAND]`，列是预测 `[继续, LAND]`，因此排列为 `[[TN, FP], [FN, TP]]`。先做原始计数版；需要行归一化时另作图并标“行百分比”，该行真值样本数为 0 时标 NA。图注注明归一化阈值 >=0、last OR penultimate 标签、原程序 zero_division=0。不能只凭总体 accuracy 高就说停止能力强。

### V06：同任务轨迹

选定规则预先固定：在自己负责场景的五模型共同完整任务中，按 UUID 排序，优先取第一条 M1 成功任务和第一条 M1 失败任务；若无该类就如实标缺，不从别的任务冒充。各场景记录候选索引，每人至少选两个代表 UUID 制作图。可增加 M1/M4 改善与退化案例，但须说明选择条件，不能只选改善。

从每个模型的 `log/*.json` 按数字帧号读取 `sensors.state.position`，参考轨迹从该结果 `ori_info.json` 指向的 `merged_data.json` 中读取 `trajectory_raw_detailed[*].position`。两者保留相同世界坐标；不要分别减去各模型起点后把真实偏移消掉。

使用 XY 散点连线图，横纵轴同单位、等比例；标实际起点、末点，参考轨迹用黑虚线，五模型用固定颜色。拥挤时改成五个并排小面板，但轴范围相同。可另画 `-z` 高度随记录序号的图，注明 NED 的 z 向下；没有可靠时间戳不能把记录序号写为秒。

参考轨迹终点只标“参考终点”，不能自动标为真实目标。目标位置经过现有环境的场景元数据匹配，未经核对不画目标圆，也不能依据距参考终点 <=20m 重新判成功。图注列 UUID、各模型结果前缀、NE/SPL 来源和异常/续跑情况；折线直线段不代表逐帧可见的实际碰撞路径。

不想手工复制坐标时，可在终端临时导出所选结果到 CSV，再用表格软件画 XY 散点连线图；这只读取已有文件，不修改推理代码。将示例路径替换为已核验的真实任务目录：

```bash
export CASE_RESULT_DIR="./eval_results/course_inference/a/formal/diffusion_head_best/seen_valset/NYCEnvironmentMegapa/填写真实结果目录名"
export CASE_CSV_DIR="./eval_results/course_inference/a/analysis/case_coordinates/填写UUID/diffusion_head_best"
python - <<'PYCOORD'
import csv, json, math, os
from pathlib import Path
source = Path(os.environ['CASE_RESULT_DIR'])
output = Path(os.environ['CASE_CSV_DIR'])
files = sorted((source / 'log').glob('*.json'), key=lambda p: int(p.stem))
assert files, 'No saved trajectory logs'
points = [json.loads(p.read_text())['sensors']['state']['position'] for p in files]
reference = Path(json.loads((source / 'ori_info.json').read_text())['ori_traj_dir'])
ref_points = [x['position'] for x in json.loads((reference / 'merged_data.json').read_text())['trajectory_raw_detailed']]
assert points and ref_points
assert all(len(p) == 3 and all(math.isfinite(float(v)) for v in p) for p in points + ref_points)
output.mkdir(parents=True, exist_ok=True)
for name, series, indices in [('predicted', points, [int(p.stem) for p in files]), ('reference', ref_points, range(len(ref_points)))]:
    target = output / (name + '.csv')
    assert not target.exists(), f'Use a new output directory: {target}'
    with target.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['record_index', 'x', 'y', 'z_down', 'height_minus_z'])
        for i, p in zip(indices, series):
            writer.writerow([i, p[0], p[1], p[2], -p[2]])
print('Coordinates exported; no model or simulator was run.')
PYCOORD
```

对五模型分别导出，检查它们的 ori_info 都对应同一轨迹。参考序号与模型序号不是对齐时间，不能把相同序号位置差当成逐时刻跟踪误差。

### V07：关键帧与可选视频

每个案例按“开始、首次明显偏离或关键事件、终止前、最后可用图像”选帧。保留 front/down 视角和帧号，截图不得剪掉影响解释的重要障碍。事件无法定位时选可复核的记录位置并说明，不编造模型注意力或因果解释。五模型关键事件不一定发生在同一序号，不把并排画面解释成时间同步。

视频只作为可选演示。已有 get_video.py 的 main 使用旧机器硬编码路径，不直接执行该 main，也无需编辑源码；可以调用现有函数：

```bash
export CASE_IMAGE_DIR="./填写真实任务目录/frontcamera"
export CASE_VIDEO_PATH="./videos/course_inference/填写成员_模型_UUID.mp4"
python - <<'PYVIDEO'
import os
from pathlib import Path
from tools.visualization.get_video import images_to_video
source = Path(os.environ['CASE_IMAGE_DIR'])
output = Path(os.environ['CASE_VIDEO_PATH'])
assert source.is_dir() and not output.exists()
output.parent.mkdir(parents=True, exist_ok=True)
assert images_to_video(str(source), str(output), fps=10)
PYVIDEO
```

fps=10 是演示播放速度，不是实测相机或控制频率。原 BEV/点云脚本有历史路径与坐标假设，本阶段不要求修改它们；普通 XY 轨迹图和关键帧已满足必做可视化。

## 4. 图表交付规则

每张图登记 figure_id、owner、source_run_ids、原始数据/作图表路径、统计分母、筛选规则、单位、导出文件和简短结论。图注先说看到的事实，再说明证据允许的解释；“动作误差更低”与“闭环导航更好”分开判断。

文件名示例 `V01_all_navigation.png`、`V02_scene_sr.pdf`、`V06_a_<uuid>_xy.png`；保存到各自 analysis/figures。图表清单初始状态是 not_run，完成后在工作副本更新，不将空模板或历史结果当作本次完成成果。

结论至少指出一个改善、一个无明显差异或退化，以及对应样本量和局限；若真实数据不支持改善就如实报告。只做单次固定策略评测，不使用“显著优于”“稳健提升”等没有统计证据支撑的措辞。
