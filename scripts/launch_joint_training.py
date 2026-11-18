"""Supervise a joint training command and refresh a local Markdown report.

Run this script in tmux. Everything after -- is passed to train_diffusion_joint.sh.
The trainer owns its checkpoint directory; this process writes only log/report files.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_metrics(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines()[-51:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def write_report(report, output, log, command, returncode, pid):
    config = read_json(output / "run_config.json")
    status = read_json(output / "status.json")
    metrics = read_metrics(output / "metrics.jsonl")
    current = metrics[-1] if metrics else {}
    completed = returncode == 0 and status.get("status") == "completed"
    phase = "完成" if completed else ("失败／中断" if returncode is not None else "运行中")
    step, total = current.get("step", 0), current.get("total_steps")
    progress = f"{step} / {total}（{step / total:.2%}）" if total else f"{step}（加载或初始验证阶段）"
    timing = "等待足够的已完成训练步"
    if len(metrics) >= 2:
        first = metrics[0]
        seconds = (current["elapsed_seconds"] - first["elapsed_seconds"]) / (step - first["step"])
        timing = f"最近 {step - first['step']} 步平均 {seconds:.2f} 秒/步"
        if total and not completed:
            timing += f"；剩余约 {(total-step)*seconds/3600:.1f} 小时（验证/保存耗时会使估算波动）"
    lines = ["# AeroVLA 联合 LoRA 训练实时进度", "",
             "此文件由监督进程自动更新；配置与验证说明见 [训练记录](joint_lora_20261103.md)。", "",
             f"更新时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}", "",
             f"- 状态：{phase}；子进程 PID：{pid}；退出码：{returncode}",
             f"- GPU：{os.environ.get('CUDA_VISIBLE_DEVICES', '未指定')}；进程数：{config.get('world_size', '加载中')}",
             f"- optimizer step：{progress}",
             f"- 速度：{timing}",
             f"- 最近 loss：{current.get('loss', '待生成')}",
             f"- rank 0 峰值分配显存：{current.get('peak_memory_allocated_gib', '待生成')} GiB",
             f"- 输出目录：`{output}`", f"- 日志：`{log}`", ""]
    best = read_json(output / "best_metrics.json")
    if best:
        lines += [f"固定 val 子集最佳 noise MSE：{best['noise_mse']:.8f}（step {best['step']}，{best['num_samples']} 帧）。",
                  "这是离线动作指标，不代表闭环导航成功率。", ""]
    if status.get("error"):
        lines += [f"失败信息：`{status['error']}`", ""]
    lines += ["实际启动命令：", "", "```bash", shlex.join(command), "```", ""]
    temporary = report.with_name(report.name + ".tmp")
    temporary.write_text("\n".join(lines))
    os.replace(temporary, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output, log, report = (Path(value).resolve() for value in (args.output_dir, args.log, args.report))
    if log.exists():
        raise FileExistsError(f"Refusing to replace an existing training log: {log}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output already exists: {output}")
    log.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    extra = args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args
    if any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in extra):
        raise ValueError("Set output-dir on the supervisor only")
    command = ["bash", "scripts/train_diffusion_joint.sh", "--output-dir", str(output), *extra]
    environment = dict(os.environ, PYTHON=sys.executable)
    with log.open("x") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        while True:
            returncode = child.poll()
            write_report(report, output, log, command, returncode, child.pid)
            if returncode is not None:
                return returncode
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
