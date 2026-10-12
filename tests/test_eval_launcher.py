"""Closed-loop launcher checks without importing or contacting the simulator."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "eval_aerovla.sh"
OVERRIDES = (
    "PYTHON", "MODEL_DIR", "MODEL_BASE", "TASK_ID", "GPU_ID", "SIMULATOR_GPU_ID",
    "SIMULATOR_PORT", "DDP_MASTER_PORT", "TEST_JSON", "SAVE_DIR", "DATASET_PATH",
    "BATCH_SIZE", "MAX_WAYPOINTS", "MAP_SPAWN_AREA_JSON_PATH", "OBJECT_NAME_JSON_PATH",
    "HF_HOME", "HF_MODULES_CACHE",
)


@pytest.fixture
def launcher(tmp_path):
    project = tmp_path / "project with spaces"
    script = project / "scripts" / LAUNCHER.name
    script.parent.mkdir(parents=True)
    shutil.copyfile(LAUNCHER, script)
    stub = tmp_path / "stub python"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['STUB_OUTPUT']).write_text(json.dumps({\n"
        "    'argv': sys.argv[1:], 'cwd': os.getcwd(),\n"
        "    'cuda': os.environ['CUDA_VISIBLE_DEVICES'],\n"
        "    'hf_home': os.environ['HF_HOME'],\n"
        "    'hf_modules_cache': os.environ['HF_MODULES_CACHE'],\n"
        "}))\n"
    )
    stub.chmod(0o755)
    output = tmp_path / "stub.json"
    environment = {key: value for key, value in os.environ.items() if key not in OVERRIDES}
    environment.update(PYTHON=str(stub), STUB_OUTPUT=str(output))

    def run(*args, **overrides):
        result = subprocess.run(
            ["bash", str(script), *args], cwd=tmp_path,
            env={**environment, **overrides}, capture_output=True, text=True,
        )
        return result, json.loads(output.read_text()) if output.exists() else None

    return project, run


def option(record, name):
    arguments = record["argv"]
    return arguments[arguments.index(name) + 1]


def test_defaults_resolve_from_project_root(launcher):
    project, run = launcher
    result, record = run()
    assert result.returncode == 0, result.stderr
    assert record["cwd"] == str(project)
    assert record["cuda"] == "0"
    assert record["argv"][:2] == ["-u", str(project / "src/vlnce_src/eval_aerovla.py")]
    assert option(record, "--project_prefix") == str(project)
    assert option(record, "--gpu_id") == "0"
    assert option(record, "--simulator_tool_port") == "30000"
    assert option(record, "--DDP_MASTER_PORT") == "20001"
    assert option(record, "--eval_json_path") == str(
        project / "data/uav_dataset/seen_valset_splits/NYCEnvironmentMegapa.json"
    )
    assert option(record, "--model_path") == str(project / "checkpoints/aero_vla")
    assert record["hf_home"] == str(project / ".cache/huggingface")
    assert record["hf_modules_cache"] == str(project / ".cache/huggingface/modules")
    assert "--model_base" not in record["argv"]


def test_gpu_separation_and_quoted_overrides(launcher):
    _, run = launcher
    result, record = run(
        "--", "--name", "experiment with spaces", GPU_ID="2", SIMULATOR_GPU_ID="0",
        SIMULATOR_PORT="31000", DDP_MASTER_PORT="21000",
        TASK_ID="unseen_map_valset/ModularPark", MODEL_DIR="models/diffusion head",
        MODEL_BASE="models/base with spaces", TEST_JSON="data/custom episodes.json",
        SAVE_DIR="output/evaluation results", DATASET_PATH="raw images/",
        BATCH_SIZE="2", MAX_WAYPOINTS="50",
        MAP_SPAWN_AREA_JSON_PATH="metadata/map areas.json",
        OBJECT_NAME_JSON_PATH="metadata/object names.json",
        HF_HOME="cache with spaces", HF_MODULES_CACHE="module cache with spaces",
    )
    assert result.returncode == 0, result.stderr
    assert record["cuda"] == "2"
    for name, expected in {
        "--gpu_id": "0", "--simulator_tool_port": "31000", "--DDP_MASTER_PORT": "21000",
        "--model_path": "models/diffusion head", "--model_base": "models/base with spaces",
        "--eval_json_path": "data/custom episodes.json", "--eval_save_path": "output/evaluation results",
        "--dataset_path": "raw images/", "--batchSize": "2", "--maxWaypoints": "50",
        "--map_spawn_area_json_path": "metadata/map areas.json",
        "--object_name_json_path": "metadata/object names.json",
    }.items():
        assert option(record, name) == expected
    assert record["argv"][-2:] == ["--name", "experiment with spaces"]
    assert record["hf_home"] == "cache with spaces"
    assert record["hf_modules_cache"] == "module cache with spaces"


def test_legacy_gpu_default_controls_simulator_and_port(launcher):
    _, run = launcher
    result, record = run(GPU_ID="1")
    assert result.returncode == 0, result.stderr
    assert record["cuda"] == "1"
    assert option(record, "--gpu_id") == "1"
    assert option(record, "--simulator_tool_port") == "35000"


def test_dry_run_prints_complete_quoted_command_without_python(launcher):
    project, run = launcher
    result, record = run(
        "--dry-run", GPU_ID="3", SIMULATOR_GPU_ID="0", MODEL_DIR="model with spaces",
        MODEL_BASE="base with spaces", TASK_ID="unseen_map_valset/ModularPark",
    )
    assert result.returncode == 0, result.stderr
    assert record is None
    line = next(line for line in result.stdout.splitlines() if line.startswith("Command:"))
    tokens = shlex.split(line.removeprefix("Command:"))
    assert tokens[:2] == ["env", "CUDA_VISIBLE_DEVICES=3"]
    assert tokens[3:5] == ["-u", str(project / "src/vlnce_src/eval_aerovla.py")]
    command_record = {"argv": tokens}
    assert option(command_record, "--model_path") == "model with spaces"
    assert option(command_record, "--model_base") == "base with spaces"
    assert option(command_record, "--gpu_id") == "0"
    assert option(command_record, "--simulator_tool_port") == "30000"
    assert option(command_record, "--eval_json_path").endswith("unseen_map_valset_splits/ModularPark.json")


@pytest.mark.parametrize("overrides", [
    {"DDP_MASTER_PORT": "80005"}, {"DDP_MASTER_PORT": "0"},
    {"DDP_MASTER_PORT": "not-a-port"}, {"SIMULATOR_PORT": "65536"},
    {"SIMULATOR_GPU_ID": "-1"}, {"TASK_ID": "missing-map"},
])
def test_invalid_configuration_fails_before_python(launcher, overrides):
    _, run = launcher
    result, record = run("--dry-run", **overrides)
    assert result.returncode == 2
    assert result.stderr
    assert record is None
