#!/usr/bin/env python3
"""Recompute paired metrics from saved trajectories, using CPU and stdlib only.

Reads both existing evaluations without modifying them. The ten UUIDs come
from the diffusion run's fixed selection, not from outcome-based filtering.
Baseline success/OSR are inferred from original result-directory prefixes;
action counts are inferred from the verified five-state saving convention.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[2]
BASE_PROJECT = Path("/data/xiyang/project/AeroVLA")
BASE = BASE_PROJECT / "eval_results/aero_vla/seen_valset/NYCEnvironmentMegapa"
RUN = PROJECT / "eval_results/diffusion_ckpt70000_nyc10_seed42_gpu2_20261011"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def same(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7), (actual, expected)


def identity(name):
    return name.removeprefix("success_").removeprefix("oracle_")


def original_path(directory, project):
    path = Path(read(directory / "ori_info.json")["ori_traj_dir"])
    return path if path.is_absolute() else project / path


def quat_angle_degrees(q1, q2):
    n1, n2 = math.sqrt(sum(x*x for x in q1)), math.sqrt(sum(x*x for x in q2))
    assert n1 > 0 and n2 > 0
    dot = abs(sum(a*b for a, b in zip(q1, q2)) / (n1*n2))
    return math.degrees(2 * math.acos(min(1.0, dot)))


def load_result(directory, reference, target):
    logs = sorted((directory / "log").glob("*.json"), key=lambda p: int(p.stem))
    assert logs and [int(p.stem) for p in logs] == list(range(len(logs))), directory
    assert (len(logs) - 1) % 5 == 0, directory
    records, digest = [], hashlib.sha256()
    for path in logs:
        raw = path.read_bytes()
        digest.update(path.name.encode() + b"\0" + raw + b"\0")
        record = json.loads(raw)
        assert record["frame"] == int(path.stem)
        records.append(record["sensors"]["state"])
    points = [r["position"] for r in records]
    assert all(len(p) == 3 and all(math.isfinite(v) for v in p) for p in points)
    action_count = (len(logs) - 1) // 5
    # The original controller duplicates one post-action sample five times.
    # These are saved records, not five independent measurement instants.
    for start in range(1, len(records), 5):
        first = records[start]
        assert all(r["position"] == first["position"] and r["timestamp"] == first["timestamp"]
                   for r in records[start:start+5]), directory
    expected_images = [f"{5*i:06d}" for i in range(action_count + 1)]
    camera_count = 0
    for camera in ("frontcamera", "leftcamera", "rightcamera", "rearcamera", "downcamera"):
        for suffix in ("", "_depth"):
            frames = sorted((directory / (camera + suffix)).glob("*.png"))
            assert [p.stem for p in frames] == expected_images, (directory, camera + suffix)
            assert all(p.stat().st_size > 0 for p in frames)
            camera_count += len(frames)
    success = directory.name.startswith("success_")
    oracle = success or directory.name.startswith("oracle_")
    pred_length, ref_length = length(points), length(reference)
    adjusted = ref_length - 20.0
    assert adjusted > 0, "This fixed comparison expects all reference paths longer than 20 m"
    return {
        "result_directory": str(directory),
        "success": success, "oracle_success": oracle,
        "success_source": "original_saved_directory_prefix",
        "action_count": action_count,
        "action_count_source": "(saved_state_count - 1) / 5; verified camera indices",
        "saved_state_count": len(logs), "sampled_observation_count": action_count + 1,
        "saved_camera_pngs": camera_count,
        "ne_m": math.dist(points[-1], reference[-1]),
        "final_target_distance_m": math.dist(points[-1], target),
        "min_sampled_target_distance_m": min(math.dist(p, target) for p in points),
        "path_length_m": pred_length,
        "legacy_spl": max(0.0, adjusted / max(adjusted, pred_length)) if success else 0.0,
        "initial_orientation_xyzw": records[0]["orientation"],
        "sensor_initial_has_collided": bool(records[0]["collision"]["has_collided"]),
        "sensor_final_has_collided": bool(records[-1]["collision"]["has_collided"]),
        "sensor_any_has_collided": any(r["collision"]["has_collided"] for r in records),
        "max_height_above_target_m": max(target[2] - p[2] for p in points),
        "positions": points,
        "input_provenance": {
            "log_directory": str(directory / "log"),
            "logs_ordered_content_sha256": digest.hexdigest(),
            "digest_rule": "SHA256 of filename + NUL + original JSON bytes + NUL, in numeric frame order",
            "ori_info_sha256": sha(directory / "ori_info.json"),
            "object_description_sha256": sha(directory / "object_description.json"),
        },
    }


def aggregate(episodes, side):
    items = [e[side] for e in episodes]
    if not items:
        return None
    result = {"n": len(items), "success_count": sum(r["success"] for r in items),
              "oracle_success_count": sum(r["oracle_success"] for r in items)}
    result.update(sr=result["success_count"]/len(items), osr=result["oracle_success_count"]/len(items))
    for field in ("legacy_spl", "ne_m", "final_target_distance_m", "path_length_m", "action_count"):
        result[field] = statistics.mean(r[field] for r in items)
    result["action_count_total"] = sum(r["action_count"] for r in items)
    return result


def main():
    status = read(RUN / "status.json")
    assert status["state"] == "completed" and status["completed_episodes"] == 10
    selected = read(RUN / "selected_episodes.json")
    ids = [Path(row["json"]).parent.name for row in selected]
    assert len(ids) == len(set(ids)) == 10
    rows = [json.loads(line) for line in (RUN / "episode_metrics.jsonl").read_text().splitlines() if line]
    metrics = {r["episode_uuid"]: r for r in rows}
    assert len(rows) == 10 and set(metrics) == set(ids)
    baseline_dirs = [p for p in BASE.iterdir() if p.is_dir()]
    protocol_files = (
        "src/vlnce_src/eval_aerovla.py", "src/vlnce_src/closeloop_util.py", "src/vlnce_src/assist.py",
        "airsim_plugin/AirVLNSimulatorClientTool_AeroVLA.py", "airsim_plugin/AirVLNSimulatorServerTool.py",
        "utils/metric.py", "data/meta/map_spawnarea_info.json", "data/meta/object_description.json",
    )
    protocol_audit = {}
    for name in protocol_files:
        baseline_sha, diffusion_sha = sha(BASE_PROJECT/name), sha(PROJECT/name)
        assert baseline_sha == diffusion_sha, name
        protocol_audit[name] = {"baseline_path": str(BASE_PROJECT/name), "diffusion_path": str(PROJECT/name),
                                "current_files_identical": True, "sha256": baseline_sha}
    areas = read(BASE_PROJECT / "data/meta/map_spawnarea_info.json")["NYCEnvironmentMegapa"]
    episodes = []
    for index, uid in enumerate(ids, 1):
        matches = [p for p in baseline_dirs if identity(p.name) == uid]
        assert len(matches) == 1, (uid, matches)
        base_dir, metric = matches[0], metrics[uid]
        diff_dir = Path(metric["result_directory"])
        assert identity(diff_dir.name) == uid and diff_dir.parent == RUN / "episodes"
        base_ref = original_path(base_dir, BASE_PROJECT)
        diff_ref = original_path(diff_dir, PROJECT)
        assert base_ref.name == diff_ref.name == uid
        metadata = {}
        for filename in ("merged_data.json", "mark.json", "object_description.json"):
            bp, dp = base_ref / filename, diff_ref / filename
            assert sha(bp) == sha(dp), (uid, filename)
            metadata[filename] = {"baseline_path": str(bp), "diffusion_path": str(dp), "sha256": sha(bp)}
        assert sha(base_dir / "object_description.json") == sha(diff_dir / "object_description.json")
        reference = [r["position"] for r in read(diff_ref / "merged_data.json")["trajectory_raw_detailed"]]
        target = read(diff_ref / "mark.json")["target"]["position"]
        assert target == metric["target_position"]
        closest = min((a for a in areas if len(a) >= 18),
                      key=lambda a: math.dist(target, [a[0]+1, a[1]+1, a[2]+0.5]))
        assert list(closest[9:12]) == target, (uid, "resolved target differs from annotation")
        base = load_result(base_dir, reference, target)
        diff = load_result(diff_dir, reference, target)
        assert diff["success"] == metric["success"]
        assert diff["oracle_success"] == bool(metric["success"] or metric["oracle_success"])
        assert diff["action_count"] == metric["step_count"]
        for new, old in (("ne_m", "reference_endpoint_distance"), ("final_target_distance_m", "final_target_distance"),
                         ("path_length_m", "pred_length"), ("legacy_spl", "legacy_spl")):
            same(diff[new], metric[old])
        same(length(reference), metric["reference_length"])
        outcome = ("both_success" if base["success"] else "diffusion_only_success") if diff["success"] else (
            "baseline_only_success" if base["success"] else "both_fail")
        episodes.append({"index": index, "uuid": uid, "short_id": uid.split("-")[0],
                         "target": target,
                         "target_basis": "shared mark.json target; reproduced nearest-area spawn rule and diffusion actual target both match",
                         "reference": reference, "reference_length_m": length(reference),
                         "baseline": base, "diffusion": diff, "outcome": outcome,
                         "initial_position_gap_m": math.dist(base["positions"][0], diff["positions"][0]),
                         "initial_orientation_gap_degrees": quat_angle_degrees(base["initial_orientation_xyzw"], diff["initial_orientation_xyzw"]),
                         "source_metadata": metadata,
                         "delta_diffusion_minus_baseline": {field: diff[field]-base[field] for field in (
                             "success", "ne_m", "final_target_distance_m", "path_length_m", "legacy_spl", "action_count")}})
    groups = {name: [e for e in episodes if e["outcome"] == name] for name in (
        "both_success", "diffusion_only_success", "baseline_only_success", "both_fail")}
    all_aggregates = {side: aggregate(episodes, side) for side in ("baseline", "diffusion")}
    paired = {side: aggregate(groups["both_success"], side) for side in ("baseline", "diffusion")}
    output = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "labels": {"baseline": "AeroVLA historical baseline", "diffusion": "Diffusion checkpoint-70000"},
              "sources": {"baseline_root": str(BASE), "diffusion_run": str(RUN),
                          "selection_path": str(RUN / "selected_episodes.json"),
                          "selection_sha256": sha(RUN / "selected_episodes.json"),
                          "diffusion_metrics_sha256": sha(RUN / "episode_metrics.jsonl"),
                          "baseline_candidate_directory_count": len(baseline_dirs)},
              "matched_count": len(episodes), "aggregates": {"all": all_aggregates, "both_success": paired},
              "current_protocol_file_audit": protocol_audit,
              "outcome_groups": {k: [e["uuid"] for e in v] for k, v in groups.items()},
              "metric_definitions": {
                  "ne_m": "3D final position distance to the shared reference trajectory endpoint (original metric.py NE)",
                  "final_target_distance_m": "3D final position distance to shared mark.json target",
                  "legacy_spl": "success * max(0, (reference_length-20)/max(reference_length-20, predicted_length))",
                  "action_count": "(saved_state_count-1)/5, supported by camera indices; directly checked against diffusion trace counts",
                  "initial_orientation_gap_degrees": "normalized quaternion geodesic rotation difference; includes roll, pitch and yaw",
                  "sensor_has_collided": "saved sensor contact flag, not equivalent to evaluator collision/heuristic termination flag",
                  "saved_state_count": "includes five identical position/timestamp copies per post-action sample",
                  "min_sampled_target_distance_m": "minimum over saved positions, not the continuous trajectory or official OSR",
              },
              "comparability_limits": [
                  "Paired historical results, not a controlled rerun; observed initial poses differ.",
                  "Historical baseline checkpoint, RNG and runtime are not identified by the result directories alone.",
                  "Goal distance uses a common annotated target; no historical per-step target trace is stored.",
                  "Raw all-task path/action averages include early failures; use the five jointly successful tasks for paired efficiency analysis.",
                  "Original evaluation includes GT-position-derived semantic direction and a 20 m success threshold.",
              ], "episodes": episodes}
    write_json(OUTPUT / "comparison_data.json", output)
    csv_fields = ["index", "uuid", "outcome", "initial_position_gap_m", "initial_orientation_gap_degrees"]
    metric_fields = ("success", "oracle_success", "action_count", "ne_m", "final_target_distance_m", "legacy_spl", "path_length_m", "max_height_above_target_m")
    csv_fields += [f"{side}_{field}" for side in ("baseline", "diffusion") for field in metric_fields]
    csv_fields += ["delta_ne_m", "delta_legacy_spl", "delta_action_count", "baseline_result_directory", "diffusion_result_directory"]
    with (OUTPUT / "paired_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for e in episodes:
            row = {field: e[field] for field in csv_fields[:5]}
            row.update({f"{side}_{field}": e[side][field] for side in ("baseline", "diffusion") for field in metric_fields})
            row.update({f"delta_{field}": e["delta_diffusion_minus_baseline"][field] for field in ("ne_m", "legacy_spl", "action_count")})
            row.update({f"{side}_result_directory": e[side]["result_directory"] for side in ("baseline", "diffusion")})
            writer.writerow(row)
    matching = [{"index": e["index"], "uuid": e["uuid"],
                 "baseline_directory": e["baseline"]["result_directory"],
                 "diffusion_directory": e["diffusion"]["result_directory"]} for e in episodes]
    write_json(OUTPUT / "matched_tasks.json", matching)
    verification = {"passed": True, "verified_utc": output["generated_utc"], "matched_unique_tasks": 10,
                    "all_reference_and_mark_files_identical": True,
                    "all_saved_object_descriptions_identical": True,
                    "diffusion_metrics_match_original_run": True,
                    "contiguous_log_indices_and_ten_camera_streams_checked": True,
                    "five_duplicate_states_per_action_verified": True,
                    "current_protocol_files_identical": True,
                    "spawn_rule_resolved_targets_equal_in_both_projects": True,
                    "baseline_saved_states": sum(e["baseline"]["saved_state_count"] for e in episodes),
                    "diffusion_saved_states": sum(e["diffusion"]["saved_state_count"] for e in episodes),
                    "baseline_camera_pngs": sum(e["baseline"]["saved_camera_pngs"] for e in episodes),
                    "diffusion_camera_pngs": sum(e["diffusion"]["saved_camera_pngs"] for e in episodes),
                    "output_sha256": {name: sha(OUTPUT/name) for name in ("build_comparison.py", "comparison_data.json", "paired_results.csv", "matched_tasks.json")}}
    write_json(OUTPUT / "verification.json", verification)
    print(json.dumps({"matched": 10, "aggregates": output["aggregates"],
                      "outcome_counts": {k: len(v) for k, v in groups.items()},
                      "initial_position_gap_range_m": [min(e["initial_position_gap_m"] for e in episodes), max(e["initial_position_gap_m"] for e in episodes)],
                      "initial_orientation_gap_range_degrees": [min(e["initial_orientation_gap_degrees"] for e in episodes), max(e["initial_orientation_gap_degrees"] for e in episodes)]}, indent=2))


if __name__ == "__main__":
    main()
