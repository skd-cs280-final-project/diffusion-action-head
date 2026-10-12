"""Read-only monitoring regressions; no GPU or model weights are needed."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.monitor_diffusion_training import (
    JsonlTail, TrainingMonitor, checkpoint_inventory, main, read_json_object,
    rolling_mean, status_name, training_schedule,
)


class TrainingMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.log_dir = self.root / "logs"
        self.run_dir.mkdir()
        self.log_dir.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def config(self):
        config = {"train_split": {"num_samples": 17}, "batch_size": 2,
                  "gradient_accumulation_steps": 2, "epochs": 2,
                  "max_steps": None, "eval_steps": 0, "max_grad_norm": 1.0}
        (self.run_dir / "run_config.json").write_text(json.dumps(config))
        return config

    def append(self, filename, record):
        with (self.run_dir / filename).open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def monitor(self, **kwargs):
        return TrainingMonitor(self.run_dir, self.log_dir, make_plots=False, **kwargs)

    def test_partial_utf8_tail_is_delayed_then_read_exactly_once(self):
        path = self.run_dir / "metrics.jsonl"
        complete = '{"step": 1, "note": "中文"}\n'.encode()
        first = complete.index("中".encode()) + 1
        path.write_bytes(complete[:first])
        reader = JsonlTail(path)
        reader.poll()
        self.assertEqual(reader.rows, [])
        with path.open("ab") as stream:
            stream.write(complete[first:])
        reader.poll()
        reader.poll()
        self.assertEqual(reader.rows, [{"step": 1, "note": "中文"}])
        self.assertEqual(reader.warnings, [])
        self.assertEqual(path.read_bytes(), complete)

    def test_missing_sources_export_headers_and_wait(self):
        monitor = self.monitor()
        result = monitor.refresh(monotonic_now=10, wall_now=1000)
        self.assertFalse(result["terminal"])
        self.assertEqual(result["status"], "waiting")
        self.assertEqual(result["phase"], "waiting_for_training_metrics")
        self.assertIsNone(result["estimated_remaining_training_seconds"])
        self.assertFalse(result["config_available"])
        self.assertTrue((monitor.output_dir / "status.md").exists())
        with (monitor.output_dir / "train_metrics.csv").open() as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])
        self.config()
        self.append("metrics.jsonl", {"step": 1, "epoch": 1, "loss": 1.2})
        result = monitor.refresh(monotonic_now=20, wall_now=1010)
        self.assertTrue(result["config_available"])
        self.assertEqual(result["total_optimizer_steps"], 10)

    def test_csv_keeps_all_fields_and_flattens_validation(self):
        self.append("metrics.jsonl", {"step": 1, "epoch": 1, "loss": 1.2, "new_field": "abc"})
        self.append("metrics.jsonl", {"step": 2, "epoch": 1, "loss": 1.1, "future_metric": [1, 2]})
        self.append("validation_metrics.jsonl", {"step": 2, "noise_mse": 0.7,
                    "physical": {"yaw": {"mae": 0.4, "unit": "radians"}}, "land": {"f1": 0.6}})
        monitor = self.monitor()
        monitor.refresh()
        with (monitor.output_dir / "train_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["new_field"], "abc")
        self.assertEqual(rows[1]["future_metric"], "[1, 2]")
        with (monitor.output_dir / "validation_metrics.csv").open() as stream:
            validation = list(csv.DictReader(stream))
        self.assertEqual(validation[0]["physical.yaw.mae"], "0.4")
        self.assertEqual(validation[0]["land.f1"], "0.6")
        self.assertEqual(validation[0]["physical.yaw.unit"], "radians")

    def test_nonfinite_and_invalid_records_have_warning_table_and_strict_json(self):
        path = self.run_dir / "metrics.jsonl"
        source = '{"step": 1, "epoch": 1, "loss": NaN, "grad_norm": Infinity}\ninvalid\n[]\n{"step":'
        path.write_text(source)
        monitor = self.monitor()
        result = monitor.refresh()
        self.assertEqual(result["training_records"], 1)
        self.assertEqual(result["warning_count"], 4)
        self.assertIsNone(result["latest_training"]["loss"])
        self.assertGreater(result["pending_training_tail_bytes"], 0)
        raw = (monitor.output_dir / "latest_summary.json").read_text()
        # parse_constant raises if any unquoted NaN/Infinity leaks into machine-readable JSON.
        json.loads(raw, parse_constant=lambda value: self.fail(f"non-standard JSON: {value}"))
        with (monitor.output_dir / "warnings.csv").open() as stream:
            warnings = list(csv.DictReader(stream))
        self.assertEqual(len(warnings), 4)
        self.assertIn("nonfinite", (monitor.output_dir / "status.md").read_text())
        self.assertEqual(path.read_text(), source)

    def test_throughput_uses_observations_and_resets_after_validation(self):
        self.config()
        self.append("metrics.jsonl", {"step": 1, "epoch": 1, "loss": 1.2, "samples": 4})
        monitor = self.monitor()
        first = monitor.refresh(monotonic_now=100, wall_now=1000)
        self.assertIsNone(first["throughput"])
        self.append("metrics.jsonl", {"step": 2, "epoch": 1, "loss": 1.1, "samples": 4})
        second = monitor.refresh(monotonic_now=110, wall_now=1010)
        self.assertAlmostEqual(second["throughput"]["optimizer_steps_per_second"], 0.1)
        self.assertAlmostEqual(second["throughput"]["samples_per_second"], 0.4)
        self.assertAlmostEqual(second["estimated_remaining_training_seconds"], 80)
        self.assertFalse(second["eta_includes_validation"])
        stationary = monitor.refresh(monotonic_now=200, wall_now=1100)
        self.assertEqual(stationary["throughput"], second["throughput"])
        self.append("validation_metrics.jsonl", {"step": 2, "epoch": 1, "noise_mse": 0.9})
        validated = monitor.refresh(monotonic_now=210, wall_now=1110)
        self.assertIsNone(validated["throughput"])
        self.append("metrics.jsonl", {"step": 3, "epoch": 1, "loss": 1.0, "samples": 4})
        third = monitor.refresh(monotonic_now=220, wall_now=1120)
        self.assertEqual(third["throughput"]["first_observed_step"], 2)
        self.assertEqual(third["throughput"]["observed_window_seconds"], 10)

    def test_epoch_boundary_marks_validation_and_epoch_change_resets_timing(self):
        self.config()
        self.append("metrics.jsonl", {"step": 4, "epoch": 1, "samples": 4})
        monitor = self.monitor()
        monitor.refresh(monotonic_now=0)
        self.append("metrics.jsonl", {"step": 5, "epoch": 1, "samples": 1})
        boundary = monitor.refresh(monotonic_now=10)
        self.assertEqual(boundary["phase"], "validation_or_checkpoint")
        self.assertAlmostEqual(boundary["throughput"]["samples_per_second"], 0.1)
        self.append("metrics.jsonl", {"step": 6, "epoch": 2, "samples": 4})
        resumed = monitor.refresh(monotonic_now=100)
        self.assertIsNone(resumed["throughput"])

    def test_log_truncation_starts_new_generation(self):
        path = self.run_dir / "metrics.jsonl"
        path.write_text('{"step": 12345, "loss": 1.0}\n')
        reader = JsonlTail(path)
        reader.poll()
        path.write_text('{"step": 1}\n')
        reader.poll()
        self.assertEqual(reader.rows, [{"step": 1}])
        self.assertEqual(reader.generation, 1)

    def test_completed_and_failed_exit_after_final_export_without_sleep(self):
        for state, expected in [("completed", 0), ("failed", 1)]:
            with self.subTest(state=state):
                (self.log_dir / "status.json").write_text(json.dumps({"status": state}))
                with mock.patch("src.monitor_diffusion_training.time.sleep") as sleep:
                    code = main(["--run-dir", str(self.run_dir), "--log-dir", str(self.log_dir), "--no-plots"])
                sleep.assert_not_called()
                self.assertEqual(code, expected)
                summary = json.loads((self.run_dir / "monitoring/latest_summary.json").read_text())
                self.assertTrue(summary["terminal"])
                self.assertEqual(summary["status"], state)
        self.assertEqual(status_name({"completed": True}), "completed")
        self.assertEqual(status_name({"failed": True}), "failed")

    def test_plot_failure_does_not_prevent_supervision_exports(self):
        monitor = TrainingMonitor(self.run_dir, self.log_dir)
        with mock.patch("src.monitor_diffusion_training.plot_curves", side_effect=ImportError("no matplotlib")):
            result = monitor.refresh()
        self.assertEqual(result["warning_count"], 1)
        self.assertEqual(result["recent_warnings"][0]["kind"], "plot_unavailable")
        self.assertTrue((monitor.output_dir / "train_metrics.csv").exists())
        self.assertTrue((monitor.output_dir / "latest_summary.json").exists())

    def test_terminal_status_repolls_metrics_written_during_status_read(self):
        self.append("metrics.jsonl", {"step": 1, "epoch": 1})

        def read_with_completion(path):
            if Path(path).name == "status.json":
                self.append("metrics.jsonl", {"step": 2, "epoch": 1})
                self.append("validation_metrics.jsonl", {"step": 2, "noise_mse": 0.5})
                return {"status": "completed"}
            return read_json_object(path)

        with mock.patch("src.monitor_diffusion_training.read_json_object", side_effect=read_with_completion):
            summary = self.monitor().refresh()
        self.assertTrue(summary["terminal"])
        self.assertEqual(summary["training_records"], 2)
        self.assertEqual(summary["latest_training"]["step"], 2)
        self.assertEqual(summary["latest_validation"]["step"], 2)

    def test_checkpoint_inventory_only_stats_weights_and_reports_file_presence(self):
        self.assertEqual(checkpoint_inventory(self.run_dir), [])
        checkpoint = self.run_dir / "checkpoint-2"
        checkpoint.mkdir()
        # Deliberately not torch data: inventory must never load it or claim validation.
        (checkpoint / "diffusion_head.pt").write_bytes(b"unfinished weights")
        (checkpoint / "diffusion_config.json").write_text(json.dumps({"training_metadata": {"global_step": 2}}))
        (self.run_dir / "best").mkdir()
        records = checkpoint_inventory(self.run_dir)
        self.assertEqual(len(records), 2)
        ready = next(row for row in records if row["directory"] == "checkpoint-2")
        self.assertEqual(ready["global_step"], 2)
        self.assertTrue(ready["files_present"])
        self.assertEqual(ready["weights_size_bytes"], len(b"unfinished weights"))
        self.assertFalse(next(row for row in records if row["directory"] == "best")["files_present"])
        monitor = self.monitor()
        summary = monitor.refresh()
        self.assertEqual(summary["checkpoint_records"], 2)
        self.assertTrue((monitor.output_dir / "checkpoints.csv").exists())

    def test_schedule_handles_tail_accumulation_and_early_stop(self):
        config = self.config()
        self.assertEqual(training_schedule(config), (5, 10))
        config["max_steps"] = 6
        self.assertEqual(training_schedule(config), (5, 6))
        config["max_samples"] = 3
        self.assertEqual(training_schedule(config), (1, 2))
        self.assertEqual(training_schedule({}), (None, None))

    def test_rolling_mean_ignores_nonfinite_without_losing_window(self):
        self.assertEqual(rolling_mean([1, float("nan"), 3, 5], window=2), [1, 1, 3, 4])


if __name__ == "__main__":
    unittest.main()
