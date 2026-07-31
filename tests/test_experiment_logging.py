import csv
from datetime import datetime, timezone

import numpy as np
from active_inference_navigation.models import AxisAction, NavigationAction, Observation
from active_inference_navigation.runtime import NavigationRuntimeResult

from active_inference_meta.experiment_logging import ExperimentRunLogger
from active_inference_meta.models import MetaObservation, TaskTelemetry


def telemetry(step=0, *, meta_latency=4.0):
    return TaskTelemetry(
        step=step,
        resolution=2,
        belief=np.asarray([0.1, 0.2, 0.6, 0.1]),
        robot_x=1.0,
        robot_y=1.0,
        rssi=-64.0,
        selected_action=(2, 0),
        inference_latency_ms=10.0,
        meta_inference_latency_ms=meta_latency,
        meta_observation=MetaObservation(0.08, 24.0, 10.0, 82.0),
        meta_observation_step=step,
        selected_resolution=5,
        model_switched=True,
    )


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_experiment_logger_writes_steps_and_paper_summary(tmp_path):
    logger = ExperimentRunLogger(
        tmp_path,
        meta_inference_enabled=True,
        fixed_resolution=None,
        arena_width=7.0,
        arena_height=7.0,
        task_temporal_horizon=3,
        source_x=1.35,
        source_y=1.0,
        cpu_condition="low",
        run_label="corner-1",
        run_id="abc12345",
        started_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
    )
    logger.record(telemetry())
    logger.finish(
        NavigationRuntimeResult(
            observations=(
                Observation(1.0, 1.0, -64.0),
                Observation(1.2, 1.0, -59.0),
            ),
            actions=(NavigationAction(AxisAction.POSITIVE, AxisAction.NONE),),
            terminated=True,
        )
    )
    logger.close()

    rows = read_rows(logger.path)
    assert logger.path.name == "robot_run_20260731T000000.000000Z_meta_abc12345.csv"
    assert [row["event"] for row in rows] == ["task_step", "run_summary"]
    assert rows[0]["method"] == "meta_inference"
    assert rows[0]["meta_inference_enabled"] == "True"
    assert rows[0]["cpu_condition"] == "low"
    assert rows[0]["temporal_phase"] == "0"
    assert rows[0]["map_source_error_m"] != ""
    assert rows[1]["success"] == "True"
    assert rows[1]["action_count"] == "1"
    assert rows[1]["total_task_inference_ms"] == "10.0"
    assert rows[1]["total_meta_inference_ms"] == "4.0"
    assert rows[1]["total_compute_ms"] == "14.0"
    assert rows[1]["final_prediction_error"] == "24.0"


def test_fixed_run_filename_and_method_are_automatic_and_unique(tmp_path):
    started = datetime(2026, 7, 31, tzinfo=timezone.utc)
    first = ExperimentRunLogger(
        tmp_path,
        meta_inference_enabled=False,
        fixed_resolution=10,
        arena_width=7.0,
        arena_height=7.0,
        task_temporal_horizon=3,
        source_x=3.0,
        source_y=4.0,
        run_id="first-id",
        started_at=started,
    )
    second = ExperimentRunLogger(
        tmp_path,
        meta_inference_enabled=False,
        fixed_resolution=10,
        arena_width=7.0,
        arena_height=7.0,
        task_temporal_horizon=3,
        source_x=3.0,
        source_y=4.0,
        run_id="second-id",
        started_at=started,
    )
    first.record(telemetry(meta_latency=0.0))
    second.record(telemetry(meta_latency=0.0))
    first.close()
    second.close()

    assert first.path != second.path
    assert read_rows(first.path)[0]["method"] == "fixed_10x10"
    assert read_rows(first.path)[0]["fixed_resolution"] == "10"


def test_failed_run_keeps_partial_compute_and_error(tmp_path):
    logger = ExperimentRunLogger(
        tmp_path,
        meta_inference_enabled=True,
        fixed_resolution=None,
        arena_width=7.0,
        arena_height=7.0,
        task_temporal_horizon=3,
        source_x=3.0,
        source_y=4.0,
    )
    logger.record(telemetry())
    logger.fail(TimeoutError("Robot action timed out."))
    logger.close()

    summary = read_rows(logger.path)[-1]
    assert summary["event"] == "run_error"
    assert summary["total_compute_ms"] == "14.0"
    assert summary["error_type"] == "TimeoutError"
    assert summary["error_message"] == "Robot action timed out."
