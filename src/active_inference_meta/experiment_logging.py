"""Durable, non-overwriting CSV records for physical navigation experiments."""

from __future__ import annotations

import csv
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from active_inference_navigation.runtime import NavigationRuntimeResult

from .models import TaskTelemetry


def _safe_filename_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return cleaned or fallback


class ExperimentRunLogger:
    """Write one detailed CSV per run and finish it with a summary row."""

    fieldnames = (
        "event",
        "run_id",
        "run_started_utc",
        "run_finished_utc",
        "method",
        "meta_inference_enabled",
        "fixed_resolution",
        "cpu_condition",
        "run_label",
        "success_distance_m",
        "step",
        "temporal_phase",
        "active_resolution",
        "robot_x_m",
        "robot_y_m",
        "source_x_m",
        "source_y_m",
        "distance_to_source_m",
        "rssi_dbm",
        "action_x",
        "action_y",
        "task_inference_ms",
        "meta_inference_ms",
        "cumulative_task_inference_ms",
        "cumulative_meta_inference_ms",
        "information_gain_proxy",
        "prediction_error",
        "cpu_availability",
        "meta_observation_step",
        "meta_observation_is_new",
        "meta_selected_resolution",
        "model_switched",
        "map_source_x_m",
        "map_source_y_m",
        "map_source_error_m",
        "threshold_reached",
        "terminated",
        "success",
        "action_count",
        "observation_count",
        "minimum_distance_m",
        "total_task_inference_ms",
        "total_meta_inference_ms",
        "total_compute_ms",
        "final_prediction_error",
        "final_map_source_error_m",
        "error_type",
        "error_message",
    )

    def __init__(
        self,
        output_directory: str | Path,
        *,
        meta_inference_enabled: bool,
        fixed_resolution: int | None,
        arena_width: float,
        arena_height: float,
        task_temporal_horizon: int,
        source_x: float,
        source_y: float,
        success_distance_m: float = 0.5,
        cpu_condition: str = "uncontrolled",
        run_label: str = "",
        filename_prefix: str = "robot_run",
        run_id: str | None = None,
        started_at: datetime | None = None,
    ) -> None:
        if arena_width <= 0.0 or arena_height <= 0.0:
            raise ValueError("Experiment arena dimensions must be positive.")
        if task_temporal_horizon < 1:
            raise ValueError("The task temporal horizon must be positive.")
        if success_distance_m <= 0.0:
            raise ValueError("The success distance must be positive.")
        if not 0.0 <= source_x <= arena_width or not 0.0 <= source_y <= arena_height:
            raise ValueError("The experiment source must lie inside the arena.")
        if meta_inference_enabled and fixed_resolution is not None:
            raise ValueError("Adaptive runs must not declare a fixed resolution.")
        if not meta_inference_enabled and fixed_resolution is None:
            raise ValueError("Fixed runs must declare their resolution.")

        self.meta_inference_enabled = meta_inference_enabled
        self.fixed_resolution = fixed_resolution
        self.arena_width = float(arena_width)
        self.arena_height = float(arena_height)
        self.task_temporal_horizon = task_temporal_horizon
        self.source_x = float(source_x)
        self.source_y = float(source_y)
        self.success_distance_m = float(success_distance_m)
        self.cpu_condition = cpu_condition
        self.run_label = run_label
        self.run_id = run_id or uuid4().hex
        self.started_at = started_at or datetime.now(timezone.utc)
        self._task_compute_ms = 0.0
        self._meta_compute_ms = 0.0
        self._last_prediction_error: float | None = None
        self._last_map_source_error_m: float | None = None
        self._closed = False
        self._finished = False

        mode = "meta" if meta_inference_enabled else f"fixed-{fixed_resolution}x{fixed_resolution}"
        timestamp = self.started_at.astimezone(timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        filename = "_".join(
            (
                _safe_filename_component(filename_prefix, "robot-run"),
                timestamp,
                _safe_filename_component(mode, "run"),
                _safe_filename_component(self.run_id[:8], "id"),
            )
        )
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{filename}.csv"
        self._stream: TextIO = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._stream.flush()

    @property
    def method(self) -> str:
        """Return the table-ready experiment method label."""

        if self.meta_inference_enabled:
            return "meta_inference"
        return f"fixed_{self.fixed_resolution}x{self.fixed_resolution}"

    @property
    def finished(self) -> bool:
        """Return whether a summary or error row has finalized the run."""

        return self._finished

    def record(self, telemetry: TaskTelemetry) -> None:
        """Append and flush one task-inference event."""

        self._require_open()
        self._task_compute_ms += telemetry.inference_latency_ms
        self._meta_compute_ms += telemetry.meta_inference_latency_ms
        meta = telemetry.meta_observation
        if meta is not None:
            self._last_prediction_error = meta.prediction_error

        distance = math.hypot(
            telemetry.robot_x - self.source_x,
            telemetry.robot_y - self.source_y,
        )
        peak_index = int(telemetry.belief.argmax())
        peak_column = peak_index % telemetry.resolution
        peak_row = peak_index // telemetry.resolution
        map_x = (peak_column + 0.5) * self.arena_width / telemetry.resolution
        map_y = (peak_row + 0.5) * self.arena_height / telemetry.resolution
        map_error = math.hypot(map_x - self.source_x, map_y - self.source_y)
        self._last_map_source_error_m = map_error

        action_x = action_y = ""
        if telemetry.selected_action is not None:
            action_x, action_y = telemetry.selected_action
        self._write(
            "task_step",
            {
                "step": telemetry.step,
                "temporal_phase": telemetry.step % self.task_temporal_horizon,
                "active_resolution": telemetry.resolution,
                "robot_x_m": telemetry.robot_x,
                "robot_y_m": telemetry.robot_y,
                "distance_to_source_m": distance,
                "rssi_dbm": telemetry.rssi,
                "action_x": action_x,
                "action_y": action_y,
                "task_inference_ms": telemetry.inference_latency_ms,
                "meta_inference_ms": telemetry.meta_inference_latency_ms,
                "cumulative_task_inference_ms": self._task_compute_ms,
                "cumulative_meta_inference_ms": self._meta_compute_ms,
                "information_gain_proxy": (
                    "" if meta is None else meta.information_gain_proxy
                ),
                "prediction_error": "" if meta is None else meta.prediction_error,
                "cpu_availability": "" if meta is None else meta.cpu_availability,
                "meta_observation_step": telemetry.meta_observation_step,
                "meta_observation_is_new": (
                    meta is not None and telemetry.meta_observation_step == telemetry.step
                ),
                "meta_selected_resolution": telemetry.selected_resolution,
                "model_switched": telemetry.model_switched,
                "map_source_x_m": map_x,
                "map_source_y_m": map_y,
                "map_source_error_m": map_error,
                "threshold_reached": distance <= self.success_distance_m,
            },
        )

    def finish(self, result: NavigationRuntimeResult) -> None:
        """Append the paper-ready summary for a normally completed run."""

        self._require_open()
        distances = [
            math.hypot(observation.x - self.source_x, observation.y - self.source_y)
            for observation in result.observations
        ]
        minimum_distance = min(distances)
        success = minimum_distance <= self.success_distance_m
        self._write(
            "run_summary",
            {
                "run_finished_utc": datetime.now(timezone.utc).isoformat(),
                "terminated": result.terminated,
                "success": success,
                "action_count": len(result.actions),
                "observation_count": len(result.observations),
                "minimum_distance_m": minimum_distance,
                "total_task_inference_ms": self._task_compute_ms,
                "total_meta_inference_ms": self._meta_compute_ms,
                "total_compute_ms": self._task_compute_ms + self._meta_compute_ms,
                "final_prediction_error": self._last_prediction_error,
                "final_map_source_error_m": self._last_map_source_error_m,
            },
        )
        self._finished = True

    def fail(self, error: BaseException) -> None:
        """Record partial compute totals and the failure without hiding the error."""

        self._require_open()
        self._write(
            "run_error",
            {
                "run_finished_utc": datetime.now(timezone.utc).isoformat(),
                "total_task_inference_ms": self._task_compute_ms,
                "total_meta_inference_ms": self._meta_compute_ms,
                "total_compute_ms": self._task_compute_ms + self._meta_compute_ms,
                "final_prediction_error": self._last_prediction_error,
                "final_map_source_error_m": self._last_map_source_error_m,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        self._finished = True

    def close(self) -> None:
        """Close the run file."""

        if not self._closed:
            self._stream.close()
            self._closed = True

    def _base_row(self, event: str) -> dict[str, Any]:
        return {
            "event": event,
            "run_id": self.run_id,
            "run_started_utc": self.started_at.isoformat(),
            "method": self.method,
            "meta_inference_enabled": self.meta_inference_enabled,
            "fixed_resolution": (
                "" if self.fixed_resolution is None else self.fixed_resolution
            ),
            "cpu_condition": self.cpu_condition,
            "run_label": self.run_label,
            "success_distance_m": self.success_distance_m,
            "source_x_m": self.source_x,
            "source_y_m": self.source_y,
        }

    def _write(self, event: str, values: dict[str, Any]) -> None:
        row = self._base_row(event)
        row.update(values)
        self._writer.writerow(row)
        self._stream.flush()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The experiment run logger is closed.")
        if self._finished:
            raise RuntimeError("The experiment run has already been finalized.")
