import json
import time
from urllib.request import urlopen

import numpy as np

from active_inference_meta.dashboard import LiveDashboardServer
from active_inference_meta.models import MetaObservation, TaskTelemetry


def telemetry(step=3):
    return TaskTelemetry(
        step=step,
        resolution=2,
        belief=np.asarray([0.1, 0.2, 0.6, 0.1]),
        robot_x=1.225,
        robot_y=2.275,
        rssi=-61.0,
        selected_action=(2, 0),
        inference_latency_ms=125.0,
        meta_observation=MetaObservation(0.08, 24.0, 125.0, 82.0),
        meta_observation_step=3,
        selected_resolution=5,
        model_switched=True,
    )


def test_dashboard_serves_browser_and_latest_telemetry():
    server = LiveDashboardServer(
        arena_width=7.0,
        arena_height=7.0,
        host="127.0.0.1",
        port=0,
        ground_truth_source=(2.975, 4.375),
    )
    try:
        server.submit(telemetry())
        deadline = time.monotonic() + 2.0
        state = None
        while time.monotonic() < deadline:
            with urlopen(f"{server.url}api/state", timeout=1.0) as response:
                state = json.load(response)
            if state["telemetry"] is not None:
                break
            time.sleep(0.01)

        assert state is not None
        assert state["status"] == "running"
        assert state["telemetry"]["resolution"] == 2
        assert state["telemetry"]["selected_action"] == [2, 0]
        assert state["telemetry"]["meta_observation"]["cpu_availability"] == 82.0
        assert state["ground_truth_source"] == {"x": 2.975, "y": 4.375}
        assert state["path"] == [{"x": 1.225, "y": 2.275}]

        with urlopen(server.url, timeout=1.0) as response:
            html = response.read().decode("utf-8")
        assert "Active Inference Navigation" in html
        assert 'fetch("/api/state"' in html
        assert 'let actionLabel = "NO ACTION"' in html
        assert "telemetry.meta_decision?.switched" in html
    finally:
        server.close()


def test_dashboard_marks_runtime_outcome():
    server = LiveDashboardServer(
        arena_width=7.0,
        arena_height=7.0,
        host="127.0.0.1",
        port=0,
    )
    try:
        server.submit(telemetry())
        deadline = time.monotonic() + 2.0
        while server.snapshot()["telemetry"] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        server.mark_complete(
            terminated=True,
            robot_x=2.625,
            robot_y=4.375,
            rssi=-52.0,
        )
        state = server.snapshot()
        assert state["status"] == "goal reached"
        assert state["telemetry"]["robot"]["x"] == 2.625
        assert state["telemetry"]["rssi_dbm"] == -52.0
        assert state["path"][-1] == {"x": 2.625, "y": 4.375}
    finally:
        server.close()
