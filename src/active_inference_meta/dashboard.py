"""Non-blocking telemetry server for the browser-rendered robot dashboard."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import time
from typing import Any

from .models import TaskTelemetry


class LiveDashboardServer:
    """Serve the newest telemetry while keeping inference submission non-blocking."""

    def __init__(
        self,
        *,
        arena_width: float,
        arena_height: float,
        host: str = "0.0.0.0",
        port: int = 8000,
        refresh_steps: int = 1,
        history_limit: int = 500,
        ground_truth_source: tuple[float, float] | None = None,
    ) -> None:
        if arena_width <= 0.0 or arena_height <= 0.0:
            raise ValueError("Dashboard arena dimensions must be positive.")
        if not host.strip():
            raise ValueError("Dashboard host must not be empty.")
        if not 0 <= port <= 65535:
            raise ValueError("Dashboard port must lie between 0 and 65535.")
        if refresh_steps < 1 or history_limit < 1:
            raise ValueError("Dashboard refresh and history limits must be positive.")
        if ground_truth_source is not None:
            source_x, source_y = ground_truth_source
            if not 0.0 <= source_x <= arena_width or not 0.0 <= source_y <= arena_height:
                raise ValueError("Dashboard ground truth must lie inside the arena.")

        self.arena_width = float(arena_width)
        self.arena_height = float(arena_height)
        self.host = host
        self.refresh_steps = refresh_steps
        self.history_limit = history_limit
        self._queue: Queue[TaskTelemetry | None] = Queue(maxsize=1)
        self._lock = Lock()
        self._state: dict[str, Any] = {
            "status": "waiting",
            "arena": {"width": self.arena_width, "height": self.arena_height},
            "ground_truth_source": (
                None
                if ground_truth_source is None
                else {"x": ground_truth_source[0], "y": ground_truth_source[1]}
            ),
            "path": [],
            "telemetry": None,
            "updated_at_unix_s": None,
        }
        self._html = files("active_inference_meta.resources").joinpath(
            "dashboard.html"
        ).read_bytes()
        handler = self._handler_type()
        self._server = ThreadingHTTPServer((host, port), handler)
        self.port = int(self._server.server_address[1])
        self._worker = Thread(target=self._run_worker, daemon=True)
        self._server_thread = Thread(target=self._server.serve_forever, daemon=True)
        self._worker.start()
        self._server_thread.start()

    @property
    def url(self) -> str:
        """Return a locally usable URL; remote clients should substitute the robot IP."""

        display_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{display_host}:{self.port}/"

    def submit(self, telemetry: TaskTelemetry) -> None:
        """Queue only the newest selected step without blocking inference."""

        if telemetry.step % self.refresh_steps != 0:
            return
        try:
            self._queue.put_nowait(telemetry)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(telemetry)
            except Full:
                pass

    def mark_complete(self, *, terminated: bool) -> None:
        """Expose the final runtime outcome to connected browsers."""

        with self._lock:
            self._state["status"] = "goal reached" if terminated else "planning limit"
            self._state["updated_at_unix_s"] = time()

    def snapshot(self) -> dict[str, Any]:
        """Return an isolated JSON-compatible state snapshot."""

        with self._lock:
            return json.loads(json.dumps(self._state))

    def close(self) -> None:
        """Stop background threads without waiting on a full telemetry queue."""

        try:
            self._queue.put_nowait(None)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
        self._worker.join(timeout=2.0)
        self._server.shutdown()
        self._server.server_close()
        self._server_thread.join(timeout=2.0)

    def _run_worker(self) -> None:
        while True:
            telemetry = self._queue.get()
            if telemetry is None:
                return
            self._apply(telemetry)

    def _apply(self, telemetry: TaskTelemetry) -> None:
        meta = telemetry.meta_observation
        decision = None
        if telemetry.selected_resolution is not None:
            decision = {
                "selected_resolution": telemetry.selected_resolution,
                "switched": telemetry.model_switched,
            }
        payload = {
            "step": telemetry.step,
            "resolution": telemetry.resolution,
            "belief": telemetry.belief.tolist(),
            "robot": {
                "x": telemetry.robot_x,
                "y": telemetry.robot_y,
                "heading": "positive_x",
            },
            "rssi_dbm": telemetry.rssi,
            "selected_action": (
                None
                if telemetry.selected_action is None
                else list(telemetry.selected_action)
            ),
            "inference_latency_ms": telemetry.inference_latency_ms,
            "meta_observation_step": telemetry.meta_observation_step,
            "meta_observation": (
                None
                if meta is None
                else {
                    "information_gain_proxy": meta.information_gain_proxy,
                    "prediction_error": meta.prediction_error,
                    "inference_latency_ms": meta.inference_latency_ms,
                    "cpu_availability": meta.cpu_availability,
                }
            ),
            "meta_decision": decision,
        }
        with self._lock:
            path = self._state["path"]
            point = {"x": telemetry.robot_x, "y": telemetry.robot_y}
            if not path or path[-1] != point:
                path.append(point)
                del path[:-self.history_limit]
            self._state["status"] = "running"
            self._state["telemetry"] = payload
            self._state["updated_at_unix_s"] = time()

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        dashboard = self

        class DashboardHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in {"/", "/index.html"}:
                    self._write(HTTPStatus.OK, "text/html; charset=utf-8", dashboard._html)
                    return
                if self.path == "/api/state":
                    body = json.dumps(
                        dashboard.snapshot(),
                        separators=(",", ":"),
                    ).encode("utf-8")
                    self._write(HTTPStatus.OK, "application/json", body)
                    return
                if self.path == "/health":
                    self._write(HTTPStatus.OK, "text/plain; charset=utf-8", b"ok\n")
                    return
                self._write(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")

            def _write(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return DashboardHandler
