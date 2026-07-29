"""Non-blocking terminal visualization of the task-level source posterior."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread
from typing import TextIO

import numpy as np


@dataclass(frozen=True)
class BeliefSnapshot:
    """One immutable task-level source-belief frame."""

    step: int
    resolution: int
    belief: np.ndarray
    robot_x: float
    robot_y: float

    def __post_init__(self) -> None:
        values = np.asarray(self.belief, dtype=float)
        if values.shape != (self.resolution**2,):
            raise ValueError("Belief size must match the square task resolution.")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Belief probabilities must be finite and nonnegative.")
        if values.sum() <= 0.0:
            raise ValueError("Belief probabilities must have positive mass.")
        if not np.all(np.isfinite([self.robot_x, self.robot_y])):
            raise ValueError("Robot position must be finite.")
        object.__setattr__(self, "belief", values.copy() / values.sum())


def render_belief_heatmap(snapshot: BeliefSnapshot) -> str:
    """Render a compact probability heatmap suitable for an SSH terminal."""

    grid = snapshot.belief.reshape(snapshot.resolution, snapshot.resolution)
    peak_index = int(np.argmax(snapshot.belief))
    peak_y, peak_x = divmod(peak_index, snapshot.resolution)
    peak_probability = float(snapshot.belief[peak_index])
    symbols = " .:-=+*#%@"
    scaled = grid / max(float(grid.max()), np.finfo(float).eps)
    rows = []
    for y in reversed(range(snapshot.resolution)):
        row = "".join(
            symbols[min(int(value * (len(symbols) - 1)), len(symbols) - 1)]
            for value in scaled[y]
        )
        rows.append(row)
    header = (
        f"GOAL BELIEF | step={snapshot.step} | "
        f"resolution={snapshot.resolution}x{snapshot.resolution} | "
        f"MAP cell=({peak_x}, {peak_y}) | p={peak_probability:.4f}"
    )
    return "\n".join((header, *rows))


class AsyncTerminalBeliefVisualizer:
    """Render only the newest queued belief on a background thread."""

    def __init__(
        self,
        *,
        refresh_steps: int = 1,
        clear_terminal: bool = True,
        output: TextIO | None = None,
    ) -> None:
        if refresh_steps < 1:
            raise ValueError("refresh_steps must be positive.")
        self.refresh_steps = refresh_steps
        self.clear_terminal = clear_terminal
        self.output = output or sys.stdout
        self._queue: Queue[BeliefSnapshot | None] = Queue(maxsize=1)
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(
        self,
        step: int,
        resolution: int,
        belief: np.ndarray,
        robot_x: float,
        robot_y: float,
    ) -> None:
        """Queue a snapshot without ever blocking the inference thread."""

        if step % self.refresh_steps != 0:
            return
        snapshot = BeliefSnapshot(step, resolution, belief, robot_x, robot_y)
        try:
            self._queue.put_nowait(snapshot)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(snapshot)
            except Full:
                pass

    def close(self) -> None:
        """Finish pending rendering and stop the worker."""

        self._queue.put(None)
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            snapshot = self._queue.get()
            if snapshot is None:
                return
            if self.clear_terminal:
                self.output.write("\033[2J\033[H")
            self.output.write(render_belief_heatmap(snapshot))
            self.output.write("\n")
            self.output.flush()


class AsyncMapBeliefVisualizer:
    """Update a metric arena heatmap on a background thread."""

    def __init__(
        self,
        output: str | Path,
        *,
        arena_width: float,
        arena_height: float,
        refresh_steps: int = 1,
    ) -> None:
        if arena_width <= 0.0 or arena_height <= 0.0:
            raise ValueError("Arena dimensions must be positive.")
        if refresh_steps < 1:
            raise ValueError("refresh_steps must be positive.")
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.arena_width = float(arena_width)
        self.arena_height = float(arena_height)
        self.refresh_steps = refresh_steps
        self._queue: Queue[BeliefSnapshot | None] = Queue(maxsize=1)
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(
        self,
        step: int,
        resolution: int,
        belief: np.ndarray,
        robot_x: float,
        robot_y: float,
    ) -> None:
        """Queue the newest metric-map frame without blocking inference."""

        if step % self.refresh_steps != 0:
            return
        snapshot = BeliefSnapshot(step, resolution, belief, robot_x, robot_y)
        try:
            self._queue.put_nowait(snapshot)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(snapshot)
            except Full:
                pass

    def close(self) -> None:
        """Render any accepted frame and stop the worker."""

        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            snapshot = self._queue.get()
            if snapshot is None:
                return
            self._render(snapshot)

    def _render(self, snapshot: BeliefSnapshot) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        grid = snapshot.belief.reshape(snapshot.resolution, snapshot.resolution)
        peak_index = int(np.argmax(snapshot.belief))
        peak_y, peak_x = divmod(peak_index, snapshot.resolution)
        goal_x = (peak_x + 0.5) * self.arena_width / snapshot.resolution
        goal_y = (peak_y + 0.5) * self.arena_height / snapshot.resolution

        figure = Figure(figsize=(7.2, 6.4), constrained_layout=True)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(1, 1, 1)
        image = axis.imshow(
            grid,
            origin="lower",
            extent=(0.0, self.arena_width, 0.0, self.arena_height),
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=max(float(grid.max()), np.finfo(float).eps),
        )
        ticks_x = np.linspace(0.0, self.arena_width, snapshot.resolution + 1)
        ticks_y = np.linspace(0.0, self.arena_height, snapshot.resolution + 1)
        axis.set_xticks(ticks_x, minor=True)
        axis.set_yticks(ticks_y, minor=True)
        axis.grid(which="minor", color="white", alpha=0.24, linewidth=0.55)
        axis.scatter(
            snapshot.robot_x,
            snapshot.robot_y,
            marker="o",
            s=80,
            color="#00b7ff",
            edgecolor="white",
            linewidth=1.2,
            label="robot",
        )
        axis.scatter(
            goal_x,
            goal_y,
            marker="*",
            s=190,
            color="#42ff8c",
            edgecolor="black",
            linewidth=0.8,
            label="MAP goal",
        )
        axis.set_xlim(0.0, self.arena_width)
        axis.set_ylim(0.0, self.arena_height)
        axis.set_aspect("equal")
        axis.set_xlabel("Arena x (m)")
        axis.set_ylabel("Arena y (m)")
        axis.set_title(
            f"Task-level goal belief | step {snapshot.step} | "
            f"{snapshot.resolution}x{snapshot.resolution}\n"
            f"MAP=({goal_x:.3f}, {goal_y:.3f}) m | "
            f"p={snapshot.belief[peak_index]:.4f}"
        )
        axis.legend(loc="upper right")
        figure.colorbar(image, ax=axis, label="Goal-state probability")

        temporary = self.output.with_name(
            f".{self.output.stem}.tmp{self.output.suffix}"
        )
        canvas.print_figure(temporary, dpi=120)
        temporary.replace(self.output)
