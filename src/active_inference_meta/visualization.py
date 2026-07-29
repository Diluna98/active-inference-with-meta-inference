"""Non-blocking terminal visualization of the task-level source posterior."""

from __future__ import annotations

import sys
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        values = np.asarray(self.belief, dtype=float)
        if values.shape != (self.resolution**2,):
            raise ValueError("Belief size must match the square task resolution.")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Belief probabilities must be finite and nonnegative.")
        if values.sum() <= 0.0:
            raise ValueError("Belief probabilities must have positive mass.")
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

    def submit(self, step: int, resolution: int, belief: np.ndarray) -> None:
        """Queue a snapshot without ever blocking the inference thread."""

        if step % self.refresh_steps != 0:
            return
        snapshot = BeliefSnapshot(step, resolution, belief)
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
