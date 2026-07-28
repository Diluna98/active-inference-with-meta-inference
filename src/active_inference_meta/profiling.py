"""Incremental CSV recording for fixed-resolution calibration runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

from .models import MetaObservation


class CsvMetaObservationLogger:
    """Append task diagnostics and CPU observations to a durable CSV file."""

    fieldnames = (
        "step",
        "resolution",
        "information_gain_proxy",
        "prediction_error",
        "inference_latency_ms",
        "cpu_availability",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists() and self.path.stat().st_size > 0
        self._stream: TextIO = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=self.fieldnames)
        if not existed:
            self._writer.writeheader()
            self._stream.flush()

    def record(
        self,
        step: int,
        resolution: int,
        observation: MetaObservation,
    ) -> None:
        """Write and flush one observation so shutdown does not lose prior rows."""

        self._writer.writerow(
            {
                "step": step,
                "resolution": resolution,
                "information_gain_proxy": observation.information_gain_proxy,
                "prediction_error": observation.prediction_error,
                "inference_latency_ms": observation.inference_latency_ms,
                "cpu_availability": observation.cpu_availability,
            }
        )
        self._stream.flush()

    def close(self) -> None:
        """Close the output file."""

        self._stream.close()
