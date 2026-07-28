"""Protocols that isolate meta-inference from resource-monitor implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import ComputeResourceObservation, MetaObservation


@runtime_checkable
class ComputeResourceSource(Protocol):
    """Provide processor availability without exposing platform-specific APIs."""

    def read(self) -> ComputeResourceObservation:
        """Return the latest fresh processor-availability observation."""


@runtime_checkable
class MetaObservationSource(Protocol):
    """Provide one complete, hardware-independent meta observation."""

    def read(self) -> MetaObservation:
        """Return the signals required for one meta-inference update."""
