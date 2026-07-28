"""Typed YAML configuration for adaptive simulation and TurtleBot runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from active_inference_navigation.config import (
    ActiveInferenceConfig,
    ExperimentConfig,
    FrameConfig,
    GridConfig,
    MotionConfig,
    NavigationConfig,
    RssiLikelihoodConfig,
    SensorConfig,
    TerminationConfig,
    TopicConfig,
)


@dataclass(frozen=True)
class ComputeConfig:
    """Processor observation settings."""

    provider: str = "psutil"
    median_window: int = 5
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.provider != "psutil":
            raise ValueError("The initial real-world compute provider must be 'psutil'.")
        if self.median_window < 1 or self.timeout_seconds <= 0.0:
            raise ValueError("Compute window and timeout must be positive.")


@dataclass(frozen=True)
class AdaptiveConfig:
    """Representation-selection settings."""

    initial_resolution: int = 10
    candidate_resolutions: tuple[int, int, int, int] = (2, 5, 10, 20)
    meta_interval: int = 3

    def __post_init__(self) -> None:
        if len(self.candidate_resolutions) != 4:
            raise ValueError("Exactly four candidate resolutions are required.")
        if self.initial_resolution not in self.candidate_resolutions:
            raise ValueError("initial_resolution must be a candidate resolution.")
        if self.meta_interval < 1:
            raise ValueError("meta_interval must be positive.")


@dataclass(frozen=True)
class MetaRuntimeConfig:
    """Complete configuration for adaptive real-world navigation."""

    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{key}' must be a mapping.")
    return value


def _parse(data: Any) -> MetaRuntimeConfig:
    if not isinstance(data, dict):
        raise TypeError("Meta-navigation configuration must be a YAML mapping.")
    navigation = NavigationConfig(
        grid=GridConfig(**_section(data, "grid")),
        frame=FrameConfig(**_section(data, "frame")),
        active_inference=ActiveInferenceConfig(**_section(data, "active_inference")),
        experiment=ExperimentConfig(**_section(data, "experiment")),
        termination=TerminationConfig(**_section(data, "termination")),
        topics=TopicConfig(**_section(data, "topics")),
        sensors=SensorConfig(**_section(data, "sensors")),
        motion=MotionConfig(**_section(data, "motion")),
        rssi_likelihood=RssiLikelihoodConfig(**_section(data, "rssi_likelihood")),
        likelihood_provider=str(data.get("likelihood_provider", "calibrated_dbm")),
    )
    adaptive_data = _section(data, "meta_inference")
    if "candidate_resolutions" in adaptive_data:
        adaptive_data["candidate_resolutions"] = tuple(
            adaptive_data["candidate_resolutions"]
        )
    return MetaRuntimeConfig(
        navigation=navigation,
        adaptive=AdaptiveConfig(**adaptive_data),
        compute=ComputeConfig(**_section(data, "compute")),
    )


def load_meta_runtime_config(path: str | Path) -> MetaRuntimeConfig:
    """Load an explicit adaptive-navigation YAML file."""

    with Path(path).open(encoding="utf-8") as stream:
        return _parse(yaml.safe_load(stream) or {})


def load_default_meta_runtime_config() -> MetaRuntimeConfig:
    """Load the adaptive-navigation YAML bundled with the package."""

    resource = files("active_inference_meta.resources").joinpath("meta_navigation.yaml")
    return _parse(yaml.safe_load(resource.read_text(encoding="utf-8")) or {})
