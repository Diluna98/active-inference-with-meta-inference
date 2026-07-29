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

from .controller import MetaInferenceConfig
from .likelihood import MetaLikelihoodParameters


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
    enabled: bool = True
    fixed_resolution: int | None = None

    def __post_init__(self) -> None:
        if len(self.candidate_resolutions) != 4:
            raise ValueError("Exactly four candidate resolutions are required.")
        if self.initial_resolution not in self.candidate_resolutions:
            raise ValueError("initial_resolution must be a candidate resolution.")
        if self.meta_interval < 1:
            raise ValueError("meta_interval must be positive.")
        if not self.enabled:
            if self.fixed_resolution is None:
                raise ValueError(
                    "fixed_resolution is required when meta-inference is disabled."
                )
            if self.fixed_resolution not in self.candidate_resolutions:
                raise ValueError("fixed_resolution must be a candidate resolution.")

    @property
    def task_resolution(self) -> int:
        """Return the initial or explicitly fixed task-model resolution."""

        return (
            self.initial_resolution
            if self.enabled
            else int(self.fixed_resolution)
        )


@dataclass(frozen=True)
class ProfilingConfig:
    """Incremental meta-observation logging settings."""

    enabled: bool = False
    output: Path = Path("meta_profile.csv")

    def __post_init__(self) -> None:
        if self.enabled and not str(self.output):
            raise ValueError("A profiling output path is required.")


@dataclass(frozen=True)
class VisualizationConfig:
    """Asynchronous runtime display settings."""

    enabled: bool = False
    mode: str = "map"
    refresh_steps: int = 1
    clear_terminal: bool = True
    output: Path = Path("goal_belief.png")

    def __post_init__(self) -> None:
        if self.mode not in ("map", "terminal"):
            raise ValueError("visualization.mode must be 'map' or 'terminal'.")
        if self.refresh_steps < 1:
            raise ValueError("visualization.refresh_steps must be positive.")


@dataclass(frozen=True)
class MetaRuntimeConfig:
    """Complete configuration for adaptive real-world navigation."""

    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    meta_agent: MetaInferenceConfig = field(default_factory=MetaInferenceConfig)
    meta_likelihood: MetaLikelihoodParameters = field(
        default_factory=MetaLikelihoodParameters.from_json
    )
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self) -> None:
        if self.profiling.enabled and self.adaptive.enabled:
            raise ValueError(
                "Profiling requires meta_inference.enabled: false "
                "to keep the task resolution fixed."
            )


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
    profiling_data = _section(data, "profiling")
    if "output" in profiling_data:
        profiling_data["output"] = Path(profiling_data["output"])
    visualization_data = _section(data, "visualization")
    if "output" in visualization_data:
        visualization_data["output"] = Path(visualization_data["output"])
    return MetaRuntimeConfig(
        navigation=navigation,
        adaptive=AdaptiveConfig(**adaptive_data),
        compute=ComputeConfig(**_section(data, "compute")),
        meta_agent=MetaInferenceConfig(**_section(data, "meta_agent")),
        meta_likelihood=(
            MetaLikelihoodParameters.from_mapping(_section(data, "meta_likelihood"))
            if "meta_likelihood" in data
            else MetaLikelihoodParameters.from_json()
        ),
        profiling=ProfilingConfig(**profiling_data),
        visualization=VisualizationConfig(**visualization_data),
    )


def load_meta_runtime_config(path: str | Path) -> MetaRuntimeConfig:
    """Load an explicit adaptive-navigation YAML file."""

    with Path(path).open(encoding="utf-8") as stream:
        return _parse(yaml.safe_load(stream) or {})


def load_default_meta_runtime_config() -> MetaRuntimeConfig:
    """Load the adaptive-navigation YAML bundled with the package."""

    resource = files("active_inference_meta.resources").joinpath("meta_navigation.yaml")
    return _parse(yaml.safe_load(resource.read_text(encoding="utf-8")) or {})
