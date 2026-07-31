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
from .likelihood import MetaLikelihoodParameters, MetaPreferenceParameters
from .models import MetaObservationBounds


@dataclass(frozen=True)
class ComputeConfig:
    """External processor-availability observation settings."""

    provider: str = "external_psutil"
    median_window: int = 5
    timeout_seconds: float = 2.0
    sample_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.provider not in {"external_psutil", "psutil"}:
            raise ValueError(
                "The real-world compute provider must be 'external_psutil'."
            )
        if (
            self.median_window < 1
            or self.timeout_seconds <= 0.0
            or self.sample_interval_seconds <= 0.0
        ):
            raise ValueError("Compute window, timeout, and sample interval must be positive.")


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
class ExperimentLoggingConfig:
    """Per-run evaluation logging for fixed and adaptive robot experiments."""

    enabled: bool = False
    output_directory: Path = Path("experiment_runs")
    filename_prefix: str = "robot_run"
    run_label: str = ""
    cpu_condition: str = "uncontrolled"
    success_distance_m: float = 0.5
    source_x: float | None = None
    source_y: float | None = None

    def __post_init__(self) -> None:
        if self.enabled and not str(self.output_directory):
            raise ValueError("An experiment-log output directory is required.")
        if not self.filename_prefix.strip():
            raise ValueError("The experiment-log filename prefix must not be empty.")
        if self.cpu_condition not in {"low", "medium", "high", "uncontrolled"}:
            raise ValueError(
                "cpu_condition must be low, medium, high, or uncontrolled."
            )
        if self.success_distance_m <= 0.0:
            raise ValueError("success_distance_m must be positive.")
        if (self.source_x is None) != (self.source_y is None):
            raise ValueError("Experiment source_x and source_y must be set together.")


@dataclass(frozen=True)
class VisualizationConfig:
    """Asynchronous runtime display settings."""

    enabled: bool = False
    mode: str = "map"
    refresh_steps: int = 1
    clear_terminal: bool = True
    output: Path = Path("goal_belief.png")
    host: str = "0.0.0.0"
    port: int = 8000
    history_limit: int = 500
    ground_truth_source_x: float | None = None
    ground_truth_source_y: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("dashboard", "map", "terminal"):
            raise ValueError(
                "visualization.mode must be 'dashboard', 'map', or 'terminal'."
            )
        if self.refresh_steps < 1:
            raise ValueError("visualization.refresh_steps must be positive.")
        if not self.host.strip():
            raise ValueError("visualization.host must not be empty.")
        if not 0 <= self.port <= 65535:
            raise ValueError("visualization.port must lie between 0 and 65535.")
        if self.history_limit < 1:
            raise ValueError("visualization.history_limit must be positive.")
        if (self.ground_truth_source_x is None) != (
            self.ground_truth_source_y is None
        ):
            raise ValueError(
                "Both ground-truth source coordinates must be set together."
            )


@dataclass(frozen=True)
class MetaLearningCheckpointConfig:
    """Persistence settings for learned continuous likelihood parameters."""

    checkpoint: Path = Path("learned_meta_likelihood.yaml")
    load_if_available: bool = True
    save_on_exit: bool = True

    def __post_init__(self) -> None:
        if not str(self.checkpoint):
            raise ValueError("A meta-learning checkpoint path is required.")


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
    meta_observation_bounds: MetaObservationBounds = field(
        default_factory=MetaObservationBounds
    )
    meta_preferences: MetaPreferenceParameters = field(
        default_factory=MetaPreferenceParameters
    )
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    experiment_logging: ExperimentLoggingConfig = field(
        default_factory=ExperimentLoggingConfig
    )
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    meta_learning: MetaLearningCheckpointConfig = field(
        default_factory=MetaLearningCheckpointConfig
    )

    def __post_init__(self) -> None:
        if self.profiling.enabled and self.adaptive.enabled:
            raise ValueError(
                "Profiling requires meta_inference.enabled: false "
                "to keep the task resolution fixed."
            )
        if self.experiment_logging.enabled:
            source_is_explicit = self.experiment_logging.source_x is not None
            source_is_termination = self.navigation.termination.provider == "source_distance"
            if not source_is_explicit and not source_is_termination:
                raise ValueError(
                    "Experiment logging requires source coordinates either in "
                    "experiment_logging or source_distance termination."
                )


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{key}' must be a mapping.")
    return value


def _parse(data: Any, *, base_directory: Path | None = None) -> MetaRuntimeConfig:
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
    experiment_logging_data = _section(data, "experiment_logging")
    if "output_directory" in experiment_logging_data:
        experiment_logging_data["output_directory"] = Path(
            experiment_logging_data["output_directory"]
        )
    visualization_data = _section(data, "visualization")
    if "output" in visualization_data:
        visualization_data["output"] = Path(visualization_data["output"])
    meta_learning_data = _section(data, "meta_learning")
    checkpoint = Path(
        meta_learning_data.get("checkpoint", "learned_meta_likelihood.yaml")
    )
    if not checkpoint.is_absolute() and base_directory is not None:
        checkpoint = base_directory / checkpoint
    meta_learning_data["checkpoint"] = checkpoint
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
        meta_observation_bounds=MetaObservationBounds.from_mapping(
            _section(data, "meta_observation_bounds")
        ),
        meta_preferences=MetaPreferenceParameters.from_mapping(
            _section(data, "meta_preferences")
        ),
        profiling=ProfilingConfig(**profiling_data),
        experiment_logging=ExperimentLoggingConfig(**experiment_logging_data),
        visualization=VisualizationConfig(**visualization_data),
        meta_learning=MetaLearningCheckpointConfig(**meta_learning_data),
    )


def load_meta_runtime_config(path: str | Path) -> MetaRuntimeConfig:
    """Load an explicit adaptive-navigation YAML file."""

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        return _parse(
            yaml.safe_load(stream) or {},
            base_directory=config_path.parent,
        )


def load_default_meta_runtime_config() -> MetaRuntimeConfig:
    """Load the adaptive-navigation YAML bundled with the package."""

    resource = files("active_inference_meta.resources").joinpath("meta_navigation.yaml")
    return _parse(
        yaml.safe_load(resource.read_text(encoding="utf-8")) or {},
        base_directory=Path.cwd(),
    )
