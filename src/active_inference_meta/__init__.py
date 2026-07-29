"""Hierarchical active inference for adaptive model representation."""

from .adaptive_navigation import (
    AdaptiveNavigationConfig,
    AdaptiveNavigationResult,
    AdaptiveNavigationStep,
    baseline_compute_availability,
    infer_adaptive_task_policies,
    policy_averaged_rssi_surprise,
    rebuild_navigation_agent,
    remap_spatial_belief,
    run_adaptive_navigation_episode,
)
from .compute import (
    CpuTimesSnapshot,
    ExternalCpuAvailabilitySource,
    FixedComputeResourceSource,
    PsutilComputeResourceSource,
    external_cpu_utilization,
)
from .controller import MetaInferenceConfig, MetaInferenceController
from .experiment import load_trace, run_meta_trace
from .interfaces import ComputeResourceSource, MetaObservationSource
from .likelihood import MetaLikelihood, MetaLikelihoodParameters
from .models import (
    ComputeResourceObservation,
    MetaObservation,
    MetaObservationBounds,
    ModelResolution,
    TaskInferenceMetrics,
)
from .observations import MetaDecision, MetaObservationBuilder
from .policy import AdaptiveNavigationPolicy
from .runtime import run_adaptive_simulation_runtime

__all__ = [
    "AdaptiveNavigationConfig",
    "AdaptiveNavigationPolicy",
    "AdaptiveNavigationResult",
    "AdaptiveNavigationStep",
    "ComputeResourceObservation",
    "ComputeResourceSource",
    "CpuTimesSnapshot",
    "ExternalCpuAvailabilitySource",
    "FixedComputeResourceSource",
    "MetaDecision",
    "MetaInferenceConfig",
    "MetaInferenceController",
    "MetaLikelihood",
    "MetaLikelihoodParameters",
    "MetaObservation",
    "MetaObservationBounds",
    "MetaObservationBuilder",
    "MetaObservationSource",
    "ModelResolution",
    "PsutilComputeResourceSource",
    "TaskInferenceMetrics",
    "baseline_compute_availability",
    "external_cpu_utilization",
    "infer_adaptive_task_policies",
    "load_trace",
    "policy_averaged_rssi_surprise",
    "rebuild_navigation_agent",
    "remap_spatial_belief",
    "run_adaptive_navigation_episode",
    "run_adaptive_simulation_runtime",
    "run_meta_trace",
]
