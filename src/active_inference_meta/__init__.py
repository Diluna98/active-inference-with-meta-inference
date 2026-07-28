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
from .compute import FixedComputeResourceSource, PsutilComputeResourceSource
from .controller import MetaInferenceConfig, MetaInferenceController
from .experiment import load_trace, run_meta_trace
from .interfaces import ComputeResourceSource, MetaObservationSource
from .likelihood import MetaLikelihood, MetaLikelihoodParameters
from .models import (
    ComputeResourceObservation,
    MetaObservation,
    ModelResolution,
    TaskInferenceMetrics,
)
from .observations import MetaDecision, MetaObservationBuilder
from .policy import AdaptiveNavigationPolicy

__all__ = [
    "AdaptiveNavigationConfig",
    "AdaptiveNavigationPolicy",
    "AdaptiveNavigationResult",
    "AdaptiveNavigationStep",
    "ComputeResourceObservation",
    "ComputeResourceSource",
    "FixedComputeResourceSource",
    "MetaDecision",
    "MetaInferenceConfig",
    "MetaInferenceController",
    "MetaLikelihood",
    "MetaLikelihoodParameters",
    "MetaObservation",
    "MetaObservationBuilder",
    "MetaObservationSource",
    "ModelResolution",
    "PsutilComputeResourceSource",
    "TaskInferenceMetrics",
    "baseline_compute_availability",
    "infer_adaptive_task_policies",
    "load_trace",
    "policy_averaged_rssi_surprise",
    "rebuild_navigation_agent",
    "remap_spatial_belief",
    "run_adaptive_navigation_episode",
    "run_meta_trace",
]
