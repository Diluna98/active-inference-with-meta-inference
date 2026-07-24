"""Hierarchical active inference for adaptive model representation."""

from .adaptive_navigation import (
    AdaptiveNavigationConfig,
    AdaptiveNavigationResult,
    AdaptiveNavigationStep,
    infer_paper_task_policies,
    paper_cpu_availability,
    paper_policy_averaged_surprise,
    rebuild_navigation_agent,
    remap_spatial_belief,
    run_adaptive_navigation_episode,
)
from .controller import MetaInferenceConfig, MetaInferenceController
from .experiment import load_trace, run_meta_trace
from .likelihood import MetaLikelihood, MetaLikelihoodParameters
from .observations import MetaDecision, MetaObservation

__all__ = [
    "AdaptiveNavigationConfig",
    "AdaptiveNavigationResult",
    "AdaptiveNavigationStep",
    "MetaDecision",
    "MetaInferenceConfig",
    "MetaInferenceController",
    "MetaLikelihood",
    "MetaLikelihoodParameters",
    "MetaObservation",
    "infer_paper_task_policies",
    "load_trace",
    "paper_cpu_availability",
    "paper_policy_averaged_surprise",
    "rebuild_navigation_agent",
    "remap_spatial_belief",
    "run_adaptive_navigation_episode",
    "run_meta_trace",
]
