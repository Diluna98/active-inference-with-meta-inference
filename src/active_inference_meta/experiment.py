"""Run meta-inference over task-level observation traces."""

from __future__ import annotations

import json
from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path

from .controller import MetaInferenceController
from .observations import MetaDecision, MetaObservation


def load_trace(path: str | Path | None = None) -> list[MetaObservation]:
    """Load a JSON list of task-level meta observations."""

    if path is None:
        resource = files("active_inference_meta").joinpath("data/example_trace.json")
        records = json.loads(resource.read_text(encoding="utf-8"))
    else:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("A trace must contain a non-empty JSON list.")
    return [MetaObservation(**record) for record in records]


def run_meta_trace(
    observations: Iterable[MetaObservation],
    *,
    initial_resolution: int = 2,
    controller: MetaInferenceController | None = None,
) -> list[MetaDecision]:
    """Select representations sequentially for a task observation trace."""

    active_controller = controller or MetaInferenceController()
    active_controller.reset()
    return active_controller.infer_sequence(initial_resolution, observations)


def decisions_to_records(
    observations: Iterable[MetaObservation],
    decisions: Iterable[MetaDecision],
) -> list[dict]:
    records = []
    for step, (observation, decision) in enumerate(
        zip(observations, decisions, strict=True)
    ):
        records.append(
            {
                "step": step,
                "observation": observation.as_array().tolist(),
                "action_index": decision.action_index,
                "selected_resolution": decision.selected_resolution,
                "switched": decision.switched,
                "policy_posterior": decision.policy_posterior.tolist(),
                "expected_free_energy": decision.expected_free_energy.tolist(),
                "risk": decision.risk.tolist(),
                "ambiguity": decision.ambiguity.tolist(),
            }
        )
    return records
